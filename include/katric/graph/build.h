// Building the graphs of graph/graph.h: the input graph from KaGen's CSR, the
// exchange of per-vertex values to the ghosts, the orientation, and the way
// back to KaGen's CSR for load balancing.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

#include <kagen.h>
#include <mpi.h>
#include <dstl/distribution.hpp>
#include <kassert/kassert.hpp>

#include "katric/graph/graph.h"

namespace katric::graph {

/// The undirected graph of a rank's part of the input (KaGen's CSR: vertex
/// range, xadj, adjncy in global ids), with ghosts. Collective (builds the
/// partition table). Neighborhoods keep the input's order. The cut edges are
/// sorted once by (head, position): that gives the ghosts in global-id order,
/// and each run of equal heads one ghost, so their handles need no lookup per
/// edge.
inline auto build_input_graph(kagen::Graph const& input,
                              kamping::v2::comm_view comm) -> Graph {
  KASSERT(input.representation == kagen::GraphRepresentation::CSR);
  auto const [first, last] = input.vertex_range;
  // an empty range may have any first and last
  LocalNumbering local(
      dstl::distribution(first < last ? last - first : 0, comm), comm);
  KASSERT(first >= last || local.local_range().first == first,
          "the vertex ranges are not contiguous in rank order");
  std::vector<VertexHandle> adjncy(input.adjncy.size());
  std::vector<std::pair<VertexId, std::size_t>>
      cut;  // (head, position in adjncy), sorted
  for (std::size_t position = 0; position < input.adjncy.size(); ++position) {
    VertexId const gid = input.adjncy[position];
    if (local.is_local_id(gid)) {
      adjncy[position] = local.handle(gid);
    } else {
      cut.emplace_back(gid, position);
    }
  }
  std::ranges::sort(cut);
  std::vector<VertexId> ghost_ids;
  std::ranges::unique_copy(cut | std::views::keys,
                           std::back_inserter(ghost_ids));
  VertexNumbering numbering(std::move(local), std::move(ghost_ids));
  std::size_t ghost_index = 0;
  for (std::size_t i = 0; i < cut.size(); ++i) {
    if (i > 0 && cut[i].first != cut[i - 1].first) {
      ++ghost_index;  // the next run: the next ghost
    }
    adjncy[cut[i].second] = numbering.ghost_handle(ghost_index);
  }
  decltype(cut){}.swap(cut);  // frees the cut pairs before the copy of xadj
  // copied: the input is only read (the harness reuses it for every iteration),
  // and KaGen's SInt need not be std::size_t (unsigned long long vs. unsigned
  // long)
  std::vector<std::size_t> xadj(input.xadj.begin(), input.xadj.end());
  if (xadj.empty()) {
    xadj.push_back(0);
  }
  KASSERT(xadj.size() == numbering.num_local_vertices() + 1);
  return {std::move(numbering), {std::move(xadj), std::move(adjncy)}};
}

/// `G` in KaGen's CSR representation (global ids). Consumes `G`: its arrays
/// are freed on return.
inline auto to_kagen_graph(Graph&& G) -> kagen::Graph {
  auto const [numbering, local] = std::move(G).release();
  kagen::Graph input;
  input.representation = kagen::GraphRepresentation::CSR;
  input.vertex_range = numbering.local_range();
  input.xadj.reserve(local.num_vertices() + 1);
  input.adjncy.reserve(local.num_edges());
  input.xadj.push_back(0);
  for (std::size_t index = 0; index < local.num_vertices(); ++index) {
    for (VertexHandle const u : local.neighbors(index)) {
      input.adjncy.push_back(numbering.global_id(u));
    }
    input.xadj.push_back(input.adjncy.size());
  }
  return input;
}

/// Orients the edges of the undirected graph `G` from the smaller to the larger
/// `key` and returns the out-edges, neighborhoods in no particular order.
/// Consumes `G`: each neighborhood is partitioned in place (out-neighbors
/// first) and cut to its out-neighbors, so the oriented graph keeps `G`'s
/// arrays (their capacity too) and numbering. `key` is evaluated once per
/// vertex and once per edge; it may read `G`'s degrees and numbering while the
/// orientation runs, not its neighborhoods. With `ghost_out_edges`, also the
/// ghosts' out-edges to local vertices (CETRIC), sorted.
template <typename Key>
auto orient(Graph&& G, Key const& key, bool ghost_out_edges) -> Graph {
  std::vector<std::size_t> out_degree(G.num_local_vertices());
  for (VertexHandle const v : G.local_vertices()) {
    auto const key_v = key(v);
    auto const neighbors = G.mutable_neighbors(v);
    auto const in_begin =
        std::partition(neighbors.begin(), neighbors.end(),
                       [&](VertexHandle u) -> bool { return key_v < key(u); });
    out_degree[G.to_local_index(v)] =
        static_cast<std::size_t>(in_begin - neighbors.begin());
  }
  // the ghosts' out-edges: a local vertex's in-neighbors that are ghosts
  std::optional<AdjacencyArray<VertexHandle>> ghost_out;
  if (ghost_out_edges) {
    auto const& numbering = G.numbering();
    auto const in_neighbors = [&](VertexHandle v) {
      return G.local_neighbors(v).subspan(out_degree[G.to_local_index(v)]);
    };
    std::vector<std::size_t> ghost_xadj(numbering.num_ghosts() + 1, 0);
    for (VertexHandle const v : G.local_vertices()) {
      for (VertexHandle const u : in_neighbors(v)) {
        if (numbering.is_ghost(u)) {
          ghost_xadj[numbering.to_ghost_index(u) + 1]++;
        }
      }
    }
    for (std::size_t index = 0; index + 1 < ghost_xadj.size(); ++index) {
      ghost_xadj[index + 1] += ghost_xadj[index];
    }
    std::vector<VertexHandle> ghost_adjncy(ghost_xadj.back());
    std::vector<std::size_t> next(ghost_xadj.begin(), ghost_xadj.end() - 1);
    for (VertexHandle const v : G.local_vertices()) {
      for (VertexHandle const u : in_neighbors(v)) {
        if (numbering.is_ghost(u)) {
          // v ascending: each ghost's neighborhood comes out sorted
          ghost_adjncy[next[numbering.to_ghost_index(u)]++] = v;
        }
      }
    }
    ghost_out.emplace(std::move(ghost_xadj), std::move(ghost_adjncy));
  }
  auto [numbering, local] = std::move(G).release();
  local.truncate_neighborhoods(out_degree);
  return {std::move(numbering), std::move(local), std::move(ghost_out)};
}

/// DiTriC: the out-neighborhoods of `G` (sorted by handle) as global ids. A
/// neighborhood sorted by handle is [local vertices][ghosts below the local
/// range][ghosts above]; translated in the order [ghosts below][local
/// vertices][ghosts above] it is ascending. A new array: the handles cannot be
/// reused for global ids.
inline auto to_global_ids(Graph&& G) -> GlobalIdGraph {
  std::vector<VertexId> global;
  global.reserve(G.num_local_edges());
  {
    auto const& numbering = G.numbering();
    auto const translate = [&](auto first, auto last) -> void {
      std::transform(
          first, last, std::back_inserter(global),
          [&](VertexHandle u) -> VertexId { return numbering.global_id(u); });
    };
    for (VertexHandle const v : G.local_vertices()) {
      auto const local_heads = G.local_heads(v);
      auto const ghost_heads = G.ghost_heads(v);
      auto const below_end = std::ranges::lower_bound(
          ghost_heads, numbering.first_ghost_above());
      translate(ghost_heads.begin(), below_end);
      translate(local_heads.begin(), local_heads.end());
      translate(below_end, ghost_heads.end());
    }
  }
  auto [numbering, local] = std::move(G).release();
  auto xadj = std::move(local).release().first;
  return {std::move(numbering), {std::move(xadj), std::move(global)}};
}

/// Contracts `G` (neighborhoods sorted by handle) to its cut edges: of each
/// neighborhood only the ghost part is kept ([ghosts below][ghosts above],
/// already in global order), as global ids. The kept vertices are those with a
/// cut out-edge.
inline auto contract_to_global_ids(Graph&& G) -> GlobalIdGraph {
  std::size_t num_cut_edges = 0;
  for (VertexHandle const v : G.local_vertices()) {
    num_cut_edges += G.ghost_heads(v).size();
  }
  std::vector<VertexHandle> kept;
  std::vector<std::size_t> xadj(G.num_local_vertices() + 1, 0);
  std::vector<VertexId> global;
  global.reserve(num_cut_edges);
  for (VertexHandle const v : G.local_vertices()) {
    auto const cut = G.ghost_heads(v);
    if (!cut.empty()) {
      kept.push_back(v);
    }
    for (VertexHandle const u : cut) {
      global.push_back(G.global_id(u));
    }
    xadj[G.to_local_index(v) + 1] = global.size();
  }
  auto numbering = std::move(G).release().first;
  return {std::move(numbering),
          {std::move(xadj), std::move(global)},
          std::move(kept)};
}

}  // namespace katric::graph
