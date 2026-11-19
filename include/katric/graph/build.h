// Building the graphs of graph/graph.h: the input graph from KaGen's CSR, the
// exchange of per-vertex values to the ghosts, the orientation, and the way
// back to KaGen's CSR for load balancing.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

#include <dstl/distribution.hpp>
#include <kagen.h>
#include <mpi.h>
#include <kamping/v2/types/std/unsafe/utility.hpp>
#include <kassert/kassert.hpp>

#include "katric/ghost_exchange.h"
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
  dstl::distribution distribution(first < last ? last - first : 0, comm);
  int const rank = comm.rank();
  KASSERT(first >= last || distribution.index_range_begin(rank) == first,
          "the vertex ranges are not contiguous in rank order");
  std::vector<std::pair<VertexId, std::size_t>>
      cut;  // (head, position in adjncy), sorted
  for (std::size_t position = 0; position < input.adjncy.size(); ++position) {
    VertexId const gid = input.adjncy[position];
    if (!distribution.is_local(gid, rank)) {
      cut.emplace_back(gid, position);
    }
  }
  std::ranges::sort(cut);
  std::vector<VertexId> ghost_ids;
  std::ranges::unique_copy(cut | std::views::keys,
                           std::back_inserter(ghost_ids));
  VertexLayer layer(std::move(distribution), std::move(ghost_ids), comm);
  std::vector<VertexHandle> adjncy(input.adjncy.size());
  for (std::size_t position = 0; position < input.adjncy.size(); ++position) {
    VertexId const gid = input.adjncy[position];
    if (layer.is_local_id(gid)) {
      adjncy[position] = layer.handle(gid);
    }
  }
  std::size_t ghost_index = 0;
  for (std::size_t i = 0; i < cut.size(); ++i) {
    if (i > 0 && cut[i].first != cut[i - 1].first) {
      ++ghost_index;  // the next run: the next ghost
    }
    adjncy[cut[i].second] = layer.ghost_handle(ghost_index);
  }
  decltype(cut){}.swap(cut);  // frees the cut pairs before the copy of xadj
  // copied: the input is only read (the harness reuses it for every iteration),
  // and KaGen's SInt need not be std::size_t (unsigned long long vs. unsigned
  // long)
  std::vector<std::size_t> xadj(input.xadj.begin(), input.xadj.end());
  if (xadj.empty()) {
    xadj.push_back(0);
  }
  KASSERT(xadj.size() == layer.num_local_vertices() + 1);
  return {std::move(layer), {std::move(xadj), std::move(adjncy)}};
}

/// `G` in KaGen's CSR representation (global ids).
inline auto to_kagen_graph(Graph const& G) -> kagen::Graph {
  kagen::Graph input;
  input.representation = kagen::GraphRepresentation::CSR;
  input.vertex_range = G.layer().local_range();
  input.xadj.reserve(G.num_local_vertices() + 1);
  input.adjncy.reserve(G.num_local_edges());
  input.xadj.push_back(0);
  for (VertexHandle const v : G.local_vertices()) {
    for (VertexHandle const u : G.neighbors(v)) {
      input.adjncy.push_back(G.global_id(u));
    }
    input.xadj.push_back(input.adjncy.size());
  }
  return input;
}

/// Sends `value(v)` of every local vertex v to the ranks holding v as a ghost
/// and returns the received values, indexed by ghost index. Message counters as
/// push_to_ghosts(), two words per entry (vertex and value).
template <typename ValueFunc>
auto exchange_ghost_values(Graph const& G,
                           ValueFunc const& value,
                           GhostExchangeMode mode,
                           MessageTag tag) -> std::vector<std::uint64_t> {
  // (vertex, value)
  auto const received = push_to_ghosts(
      G, G.local_vertices(),
      [&](VertexHandle v) -> std::pair<VertexId, std::uint64_t> {
        return std::pair<VertexId, std::uint64_t>{
            G.global_id(v), static_cast<std::uint64_t>(value(v))};
      },
      mode, tag, 2);
  auto const& layer = G.layer();
  std::vector<std::uint64_t> values(layer.num_ghosts());
  for (auto const& [vertex, data] : received) {
    values[layer.to_ghost_index(layer.handle(vertex))] = data;
  }
  return values;
}

/// Orients the edges of the undirected graph `G` from the smaller to the larger
/// `key` and returns the out-edges, neighborhoods unsorted (in `G`'s order).
/// Consumes `G`: the oriented graph takes over its vertex layer (`key` may read
/// `G` while the orientation runs).
/// With `ghost_out_edges`, also the ghosts' out-edges to local vertices
/// (CETRIC), sorted.
template <typename Key>
auto orient(Graph&& G, Key const& key, bool ghost_out_edges) -> Graph {
  auto const& layer = G.layer();
  std::size_t const num_local = layer.num_local_vertices();
  auto const out = [&](VertexHandle v, VertexHandle u) -> bool {
    return key(v) < key(u);
  };
  std::vector<std::size_t> xadj(num_local + 1, 0);
  std::vector<std::size_t> ghost_xadj(
      ghost_out_edges ? layer.num_ghosts() + 1 : 0, 0);
  for (VertexHandle const v : G.local_vertices()) {
    for (VertexHandle const u : G.neighbors(v)) {
      if (out(v, u)) {
        xadj[layer.to_local_index(v) + 1]++;
      } else if (ghost_out_edges && layer.is_ghost(u)) {
        ghost_xadj[layer.to_ghost_index(u) + 1]++;
      }
    }
  }
  for (std::size_t index = 0; index < num_local; ++index) {
    xadj[index + 1] += xadj[index];
  }
  for (std::size_t index = 0; index + 1 < ghost_xadj.size(); ++index) {
    ghost_xadj[index + 1] += ghost_xadj[index];
  }
  std::vector<VertexHandle> adjncy(xadj.back());
  std::vector<VertexHandle> ghost_adjncy(ghost_out_edges ? ghost_xadj.back()
                                                         : 0);
  std::vector<std::size_t> ghost_next(ghost_xadj);
  for (VertexHandle const v : G.local_vertices()) {
    std::size_t next = xadj[layer.to_local_index(v)];
    for (VertexHandle const u : G.neighbors(v)) {
      if (out(v, u)) {
        adjncy[next++] = u;
      } else if (ghost_out_edges && layer.is_ghost(u)) {
        // v ascending: each ghost's neighborhood comes out sorted
        ghost_adjncy[ghost_next[layer.to_ghost_index(u)]++] = v;
      }
    }
  }
  if (!ghost_out_edges) {
    return {std::move(G).release().first, {std::move(xadj), std::move(adjncy)}};
  }
  return {std::move(G).release().first,
          {std::move(xadj), std::move(adjncy)},
          AdjacencyArray<VertexHandle>{std::move(ghost_xadj),
                                       std::move(ghost_adjncy)}};
}

/// DiTriC: the out-neighborhoods of `G` (sorted by handle) as global ids. A
/// neighborhood sorted by handle is [local vertices][ghosts below the local
/// range][ghosts above]; translated in the order [ghosts below][local
/// vertices][ghosts above] it is ascending. A new array: the handles cannot be
/// reused for global ids.
inline auto to_global_ids(Graph&& G) -> GlobalIdGraph {
  auto [layer, local] = std::move(G).release();
  std::vector<VertexId> global;
  global.reserve(local.num_edges());
  auto const translate = [&](auto first, auto last) -> void {
    std::transform(
        first, last, std::back_inserter(global),
        [&](VertexHandle u) -> VertexId { return layer.global_id(u); });
  };
  for (std::size_t index = 0; index < local.num_vertices(); ++index) {
    auto const neighbors = local.neighbors(index);
    auto const local_end =
        std::ranges::lower_bound(neighbors, layer.first_ghost());
    auto const below_end =
        std::lower_bound(local_end, neighbors.end(), layer.first_ghost_above());
    translate(local_end, below_end);
    translate(neighbors.begin(), local_end);
    translate(below_end, neighbors.end());
  }
  auto xadj = std::move(local).release().first;
  return {std::move(layer), {std::move(xadj), std::move(global)}};
}

/// Contracts `G` (neighborhoods sorted by handle) to its cut edges: of each
/// neighborhood only the ghost part is kept ([ghosts below][ghosts above],
/// already in global order), as global ids. The kept vertices are those with a
/// cut out-edge.
inline auto contract_to_global_ids(Graph&& G) -> GlobalIdGraph {
  auto [layer, local] = std::move(G).release();
  auto const cut_neighbors = [&](std::size_t index) -> auto {
    auto const neighbors = local.neighbors(index);
    return std::ranges::subrange(
        std::ranges::lower_bound(neighbors, layer.first_ghost()),
        neighbors.end());
  };
  std::size_t num_cut_edges = 0;
  for (std::size_t index = 0; index < local.num_vertices(); ++index) {
    num_cut_edges += cut_neighbors(index).size();
  }
  std::vector<VertexHandle> kept;
  std::vector<std::size_t> xadj(local.num_vertices() + 1, 0);
  std::vector<VertexId> global;
  global.reserve(num_cut_edges);
  for (std::size_t index = 0; index < local.num_vertices(); ++index) {
    auto const cut = cut_neighbors(index);
    if (!cut.empty()) {
      kept.push_back(layer.local_handle(index));
    }
    for (VertexHandle const u : cut) {
      global.push_back(layer.global_id(u));
    }
    xadj[index + 1] = global.size();
  }
  return {
      std::move(layer), {std::move(xadj), std::move(global)}, std::move(kept)};
}

}  // namespace katric::graph
