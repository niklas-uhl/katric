// The conversions of an oriented Graph to the global phase's GlobalIdGraph.
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
/// already in global order), as global ids; vertices without cut edges keep an
/// empty neighborhood.
inline auto contract_to_global_ids(Graph&& G) -> GlobalIdGraph {
  std::size_t num_cut_edges = 0;
  for (VertexHandle const v : G.local_vertices()) {
    num_cut_edges += G.ghost_heads(v).size();
  }
  std::vector<std::size_t> xadj(G.num_local_vertices() + 1, 0);
  std::vector<VertexId> global;
  global.reserve(num_cut_edges);
  for (VertexHandle const v : G.local_vertices()) {
    for (VertexHandle const u : G.ghost_heads(v)) {
      global.push_back(G.global_id(u));
    }
    xadj[G.to_local_index(v) + 1] = global.size();
  }
  auto numbering = std::move(G).release().first;
  return {std::move(numbering), {std::move(xadj), std::move(global)}};
}

}  // namespace katric::graph
