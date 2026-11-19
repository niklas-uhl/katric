// The phases before the local phase on compact ids: building the input graph,
// the primary load balancing, and the orientation. Each phase is a Caliper
// region; there are no barriers between the phases, so a phase's time on a rank
// includes waiting for other ranks inside the next phase's first collective.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>
#include <kassert/kassert.hpp>

#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/ghost_exchange.h"
#include "katric/graph/build.h"
#include "katric/graph/graph.h"
#include "katric/load_balancing.h"
#include "katric/message_tags.h"
#include "katric/phase_counters.h"
#include "katric/util.h"

namespace katric {

/// The orientation key of DiTriC and CETRIC: (degree, global id), the ghosts'
/// degrees from `ghost_degree`.
inline auto degree_key(graph::Graph const& G,
                       std::vector<std::uint64_t> const& ghost_degree) {
  return [&G, &ghost_degree](graph::VertexHandle vertex) {
    auto const& numbering = G.numbering();
    return std::pair(numbering.is_local(vertex)
                         ? G.degree(vertex)
                         : ghost_degree[numbering.to_ghost_index(vertex)],
                     numbering.global_id(vertex));
  };
}

/// The primary load balancing's cost of every local vertex
/// (`--primary-cost-function`, PATRIC's cost functions): N = 1; DPD = Σ over
/// out-neighbors u of d+(v) + d+(u) (the tail-compute work), IDPD = the same
/// over in-neighbors (push), both with the orientation by (degree, global id).
/// Regions: `cost_function_communication` for the exchanges of the ghosts'
/// degrees and out-degrees, `cost_function_evaluation` for the out-degrees and
/// the costs (each entered twice).
inline auto vertex_costs(graph::Graph const& G, Config const& conf)
    -> std::vector<std::size_t> {
  using graph::VertexHandle;
  if (conf.primary_cost_function == CostFunction::N) {
    return std::vector<std::size_t>(G.num_local_vertices(), 1);
  }
  KASSERT(conf.primary_cost_function == CostFunction::DPD ||
          conf.primary_cost_function == CostFunction::IDPD);
  auto const& numbering = G.numbering();
  std::vector<std::uint64_t> ghost_degree;
  {
    CALI_CXX_MARK_SCOPE("cost_function_communication");
    ghost_degree = push_to_ghosts(
        G, [&](VertexHandle v) { return G.degree(v); }, conf.degree_exchange,
        MessageTag::CostFunction);
  }
  // the orientation of every edge (in adjacency order), decided once
  std::vector<bool> is_out_edge(G.num_local_edges());
  std::vector<std::uint64_t> local_out_degree(G.num_local_vertices(), 0);
  {
    CALI_CXX_MARK_SCOPE("cost_function_evaluation");
    auto const key = degree_key(G, ghost_degree);
    std::size_t edge = 0;
    for (VertexHandle const v : G.local_vertices()) {
      auto const key_v = key(v);
      std::uint64_t out_degree_v = 0;
      for (VertexHandle const u : G.neighbors(v)) {
        bool const out = key_v < key(u);
        is_out_edge[edge++] = out;
        if (out) {
          ++out_degree_v;
        }
      }
      local_out_degree[numbering.to_local_index(v)] = out_degree_v;
    }
  }
  std::vector<std::uint64_t> ghost_out_degree;
  {
    CALI_CXX_MARK_SCOPE("cost_function_communication");
    ghost_out_degree = push_to_ghosts(
        G,
        [&](VertexHandle v) {
          return local_out_degree[numbering.to_local_index(v)];
        },
        conf.degree_exchange, MessageTag::CostFunctionOutDegree);
  }
  CALI_CXX_MARK_SCOPE("cost_function_evaluation");
  auto const out_degree = [&](VertexHandle vertex) -> std::uint64_t {
    return numbering.is_local(vertex)
               ? local_out_degree[numbering.to_local_index(vertex)]
               : ghost_out_degree[numbering.to_ghost_index(vertex)];
  };
  bool const over_out_edges = conf.primary_cost_function == CostFunction::DPD;
  std::vector<std::size_t> cost(G.num_local_vertices(), 0);
  std::size_t edge = 0;
  for (VertexHandle const v : G.local_vertices()) {
    std::size_t& cost_v = cost[numbering.to_local_index(v)];
    for (VertexHandle const u : G.neighbors(v)) {
      if (is_out_edge[edge++] == over_out_edges) {
        cost_v += out_degree(v) + out_degree(u);
      }
    }
  }
  return cost;
}

namespace phases {

/// The input graph with compact ids (ghosts found, their owners looked up).
inline auto input_graph(kagen::Graph const& input, kamping::v2::comm_view comm)
    -> graph::Graph {
  CALI_CXX_MARK_SCOPE("input_graph");
  return graph::build_input_graph(input, comm);
}

/// Repartitions `G` into contiguous vertex ranges of equal cost
/// (`--primary-cost-function`; with `none`, returns `G`). Consumes `G`: it is
/// freed before the vertices move.
inline auto primary_load_balancing(graph::Graph&& G, Config const& conf)
    -> graph::Graph {
  CALI_CXX_MARK_SCOPE("primary_load_balancing");
  if (conf.primary_cost_function == CostFunction::none) {
    return std::move(G);
  }
  auto const comm = G.comm();
  auto const cost = vertex_costs(G, conf);
  return graph::build_input_graph(
      load_balancing::run(graph::to_kagen_graph(std::move(G)), cost, comm),
      comm);
}

/// The degrees of the ghosts of `G`, indexed by ghost index (region
/// `degree_exchange`).
inline std::vector<std::uint64_t> ghost_degrees(graph::Graph const& G,
                                                Config const& conf) {
  CALI_CXX_MARK_SCOPE("degree_exchange");
  return push_to_ghosts(
      G, [&](graph::VertexHandle v) { return G.degree(v); },
      conf.degree_exchange, MessageTag::Orientation);
}

/// Degree exchange and orientation by (degree, global id) (region
/// `orientation`), then the neighborhoods sorted by handle (`sorting`). With
/// `--contraction`, also the ghosts' out-edges to local vertices, so that the
/// local phase finds every triangle with two local vertices. Consumes the input
/// graph `G`.
inline graph::Graph preprocessing(graph::Graph&& G, Config const& conf) {
  CALI_CXX_MARK_SCOPE("preprocessing");
  auto const ghost_degree = ghost_degrees(G, conf);
  auto const key = degree_key(G, ghost_degree);
  auto G_oriented = [&] {
    CALI_CXX_MARK_SCOPE("orientation");
    return graph::orient(std::move(G), key, conf.contraction);
  }();
  {
    CALI_CXX_MARK_SCOPE("sorting");
    G_oriented.sort_neighborhoods();
  }
  return G_oriented;
}

/// Without contraction, after the local phase: the oriented graph converted to
/// global ids in place for the global phase (region `global_ids`).
inline graph::GlobalIdGraph global_ids(graph::Graph&& G) {
  CALI_CXX_MARK_SCOPE("global_ids");
  return graph::to_global_ids(std::move(G));
}

/// The number of cut out-edges of local vertex `v` (its neighborhood sorted by
/// handle ends with them).
inline graph::Degree cut_out_degree(graph::Graph const& G,
                                    graph::VertexHandle v) {
  auto const neighbors = G.neighbors(v);
  return static_cast<graph::Degree>(
      neighbors.end() - std::lower_bound(neighbors.begin(), neighbors.end(),
                                         G.numbering().first_ghost()));
}

/// The contraction's counters: the vertices with cut edges (compact_nodes) and
/// their cut out-edges (compact_edges).
inline void report_contraction(graph::Graph const& G) {
  std::int64_t nodes = 0;
  std::int64_t edges = 0;
  for (graph::VertexHandle const v : G.local_vertices()) {
    auto const degree = static_cast<std::int64_t>(cut_out_degree(G, v));
    nodes += degree > 0 ? 1 : 0;
    edges += degree;
  }
  profiling::counter_add("compact_nodes", nodes);
  profiling::counter_add("compact_edges", edges);
}

/// With contraction, after the local phase: the edges the local phase did not
/// handle (the cut out-edges), as global ids, contracted in place (region
/// `contraction`).
inline graph::GlobalIdGraph contraction(graph::Graph&& G) {
  CALI_CXX_MARK_SCOPE("contraction");
  report_contraction(G);
  return graph::contract_to_global_ids(std::move(G));
}

}  // namespace phases
}  // namespace katric
