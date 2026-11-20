// The local phase on compact ids: the intersections for the edges whose
// triangles a rank can find without communication. Neighborhoods are sorted by
// handle, so they list their local vertices first.
#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "katric/config.h"
#include "katric/graph/graph.h"
#include "katric/intersection.h"
#include "katric/caliper.h"
#include "katric/triangle_counter.h"

namespace katric {

/// Work counters of the local phase.
struct LocalPhaseCounters {
  std::int64_t skipped_nodes = 0;
  std::int64_t wedge_checks = 0;
  std::int64_t intersection_size_local = 0;

  /// Adds the counts to the open Caliper region.
  void report() const {
    profiling::counter_add("skipped_nodes", skipped_nodes);
    profiling::counter_add("wedge_checks", wedge_checks);
    profiling::counter_add("intersection_size_local", intersection_size_local);
  }
};


/// The work for one edge v -> u: records the triangles (v, u, w) for w in N+(v)
/// ∩ N+(u), where a ghost's neighborhood is its out-edges to local vertices. A
/// ghost's neighbors are only local vertices, so when one end is a ghost only
/// the other end's local heads are intersected; this bounds the merge and keeps
/// the work counters exact.
inline void intersect_edge(graph::Graph const& G,
                           Config const& conf,
                           graph::VertexHandle v,
                           graph::VertexHandle u,
                           LocalPhaseCounters& counters,
                           TriangleCounter& triangles) {
  auto const v_neighbors = G.is_ghost(u) ? G.local_heads(v) : G.neighbors(v);
  auto const u_neighbors = G.is_ghost(v) ? G.local_heads(u) : G.neighbors(u);
  // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
  counters.wedge_checks += static_cast<std::int64_t>(u_neighbors.size());
  counters.intersection_size_local +=
      static_cast<std::int64_t>(v_neighbors.size() + u_neighbors.size());
  katric::intersection(
      v_neighbors, u_neighbors,
      [&](graph::VertexHandle w) -> void {
        triangles.record(
            Triangle{.x=G.global_id(v), .y=G.global_id(u), .z=G.global_id(w)});
      },
      conf);
}

/// The local phase (region `local_phase`): the intersection for every edge
/// v -> u with both ends local and, if the graph has the ghosts' out-edges
/// (CETRIC), also for the edges with a ghost end, through the ghosts'
/// out-edges.
inline void local_phase(graph::Graph const& G,
                        Config const& conf,
                        TriangleCounter& triangles) {
  CALI_CXX_MARK_SCOPE("local_phase");
  auto const triangles_before = triangles.count();
  LocalPhaseCounters counters;
  bool const with_ghosts = G.has_ghost_out_edges();
  for (graph::VertexHandle const v :
       with_ghosts ? G.vertices() : G.local_vertices()) {
    // this is even correct on the oriented graph:
    // if a vertex has degree 0, we can skip it.
    // if a vertex has degree 1, then degree orientation either connects it to
    // another degree 1 vertex, or a vertex with higher degree.
    // so in the oriented graph its degree is either 1 or 0, so the skipping
    // works
    if (conf.pseudo2core && G.neighbors(v).size() < 2) {
      counters.skipped_nodes++;
      continue;
    }
    for (graph::VertexHandle const u : G.local_heads(v)) {
      intersect_edge(G, conf, v, u, counters, triangles);
    }
    // without the ghosts' out-edges, edges to ghosts are left to the global
    // phase (a ghost has no ghost heads)
    if (with_ghosts) {
      for (graph::VertexHandle const u : G.ghost_heads(v)) {
        intersect_edge(G, conf, v, u, counters, triangles);
      }
    }
  }
  counters.report();
  profiling::counter_add(
      "local_triangles",
      static_cast<std::int64_t>(triangles.count() - triangles_before));
}

}  // namespace katric
