// The local phase: the intersections for the edges whose triangles a rank can find without communication.
#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/counters/intersection.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph_definitions.h"

namespace cetric {

/// Work counters of the local phase.
struct LocalPhaseCounters {
    std::size_t skipped_nodes           = 0;
    std::size_t wedge_checks            = 0;
    std::size_t intersection_size_local = 0;
    std::size_t local_triangles         = 0;

    /// Adds the counts to the open Caliper region.
    void report() const {
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_local", intersection_size_local);
        profiling::counter_add("local_triangles", local_triangles);
    }
};

namespace local_phase {

/// Whether neighborhoods sorted by `NodeOrdering` have the local vertices first.
template <typename NodeOrdering>
constexpr bool local_first = std::is_same_v<std::remove_cvref_t<NodeOrdering>, node_ordering::id_outward>;

/// End of the local prefix of a neighborhood in the ghost-expanded graph (CETRIC). A ghost's adjacency holds only
/// local vertices, so when one side of an intersection is a ghost, only the local part of the other side can match.
/// With `id_outward` the local vertices come first, so the other side is clipped there. This bounds the merge and keeps
/// the work counters exact. Other orderings do not sort local vertices first, so the range stays as it is.
template <typename NodeOrdering, typename Graph, typename It>
[[nodiscard]] It local_prefix_end(Graph const& G, It first, It last) {
    if constexpr (local_first<NodeOrdering>) {
        return std::partition_point(first, last, [&G](graph::RankEncodedNodeId w) { return G.is_owned(w); });
    } else {
        return last;
    }
}

/// N+(v) for a local vertex, the ghost adjacency (CETRIC: its out-edges to local vertices) for a ghost.
template <typename Graph>
[[nodiscard]] auto neighbors_with_ghosts(Graph const& G, graph::RankEncodedNodeId v) {
    return G.is_owned(v) ? G.out_adj(v).neighbors() : G.ghost_adj(v).neighbors();
}

/// The work for one edge v -> u: emits the triangles (v, u, w) for w in N+(v) ∩ N+(u), where a ghost's neighborhood is
/// its ghost adjacency.
template <typename Graph, typename TriangleFunc, typename NodeOrdering>
void intersect_edge(
    Graph const&             G,
    Config const&            conf,
    graph::RankEncodedNodeId v,
    graph::RankEncodedNodeId u,
    LocalPhaseCounters&      counters,
    TriangleFunc&            emit,
    NodeOrdering const&      node_ordering
) {
    auto const v_neighbors = neighbors_with_ghosts(G, v);
    auto const u_neighbors = neighbors_with_ghosts(G, u);
    auto       v_first     = v_neighbors.begin();
    auto       v_last      = v_neighbors.end();
    auto       u_first     = u_neighbors.begin();
    auto       u_last      = u_neighbors.end();
    if (!G.is_owned(u)) {
        v_last = local_prefix_end<NodeOrdering>(G, v_first, v_last);
    } else if (!G.is_owned(v)) {
        u_last = local_prefix_end<NodeOrdering>(G, u_first, u_last);
    }
    // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
    counters.wedge_checks += u_last - u_first;
    counters.intersection_size_local += (v_last - v_first) + (u_last - u_first);
    cetric::intersection(
        v_first,
        v_last,
        u_first,
        u_last,
        [&](graph::RankEncodedNodeId w) {
            counters.local_triangles++;
            emit(graph::Triangle<graph::RankEncodedNodeId>{v, u, w});
        },
        node_ordering,
        conf
    );
}

/// The local phase: the intersection for every edge v -> u with both ends local and, `with_ghosts` (CETRIC), also for
/// the edges with a ghost end, through the ghosts' adjacencies. Returns the interface vertices (local vertices with a
/// cut out-edge) for the global phase.
template <bool with_ghosts, typename Graph, typename TriangleFunc, typename NodeOrdering>
std::vector<graph::RankEncodedNodeId>
run(Graph const& G, Config const& conf, TriangleFunc emit, NodeOrdering const& node_ordering) {
    std::vector<graph::RankEncodedNodeId> interface_nodes;
    LocalPhaseCounters                    counters;
    auto for_each_out_edge = [&](graph::RankEncodedNodeId v, graph::Degree degree) {
        if (conf.pseudo2core && degree < 2) {
            counters.skipped_nodes++;
            return;
        }
        bool is_interface = false;
        for (graph::RankEncodedNodeId u: neighbors_with_ghosts(G, v)) {
            if (!G.is_owned(u)) {
                if (G.is_owned(v) && !is_interface) {
                    interface_nodes.emplace_back(v);
                    is_interface = true;
                }
                if constexpr (!with_ghosts) {
                    // without ghost adjacencies only local heads are intersected; with the local vertices first, the
                    // first cut edge ends them
                    if constexpr (local_first<NodeOrdering>) {
                        break;
                    } else {
                        continue;
                    }
                }
            }
            intersect_edge(G, conf, v, u, counters, emit, node_ordering);
        }
    };
    for (auto v: G.local_nodes()) {
        for_each_out_edge(v, G.outdegree(v));
    }
    if constexpr (with_ghosts) {
        for (auto v: G.ghosts()) {
            for_each_out_edge(v, G.ghost_degree(v));
        }
    }
    counters.report();
    return interface_nodes;
}

} // namespace local_phase
} // namespace cetric
