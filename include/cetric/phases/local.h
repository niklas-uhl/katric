// The local phase on compact ids: the intersections for the edges whose triangles a rank can find without
// communication. Rows are sorted by handle, so a row lists its local vertices first.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "cetric/config.h"
#include "cetric/counters/intersection.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/graph/graph.h"
#include "cetric/phase_counters.h"

namespace cetric::local {

/// The row of `c`: N+(c) for a local vertex, the ghost row (CETRIC: its out-edges to local vertices) for a ghost.
inline std::span<graph::Handle const> row(graph::CsrGraph const& G, graph::Handle c) {
    return G.is_owned(c) ? G.neighbors(c) : G.ghost_neighbors(c);
}

/// The work for one edge v -> u: emits the triangles (v, u, w) for w in N+(v) ∩ N+(u), where a ghost's neighborhood is
/// its ghost row. A ghost row holds only local vertices, so when one end is a ghost the other side is clipped to its
/// local prefix; this bounds the merge and keeps the work counters exact.
template <typename TriangleFunc>
void intersect_edge(
    graph::CsrGraph const& G,
    Config const&          conf,
    graph::Handle          v,
    graph::Handle          u,
    LocalPhaseCounters&    counters,
    TriangleFunc&          emit
) {
    auto const   v_row   = row(G, v);
    auto const   u_row   = row(G, u);
    auto         v_first = v_row.begin();
    auto         v_last  = v_row.end();
    auto         u_first = u_row.begin();
    auto         u_last  = u_row.end();
    auto const   n       = static_cast<graph::Handle>(G.local_node_count());
    if (!G.is_owned(u)) {
        v_last = std::lower_bound(v_first, v_last, n);
    } else if (!G.is_owned(v)) {
        u_last = std::lower_bound(u_first, u_last, n);
    }
    // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
    counters.wedge_checks += u_last - u_first;
    counters.intersection_size_local += (v_last - v_first) + (u_last - u_first);
    cetric::intersection(
        v_first,
        v_last,
        u_first,
        u_last,
        [&](graph::Handle w) {
            counters.local_triangles++;
            emit(graph::Triangle<graph::NodeId>{G.global_id(v), G.global_id(u), G.global_id(w)});
        },
        std::less<>{},
        conf
    );
}

/// The local phase: the intersection for every edge v -> u with both ends local and, `with_ghosts` (CETRIC), also for
/// the edges with a ghost end, through the ghost rows. Returns the interface vertices (local vertices with a cut
/// out-edge), ascending.
template <bool with_ghosts, typename TriangleFunc>
std::vector<graph::Handle> run(graph::CsrGraph const& G, Config const& conf, TriangleFunc emit) {
    std::vector<graph::Handle> interface_nodes;
    LocalPhaseCounters         counters;
    auto                       for_each_out_edge = [&](graph::Handle v, graph::Degree degree) {
        if (conf.pseudo2core && degree < 2) {
            counters.skipped_nodes++;
            return;
        }
        bool is_interface = false;
        for (graph::Handle const u: row(G, v)) {
            if (!G.is_owned(u)) {
                if (G.is_owned(v) && !is_interface) {
                    interface_nodes.push_back(v);
                    is_interface = true;
                }
                if constexpr (!with_ghosts) {
                    // without ghost rows only local heads are intersected; they come first
                    break;
                }
            }
            intersect_edge(G, conf, v, u, counters, emit);
        }
    };
    for (graph::Handle const v: G.local_nodes()) {
        for_each_out_edge(v, G.degree(v));
    }
    if constexpr (with_ghosts) {
        for (graph::Handle const v: G.ghosts()) {
            for_each_out_edge(v, G.ghost_degree(v));
        }
    }
    counters.report();
    return interface_nodes;
}

} // namespace cetric::local
