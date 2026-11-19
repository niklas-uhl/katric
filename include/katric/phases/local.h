// The local phase on compact ids: the intersections for the edges whose triangles a rank can find without
// communication. Neighborhoods are sorted by handle, so they list their local vertices first.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "katric/config.h"
#include "katric/intersection.h"
#include "katric/graph/graph.h"
#include "katric/phase_counters.h"
#include "katric/triangle_counter.h"

namespace katric::local {

/// N+(c) for a local vertex, and for a ghost (CETRIC) its out-edges to local vertices.
inline std::span<graph::VertexHandle const> neighbors_with_ghosts(graph::Graph const& G, graph::VertexHandle c) {
    return G.is_local(c) ? G.neighbors(c) : G.ghost_out_neighbors(c);
}

/// The work for one edge v -> u: emits the triangles (v, u, w) for w in N+(v) ∩ N+(u), where a ghost's neighborhood is
/// its out-edges to local vertices. A ghost's neighbors are only local vertices, so when one end is a ghost the other side is clipped to its
/// local prefix; this bounds the merge and keeps the work counters exact.
template <typename TriangleFunc>
void intersect_edge(
    graph::Graph const& G,
    Config const&          conf,
    graph::VertexHandle          v,
    graph::VertexHandle          u,
    LocalPhaseCounters&    counters,
    TriangleFunc&          emit
) {
    auto const   v_neighbors   = neighbors_with_ghosts(G, v);
    auto const   u_neighbors   = neighbors_with_ghosts(G, u);
    auto         v_first = v_neighbors.begin();
    auto         v_last  = v_neighbors.end();
    auto         u_first = u_neighbors.begin();
    auto         u_last  = u_neighbors.end();
    auto const   n       = static_cast<graph::VertexHandle>(G.num_local_vertices());
    if (!G.is_local(u)) {
        v_last = std::lower_bound(v_first, v_last, n);
    } else if (!G.is_local(v)) {
        u_last = std::lower_bound(u_first, u_last, n);
    }
    // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
    counters.wedge_checks += u_last - u_first;
    counters.intersection_size_local += (v_last - v_first) + (u_last - u_first);
    katric::intersection(
        v_first,
        v_last,
        u_first,
        u_last,
        [&](graph::VertexHandle w) {
            counters.local_triangles++;
            emit(Triangle{G.global_id(v), G.global_id(u), G.global_id(w)});
        },
        std::less<>{},
        conf
    );
}

/// What the local phase hands on: the interface vertices (local vertices with a cut out-edge, ascending) and the number
/// of triangles it found.
struct Result {
    std::vector<graph::VertexHandle> interface_nodes;
    std::size_t                      triangles = 0;
};

/// The local phase: the intersection for every edge v -> u with both ends local and, `with_ghosts` (CETRIC), also for
/// the edges with a ghost end, through the ghosts' out-edges.
template <bool with_ghosts, typename TriangleFunc>
auto run(graph::Graph const& G, Config const& conf, TriangleFunc emit) -> Result {
    std::vector<graph::VertexHandle> interface_nodes;
    LocalPhaseCounters         counters;
    auto                       for_each_out_edge = [&](graph::VertexHandle v, graph::Degree degree) {
        if (conf.pseudo2core && degree < 2) {
            counters.skipped_nodes++;
            return;
        }
        bool is_interface = false;
        for (graph::VertexHandle const u: neighbors_with_ghosts(G, v)) {
            if (!G.is_local(u)) {
                if (G.is_local(v) && !is_interface) {
                    interface_nodes.push_back(v);
                    is_interface = true;
                }
                if constexpr (!with_ghosts) {
                    // without the ghosts' out-edges only local heads are intersected; they come first
                    break;
                }
            }
            intersect_edge(G, conf, v, u, counters, emit);
        }
    };
    for (graph::VertexHandle const v: G.local_vertices()) {
        for_each_out_edge(v, G.degree(v));
    }
    if constexpr (with_ghosts) {
        for (graph::VertexHandle const v: G.ghosts()) {
            for_each_out_edge(v, G.ghost_out_degree(v));
        }
    }
    counters.report();
    return {std::move(interface_nodes), static_cast<std::size_t>(counters.local_triangles)};
}

} // namespace katric::local
