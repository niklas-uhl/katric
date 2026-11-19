// The phases before the local phase on compact ids: building the input graph, the primary load balancing, and the
// orientation. Each phase is a Caliper region; there are no barriers between the phases, so a phase's time on a rank
// includes waiting for other ranks inside the next phase's first collective.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <caliper/cali.h>

#include "katric/caliper.h"
#include "katric/message_tags.h"
#include "katric/config.h"
#include <kagen.h>
#include "katric/ghost_exchange.h"
#include "katric/graph/build.h"
#include "katric/graph/graph.h"
#include "katric/load_balancing.h"
#include "katric/phase_counters.h"
#include "katric/util.h"

namespace katric {

/// The primary load balancing's cost of every local vertex (`--primary-cost-function`, PATRIC's cost functions):
/// N = 1; DPD = Σ over out-neighbors u of d+(v) + d+(u) (the tail-compute work), IDPD = the same over in-neighbors
/// (push), both with the orientation by (degree, global id). Region `cost_function_communication` for the ghosts'
/// degrees and out-degrees, `cost_function_evaluation` for the costs.
inline std::vector<std::size_t> vertex_costs(graph::CsrGraph const& G, Config const& conf) {
    using graph::Handle;
    std::string const& name = conf.primary_cost_function;
    if (name == "N") {
        return std::vector<std::size_t>(G.local_node_count(), 1);
    }
    if (name != "DPD" && name != "IDPD") {
        throw std::runtime_error("Unsupported cost function");
    }
    auto const&                layer = *G.layer();
    std::vector<std::uint64_t> ghost_degree;
    std::vector<std::uint64_t> out_degree(G.local_node_count() + G.ghost_count());
    auto const                 key = [&](Handle c) {
        return std::pair(layer.is_owned(c) ? G.degree(c) : ghost_degree[c - layer.n()], layer.global_id(c));
    };
    {
        CALI_CXX_MARK_SCOPE("cost_function_communication");
        ghost_degree = graph::exchange_ghost_values(
            G,
            [&](Handle v) { return G.degree(v); },
            conf.degree_exchange,
            MessageTag::CostFunction
        );
        for (Handle const v: G.local_nodes()) {
            out_degree[v] = std::ranges::count_if(G.neighbors(v), [&](Handle u) { return key(u) > key(v); });
        }
        auto const ghost_out_degree =
            graph::exchange_ghost_values(G, [&](Handle v) { return out_degree[v]; }, conf.degree_exchange, MessageTag::CostFunctionOutDegree);
        std::ranges::copy(ghost_out_degree, out_degree.begin() + static_cast<std::ptrdiff_t>(layer.n()));
    }
    CALI_CXX_MARK_SCOPE("cost_function_evaluation");
    bool const               out_edges = name == "DPD";
    std::vector<std::size_t> cost(G.local_node_count(), 0);
    for (Handle const v: G.local_nodes()) {
        for (Handle const u: G.neighbors(v)) {
            if ((key(u) > key(v)) == out_edges) {
                cost[v] += out_degree[v] + out_degree[u];
            }
        }
    }
    return cost;
}

namespace phases {

/// The input graph with compact ids (ghosts found, their owners looked up). Returns it and the message queue's
/// threshold (from the input graph).
inline std::pair<graph::CsrGraph, std::size_t> input_graph(kagen::Graph const& input, Config const& conf, MPI_Comm comm) {
    CALI_CXX_MARK_SCOPE("input_graph");
    auto              G         = graph::build_input_graph(input, comm);
    std::size_t const threshold = get_threshold(G, conf);
    return {std::move(G), threshold};
}

/// Repartitions `G` into contiguous vertex ranges of equal cost (`--primary-cost-function`, off with "none").
inline void primary_load_balancing(graph::CsrGraph& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("primary_load_balancing");
    auto const comm = G.comm();
    if (conf.primary_cost_function != "none") {
        auto const cost = vertex_costs(G, conf);
        G               = graph::build_input_graph(load_balancing::run(graph::to_kagen_graph(G), cost, comm), comm);
    }
}

/// The degrees of the ghosts of `G`, indexed by ghost slot (region `degree_exchange`).
inline std::vector<std::uint64_t> ghost_degrees(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("degree_exchange");
    return graph::exchange_ghost_values(
        G,
        [&](graph::Handle v) { return G.degree(v); },
        conf.degree_exchange,
        MessageTag::Orientation
    );
}

/// Orients the edges of `G` from the smaller to the larger `key` (region `orientation`) and sorts the neighborhoods by
/// handle (`sorting`); with `ghost_out_edges`, also the ghosts' out-edges to local vertices.
template <typename Key>
graph::CsrGraph orient_and_sort(graph::CsrGraph const& G, Key&& key, bool ghost_out_edges) {
    graph::CsrGraph G_oriented;
    {
        CALI_CXX_MARK_SCOPE("orientation");
        G_oriented = graph::orient(G, key, ghost_out_edges);
    }
    CALI_CXX_MARK_SCOPE("sorting");
    G_oriented.sort_neighborhoods();
    return G_oriented;
}

/// The orientation key of DiTriC and CETRIC: (degree, global id), the ghosts' degrees from `ghost_degree`.
inline auto degree_key(graph::CsrGraph const& G, std::vector<std::uint64_t> const& ghost_degree) {
    return [&G, &ghost_degree](graph::Handle c) {
        auto const& layer = *G.layer();
        return std::pair(layer.is_owned(c) ? G.degree(c) : ghost_degree[c - layer.n()], layer.global_id(c));
    };
}

/// Degree exchange and orientation by (degree, global id). With `--contraction`, also the ghosts' out-edges to local
/// vertices (the ghost rows), so that the local phase finds every triangle with two local vertices.
inline graph::CsrGraph preprocessing(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    auto const ghost_degree = ghost_degrees(G, conf);
    return orient_and_sort(G, degree_key(G, ghost_degree), conf.contraction);
}

/// Without contraction, after the local phase: the oriented graph converted to global ids in place for the global phase (region
/// `global_ids`).
inline graph::GlobalIdGraph global_ids(graph::CsrGraph&& G) {
    CALI_CXX_MARK_SCOPE("global_ids");
    return graph::to_global_ids(std::move(G));
}

/// The number of cut out-edges of local vertex `v` (its neighborhood sorted by handle ends with them).
inline Degree cut_out_degree(graph::CsrGraph const& G, graph::Handle v) {
    auto const neighbors = G.neighbors(v);
    return static_cast<Degree>(
        neighbors.end() - std::lower_bound(neighbors.begin(), neighbors.end(), graph::Handle{G.local_node_count()})
    );
}

/// The contraction's counters: the vertices with cut edges (compact_nodes) and their cut out-edges (compact_edges).
inline void report_contraction(graph::CsrGraph const& G) {
    std::int64_t nodes = 0;
    std::int64_t edges = 0;
    for (graph::Handle const v: G.local_nodes()) {
        auto const degree = static_cast<std::int64_t>(cut_out_degree(G, v));
        nodes += degree > 0 ? 1 : 0;
        edges += degree;
    }
    profiling::counter_add("compact_nodes", nodes);
    profiling::counter_add("compact_edges", edges);
}

/// With contraction, after the local phase: the edges the local phase did not handle (the cut out-edges), as global
/// ids, contracted in place (region `contraction`).
inline graph::GlobalIdGraph contraction(graph::CsrGraph&& G) {
    CALI_CXX_MARK_SCOPE("contraction");
    report_contraction(G);
    return graph::contract_to_global_ids(std::move(G));
}

} // namespace phases
} // namespace katric
