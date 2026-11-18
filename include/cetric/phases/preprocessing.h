// The phases before the local phase on compact ids: building the input graph, the primary load balancing, and the
// orientation. Each phase is a Caliper region; there are no barriers between the phases, so a phase's time on a rank
// includes waiting for other ranks inside the next phase's first collective.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <caliper/cali.h>

#include "cetric/caliper.h"
#include "cetric/message_tags.h"
#include "cetric/config.h"
#include <kagen.h>
#include "cetric/ghost_exchange.h"
#include "cetric/graph/build.h"
#include "cetric/graph/graph.h"
#include "cetric/load_balancing.h"
#include "cetric/phase_counters.h"
#include "cetric/util.h"

namespace cetric {

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
inline std::pair<graph::CsrGraph, std::size_t> input_graph(kagen::Graph&& input, Config const& conf, MPI_Comm comm) {
    CALI_CXX_MARK_SCOPE("input_graph");
    auto              G         = graph::build_input_graph(std::move(input), comm);
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

/// The degrees of the ghosts of `G` (a CsrGraph or a ContractedGraph), indexed by ghost slot (region
/// `degree_exchange`).
template <typename Graph>
std::vector<std::uint64_t> ghost_degrees(Graph const& G, Config const& conf) {
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

/// DiTriC: degree exchange, orientation by degree, and the neighbors as global ids for the global phase (region
/// `neighbors_as_global_ids`).
inline graph::CsrGraph preprocessing_patric(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    auto const ghost_degree = ghost_degrees(G, conf);
    auto       G_oriented   = orient_and_sort(G, degree_key(G, ghost_degree), false);
    {
        CALI_CXX_MARK_SCOPE("neighbors_as_global_ids");
        G_oriented.add_neighbors_as_global_ids();
    }
    return G_oriented;
}

/// CETRIC: degree exchange, orientation by degree, with the ghosts' out-edges to local vertices (which replace the
/// old ghost expansion).
inline graph::CsrGraph preprocessing_cetric_x(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    auto const ghost_degree = ghost_degrees(G, conf);
    auto       G_oriented   = orient_and_sort(G, degree_key(G, ghost_degree), true);
    return G_oriented;
}

/// pre-CETRIC: no degree exchange; local edges by (degree, global id), cut edges outward.
inline graph::CsrGraph preprocessing_cetric(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    auto const& layer      = *G.layer();
    auto        G_oriented = orient_and_sort(
        G,
        [&](graph::Handle c) {
            Degree const degree = layer.is_owned(c) ? G.degree(c) : std::numeric_limits<Degree>::max();
            return std::pair(degree, layer.global_id(c));
        },
        false
    );
    return G_oriented;
}

/// CETRIC and pre-CETRIC: the edges the local phase did not handle, as the contracted graph.
inline graph::ContractedGraph contraction(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("contraction");
    auto G_contracted = graph::contract(G);
    profiling::counter_add("compact_nodes", static_cast<std::int64_t>(G_contracted.local_node_count()));
    profiling::counter_add("compact_edges", static_cast<std::int64_t>(G_contracted.local_edge_count()));
    return G_contracted;
}

/// pre-CETRIC: degree exchange on the contracted graph and its orientation by (contracted degree, global id).
inline graph::ContractedGraph preprocessing_global(graph::ContractedGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing_global");
    auto const             ghost_degree = ghost_degrees(G, conf);
    graph::ContractedGraph G_oriented;
    {
        CALI_CXX_MARK_SCOPE("orientation");
        auto const& layer = *G.layer();
        G_oriented        = graph::orient(G, [&](graph::Handle c) {
            return std::pair(layer.is_owned(c) ? G.degree(c) : ghost_degree[c - layer.n()], layer.global_id(c));
        });
    }
    return G_oriented;
}

} // namespace phases
} // namespace cetric
