// The phases before the local phase on compact ids: building the input graph, the primary load balancing, and the
// orientation. Each phase is a Caliper region; there are no barriers between the phases, so a phase's time on a rank
// includes waiting for other ranks inside the next phase's first collective.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <caliper/cali.h>

#include "cetric/caliper.h"
#include "cetric/communicator.h"
#include "cetric/config.h"
#include "cetric/datastructures/local_graph_view.h"
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
            degree_exchange(conf),
            as_int(MessageTag::CostFunction)
        );
        for (Handle const v: G.local_nodes()) {
            out_degree[v] = std::ranges::count_if(G.neighbors(v), [&](Handle u) { return key(u) > key(v); });
        }
        auto const ghost_out_degree =
            graph::exchange_ghost_values(G, [&](Handle v) { return out_degree[v]; }, degree_exchange(conf), as_int(MessageTag::CostFunctionOutDegree));
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
inline std::pair<graph::CsrGraph, std::size_t> input_graph(graph::LocalInput&& input, Config const& conf, MPI_Comm comm) {
    CALI_CXX_MARK_SCOPE("input_graph");
    auto              G         = graph::build_input_graph(std::move(input.G), input.node_range, comm);
    std::size_t const threshold = get_threshold(G, conf);
    return {std::move(G), threshold};
}

/// Repartitions `G` into contiguous vertex ranges of equal cost (`--primary-cost-function`, off with "none").
inline void primary_load_balancing(graph::CsrGraph& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("primary_load_balancing");
    MPI_Comm const comm = G.comm();
    if (conf.primary_cost_function != "none") {
        auto const cost      = vertex_costs(G, conf);
        auto       cost_of   = [&cost](graph::LocalGraphView const&, graph::NodeId v) { return cost[v]; };
        auto       view      = load_balancing::LoadBalancer::run(graph::to_local_graph_view(G), cost_of, conf, comm);
        auto const range     = view.node_range();
        G                    = graph::build_input_graph(std::move(view), range, comm);
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
        degree_exchange(conf),
        as_int(MessageTag::Orientation)
    );
}

/// Orients the edges of `G` by `orientation` (region `orientation`) and sorts the rows by handle (`sorting`); with
/// `ghost_rows`, also the ghosts' out-edges to local vertices.
inline graph::CsrGraph
orient_and_sort(graph::CsrGraph const& G, std::shared_ptr<graph::Orientation const> orientation, bool ghost_rows) {
    graph::CsrGraph G_oriented;
    {
        CALI_CXX_MARK_SCOPE("orientation");
        G_oriented = graph::orient(G, std::move(orientation), ghost_rows);
    }
    CALI_CXX_MARK_SCOPE("sorting");
    G_oriented.sort_rows();
    return G_oriented;
}

/// An orientation by (degree, global id): `local_degree(v)` for the local vertices, `ghost_degree` (by ghost slot) for
/// the ghosts.
template <typename LocalDegree>
std::shared_ptr<graph::Orientation const> degree_orientation(
    std::shared_ptr<graph::VertexLayer const> const& layer, LocalDegree&& local_degree, std::vector<std::uint64_t> const& ghost_degree
) {
    std::vector<Degree> degree(layer->n() + layer->ghost_count());
    for (graph::Handle c = 0; c < layer->n(); ++c) {
        degree[c] = local_degree(c);
    }
    std::ranges::copy(ghost_degree, degree.begin() + static_cast<std::ptrdiff_t>(layer->n()));
    return std::make_shared<graph::Orientation const>(layer, std::move(degree));
}

/// DiTriC: degree exchange, orientation by degree, and the rows in global ids for the global phase (`global_rows`).
inline graph::CsrGraph preprocessing_patric(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    auto const ghost_degree = ghost_degrees(G, conf);
    auto       orientation  = degree_orientation(G.layer(), [&](graph::Handle v) { return G.degree(v); }, ghost_degree);
    auto       G_oriented   = orient_and_sort(G, std::move(orientation), false);
    {
        CALI_CXX_MARK_SCOPE("global_rows");
        G_oriented.add_global_rows();
    }
    return G_oriented;
}

/// CETRIC: degree exchange, orientation by degree, with the ghosts' out-edges to local vertices (the ghost rows, which
/// replace the old ghost expansion).
inline graph::CsrGraph preprocessing_cetric_x(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    auto const ghost_degree = ghost_degrees(G, conf);
    auto       orientation  = degree_orientation(G.layer(), [&](graph::Handle v) { return G.degree(v); }, ghost_degree);
    return orient_and_sort(G, std::move(orientation), true);
}

/// pre-CETRIC: no degree exchange; local edges by (degree, global id), cut edges outward (the ghosts' degree is the
/// maximum).
inline graph::CsrGraph preprocessing_cetric(graph::CsrGraph const& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    std::vector<std::uint64_t> const outward(G.ghost_count(), std::numeric_limits<Degree>::max());
    auto orientation = degree_orientation(G.layer(), [&](graph::Handle v) { return G.degree(v); }, outward);
    return orient_and_sort(G, std::move(orientation), false);
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
        // local vertices without a row have no cut edges and are never compared
        auto orientation = degree_orientation(
            G.layer(), [&](graph::Handle v) { return G.is_local(v) ? G.degree(v) : 0; }, ghost_degree
        );
        G_oriented = graph::orient(G, std::move(orientation));
    }
    return G_oriented;
}

} // namespace phases
} // namespace cetric
