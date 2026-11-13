// The phases before the local phase: ghost ranks, primary load balancing, and the orientation of each algorithm.
// Each phase is a Caliper region and ends in a barrier with `--global-synchronization`.
#pragma once

#include <cstddef>
#include <limits>
#include <utility>

#include <caliper/cali.h>

#include "cetric/config.h"
#include "cetric/cost_function.h"
#include "cetric/counters/cetric_edge_iterator.h"
#include "cetric/datastructures/auxiliary_node_data.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/distributed/graph_communicator.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/execution.h"
#include "cetric/ghost_exchange.h"
#include "cetric/load_balancing.h"
#include "cetric/util.h"

namespace cetric {

/// Optionally fetches the ghosts' degrees (region `degree_exchange`), then orients every edge by `OrientationOrdering`
/// (`orientation`) and sorts the neighborhoods by `NodeOrdering` (`sorting`).
template <typename NodeOrdering, typename OrientationOrdering, typename NodeIndexer>
void orient_and_sort(DistributedGraph<NodeIndexer>& G, Config const& conf, bool fetch_ghost_degree) {
    node_set                  ghosts;
    AuxiliaryNodeData<Degree> ghost_degree;
    if (fetch_ghost_degree) {
        CALI_CXX_MARK_SCOPE("degree_exchange");
        find_ghosts(G, ghosts);
        ghost_degree = AuxiliaryNodeData<Degree>{ghosts.begin(), ghosts.end()};
        DegreeCommunicator comm(G, as_int(MessageTag::Orientation));
        comm.get_ghost_degree(
            [&](RankEncodedNodeId node, Degree degree) { ghost_degree[node] = degree; },
            degree_exchange(conf)
        );
    }
    bool const parallel = conf.num_threads > 1;
    {
        CALI_CXX_MARK_SCOPE("orientation");
        for_each_local_node(G, conf, parallel, [&](auto node) { G.orient(node, OrientationOrdering(G, ghost_degree)); });
    }
    CALI_CXX_MARK_SCOPE("sorting");
    for_each_local_node(G, conf, parallel, [&](auto node) { G.sort_neighborhoods(node, NodeOrdering(G, ghost_degree)); });
}

/// Sets the owner rank of every neighbor (RankEncodedNodeId).
template <typename Graph>
void compute_ghost_ranks(Graph& G, Config const& conf) {
    with_execution_policy(conf, [&](auto policy) { G.find_ghost_ranks(std::move(policy)); });
}

/// The half-open global id range of a local graph view; (max, max) without vertices.
inline std::pair<NodeId, NodeId> node_range_of(LocalGraphView const& view) {
    if (view.local_node_count() == 0) {
        return {std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max()};
    }
    return {view.node_info.front().global_id, view.node_info.back().global_id + 1};
}

namespace phases {

/// Sets the neighbors' ranks; returns the message queue's threshold (from the input graph).
inline std::size_t ghost_ranks(DistributedGraph<>& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("ghost_ranks");
    std::size_t const threshold = get_threshold(G, conf);
    compute_ghost_ranks(G, conf);
    ConditionalBarrier(true, G.comm());
    return threshold;
}

/// Repartitions `G` into contiguous vertex ranges of equal cost (`--primary-cost-function`, off with "none").
inline void primary_load_balancing(DistributedGraph<>& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("primary_load_balancing");
    if (conf.primary_cost_function != "none") {
        auto cost_function = with_execution_policy(conf, [&](auto policy) {
            return CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf, std::move(policy));
        });
        LocalGraphView view = load_balancing::LoadBalancer::run(G.to_local_graph_view(), cost_function, conf, G.comm());
        auto const     range = node_range_of(view);
        G                    = DistributedGraph(std::move(view), range, G.comm());
        compute_ghost_ranks(G, conf);
    }
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// DiTriC: degree exchange, orientation by degree, neighborhoods by id.
inline void preprocessing_patric(DistributedGraph<>& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    orient_and_sort<node_ordering::id, node_ordering::degree<DistributedGraph<>>>(G, conf, true);
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// pre-CETRIC: no degree exchange; cut edges point outward, the local vertices first in each neighborhood.
inline void preprocessing_cetric(DistributedGraph<>& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    orient_and_sort<node_ordering::id_outward, node_ordering::degree_outward<DistributedGraph<>>>(G, conf, false);
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// CETRIC: degree exchange, orientation by degree, the local vertices first; then the ghost expansion.
inline void preprocessing_cetric_x(DistributedGraph<>& G, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    orient_and_sort<node_ordering::id_outward, node_ordering::degree<DistributedGraph<>>>(G, conf, true);
    {
        CALI_CXX_MARK_SCOPE("ghost_expansion");
        with_execution_policy(conf, conf.parallel_compact, [&](auto policy) {
            G.remove_in_edges_and_expand_ghosts(node_ordering::id_outward(conf.rank), std::move(policy));
        });
    }
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

} // namespace phases
} // namespace cetric
