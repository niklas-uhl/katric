// CETRIC and pre-CETRIC between the local and the global phase: contraction to the cut edges, and for pre-CETRIC the
// optional secondary load balancing and the contracted graph's orientation. Each phase is a Caliper region and ends
// in a barrier with `--global-synchronization`.
#pragma once

#include <cstdint>
#include <ranges>
#include <utility>

#include <caliper/cali.h>

#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/cost_function.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/execution.h"
#include "cetric/preprocessing.h"
#include "cetric/util.h"

namespace cetric {

/// The contracted graph: the vertices with cut edges and their (oriented) cut edges.
using CompactGraph = DistributedGraph<SparseNodeIndexer>;

namespace phases {

/// Removes the edges the local phase has handled and returns the rest as the contracted graph. `ordering` only tells
/// remove_internal_edges how the neighborhoods are sorted.
template <typename Ordering>
CompactGraph contraction(DistributedGraph<>& G, Config const& conf, Ordering const& ordering) {
    CALI_CXX_MARK_SCOPE("contraction");
    for_each_local_node(G, conf, conf.local_parallel, [&](auto node) {
        // a prvalue: remove_internal_edges picks its fast path by the exact ordering type
        G.remove_internal_edges(node, Ordering(ordering));
    });
    CompactGraph G_compact =
        with_execution_policy(conf, conf.parallel_compact, [&](auto policy) { return G.compact(std::move(policy)); });
    profiling::counter_add("compact_nodes", static_cast<std::int64_t>(std::ranges::size(G_compact.local_nodes())));
    profiling::counter_add("compact_edges", static_cast<std::int64_t>(G_compact.local_edge_count_with_ghost_edges()));
    ConditionalBarrier(conf.global_synchronization, G.comm());
    return G_compact;
}

/// pre-CETRIC: repartitions the contracted graph (`--secondary-cost-function`, off with "none"). Returns whether it
/// did; then some of the contracted graph's edges are local again.
inline bool secondary_load_balancing(CompactGraph& G_compact, Config const& conf) {
    CALI_CXX_MARK_SCOPE("secondary_load_balancing");
    bool const rebalance = conf.secondary_cost_function != "none";
    if (rebalance) {
        auto cost_function [[maybe_unused]] = with_execution_policy(conf, [&](auto policy) {
            return CostFunctionRegistry<CompactGraph>::get(conf.secondary_cost_function, G_compact, conf, std::move(policy));
        });
        auto       view  = G_compact.to_local_graph_view();
        auto const range = node_range_of(view);
        G_compact        = CompactGraph(std::move(view), range, G_compact.comm());
        compute_ghost_ranks(G_compact, conf);
    }
    ConditionalBarrier(conf.global_synchronization, G_compact.comm());
    return rebalance;
}

/// pre-CETRIC: degree exchange and orientation on the contracted graph.
inline void preprocessing_global(CompactGraph& G_compact, Config const& conf) {
    CALI_CXX_MARK_SCOPE("preprocessing_global");
    orient_and_sort<node_ordering::id_outward, node_ordering::degree<CompactGraph>>(G_compact, conf, true);
    ConditionalBarrier(conf.global_synchronization, G_compact.comm());
}

} // namespace phases
} // namespace cetric
