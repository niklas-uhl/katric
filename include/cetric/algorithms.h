// The algorithms' entry points. Each overload is compiled in its own translation unit (src/), so including this
// header does not instantiate the edge iterator.
#pragma once

#include <cstddef>

#include "cetric/communication_policy.h"
#include "cetric/config.h"
#include "cetric/datastructures/distributed/distributed_graph.h"

namespace cetric {
// Each run modifies G (orientation, load balancing, ghost expansion). The returned triangle count is only valid on
// rank 0.

/// DiTriC
std::size_t run_patric(graph::DistributedGraph<>& G, Config const& conf, MessageQueuePolicy policy);
std::size_t run_patric(graph::DistributedGraph<>& G, Config const& conf, GridPolicy policy);
/// pre-CETRIC: outward orientation of cut edges, degree exchange only on the contracted graph
std::size_t run_cetric(graph::DistributedGraph<>& G, Config const& conf, MessageQueuePolicy policy);
std::size_t run_cetric(graph::DistributedGraph<>& G, Config const& conf, GridPolicy policy);
/// CETRIC (ghost expansion)
std::size_t run_cetric_x(graph::DistributedGraph<>& G, Config const& conf, MessageQueuePolicy policy);
std::size_t run_cetric_x(graph::DistributedGraph<>& G, Config const& conf, GridPolicy policy);
} // namespace cetric
