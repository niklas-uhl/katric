// One translation unit per algorithm and communication policy (see cetric/algorithms.h).
#include "cetric/algorithms.h"
#include "cetric/counters/cetric.h"

namespace cetric {
std::size_t run_cetric(graph::DistributedGraph<>& G, Config const& conf, GridPolicy /*policy*/) {
    return run_cetric(G, conf, conf.rank, conf.PEs, GridPolicy{}, conf.id_node_ordering);
}
} // namespace cetric
