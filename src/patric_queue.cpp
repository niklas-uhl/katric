// One translation unit per algorithm and communication policy (see cetric/algorithms.h).
#include "cetric/algorithms.h"
#include "cetric/counters/cetric.h"

namespace cetric {
std::size_t run_patric(graph::DistributedGraph<>& G, Config const& conf, MessageQueuePolicy /*policy*/) {
    return run_patric(G, conf, conf.rank, conf.PEs, MessageQueuePolicy{});
}
} // namespace cetric
