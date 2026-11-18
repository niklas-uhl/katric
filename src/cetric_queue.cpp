// One translation unit per algorithm and communication policy (see cetric/algorithms.h).
#include <utility>

#include "cetric/algorithms.h"
#include "cetric/phases/drivers.h"

namespace cetric {
std::size_t run_cetric(graph::LocalInput&& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy /*policy*/) {
    return run_cetric<MessageQueuePolicy>(std::move(input), conf, comm);
}
} // namespace cetric
