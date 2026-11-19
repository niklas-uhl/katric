// One translation unit per communication policy (see cetric/algorithms.h).
#include "cetric/algorithms.h"
#include "cetric/phases/drivers.h"

namespace cetric {
std::size_t count_triangles(kagen::Graph const& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy /*policy*/) {
    return count_triangles<MessageQueuePolicy>(input, conf, comm);
}
} // namespace cetric
