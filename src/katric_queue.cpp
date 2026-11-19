// One translation unit per communication policy (see katric/algorithms.h).
#include "katric/algorithms.h"
#include "katric/phases/drivers.h"

namespace katric {
std::size_t count_triangles(kagen::Graph const& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy /*policy*/) {
    return count_triangles<MessageQueuePolicy>(input, conf, comm);
}
} // namespace katric
