// One translation unit per communication policy (see cetric/algorithms.h).
#include "cetric/algorithms.h"
#include "cetric/phases/drivers.h"

namespace cetric {
std::size_t count_triangles(kagen::Graph const& input, Config const& conf, MPI_Comm comm, GridPolicy /*policy*/) {
    return count_triangles<GridPolicy>(input, conf, comm);
}
} // namespace cetric
