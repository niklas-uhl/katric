// One translation unit per algorithm and communication policy (see cetric/algorithms.h).
#include <utility>

#include "cetric/algorithms.h"
#include "cetric/phases/drivers.h"

namespace cetric {
std::size_t run_patric(kagen::Graph&& input, Config const& conf, MPI_Comm comm, GridPolicy /*policy*/) {
    return run_patric<GridPolicy>(std::move(input), conf, comm);
}
} // namespace cetric
