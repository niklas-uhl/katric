// One translation unit per communication policy (see katric/algorithms.h).
#include <kamping/v2/comm.hpp>
#include "katric/algorithms.h"
#include "katric/phases/drivers.h"

namespace katric {
auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm,
                     GridPolicy /*policy*/) -> std::size_t {
  return count_triangles<GridPolicy>(input, conf, comm);
}
}  // namespace katric
