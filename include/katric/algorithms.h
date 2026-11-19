// The algorithm's entry points. Each overload is compiled in its own
// translation unit (src/), so including this header does not instantiate the
// phases.
#pragma once

#include <cstddef>

#include <kagen.h>
#include <mpi.h>
#include <kamping/v2/comm.hpp>

#include "katric/communication_policy.h"
#include "katric/config.h"

namespace katric {
// Builds its graphs from `input` (KaGen's CSR representation, only read). The
// returned triangle count is only valid on rank 0.

/// KaTric: DiTriC, or CETRIC with `--contraction` (katric/phases/drivers.h)
auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm,
                     MessageQueuePolicy policy) -> std::size_t;
auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm,
                     GridPolicy policy) -> std::size_t;
}  // namespace katric
