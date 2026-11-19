// The algorithm's entry points. Each overload is compiled in its own translation unit (src/), so including this
// header does not instantiate the phases.
#pragma once

#include <cstddef>

#include <mpi.h>

#include "cetric/communication_policy.h"
#include "cetric/config.h"
#include <kagen.h>

namespace cetric {
// Builds its graphs from `input` (KaGen's CSR representation, only read). The returned triangle count is only valid on
// rank 0.

/// KaTric: DiTriC, or CETRIC with `--contraction` (cetric/phases/drivers.h)
std::size_t count_triangles(kagen::Graph const& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy policy);
std::size_t count_triangles(kagen::Graph const& input, Config const& conf, MPI_Comm comm, GridPolicy policy);
} // namespace cetric
