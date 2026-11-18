// The algorithms' entry points. Each overload is compiled in its own translation unit (src/), so including this
// header does not instantiate the phases.
#pragma once

#include <cstddef>

#include <mpi.h>

#include "cetric/communication_policy.h"
#include "cetric/config.h"
#include <kagen.h>

namespace cetric {
// Each run builds its graphs from `input` (KaGen's CSR representation, only read). The returned triangle count is only valid on rank 0.

/// DiTriC
std::size_t run_patric(kagen::Graph const& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy policy);
std::size_t run_patric(kagen::Graph const& input, Config const& conf, MPI_Comm comm, GridPolicy policy);
/// pre-CETRIC: outward orientation of cut edges, degree exchange only on the contracted graph
std::size_t run_cetric(kagen::Graph const& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy policy);
std::size_t run_cetric(kagen::Graph const& input, Config const& conf, MPI_Comm comm, GridPolicy policy);
/// CETRIC (ghost rows)
std::size_t run_cetric_x(kagen::Graph const& input, Config const& conf, MPI_Comm comm, MessageQueuePolicy policy);
std::size_t run_cetric_x(kagen::Graph const& input, Config const& conf, MPI_Comm comm, GridPolicy policy);
} // namespace cetric
