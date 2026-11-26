// The 2D triangle counter of Tom & Karypis as --algorithm tom2d. Its headers
// are included only by tom2d.cpp.
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// Tom2D on the ranks of `comm`, which must be MPI_COMM_WORLD (hard-coded in
/// it) with a perfect square number of ranks; throws otherwise.
auto make_tom2d(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
