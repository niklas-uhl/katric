// GRAPE / libgrape-lite's optimized LCC (Fan et al.) as --algorithm grape.
// GRAPE's headers are included only by grape.cpp.
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// GRAPE's LCCOpt on the ranks of `comm`, which must be MPI_COMM_WORLD. Needs
/// MPI_THREAD_MULTIPLE (GRAPE's message manager sends and receives from its
/// own threads); throws if MPI was initialized with less.
auto make_grape(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
