// HavoqGT's triangle count (Pearce et al.) as --algorithm havoqgt. HavoqGT's
// headers are included only by havoqgt.cpp.
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// HavoqGT's triangle count on the ranks of `comm`, which must be
/// MPI_COMM_WORLD (HavoqGT hard-codes it). Throws if the ranks are not placed
/// in equal, contiguous blocks per node.
auto make_havoqgt(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
