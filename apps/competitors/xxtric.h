// xxtric (Ghosh, 2026; TriC's successor), its MPI CPU path, as --algorithm
// xxtric. xxtric's headers are included only by xxtric.cpp.
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// xxtric's CPU triangle count (class TriangleCounting) on the ranks of `comm`.
auto make_xxtric(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
