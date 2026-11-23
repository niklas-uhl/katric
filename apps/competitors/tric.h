// TriC (Ghosh & Halappanavar, HPEC'20) as --algorithm tric. TriC's headers are
// included only by tric.cpp.
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// TriC in the variant `conf.tric_variant`, on the ranks of `comm`.
auto make_tric(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
