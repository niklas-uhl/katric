// LCC (Strausz et al., IPDPS'22: asynchronous triangle counting and LCC with
// RMA caching) as --algorithm lcc. GPL-3.0: compiled in only with
// KATRIC_COMPETITOR_LCC=ON (doc/competitors.md, decision 5).
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// LCC's triangle count on the ranks of `comm`, which must be MPI_COMM_WORLD
/// (LCC uses it throughout).
auto make_lcc(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
