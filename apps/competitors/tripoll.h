// TriPoll (Steil et al., SC'21) as --algorithm tripoll. TriPoll's and YGM's
// headers are included only by tripoll.cpp.
#pragma once

#include <memory>

#include <kamping/v2/comm.hpp>

#include "../algorithm.h"
#include "katric/config.h"

namespace katric {

/// TriPoll's triangle count (push-only or push-pull, --tripoll-variant) on
/// the ranks of `comm`, through a YGM communicator created once from it.
auto make_tripoll(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

}  // namespace katric
