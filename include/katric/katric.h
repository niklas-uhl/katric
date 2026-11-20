// The algorithm's entry point, compiled in src/katric.cpp, so including this
// header does not instantiate the phases.
#pragma once

#include <cstddef>

#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/config.h"

namespace katric {
/// KaTric: DiTriC, or CETRIC with `--contraction` (the local phase also covers
/// the triangles with two local vertices, through the ghosts' out-edges; the
/// global phase sends only the cut edges), with `--communication-policy` in the
/// global phase. Builds its graphs from `input` (KaGen's CSR representation,
/// only read). The returned triangle count is only valid on rank 0.
auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm) -> std::size_t;
}  // namespace katric
