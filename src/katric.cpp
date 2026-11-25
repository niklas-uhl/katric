// The algorithm (declared in katric/katric.h), compiled once here so that the
// harness does not instantiate the phases. Each phase function opens its
// Caliper region and reports its counters; a single call gets a block here.
#include "katric/katric.h"

#include <cstddef>
#include <utility>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/config.h"
#include "katric/graph/graph.h"
#include "katric/phases/global.h"
#include "katric/phases/load_balancing.h"
#include "katric/phases/local.h"
#include "katric/phases/preprocessing.h"
#include "katric/triangle_counter.h"

namespace katric {

auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm) -> std::size_t {
  TriangleCounter triangles;
  auto G = [&] {
    CALI_CXX_MARK_SCOPE("input_graph");
    return graph::Graph::from_kagen(input, comm);
  }();
  G = primary_load_balancing(std::move(G), conf);  // vertices move: a new graph
  std::size_t const threshold = get_threshold(G, conf);
  preprocessing(G, conf);
  local_phase(G, conf, triangles);
  global_ids(G, conf);
  global_phase(G, conf, threshold, triangles);
  return reduce(triangles, comm);
}

}  // namespace katric
