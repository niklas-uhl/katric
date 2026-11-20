// The algorithm (declared in katric/algorithms.h), compiled once here so that
// the harness does not instantiate the phases.
#include <cstddef>
#include <utility>

#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/algorithms.h"
#include "katric/config.h"
#include "katric/phase_counters.h"
#include "katric/phases/drivers.h"
#include "katric/triangle_counter.h"

namespace katric {

auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm) -> std::size_t {
  TriangleCounter triangles;
  auto G =
      phases::primary_load_balancing(phases::input_graph(input, comm), conf);
  // from the graph the phases run on (after the load balancing)
  std::size_t const threshold = get_threshold(G, conf);
  phases::preprocessing(G, conf);
  phases::local_phase(G, conf, triangles);
  phases::global_ids(G, conf);
  phases::global_phase(G, conf, threshold, triangles);
  return phases::reduce(triangles, comm);
}

}  // namespace katric
