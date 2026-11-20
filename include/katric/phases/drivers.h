// The algorithm on compact ids as a sequence of phases (phases/preprocessing.h
// and below). Each phase is a Caliper region; counters
// (katric::profiling::counter_add) go to the region that is open. The returned
// triangle count is only valid on rank 0.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/graph/graph.h"
#include "katric/phases/global.h"
#include "katric/phases/local.h"
#include "katric/phases/preprocessing.h"
#include "katric/triangle_counter.h"

namespace katric {
namespace phases {

/// The intersections that need no communication (CETRIC also through the
/// ghosts' out-edges, if preprocessing built them).
inline void local_phase(graph::Graph const& G,
                        Config const& conf,
                        TriangleCounter& triangles) {
  CALI_CXX_MARK_SCOPE("local_phase");
  auto const before = triangles.count();
  local::run(G, conf, triangles);
  profiling::counter_add("local_triangles", static_cast<std::int64_t>(triangles.count() - before));
}

/// The global phase: sends the neighborhoods of the vertices with cut edges.
template <class CommunicationPolicy>
void global_phase(graph::GlobalIdGraph const& G,
                  Config const& conf,
                  std::size_t threshold,
                  TriangleCounter& triangles) {
  CALI_CXX_MARK_SCOPE("global_phase");
  auto const before = triangles.count();
  global::run<CommunicationPolicy>(G, conf, threshold, triangles);
  profiling::counter_add("type3_triangles", static_cast<std::int64_t>(triangles.count() - before));
}

}  // namespace phases

/// KaTric: DiTriC, or CETRIC with `--contraction` (the local phase also covers
/// the triangles with two local vertices, through the ghosts' out-edges; the
/// global phase sends only the cut edges).
template <class CommunicationPolicy>
auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm) -> std::size_t {
  TriangleCounter triangles;
  auto G =
      phases::primary_load_balancing(phases::input_graph(input, comm), conf);
  // from the graph the phases run on (after the load balancing)
  std::size_t const threshold = get_threshold(G, conf);
  auto G_oriented = phases::preprocessing(std::move(G), conf);
  phases::local_phase(G_oriented, conf, triangles);
  auto const G_global = conf.contraction
                            ? phases::contraction(std::move(G_oriented))
                            : phases::global_ids(std::move(G_oriented));
  phases::global_phase<CommunicationPolicy>(G_global, conf, threshold,
                                            triangles);
  return phases::reduce(triangles, comm);
}

}  // namespace katric
