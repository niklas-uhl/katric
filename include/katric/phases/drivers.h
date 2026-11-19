// The algorithm on compact ids as a sequence of phases (phases/preprocessing.h
// and below). Each phase is a Caliper region; counters
// (katric::profiling::counter_add) go to the region that is open. The returned
// triangle count is only valid on rank 0.
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/config.h"
#include "katric/graph/graph.h"
#include "katric/phases/global.h"
#include "katric/phases/local.h"
#include "katric/phases/preprocessing.h"
#include "katric/triangle_counter.h"

namespace katric {
namespace phases {

/// The intersections that need no communication (CETRIC also through the
/// ghosts' out-edges). Returns the interface vertices, the senders of DiTriC's
/// global phase.
inline auto local_phase(graph::CsrGraph const& G,
                        Config const& conf,
                        bool with_ghosts,
                        TriangleCounter& triangles)
    -> std::vector<graph::Handle> {
  CALI_CXX_MARK_SCOPE("local_phase");
  std::vector<graph::Handle> interface_nodes;
  triangles([&](auto emit) {
    interface_nodes = with_ghosts ? local::run<true>(G, conf, emit)
                                  : local::run<false>(G, conf, emit);
  });
  return interface_nodes;
}

/// The global phase: sends the neighborhoods of `senders`.
template <class CommunicationPolicy, typename Senders>
void global_phase(graph::GlobalIdGraph const& G,
                  Config const& conf,
                  std::size_t threshold,
                  Senders const& senders,
                  TriangleCounter& triangles) {
  CALI_CXX_MARK_SCOPE("global_phase");
  triangles([&](auto emit) {
    global::run<CommunicationPolicy>(G, conf, threshold, senders, emit);
  });
}

}  // namespace phases

/// KaTric: DiTriC, or CETRIC with `--contraction` (the local phase also covers
/// the triangles with two local vertices, through the ghost rows; the global
/// phase sends only the cut edges).
template <class CommunicationPolicy>
auto count_triangles(kagen::Graph const& input,
                     Config const& conf,
                     kamping::v2::comm_view comm) -> std::size_t {
  TriangleCounter triangles;
  auto [G, threshold] = phases::input_graph(input, conf, comm);
  phases::primary_load_balancing(G, conf);
  auto G_oriented = phases::preprocessing(G, conf);
  G = {};
  auto const interface_nodes =
      phases::local_phase(G_oriented, conf, conf.contraction, triangles);
  if (conf.contraction) {
    auto const G_contracted = phases::contraction(std::move(G_oriented));
    phases::global_phase<CommunicationPolicy>(G_contracted, conf, threshold,
                                              G_contracted.kept(), triangles);
  } else {
    auto const G_global = phases::global_ids(std::move(G_oriented));
    phases::global_phase<CommunicationPolicy>(G_global, conf, threshold,
                                              interface_nodes, triangles);
  }
  return phases::reduce(triangles, comm);
}

}  // namespace katric
