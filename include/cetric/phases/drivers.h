// The algorithms on compact ids as sequences of phases (phases/preprocessing.h and below). Each phase is a Caliper
// region; counters (cetric::profiling::counter_add) go to the region that is open. The returned triangle count is only
// valid on rank 0.
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <caliper/cali.h>

#include "cetric/config.h"
#include "cetric/datastructures/local_graph_view.h"
#include "cetric/graph/graph.h"
#include "cetric/phases/global.h"
#include "cetric/phases/local.h"
#include "cetric/phases/preprocessing.h"
#include "cetric/triangle_counter.h"
#include "cetric/util.h"

namespace cetric {
namespace phases {

/// The intersections that need no communication (CETRIC also through the ghost rows). Returns the interface vertices,
/// the senders of DiTriC's global phase.
inline std::vector<graph::Handle>
local_phase(graph::CsrGraph const& G, Config const& conf, bool with_ghosts, TriangleCounter& triangles) {
    CALI_CXX_MARK_SCOPE("local_phase");
    std::vector<graph::Handle> interface_nodes;
    triangles([&](auto emit) {
        interface_nodes = with_ghosts ? local::run<true>(G, conf, emit) : local::run<false>(G, conf, emit);
    });
    return interface_nodes;
}

/// The global phase: sends the neighborhoods of `senders`.
template <class CommunicationPolicy, typename Graph, typename Senders>
void global_phase(
    Graph const& G, Config const& conf, std::size_t threshold, Senders const& senders, TriangleCounter& triangles
) {
    CALI_CXX_MARK_SCOPE("global_phase");
    triangles([&](auto emit) { global::run<CommunicationPolicy>(G, conf, threshold, senders, emit); });
}

} // namespace phases

/// DiTriC
template <class CommunicationPolicy>
std::size_t run_patric(graph::LocalInput&& input, Config const& conf, MPI_Comm comm) {
    TriangleCounter triangles;
    auto [G, threshold] = phases::input_graph(std::move(input), conf, comm);
    phases::primary_load_balancing(G, conf);
    auto const G_oriented      = phases::preprocessing_patric(G, conf);
    G                          = {};
    auto const interface_nodes = phases::local_phase(G_oriented, conf, false, triangles);
    phases::global_phase<CommunicationPolicy>(G_oriented, conf, threshold, interface_nodes, triangles);
    return phases::reduce(triangles, comm);
}

/// pre-CETRIC: outward orientation of cut edges, degree exchange only on the contracted graph
template <class CommunicationPolicy>
std::size_t run_cetric(graph::LocalInput&& input, Config const& conf, MPI_Comm comm) {
    TriangleCounter triangles;
    auto [G, threshold] = phases::input_graph(std::move(input), conf, comm);
    phases::primary_load_balancing(G, conf);
    auto const G_oriented = phases::preprocessing_cetric(G, conf);
    G                     = {};
    phases::local_phase(G_oriented, conf, false, triangles);
    auto const G_contracted = phases::preprocessing_global(phases::contraction(G_oriented, conf), conf);
    phases::global_phase<CommunicationPolicy>(G_contracted, conf, threshold, G_contracted.local_nodes(), triangles);
    return phases::reduce(triangles, comm);
}

/// CETRIC (ghost rows)
template <class CommunicationPolicy>
std::size_t run_cetric_x(graph::LocalInput&& input, Config const& conf, MPI_Comm comm) {
    TriangleCounter triangles;
    auto [G, threshold] = phases::input_graph(std::move(input), conf, comm);
    phases::primary_load_balancing(G, conf);
    auto const G_oriented = phases::preprocessing_cetric_x(G, conf);
    G                     = {};
    phases::local_phase(G_oriented, conf, true, triangles);
    auto const G_contracted = phases::contraction(G_oriented, conf);
    phases::global_phase<CommunicationPolicy>(G_contracted, conf, threshold, G_contracted.local_nodes(), triangles);
    return phases::reduce(triangles, comm);
}

} // namespace cetric
