/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CETRIC_H_1MZUS6LP
#define CETRIC_H_1MZUS6LP

#include <cstddef>
#include <vector>

#include <caliper/cali.h>
#include <kamping/types/mpi_type_traits.hpp>
#include <mpi.h>

#include "cetric/config.h"
#include "cetric/contraction.h"
#include "cetric/counters/global_phase.h"
#include "cetric/counters/local_phase.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/execution.h"
#include "cetric/preprocessing.h"
#include "cetric/triangle_counter.h"
#include "cetric/util.h"

// The three algorithms as sequences of phases (preprocessing.h, contraction.h and below). Each phase is a Caliper
// region; counters (cetric::profiling::counter_add) go to the region that is open. The returned triangle count is
// only valid on rank 0.

namespace cetric {
namespace phases {

/// The intersections that need no communication (CETRIC also through the ghosts' adjacencies). `NodeOrdering` is the
/// neighborhoods' order from preprocessing. Returns the interface vertices, the senders of DiTriC's global phase.
template <typename NodeOrdering>
std::vector<RankEncodedNodeId> local_phase(DistributedGraph<>& G, Config const& conf, TriangleCounter& triangles) {
    CALI_CXX_MARK_SCOPE("local_phase");
    std::vector<RankEncodedNodeId> interface_nodes;
    NodeOrdering const             node_ordering(G, AuxiliaryNodeData<Degree>{});
    triangles([&](auto emit) {
        if (conf.algorithm == Algorithm::cetric_x) {
            interface_nodes = cetric::local_phase::run<true>(G, conf, emit, node_ordering);
        } else {
            interface_nodes = cetric::local_phase::run<false>(G, conf, emit, node_ordering);
        }
    });
    ConditionalBarrier(conf.global_synchronization, G.comm());
    return interface_nodes;
}

/// The global phase on the contracted graph (CETRIC, pre-CETRIC): every vertex left in it sends its neighborhood.
template <class CommunicationPolicy>
void global_phase(CompactGraph& G_compact, Config const& conf, std::size_t threshold, TriangleCounter& triangles) {
    CALI_CXX_MARK_SCOPE("global_phase");
    triangles([&](auto emit) {
        cetric::global_phase::run<CommunicationPolicy>(
            G_compact, conf, threshold, G_compact.local_nodes(), emit, node_ordering::id()
        );
    });
}

} // namespace phases

/// pre-CETRIC
template <class CommunicationPolicy>
std::size_t run_cetric(DistributedGraph<>& G, Config const& conf) {
    TriangleCounter   triangles;
    std::size_t const threshold = phases::ghost_ranks(G, conf);
    phases::primary_load_balancing(G, conf);
    phases::preprocessing_cetric(G, conf);
    phases::local_phase<node_ordering::id_outward>(G, conf, triangles);
    auto       G_compact  = phases::contraction(G, conf, node_ordering::degree_outward(G));
    phases::preprocessing_global(G_compact, conf);
    phases::global_phase<CommunicationPolicy>(G_compact, conf, threshold, triangles);
    return phases::reduce(triangles, G.comm());
}

/// CETRIC
template <class CommunicationPolicy>
std::size_t run_cetric_x(DistributedGraph<>& G, Config const& conf) {
    TriangleCounter   triangles;
    std::size_t const threshold = phases::ghost_ranks(G, conf);
    phases::primary_load_balancing(G, conf);
    phases::preprocessing_cetric_x(G, conf);
    phases::local_phase<node_ordering::id_outward>(G, conf, triangles);
    auto G_compact = phases::contraction(G, conf, node_ordering::id_outward(conf.rank));
    phases::global_phase<CommunicationPolicy>(G_compact, conf, threshold, triangles);
    return phases::reduce(triangles, G.comm());
}

} // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
