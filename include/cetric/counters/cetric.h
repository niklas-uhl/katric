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

#include <caliper/cali.h>
#include <kamping/types/mpi_type_traits.hpp>
#include <mpi.h>

#include "cetric/config.h"
#include "cetric/contraction.h"
#include "cetric/counters/cetric_edge_iterator.h"
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

/// The intersections that need no communication. `NodeOrdering` is the neighborhoods' order from preprocessing.
/// Returns the edge iterator, which holds the interface vertices for DiTriC's global phase.
template <class CommunicationPolicy, typename NodeOrdering>
auto local_phase(DistributedGraph<>& G, Config const& conf, std::size_t threshold, TriangleCounter& triangles) {
    CALI_CXX_MARK_SCOPE("local_phase");
    CetricEdgeIterator<DistributedGraph<>, CommunicationPolicy> ctr(G, conf, conf.rank, conf.PEs, CommunicationPolicy{});
    ctr.set_threshold(threshold);
    triangles([&](auto emit) { ctr.run_local(emit, NodeOrdering(G, AuxiliaryNodeData<Degree>{})); });
    ConditionalBarrier(conf.global_synchronization, G.comm());
    return ctr;
}

/// DiTriC's global phase: sends the neighborhoods of the local phase's interface vertices.
template <typename EdgeIterator>
void global_phase(EdgeIterator& ctr, TriangleCounter& triangles) {
    CALI_CXX_MARK_SCOPE("global_phase");
    triangles([&](auto emit) { ctr.run_distributed(emit, node_ordering::id()); });
}

/// The global phase on the contracted graph (CETRIC, pre-CETRIC). After a secondary load balancing (`rebalanced`)
/// some of its edges are local again; those go through a local phase first.
template <class CommunicationPolicy>
void global_phase(
    CompactGraph& G_compact, Config const& conf, std::size_t threshold, bool rebalanced, TriangleCounter& triangles
) {
    CALI_CXX_MARK_SCOPE("global_phase");
    CetricEdgeIterator<CompactGraph, CommunicationPolicy> ctr(G_compact, conf, conf.rank, conf.PEs, CommunicationPolicy{});
    ctr.set_threshold(threshold);
    // the neighborhoods are sorted by id_outward; without internal edges that is the same as id
    if (rebalanced) {
        triangles([&](auto emit) { ctr.run_local(emit, node_ordering::id_outward(conf.rank)); });
        for_each_local_node(G_compact, conf, conf.local_parallel, [&](auto node) {
            G_compact.remove_internal_edges(node, node_ordering::id_outward(conf.rank));
        });
        triangles([&](auto emit) { ctr.run_distributed(emit, node_ordering::id()); });
    } else {
        auto nodes = G_compact.local_nodes();
        triangles([&](auto emit) { ctr.run_distributed(emit, nodes.begin(), nodes.end(), node_ordering::id()); });
    }
}

/// The sum of the ranks' triangle counts, on rank 0.
inline std::size_t reduce(TriangleCounter& triangles, MPI_Comm comm) {
    std::size_t total = 0;
    {
        CALI_CXX_MARK_SCOPE("reduce");
        std::size_t const local = triangles.count();
        MPI_Reduce(&local, &total, 1, kamping::types::mpi_type_traits<std::size_t>::data_type(), MPI_SUM, 0, comm);
    }
    triangles.verify();
    return total;
}

} // namespace phases

/// DiTriC
template <class CommunicationPolicy>
std::size_t run_patric(DistributedGraph<>& G, Config const& conf) {
    TriangleCounter   triangles;
    std::size_t const threshold = phases::ghost_ranks(G, conf);
    phases::primary_load_balancing(G, conf);
    phases::preprocessing_patric(G, conf);
    auto ctr = phases::local_phase<CommunicationPolicy, node_ordering::id>(G, conf, threshold, triangles);
    phases::global_phase(ctr, triangles);
    return phases::reduce(triangles, G.comm());
}

/// pre-CETRIC
template <class CommunicationPolicy>
std::size_t run_cetric(DistributedGraph<>& G, Config const& conf) {
    TriangleCounter   triangles;
    std::size_t const threshold = phases::ghost_ranks(G, conf);
    phases::primary_load_balancing(G, conf);
    phases::preprocessing_cetric(G, conf);
    phases::local_phase<CommunicationPolicy, node_ordering::id_outward>(G, conf, threshold, triangles);
    auto       G_compact  = phases::contraction(G, conf, node_ordering::degree_outward(G));
    bool const rebalanced = phases::secondary_load_balancing(G_compact, conf);
    phases::preprocessing_global(G_compact, conf);
    phases::global_phase<CommunicationPolicy>(G_compact, conf, threshold, rebalanced, triangles);
    return phases::reduce(triangles, G.comm());
}

/// CETRIC
template <class CommunicationPolicy>
std::size_t run_cetric_x(DistributedGraph<>& G, Config const& conf) {
    TriangleCounter   triangles;
    std::size_t const threshold = phases::ghost_ranks(G, conf);
    phases::primary_load_balancing(G, conf);
    phases::preprocessing_cetric_x(G, conf);
    phases::local_phase<CommunicationPolicy, node_ordering::id_outward>(G, conf, threshold, triangles);
    auto G_compact = phases::contraction(G, conf, node_ordering::id_outward(conf.rank));
    phases::global_phase<CommunicationPolicy>(G_compact, conf, threshold, false, triangles);
    return phases::reduce(triangles, G.comm());
}

} // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
