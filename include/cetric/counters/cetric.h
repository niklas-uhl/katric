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
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <utility>

#include "cetric/datastructures/local_graph_view.h"
#include <kassert/internal/assertion_macros.hpp>
#include <kassert/kassert.hpp>
#include <kamping/types/mpi_type_traits.hpp>
#include <mpi.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
#include <tlx/algorithm/multiway_merge.hpp>
#include <tlx/multi_timer.hpp>

#include <caliper/cali.h>

#include <spdlog/spdlog.h>
#include "cetric/config.h"
#include "cetric/cost_function.h"
#include "cetric/counters/cetric_edge_iterator.h"
#include "cetric/datastructures/auxiliary_node_data.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/distributed/graph_communicator.h"
#include "cetric/datastructures/distributed/helpers.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/load_balancing.h"
#include "cetric/timer.h"
#include "cetric/util.h"

namespace cetric {

/// Calls `f` with the execution policy: parallel on `conf.num_threads` threads if `parallel`, else sequential. `f`
/// gets the policy by value and should pass it on as an rvalue (`std::move`): the graph's methods compare the policy
/// type exactly.
template <typename F>
decltype(auto) with_execution_policy(Config const& conf, bool parallel, F&& f) {
    if (parallel) {
        return f(execution_policy::parallel{conf.num_threads});
    }
    return f(execution_policy::sequential{});
}

/// with_execution_policy(), parallel if there is more than one thread.
template <typename F>
decltype(auto) with_execution_policy(Config const& conf, F&& f) {
    return with_execution_policy(conf, conf.num_threads > 1, std::forward<F>(f));
}

/// Calls `f(node)` for every local vertex of `G`, in a task arena of `conf.num_threads` threads if `parallel`.
template <typename Graph, typename F>
void for_each_local_node(Graph& G, Config const& conf, bool parallel, F&& f) {
    auto nodes = G.local_nodes();
    if (!parallel) {
        for (auto node: nodes) {
            f(node);
        }
        return;
    }
    tbb::task_arena arena(conf.num_threads);
    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
            for (auto node: r) {
                f(node);
            }
        });
    });
}

/// Optionally fetches the ghosts' degrees, then orients every edge by `OrientationOrdering` and sorts the
/// neighborhoods by `NodeOrdering`.
template <typename NodeOrdering, typename OrientationOrdering, typename NodeIndexer, typename GhostSet>
inline void preprocessing(
    DistributedGraph<NodeIndexer>& G,
    AuxiliaryNodeData<Degree>&     ghost_degree,
    GhostSet&                      ghosts,
    const Config&                  conf,
    bool                           fetch_ghost_degree
) {
    if (fetch_ghost_degree) {
        CALI_CXX_MARK_SCOPE("degree_exchange");
        find_ghosts(G, ghosts);
        ghost_degree = AuxiliaryNodeData<Degree>{ghosts.begin(), ghosts.end()};
        DegreeCommunicator comm(G, as_int(MessageTag::Orientation));
        comm.get_ghost_degree(
            [&](RankEncodedNodeId node, Degree degree) { ghost_degree[node] = degree; },
            degree_exchange(conf)
        );
    }
    bool const parallel = conf.num_threads > 1;
    {
        CALI_CXX_MARK_SCOPE("orientation");
        for_each_local_node(G, conf, parallel, [&](auto node) { G.orient(node, OrientationOrdering(G, ghost_degree)); });
    }
    CALI_CXX_MARK_SCOPE("sorting");
    for_each_local_node(G, conf, parallel, [&](auto node) { G.sort_neighborhoods(node, NodeOrdering(G, ghost_degree)); });
}

template <typename Graph>
inline void compute_ghost_ranks(Graph& G, const Config& conf) {
    with_execution_policy(conf, [&](auto policy) {
        G.find_ghost_ranks(std::move(policy));
    });
}

/// Repartitions `G` into contiguous vertex ranges of equal cost (`--primary-cost-function`).
inline void primary_load_balancing(DistributedGraph<>& G, const Config& conf) {
    if (conf.primary_cost_function == "none") {
        return;
    }
    auto cost_function = with_execution_policy(conf, [&](auto policy) {
        return CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf, std::move(policy));
    });
    LocalGraphView tmp = G.to_local_graph_view();
    tmp                = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf, G.comm());
    auto node_range    = tmp.local_node_count() == 0
                             ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                             : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
    G                  = DistributedGraph(std::move(tmp), std::move(node_range), G.comm());
    compute_ghost_ranks(G, conf);
}

/// Checks that no triangle was emitted twice (only with assertions enabled).
inline void verify_triangles(
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>>& triangles [[maybe_unused]],
    size_t                                               triangle_count [[maybe_unused]]
) {
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        SPDLOG_DEBUG("Verification started");
        KASSERT(triangles.size() == triangle_count);
        std::sort(triangles.begin(), triangles.end(), [](auto const& lhs, auto const& rhs) {
            return std::tuple(lhs.x.id(), lhs.y.id(), lhs.z.id()) < std::tuple(rhs.x.id(), rhs.y.id(), rhs.z.id());
        });
        auto current = triangles.begin();
        while (true) {
            current = std::adjacent_find(current, triangles.end());
            if (current == triangles.end()) {
                break;
            }
            KASSERT(false, "Duplicate triangle " << *current);
            current++;
        }
        SPDLOG_DEBUG("Verification finished");
    }
}

/// Returns a triangle callback that counts into `count` and, with assertions enabled, records the triangle.
inline auto
make_emit(tbb::concurrent_vector<Triangle<RankEncodedNodeId>>& triangles, tbb::combinable<size_t>& count) {
    return [&triangles, &count](Triangle<RankEncodedNodeId> t) {
        (void)t;
        (void)triangles;
        SPDLOG_DEBUG("{}", t);
        if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
            t.normalize();
            triangles.push_back(t);
        }
        count.local()++;
    };
}

/// What the phases of one run hand on to each other.
template <class CommunicationPolicy>
struct PhaseState {
    using CompactGraph = DistributedGraph<SparseNodeIndexer>;

    size_t                    threshold = 0; // message queue threshold, from the input graph
    node_set                  ghosts;
    AuxiliaryNodeData<Degree> ghost_degrees;
    /// The local phase's iterator; it holds the interface vertices that patric's global phase sends.
    std::optional<CetricEdgeIterator<DistributedGraph<>, CommunicationPolicy>> local_iterator;
    /// The contracted graph (vertices with cut edges and their cut out-edges) for the global phase.
    std::optional<CompactGraph> G_compact;
    /// Whether the secondary load balancing redistributed G_compact (then it has internal edges again).
    bool rebalanced = false;

    size_t                                              triangle_count    = 0; // found on this rank
    size_t                                              counted_triangles = 0; // sum over all ranks, on rank 0
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;             // only with assertions

    /// Calls `find(emit)` and adds the triangles emitted to the rank's count.
    template <typename Find>
    void count_triangles(Find&& find) {
        tbb::combinable<size_t> count{0};
        find(make_emit(triangles, count));
        triangle_count += count.combine(std::plus<>{});
    }
};

/// A phase of an algorithm. Each is a Caliper region (counters from cetric::profiling::counter_add go to the region
/// that is open), and all but the last end in a barrier with `--global-synchronization`.
template <class CommunicationPolicy>
using PhaseFunction = void (*)(DistributedGraph<>&, Config const&, PhaseState<CommunicationPolicy>&);

namespace phases {
using Graph        = DistributedGraph<>;
using CompactGraph = DistributedGraph<SparseNodeIndexer>;

template <class P>
void ghost_ranks(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("ghost_ranks");
    s.threshold = get_threshold(G, conf);
    compute_ghost_ranks(G, conf);
    ConditionalBarrier(true, G.comm());
}

template <class P>
void primary_load_balancing(Graph& G, Config const& conf, PhaseState<P>&) {
    CALI_CXX_MARK_SCOPE("primary_load_balancing");
    cetric::primary_load_balancing(G, conf);
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// DiTriC: degree exchange, orientation by degree, neighborhoods by id.
template <class P>
void preprocessing_patric(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    preprocessing<node_ordering::id, node_ordering::degree<Graph>>(G, s.ghost_degrees, s.ghosts, conf, true);
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// pre-CETRIC: no degree exchange; cut edges point outward, the local vertices first in each neighborhood.
template <class P>
void preprocessing_cetric(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    preprocessing<node_ordering::id_outward, node_ordering::degree_outward<Graph>>(
        G,
        s.ghost_degrees,
        s.ghosts,
        conf,
        false
    );
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// CETRIC: degree exchange, orientation by degree, the local vertices first; then the ghost expansion.
template <class P>
void preprocessing_cetric_x(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("preprocessing");
    preprocessing<node_ordering::id_outward, node_ordering::degree<Graph>>(G, s.ghost_degrees, s.ghosts, conf, true);
    {
        CALI_CXX_MARK_SCOPE("ghost_expansion");
        with_execution_policy(conf, conf.parallel_compact, [&](auto policy) {
            G.remove_in_edges_and_expand_ghosts(node_ordering::id_outward(conf.rank), std::move(policy));
        });
    }
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// The intersections that need no communication; `NodeOrdering` is the neighborhoods' order from preprocessing.
template <class P, typename NodeOrdering>
void local_phase(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("local_phase");
    s.local_iterator.emplace(G, conf, conf.rank, conf.PEs, P{});
    s.local_iterator->set_threshold(s.threshold);
    s.count_triangles([&](auto emit) { s.local_iterator->run_local(emit, NodeOrdering(G, s.ghost_degrees)); });
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// DiTriC's global phase: sends the neighborhoods of the local phase's interface vertices.
template <class P>
void global_phase_patric(Graph&, Config const&, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("global_phase");
    s.count_triangles([&](auto emit) { s.local_iterator->run_distributed(emit, node_ordering::id()); });
}

/// Removes the edges the local phase has handled and keeps the rest as the contracted graph. `Ordering` only tells
/// remove_internal_edges how the neighborhoods are sorted.
template <class P, typename Ordering>
void contraction(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("contraction");
    s.local_iterator.reset();
    for_each_local_node(G, conf, conf.local_parallel, [&](auto node) {
        G.remove_internal_edges(node, Ordering(G, s.ghost_degrees));
    });
    s.G_compact.emplace(
        with_execution_policy(conf, conf.parallel_compact, [&](auto policy) { return G.compact(std::move(policy)); })
    );
    // what is left for the global phase: vertices with cut edges and their (oriented) cut edges
    profiling::counter_add("compact_nodes", static_cast<std::int64_t>(std::ranges::size(s.G_compact->local_nodes())));
    profiling::counter_add("compact_edges", static_cast<std::int64_t>(s.G_compact->local_edge_count_with_ghost_edges()));
    s.ghosts = node_set{};
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// pre-CETRIC: repartitions the contracted graph (`--secondary-cost-function`, off by default).
template <class P>
void secondary_load_balancing(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("secondary_load_balancing");
    if (conf.secondary_cost_function != "none") {
        auto cost_function [[maybe_unused]] = with_execution_policy(conf, [&](auto policy) {
            return CostFunctionRegistry<CompactGraph>::get(
                conf.secondary_cost_function,
                *s.G_compact,
                conf,
                std::move(policy)
            );
        });
        auto tmp        = s.G_compact->to_local_graph_view();
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        s.G_compact.emplace(std::move(tmp), std::move(node_range), G.comm());
        compute_ghost_ranks(*s.G_compact, conf);
        s.rebalanced = true;
    }
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// pre-CETRIC: degree exchange and orientation on the contracted graph.
template <class P>
void preprocessing_global(Graph& G, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("preprocessing_global");
    preprocessing<node_ordering::id_outward, node_ordering::degree<CompactGraph>>(
        *s.G_compact,
        s.ghost_degrees,
        s.ghosts,
        conf,
        true
    );
    ConditionalBarrier(conf.global_synchronization, G.comm());
}

/// The global phase on the contracted graph (CETRIC, pre-CETRIC). After a secondary load balancing some of its
/// edges are local again; those go through a local phase first.
template <class P>
void global_phase_compact(Graph&, Config const& conf, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("global_phase");
    s.ghost_degrees = AuxiliaryNodeData<Degree>();
    auto&                                 G_compact = *s.G_compact;
    CetricEdgeIterator<CompactGraph, P> ctr(G_compact, conf, conf.rank, conf.PEs, P{});
    ctr.set_threshold(s.threshold);
    // the neighborhoods are sorted by id_outward; without internal edges that is the same as id
    if (s.rebalanced) {
        s.count_triangles([&](auto emit) { ctr.run_local(emit, node_ordering::id_outward(conf.rank)); });
        for_each_local_node(G_compact, conf, conf.local_parallel, [&](auto node) {
            G_compact.remove_internal_edges(node, node_ordering::id_outward(conf.rank));
        });
        s.count_triangles([&](auto emit) { ctr.run_distributed(emit, node_ordering::id()); });
    } else {
        auto nodes = G_compact.local_nodes();
        s.count_triangles([&](auto emit) { ctr.run_distributed(emit, nodes.begin(), nodes.end(), node_ordering::id()); });
    }
}

template <class P>
void reduce(Graph& G, Config const&, PhaseState<P>& s) {
    CALI_CXX_MARK_SCOPE("reduce");
    MPI_Reduce(
        &s.triangle_count,
        &s.counted_triangles,
        1,
        kamping::types::mpi_type_traits<size_t>::data_type(),
        MPI_SUM,
        0,
        G.comm()
    );
}
} // namespace phases

/// Runs the phases in order and returns the triangle count (valid on rank 0).
template <class CommunicationPolicy>
size_t
run_phases(DistributedGraph<>& G, Config const& conf, std::initializer_list<PhaseFunction<CommunicationPolicy>> phases) {
    PhaseState<CommunicationPolicy> state;
    for (auto phase: phases) {
        phase(G, conf, state);
    }
    verify_triangles(state.triangles, state.triangle_count);
    return state.counted_triangles;
}

/// DiTriC
template <class P>
size_t run_patric(DistributedGraph<>& G, Config const& conf) {
    return run_phases<P>(
        G,
        conf,
        {phases::ghost_ranks<P>,
         phases::primary_load_balancing<P>,
         phases::preprocessing_patric<P>,
         phases::local_phase<P, node_ordering::id>,
         phases::global_phase_patric<P>,
         phases::reduce<P>}
    );
}

/// pre-CETRIC
template <class P>
size_t run_cetric(DistributedGraph<>& G, Config const& conf) {
    return run_phases<P>(
        G,
        conf,
        {phases::ghost_ranks<P>,
         phases::primary_load_balancing<P>,
         phases::preprocessing_cetric<P>,
         phases::local_phase<P, node_ordering::id_outward>,
         phases::contraction<P, node_ordering::degree_outward<DistributedGraph<>>>,
         phases::secondary_load_balancing<P>,
         phases::preprocessing_global<P>,
         phases::global_phase_compact<P>,
         phases::reduce<P>}
    );
}

/// CETRIC
template <class P>
size_t run_cetric_x(DistributedGraph<>& G, Config const& conf) {
    return run_phases<P>(
        G,
        conf,
        {phases::ghost_ranks<P>,
         phases::primary_load_balancing<P>,
         phases::preprocessing_cetric_x<P>,
         phases::local_phase<P, node_ordering::id_outward>,
         phases::contraction<P, node_ordering::id_outward>,
         phases::global_phase_compact<P>,
         phases::reduce<P>}
    );
}
} // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
