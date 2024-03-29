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
#include <limits>
#include <numeric>
#include <sparsehash/dense_hash_set>
#include <sstream>
#include <utility>

#include <graph-io/local_graph_view.h>
#include <kassert/internal/assertion_macros.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
#include <tlx/algorithm/multiway_merge.hpp>
#include <tlx/logger.hpp>
#include <tlx/multi_timer.hpp>

#include "cetric/atomic_debug.h"
#include "cetric/config.h"
#include "cetric/cost_function.h"
#include "cetric/counters/cetric_edge_iterator.h"
#include "cetric/datastructures/auxiliary_node_data.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/distributed/graph_communicator.h"
#include "cetric/datastructures/distributed/helpers.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/load_balancing.h"
#include "cetric/statistics.h"
#include "cetric/timer.h"
#include "cetric/util.h"

namespace cetric {
enum class Phase { Local, Global };

template <typename NodeOrdering, typename OrientationOrdering, typename NodeIndexer, typename GhostSet>
inline void preprocessing(
    DistributedGraph<NodeIndexer>&              G,
    cetric::profiling::PreprocessingStatistics& stats,
    AuxiliaryNodeData<Degree>&                  ghost_degree,
    GhostSet&                                   ghosts,
    const Config&                               conf,
    bool                                        fetch_ghost_degree
) {
    tlx::MultiTimer phase_timer;
    // if (phase == Phase::Global || conf.algorithm == Algorithm::Patric) {
    if (fetch_ghost_degree) {
        phase_timer.start("degree_exchange");
        find_ghosts(G, ghosts);
        ghost_degree = AuxiliaryNodeData<Degree>{ghosts.begin(), ghosts.end()};
        DegreeCommunicator comm(G, conf.rank, conf.PEs, as_int(MessageTag::Orientation));
        comm.get_ghost_degree(
            [&](RankEncodedNodeId node, Degree degree) { ghost_degree[node] = degree; },
            stats.message_statistics,
            !conf.dense_degree_exchange,
            conf.compact_degree_exchange
        );
    }

    phase_timer.start("orientation");
    auto nodes = G.local_nodes();
    if (conf.num_threads > 1) {
        tbb::task_arena arena(conf.num_threads);
        arena.execute([&] {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &ghost_degree](auto const& r) {
                for (auto node: r) {
                    G.orient(node, OrientationOrdering(G, ghost_degree));
                }
            });
            phase_timer.start("sorting");
            tbb::parallel_for(
                tbb::blocked_range(nodes.begin(), nodes.end()),
                [&G, &conf, &ghost_degree](auto const& r) {
                    for (auto node: r) {
                        G.sort_neighborhoods(node, NodeOrdering(G, ghost_degree));
                        // if (conf.algorithm == Algorithm::Patric) {
                        //     G.sort_neighborhoods(node, node_ordering::id());
                        // } else if (conf.algorithm == Algorithm::CetricX) {
                        //     G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
                        // } else {
                        //     G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
                        // }
                    }
                }
            );
        });
    } else {
        for (auto node: nodes) {
            G.orient(node, OrientationOrdering(G, ghost_degree));
        }
        phase_timer.start("sorting");
        for (auto node: nodes) {
            G.sort_neighborhoods(node, NodeOrdering(G, ghost_degree));
            // if (conf.algorithm == Algorithm::Patric) {
            //   G.sort_neighborhoods(node, node_ordering::id());
            // } else if (conf.algorithm == Algorithm::CetricX) {
            //   G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
            // } else {
            //   G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
            // }
        }
    }
    // } else {
    //     phase_timer.start("orientation");
    //     auto nodes = G.local_nodes();
    //     if (conf.num_threads > 1) {
    //         tbb::task_arena arena(conf.num_threads, 0);
    //         arena.execute([&] {
    //             tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
    //                 for (auto node: r) {
    //                     G.orient(node, node_ordering::degree_outward(G));
    //                 }
    //             });
    //             phase_timer.start("sorting");
    //             tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
    //                 for (auto node: r) {
    //                     G.sort_neighborhoods(node, node_ordering::degree_outward(G));
    //                 }
    //             });
    //         });
    //     } else {
    //         for (auto node: nodes) {
    //             G.orient(node, node_ordering::degree_outward(G));
    //         }
    //         phase_timer.start("sorting");
    //         for (auto node: nodes) {
    //             G.sort_neighborhoods(node, node_ordering::degree_outward(G));
    //         }
    //     }
    // }
    phase_timer.stop();
    stats.ingest(phase_timer);
}

template <class CommunicationPolicy>
inline size_t
run_patric(DistributedGraph<>& G, cetric::profiling::Statistics& stats, const Config& conf, PEID rank, PEID size, CommunicationPolicy&&) {
    tlx::MultiTimer phase_timer;
    size_t          threshold          = get_threshold(G, conf);
    stats.local.global_phase_threshold = threshold;
    bool debug                         = false;
    if (conf.num_threads > 1) {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
        } else {
            G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
        }
    } else {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::sequential{});
        } else {
            G.find_ghost_ranks<false>(execution_policy::sequential{});
        }
    }
    node_set ghosts;
    ConditionalBarrier(true);
    phase_timer.start("primary_load_balancing");
    bool no_load_balancing_required = (conf.primary_cost_function == "none");
    if (!no_load_balancing_required) {
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<DistributedGraph<>>::get(
                    conf.primary_cost_function,
                    G,
                    conf,
                    stats.local.primary_load_balancing,
                    execution_policy::parallel{conf.num_threads}
                );
            } else {
                return CostFunctionRegistry<DistributedGraph<>>::get(
                    conf.primary_cost_function,
                    G,
                    conf,
                    stats.local.primary_load_balancing,
                    execution_policy::sequential{}
                );
            }
        }();
        LocalGraphView tmp = G.to_local_graph_view();
        tmp                = cetric::load_balancing::LoadBalancer::run(
            std::move(tmp),
            cost_function,
            conf,
            stats.local.primary_load_balancing
        );
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        G               = DistributedGraph(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            if (conf.binary_rank_search) {
                G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
            } else {
                G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
            }
        } else {
            if (conf.binary_rank_search) {
                G.find_ghost_ranks<true>(execution_policy::sequential{});
            } else {
                G.find_ghost_ranks<false>(execution_policy::sequential{});
            }
        }
    }
    ConditionalBarrier(conf.global_synchronization);
    AuxiliaryNodeData<Degree> ghost_degrees;
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished ";
    // G.expand_ghosts();
    phase_timer.start("preprocessing");
    preprocessing<node_ordering ::id, node_ordering::degree<decltype(G)>>(
        G,
        stats.local.preprocessing_local_phase,
        ghost_degrees,
        ghosts,
        conf,
        true
    );
    LOG << "[R" << rank << "] "
        << "Preprocessing finished";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("local_phase");
    auto ctr = cetric::CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
    ctr.set_threshold(threshold);
    size_t                                              triangle_count = 0;
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;
    tbb::combinable<size_t>                             triangle_count_local_phase{0};
    ctr.run_local(
        [&](Triangle<RankEncodedNodeId> t) {
            (void)t;
            // atomic_debug(t);

            if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                t.normalize();
                triangles.push_back(t);
            }
            // atomic_debug(tbb::this_task_arena::current_thread_index());
            // triangle_count++;
            triangle_count_local_phase.local()++;
        },
        stats,
        node_ordering::id(),
        node_ordering::degree(G, ghost_degrees)
    );
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Local phase finished ";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("global_phase");
    tbb::combinable<size_t> triangle_count_global_phase{0};
    ctr.run_distributed(
        [&](Triangle<RankEncodedNodeId> t) {
            (void)t;
            if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                t.normalize();
                triangles.push_back(t);
            }
            // atomic_debug(t);
            // triangle_count++;
            triangle_count_global_phase.local()++;
        },
        stats,
        node_ordering::id()
    );
    triangle_count += triangle_count_global_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Global phase finished ";
    phase_timer.start("reduction");
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    phase_timer.stop();
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        LOG << "[R" << rank << "] "
            << "Verification started";
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
        LOG << "[R" << rank << "] "
            << "Verification finished";
    }
    stats.local.ingest(phase_timer);
    return stats.counted_triangles;
}

template <typename CommunicationPolicy>
inline size_t run_cetric(
    DistributedGraph<>&            G,
    cetric::profiling::Statistics& stats,
    const Config&                  conf,
    PEID                           rank,
    PEID                           size,
    CommunicationPolicy&&,
    bool id_node_ordering = false
) {
    tlx::MultiTimer phase_timer;
    phase_timer.start("ghost_ranks");
    size_t threshold                   = get_threshold(G, conf);
    stats.local.global_phase_threshold = threshold;
    bool debug                         = false;
    if (conf.num_threads > 1) {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
        } else {
            G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
        }
    } else {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::sequential{});
        } else {
            G.find_ghost_ranks<false>(execution_policy::sequential{});
        }
    }
    node_set ghosts;
    ConditionalBarrier(true);
    phase_timer.start("primary_load_balancing");
    bool no_load_balancing_required = (conf.primary_cost_function == "none");
    if (!no_load_balancing_required) {
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<DistributedGraph<>>::get(
                    conf.primary_cost_function,
                    G,
                    conf,
                    stats.local.primary_load_balancing,
                    execution_policy::parallel{conf.num_threads}
                );
            } else {
                return CostFunctionRegistry<DistributedGraph<>>::get(
                    conf.primary_cost_function,
                    G,
                    conf,
                    stats.local.primary_load_balancing,
                    execution_policy::sequential{}
                );
            }
        }();
        LocalGraphView tmp = G.to_local_graph_view();
        tmp                = cetric::load_balancing::LoadBalancer::run(
            std::move(tmp),
            cost_function,
            conf,
            stats.local.primary_load_balancing
        );
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        G               = DistributedGraph(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            if (conf.binary_rank_search) {
                G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
            } else {
                G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
            }
        } else {
            if (conf.binary_rank_search) {
                G.find_ghost_ranks<true>(execution_policy::sequential{});
            } else {
                G.find_ghost_ranks<false>(execution_policy::sequential{});
            }
        }
    }
    AuxiliaryNodeData<Degree> ghost_degrees;
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished ";
    // G.expand_ghosts();

    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("preprocessing");
    if (!id_node_ordering) {
        preprocessing<node_ordering::degree_outward<decltype(G)>, node_ordering::degree_outward<decltype(G)>>(
            G,
            stats.local.preprocessing_local_phase,
            ghost_degrees,
            ghosts,
            conf,
            false
        );
    } else {
        preprocessing<node_ordering::id_outward, node_ordering::degree_outward<decltype(G)>>(
            G,
            stats.local.preprocessing_local_phase,
            ghost_degrees,
            ghosts,
            conf,
            false
        );
    }
    LOG << "[R" << rank << "] "
        << "Preprocessing finished ";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("local_phase");
    auto ctr = cetric::CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
    ctr.set_threshold(threshold);
    size_t                                              triangle_count = 0;
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;
    tbb::combinable<size_t>                             triangle_count_local_phase{0};
    if (!id_node_ordering) {
        ctr.run_local(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                // atomic_debug(t);

                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(tbb::this_task_arena::current_thread_index());
                // triangle_count++;
                // atomic_debug(omp_get_thread_num());
                triangle_count_local_phase.local()++;
            },
            stats,
            node_ordering::degree_outward(G),
            node_ordering::degree_outward(G)
        );
    } else {
        ctr.run_local(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                // atomic_debug(t);

                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(tbb::this_task_arena::current_thread_index());
                // triangle_count++;
                // atomic_debug(omp_get_thread_num());
                triangle_count_local_phase.local()++;
            },
            stats,
            node_ordering::id_outward(rank),
            node_ordering::degree_outward(G)
        );
    }
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Local phase finished ";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("contraction");
    auto nodes = G.local_nodes();
    if (conf.local_parallel) {
        tbb::task_arena arena(conf.num_threads);
        arena.execute([&] {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
                for (auto node: r) {
                    G.remove_internal_edges(node, node_ordering::degree_outward(G));
                }
            });
        });
    } else {
        for (auto node: nodes) {
            G.remove_internal_edges(node, node_ordering::degree_outward(G));
        }
    }
    auto G_compact = conf.parallel_compact ? G.compact(execution_policy::parallel{conf.num_threads})
                                           : G.compact(execution_policy::sequential{});
    LOG << "[R" << rank << "] "
        << "Contraction finished ";
    ghosts = decltype(ghosts){};
    // atomic_debug(fmt::format("Found {} triangles in local phase",
    // triangle_count_local_phase.combine(std::plus<>{})));
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("secondary_load_balancing");
    tbb::combinable<size_t> triangle_count_global_phase{0};
    if (conf.secondary_cost_function != "none") {
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<decltype(G_compact)>::get(
                    conf.secondary_cost_function,
                    G_compact,
                    conf,
                    stats.local.secondary_load_balancing,
                    execution_policy::parallel{conf.num_threads}
                );
            } else {
                return CostFunctionRegistry<decltype(G_compact)>::get(
                    conf.secondary_cost_function,
                    G_compact,
                    conf,
                    stats.local.secondary_load_balancing,
                    execution_policy::sequential{}
                );
            }
        }();
        auto tmp            = G_compact.to_local_graph_view();
        auto node_range     = tmp.local_node_count() == 0
                                  ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                                  : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        auto G_global_phase = DistributedGraph<SparseNodeIndexer>(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            if (conf.binary_rank_search) {
                G_global_phase.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
            } else {
                G_global_phase.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
            }
        } else {
            if (conf.binary_rank_search) {
                G_global_phase.find_ghost_ranks<true>(execution_policy::sequential{});
            } else {
                G_global_phase.find_ghost_ranks<false>(execution_policy::sequential{});
            }
        }
        LOG << "[R" << rank << "] "
            << "Secondary load balancing finished ";
        ConditionalBarrier(conf.global_synchronization);
        phase_timer.start("preprocessing_global");
        preprocessing<node_ordering::id_outward, node_ordering::degree<decltype(G_global_phase)>>(
            G_global_phase,
            stats.local.preprocessing_global_phase,
            ghost_degrees,
            ghosts,
            conf,
            true
        );
        ConditionalBarrier(conf.global_synchronization);
        phase_timer.start("global_phase");
        ghost_degrees = AuxiliaryNodeData<Degree>();
        cetric::CetricEdgeIterator ctr_global(G_global_phase, conf, rank, size, CommunicationPolicy{});
        ctr_global.set_threshold(threshold);
        ctr_global.run_local(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(t);
                // triangle_count++;
                triangle_count_global_phase.local()++;
            },
            stats,
            node_ordering::id_outward(rank),
            node_ordering::degree_outward(G_global_phase, ghost_degrees)
        );
        // atomic_debug(
        //     fmt::format("Found {} triangles in global phase 1", triangle_count_global_phase.combine(std::plus<>{})));
        auto nodes = G_global_phase.local_nodes();
        if (conf.local_parallel) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G_global_phase, &rank](auto const& r) {
                for (auto node: r) {
                    G_global_phase.remove_internal_edges(node, node_ordering::id_outward(rank));
                }
            });
        } else {
            for (auto node: nodes) {
                G_global_phase.remove_internal_edges(node, node_ordering::id_outward(rank));
            }
        }
        ctr_global.run_distributed(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(t);
                // triangle_count++;
                triangle_count_global_phase.local()++;
            },
            stats,
            node_ordering::id() // even though the neighborhoods are sorted using id_outward, after removing the
                                // internal edges, this is the same as id
        );
        // atomic_debug(
        // fmt::format("Found {} triangles in global phase 2", triangle_count_global_phase.combine(std::plus<>{})));
    } else {
        ConditionalBarrier(conf.global_synchronization);
        phase_timer.start("preprocessing_global");
        preprocessing<node_ordering::id_outward, node_ordering::degree<decltype(G_compact)>>(
            G_compact,
            stats.local.preprocessing_global_phase,
            ghost_degrees,
            ghosts,
            conf,
            true
        );
        ConditionalBarrier(conf.global_synchronization);
        phase_timer.start("global_phase");
        ghost_degrees = AuxiliaryNodeData<Degree>();
        cetric::CetricEdgeIterator ctr_global(G_compact, conf, rank, size, CommunicationPolicy{});
        ctr_global.set_threshold(threshold);
        ctr_global.run_distributed(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(t);
                // triangle_count++;
                triangle_count_global_phase.local()++;
            },
            stats,
            G_compact.local_nodes().begin(),
            G_compact.local_nodes().end(),
            node_ordering::id() // even though the neighborhoods are sorted
                                // using id_outward, after removing the
                                // internal edges, this is the same as id
        );
        // }
    }
    triangle_count += triangle_count_global_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Global phase finished ";
    phase_timer.start("reduce");
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    phase_timer.stop();
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        LOG << "[R" << rank << "] "
            << "Verification started";
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
        LOG << "[R" << rank << "] "
            << "Verification finished ";
    }
    stats.local.ingest(phase_timer);
    return stats.counted_triangles;
}

template <typename CommunicationPolicy>
inline size_t
run_cetric_new(DistributedGraph<>& G, cetric::profiling::Statistics& stats, const Config& conf, PEID rank, PEID size, CommunicationPolicy&&) {
    tlx::MultiTimer phase_timer;
    phase_timer.start("ghost_ranks");
    size_t threshold                   = get_threshold(G, conf);
    stats.local.global_phase_threshold = threshold;
    bool debug                         = false;
    if (conf.num_threads > 1) {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
        } else {
            G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
        }
    } else {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::sequential{});
        } else {
            G.find_ghost_ranks<false>(execution_policy::sequential{});
        }
    }
    node_set ghosts;
    ConditionalBarrier(true);
    phase_timer.start("primary_load_balancing");
    bool no_load_balancing_required = (conf.primary_cost_function == "none");
    if (!no_load_balancing_required) {
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<DistributedGraph<>>::get(
                    conf.primary_cost_function,
                    G,
                    conf,
                    stats.local.primary_load_balancing,
                    execution_policy::parallel{conf.num_threads}
                );
            } else {
                return CostFunctionRegistry<DistributedGraph<>>::get(
                    conf.primary_cost_function,
                    G,
                    conf,
                    stats.local.primary_load_balancing,
                    execution_policy::sequential{}
                );
            }
        }();
        LocalGraphView tmp = G.to_local_graph_view();
        tmp                = cetric::load_balancing::LoadBalancer::run(
            std::move(tmp),
            cost_function,
            conf,
            stats.local.primary_load_balancing
        );
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        G               = DistributedGraph(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            if (conf.binary_rank_search) {
                G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
            } else {
                G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
            }
        } else {
            if (conf.binary_rank_search) {
                G.find_ghost_ranks<true>(execution_policy::sequential{});
            } else {
                G.find_ghost_ranks<false>(execution_policy::sequential{});
            }
        }
    }
    AuxiliaryNodeData<Degree> ghost_degrees;
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished ";
    // G.expand_ghosts();

    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("preprocessing");
    // preprocessing<node_ordering::id_outward,
    // node_ordering::degree<decltype(G)>>(
    if (!conf.id_node_ordering) {
        preprocessing<node_ordering::degree<decltype(G)>, node_ordering::degree<decltype(G)>>(
            G,
            stats.local.preprocessing_local_phase,
            ghost_degrees,
            ghosts,
            conf,
            true
        );
    } else {
        preprocessing<node_ordering::id_outward, node_ordering::degree<decltype(G)>>(
            G,
            stats.local.preprocessing_local_phase,
            ghost_degrees,
            ghosts,
            conf,
            true
        );
    }
    // TODO: work with degree ID sorting
    AuxiliaryNodeData<Degree> original_degrees;
    if (!conf.id_node_ordering) {
        original_degrees = AuxiliaryNodeData<Degree>{
            RankEncodedNodeId{G.node_range().first, static_cast<uint16_t>(rank)},
            RankEncodedNodeId{G.node_range().second, static_cast<uint16_t>(rank)}};
    }

    auto nodes = G.local_nodes();
    if (!conf.id_node_ordering) {
        if (conf.num_threads > 1) {
            tbb::task_arena arena(conf.num_threads);
            arena.execute([&] {
                tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
                    for (auto node: r) {
                        original_degrees[node] = G.degree(node);
                    }
                });
            });
        } else {
            for (auto node: nodes) {
                original_degrees[node] = G.degree(node);
            }
        }
    }
    if (!conf.id_node_ordering) {
        if (conf.parallel_compact) {
            G.remove_in_edges_and_expand_ghosts(
                node_ordering::degree(G, ghost_degrees, original_degrees),
                // node_ordering::id_outward(rank),
                execution_policy::parallel{conf.num_threads}
            );
        } else {
            G.remove_in_edges_and_expand_ghosts(
                // node_ordering::id_outward(rank),
                node_ordering::degree(G, ghost_degrees, original_degrees),
                execution_policy::sequential{}
            );
        }
    } else {
        if (conf.parallel_compact) {
            G.remove_in_edges_and_expand_ghosts(
                // node_ordering::degree(G, ghost_degrees, original_degrees),
                node_ordering::id_outward(rank),
                execution_policy::parallel{conf.num_threads}
            );
        } else {
            G.remove_in_edges_and_expand_ghosts(
                node_ordering::id_outward(rank),
                // node_ordering::degree(G, ghost_degrees, original_degrees),
                execution_policy::sequential{}
            );
        }
    }
    LOG << "[R" << rank << "] "
        << "Preprocessing finished ";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("local_phase");
    auto ctr = cetric::CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
    ctr.set_threshold(threshold);
    size_t                                              triangle_count = 0;
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;
    tbb::combinable<size_t>                             triangle_count_local_phase{0};
    if (!conf.id_node_ordering) {
        ctr.run_local(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                // atomic_debug(t);

                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(tbb::this_task_arena::current_thread_index());
                // triangle_count++;
                // atomic_debug(omp_get_thread_num());
                triangle_count_local_phase.local()++;
            },
            stats,
            // node_ordering::id_outward(rank),
            node_ordering::degree(G, ghost_degrees, original_degrees),
            node_ordering::degree(G, ghost_degrees, original_degrees)
        );
    } else {
        ctr.run_local(
            [&](Triangle<RankEncodedNodeId> t) {
                (void)t;
                // atomic_debug(t);

                if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                    t.normalize();
                    triangles.push_back(t);
                }
                // atomic_debug(tbb::this_task_arena::current_thread_index());
                // triangle_count++;
                // atomic_debug(omp_get_thread_num());
                triangle_count_local_phase.local()++;
            },
            stats,
            node_ordering::id_outward(rank),
            // node_ordering::degree(G, ghost_degrees, original_degrees),
            node_ordering::degree(G, ghost_degrees, original_degrees)
        );
    }
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Local phase finished ";
    ConditionalBarrier(conf.global_synchronization);
    original_degrees = AuxiliaryNodeData<Degree>{};
    phase_timer.start("contraction");
    if (!conf.id_node_ordering) {
        if (conf.local_parallel) {
            tbb::task_arena arena(conf.num_threads);
            arena.execute([&] {
                tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
                    for (auto node: r) {
                        G.remove_internal_edges(
                            node,
                            // node_ordering::id_outward(rank));
                            node_ordering::degree(G, ghost_degrees, original_degrees)
                        );
                    }
                });
            });
        } else {
            for (auto node: nodes) {
                G.remove_internal_edges(
                    node,
                    // node_ordering::id_outward(rank));
                    node_ordering::degree(G, ghost_degrees, original_degrees)
                );
            }
        }
    } else {
        if (conf.local_parallel) {
            tbb::task_arena arena(conf.num_threads);
            arena.execute([&] {
                tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
                    for (auto node: r) {
                        G.remove_internal_edges(node, node_ordering::id_outward(rank));
                    }
                });
            });
        } else {
            for (auto node: nodes) {
                G.remove_internal_edges(node, node_ordering::id_outward(rank));
            }
        }
    }
    auto G_compact     = conf.parallel_compact ? G.compact(execution_policy::parallel{conf.num_threads})
                                               : G.compact(execution_policy::sequential{});
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("preprocessing_global");
    auto compact_nodes = G_compact.local_nodes();
    if (!conf.id_node_ordering) {
        if (conf.local_parallel) {
            tbb::task_arena arena(conf.num_threads);
            arena.execute([&] {
                tbb::parallel_for(tbb::blocked_range(compact_nodes.begin(), compact_nodes.end()), [&](auto const& r) {
                    for (auto node: r) {
                        G_compact.sort_neighborhoods(node, node_ordering::id());
                    }
                });
            });
        } else {
            for (auto node: compact_nodes) {
                G_compact.sort_neighborhoods(node, node_ordering::id());
            }
        }
    }
    LOG << "[R" << rank << "] "
        << "Contraction finished ";
    ghosts = decltype(ghosts){};
    // atomic_debug(fmt::format("Found {} triangles in local phase",
    // triangle_count_local_phase.combine(std::plus<>{})));
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("secondary_load_balancing");
    tbb::combinable<size_t> triangle_count_global_phase{0};
    ConditionalBarrier(conf.global_synchronization);
    //phase_timer.start("preprocessing");
    // preprocessing(G_compact, stats.local.preprocessing_global_phase, ghost_degrees, ghosts, conf, Phase::Global);
    // ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("global_phase");
    ghost_degrees = AuxiliaryNodeData<Degree>();
    // atomic_debug(fmt::format("ghosts={}", ghosts));
    // atomic_debug(fmt::format("ghost_degrees={}", ghost_degrees.range()));
    cetric::CetricEdgeIterator ctr_global(G_compact, conf, rank, size, CommunicationPolicy{});
    ctr_global.set_threshold(threshold);
    ctr_global.run_distributed(
        [&](Triangle<RankEncodedNodeId> t) {
            (void)t;
            if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                t.normalize();
                triangles.push_back(t);
            }
            // atomic_debug(t);
            // triangle_count++;
            triangle_count_global_phase.local()++;
        },
        stats,
        G_compact.local_nodes().begin(),
        G_compact.local_nodes().end(),
        node_ordering::id() // even though the neighborhoods are sorted
        // node_ordering::degree(
        //     G_compact,
        //     ghost_degrees,
        //     original_degrees
        // ), // even though the neighborhoods are sorted
        // using id_outward, after removing the
        // internal edges, this is the same as id
        // ghosts
    );
    triangle_count += triangle_count_global_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Global phase finished ";
    phase_timer.start("reduce");
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    phase_timer.stop();
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        LOG << "[R" << rank << "] "
            << "Verification started";
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
        LOG << "[R" << rank << "] "
            << "Verification finished ";
    }
    stats.local.ingest(phase_timer);
    return stats.counted_triangles;
}
} // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
