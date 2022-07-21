#ifndef CETRIC_H_1MZUS6LP
#define CETRIC_H_1MZUS6LP

#include <config.h>
#include <counters/cetric_edge_iterator.h>
#include <datastructures/auxiliary_node_data.h>
#include <datastructures/distributed/graph_communicator.h>
#include <load_balancing.h>
#include <statistics.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
#include <timer.h>
#include <util.h>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sparsehash/dense_hash_set>
#include <utility>
#include "atomic_debug.h"
#include "cost_function.h"
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/helpers.h"
#include "datastructures/graph_definitions.h"
#include "debug_assert.hpp"
#include "graph-io/local_graph_view.h"
#include "kassert/internal/assertion_macros.hpp"
#include "kassert/kassert.hpp"
#include "tlx/algorithm/multiway_merge.hpp"
#include "tlx/logger.hpp"
#include "tlx/multi_timer.hpp"

namespace cetric {
enum class Phase { Local, Global };

template <typename NodeIndexer, typename GhostSet>
inline void preprocessing(DistributedGraph<NodeIndexer>& G,
                          cetric::profiling::Statistics& stats,
                          AuxiliaryNodeData<Degree>& ghost_degree,
                          GhostSet& ghosts,
                          const Config& conf,
                          Phase phase) {
    cetric::profiling::Timer timer;
    if (phase == Phase::Global || conf.algorithm == Algorithm::Patric) {
        find_ghosts(G, ghosts);
        ghost_degree = AuxiliaryNodeData<Degree>{ghosts.begin(), ghosts.end()};
        DegreeCommunicator comm(G, conf.rank, conf.PEs, as_int(MessageTag::Orientation));
        comm.get_ghost_degree([&](RankEncodedNodeId node, Degree degree) { ghost_degree[node] = degree; },
                              stats.local.preprocessing.message_statistics, !conf.dense_degree_exchange);

        timer.restart();
        auto nodes = G.local_nodes();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &ghost_degree](auto const& r) {
                for (auto node : r) {
                    G.orient(node, node_ordering::degree(G, ghost_degree));
                }
            });
        } else {
            for (auto node : nodes) {
                G.orient(node, node_ordering::degree(G, ghost_degree));
            }
        }
        stats.local.preprocessing.orientation_time += timer.elapsed_time();
        timer.restart();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &conf](auto const& r) {
                for (auto node : r) {
                    if (conf.algorithm == Algorithm::Patric) {
                        G.sort_neighborhoods(node, node_ordering::id());
                    } else {
                        G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
                    }
                }
            });
        } else {
            for (auto node : nodes) {
                if (conf.algorithm == Algorithm::Patric) {
                    G.sort_neighborhoods(node, node_ordering::id());
                } else {
                    G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
                }
            }
        }
    } else {
        // G.orient(node_ordering::degree_outward(G));
        // stats.local.preprocessing.orientation_time += timer.elapsed_time();
        // timer.restart();
        // G.sort_neighborhoods(node_ordering::degree_outward(G), execution_policy::sequential{});
        auto nodes = G.local_nodes();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
                for (auto node : r) {
                    G.orient(node, node_ordering::degree_outward(G));
                }
            });
        } else {
            for (auto node : nodes) {
                G.orient(node, node_ordering::degree_outward(G));
            }
        }
        stats.local.preprocessing.orientation_time += timer.elapsed_time();
        timer.restart();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
                for (auto node : r) {
                    G.sort_neighborhoods(node, node_ordering::degree_outward(G));
                }
            });
        } else {
            for (auto node : nodes) {
                G.sort_neighborhoods(node, node_ordering::degree_outward(G));
            }
        }
    }
    stats.local.preprocessing.sorting_time += timer.elapsed_time();
}

template <class CommunicationPolicy>
inline size_t run_patric(DistributedGraph<>& G,
                         cetric::profiling::Statistics& stats,
                         const Config& conf,
                         PEID rank,
                         PEID size,
                         CommunicationPolicy&&) {
    bool debug = true;
    if (conf.num_threads > 1) {
        G.find_ghost_ranks(execution_policy::parallel{});
    } else {
        G.find_ghost_ranks(execution_policy::sequential{});
    }
    node_set ghosts;
    cetric::profiling::Timer timer;
    if ((conf.gen.generator.empty() && conf.primary_cost_function != "N") ||
        (!conf.gen.generator.empty() && conf.primary_cost_function != "none")) {
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
                                                                     stats.local.primary_load_balancing,
                                                                     execution_policy::parallel{});
            } else {
                return CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
                                                                     stats.local.primary_load_balancing,
                                                                     execution_policy::sequential{});
            }
        }();
        LocalGraphView tmp = G.to_local_graph_view();
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
                                                        stats.local.primary_load_balancing);
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        G = DistributedGraph(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            G.find_ghost_ranks(execution_policy::parallel{});
        } else {
            G.find_ghost_ranks(execution_policy::sequential{});
        }
    }
    AuxiliaryNodeData<Degree> ghost_degrees;
    stats.local.primary_load_balancing.phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished " << stats.local.primary_load_balancing.phase_time << " s";
    // G.expand_ghosts();
    preprocessing(G, stats, ghost_degrees, ghosts, conf, Phase::Local);
    LOG << "[R" << rank << "] "
        << "Preprocessing finished";
    timer.restart();
    auto ctr = cetric::CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
    size_t triangle_count = 0;
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;
    tbb::combinable<size_t> triangle_count_local_phase{0};
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
        stats, node_ordering::id());
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    stats.local.local_phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Local phase finished " << stats.local.local_phase_time << " s";
    timer.restart();
    tbb::combinable<size_t> triangle_count_global_phase{0};
    timer.restart();
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
        stats, node_ordering::id());
    triangle_count += triangle_count_global_phase.combine(std::plus<>{});
    stats.local.global_phase_time = timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Global phase finished " << stats.local.global_phase_time << " s";
    timer.restart();
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    stats.local.reduce_time = timer.elapsed_time();
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        timer.restart();
        LOG << "[R" << rank << "] "
            << "Verification started" << stats.local.local_phase_time << " s";
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
            << "Verification finished " << timer.elapsed_time() << " s";
    }
    return stats.counted_triangles;
}

template <typename CommunicationPolicy>
inline size_t run_cetric(DistributedGraph<>& G,
                         cetric::profiling::Statistics& stats,
                         const Config& conf,
                         PEID rank,
                         PEID size,
                         CommunicationPolicy&&) {
    bool debug = true;
    if (conf.num_threads > 1) {
        G.find_ghost_ranks(execution_policy::parallel{});
    } else {
        G.find_ghost_ranks(execution_policy::sequential{});
    }
    node_set ghosts;
    cetric::profiling::Timer timer;
    if ((conf.gen.generator.empty() && conf.primary_cost_function != "N") ||
        (!conf.gen.generator.empty() && conf.primary_cost_function != "none")) {
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
                                                                     stats.local.primary_load_balancing,
                                                                     execution_policy::parallel{});
            } else {
                return CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
                                                                     stats.local.primary_load_balancing,
                                                                     execution_policy::sequential{});
            }
        }();
        LocalGraphView tmp = G.to_local_graph_view();
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
                                                        stats.local.primary_load_balancing);
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        G = DistributedGraph(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            G.find_ghost_ranks(execution_policy::parallel{});
        } else {
            G.find_ghost_ranks(execution_policy::sequential{});
        }
    }
    AuxiliaryNodeData<Degree> ghost_degrees;
    stats.local.primary_load_balancing.phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished " << stats.local.primary_load_balancing.phase_time << " s";
    // G.expand_ghosts();
    preprocessing(G, stats, ghost_degrees, ghosts, conf, Phase::Local);
    LOG << "[R" << rank << "] "
        << "Preprocessing finished";
    timer.restart();
    auto ctr = cetric::CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
    size_t triangle_count = 0;
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;
    tbb::combinable<size_t> triangle_count_local_phase{0};
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
        stats, node_ordering::degree_outward(G));
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    stats.local.local_phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Local phase finished " << stats.local.local_phase_time << " s";
    timer.restart();
    auto nodes = G.local_nodes();
    if (conf.local_parallel) {
        tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
            for (auto node : r) {
                G.remove_internal_edges(node);
            }
        });
    } else {
        for (auto node : nodes) {
            G.remove_internal_edges(node);
        }
    }
    auto G_compact = G.compact();
    stats.local.contraction_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Contraction finished " << stats.local.contraction_time << " s";
    ghosts = decltype(ghosts){};
    // atomic_debug(fmt::format("Found {} triangles in local phase",
    // triangle_count_local_phase.combine(std::plus<>{})));
    tbb::combinable<size_t> triangle_count_global_phase{0};
    if (conf.secondary_cost_function != "none") {
        timer.restart();
        auto cost_function = [&]() {
            if (conf.num_threads > 1) {
                return CostFunctionRegistry<decltype(G_compact)>::get(conf.secondary_cost_function, G_compact, conf,
                                                                      stats.local.secondary_load_balancing,
                                                                      execution_policy::parallel{});
            } else {
                return CostFunctionRegistry<decltype(G_compact)>::get(conf.secondary_cost_function, G_compact, conf,
                                                                      stats.local.secondary_load_balancing,
                                                                      execution_policy::sequential{});
            }
        }();
        auto tmp = G_compact.to_local_graph_view();
        auto node_range = tmp.local_node_count() == 0
                              ? std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::max())
                              : std::make_pair(tmp.node_info.front().global_id, tmp.node_info.back().global_id + 1);
        auto G_global_phase = DistributedGraph<SparseNodeIndexer>(std::move(tmp), std::move(node_range), rank, size);
        if (conf.num_threads > 1) {
            G_global_phase.find_ghost_ranks(execution_policy::parallel{});
        } else {
            G_global_phase.find_ghost_ranks(execution_policy::sequential{});
        }
        stats.local.secondary_load_balancing.phase_time += timer.elapsed_time();
        LOG << "[R" << rank << "] "
            << "Secondary load balancing finished " << stats.local.secondary_load_balancing.phase_time << " s";
        preprocessing(G_global_phase, stats, ghost_degrees, ghosts, conf, Phase::Global);
        ghost_degrees = AuxiliaryNodeData<Degree>();
        timer.restart();
        cetric::CetricEdgeIterator ctr_global(G_global_phase, conf, rank, size, CommunicationPolicy{});
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
            stats, node_ordering::id_outward(rank));
        // atomic_debug(
        //     fmt::format("Found {} triangles in global phase 1", triangle_count_global_phase.combine(std::plus<>{})));
        auto nodes = G_global_phase.local_nodes();
        if (conf.local_parallel) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G_global_phase](auto const& r) {
                for (auto node : r) {
                    G_global_phase.remove_internal_edges(node);
                }
            });
        } else {
            for (auto node : nodes) {
                G_global_phase.remove_internal_edges(node);
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
            stats, node_ordering::id_outward(rank));
        // atomic_debug(
        // fmt::format("Found {} triangles in global phase 2", triangle_count_global_phase.combine(std::plus<>{})));
    } else {
        preprocessing(G_compact, stats, ghost_degrees, ghosts, conf, Phase::Global);
        ghost_degrees = AuxiliaryNodeData<Degree>();
        // }
        timer.restart();
        // if (conf.secondary_cost_function != "none") {
        //     auto ctr_dist = cetric::CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
        //     ctr_dist.run(
        //         [&](Triangle<RankEncodedNodeId> t) {
        //             (void)t;
        //             t.normalize();
        //             if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
        //                 triangles.push_back(t);
        //             }
        //             // atomic_debug(t);
        //             triangle_count++;
        //             // triangle_count_global_phase.local()++;
        //         },
        //         stats, node_ordering::id());
        // } else {
        // atomic_debug(fmt::format("local nodes: {}", G.local_nodes()));
        // atomic_debug(fmt::format("ghost degrees: {}", ghost_degrees));
        cetric::CetricEdgeIterator ctr_global(G_compact, conf, rank, size, CommunicationPolicy{});
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
            stats, G_compact.local_nodes().begin(), G_compact.local_nodes().end(), node_ordering::id_outward(rank));
        // }
    }
    triangle_count += triangle_count_global_phase.combine(std::plus<>{});
    stats.local.global_phase_time = timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Global phase finished " << stats.local.global_phase_time << " s";
    timer.restart();
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    stats.local.reduce_time = timer.elapsed_time();
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        timer.restart();
        LOG << "[R" << rank << "] "
            << "Verification started" << stats.local.local_phase_time << " s";
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
            << "Verification finished " << timer.elapsed_time() << " s";
    }
    return stats.counted_triangles;
}
}  // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
