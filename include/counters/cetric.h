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
#include <numeric>
#include <sparsehash/dense_hash_set>
#include "cost_function.h"
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/helpers.h"
#include "datastructures/graph_definitions.h"
#include "debug_assert.hpp"
#include "graph-io/local_graph_view.h"
#include "kassert/kassert.hpp"
#include "tlx/algorithm/multiway_merge.hpp"
#include "tlx/logger.hpp"
#include "tlx/multi_timer.hpp"

namespace cetric {
enum class Phase { Local, Global };

template <typename GhostSet>
inline void preprocessing(DistributedGraph<>& G,
                          cetric::profiling::Statistics& stats,
                          AuxiliaryNodeData<Degree>& ghost_degree,
                          GhostSet& ghosts,
                          const Config& conf,
                          Phase phase) {
    cetric::profiling::Timer timer;
    if (phase == Phase::Global) {
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
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &ghost_degree](auto const& r) {
                for (auto node : r) {
                    G.sort_neighborhoods(node, node_ordering::id());
                }
            });
        } else {
            for (auto node : nodes) {
                G.sort_neighborhoods(node, node_ordering::id());
            }
        }
    } else {
        // G.orient(node_ordering::degree_outward(G));
        // stats.local.preprocessing.orientation_time += timer.elapsed_time();
        // timer.restart();
        // G.sort_neighborhoods(node_ordering::degree_outward(G), execution_policy::sequential{});
        auto nodes = G.local_nodes();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &ghost_degree](auto const& r) {
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
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &ghost_degree](auto const& r) {
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
    bool debug = false;
    G.find_ghost_ranks();
    google::dense_hash_set<RankEncodedNodeId, cetric::hash> ghosts;
    ghosts.set_empty_key(RankEncodedNodeId::sentinel());
    cetric::profiling::Timer timer;
    // if ((conf.gen.generator.empty() && conf.primary_cost_function != "N") ||
    //     (!conf.gen.generator.empty() && conf.primary_cost_function != "none")) {
    //     auto cost_function = CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
    //                                                                        stats.local.primary_load_balancing);
    //     LocalGraphView tmp = G.to_local_graph_view(true, false);
    //     tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
    //                                                     stats.local.primary_load_balancing);
    //     G = DistributedGraph(std::move(tmp), rank, size);
    //     G.find_ghost_ranks();
    // }
    AuxiliaryNodeData<Degree> ghost_degrees;
    stats.local.primary_load_balancing.phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished " << stats.local.primary_load_balancing.phase_time << " s";
    if (conf.skip_local_neighborhood) {
        // G.expand_ghosts();
    }
    preprocessing(G, stats, ghost_degrees, ghosts, conf, Phase::Global);
    LOG << "[R" << rank << "] "
        << "Preprocessing finished";
    timer.restart();
    auto ctr = CetricEdgeIterator(G, conf, rank, size, CommunicationPolicy{});
    size_t triangle_count = 0;
    tbb::combinable<size_t> triangle_count_local_phase{0};
    ctr.run_local(
        [&](Triangle<RankEncodedNodeId> t) {
            (void)t;
            triangle_count_local_phase.local()++;
        },
        stats, node_ordering::id());
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    stats.local.local_phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Local phase finished " << stats.local.local_phase_time << " s";
    LOG << "[R" << rank << "] "
        << "Contraction finished " << stats.local.contraction_time << " s";
    timer.restart();
    ctr.run_distributed(
        [&](Triangle<RankEncodedNodeId> t) {
            (void)t;
            triangle_count++;
        },
        stats, node_ordering::id(), ghosts);
    stats.local.global_phase_time = timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Global phase finished " << stats.local.global_phase_time << " s";
    timer.restart();
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    stats.local.reduce_time = timer.elapsed_time();
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
    google::dense_hash_set<RankEncodedNodeId, cetric::hash> ghosts;
    ghosts.set_empty_key(RankEncodedNodeId::sentinel());
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
        LocalGraphView tmp = G.to_local_graph_view(true, false);
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
                                                        stats.local.primary_load_balancing);
        G = DistributedGraph(std::move(tmp), rank, size);
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
    stats.local.contraction_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Contraction finished " << stats.local.contraction_time << " s";
    // timer.restart();
    // if (conf.secondary_cost_function != "none") {
    //     auto cost_function = CostFunctionRegistry<DistributedGraph<>>::get(conf.secondary_cost_function, G, conf,
    //                                                                        stats.local.secondary_load_balancing);
    //     auto tmp = G.to_local_graph_view(true, false);
    //     tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
    //                                                     stats.local.secondary_load_balancing);
    //     G = DistributedGraph(std::move(tmp), rank, size);
    //     G.expand_ghosts();
    //     G.find_ghost_ranks();
    //     preprocessing(G, stats, conf, Phase::Global);
    // } else {
    preprocessing(G, stats, ghost_degrees, ghosts, conf, Phase::Global);
    ghosts = decltype(ghosts){};
    ghost_degrees = AuxiliaryNodeData<Degree>();
    // }
    // stats.local.secondary_load_balancing.phase_time += timer.elapsed_time();
    // LOG << "[R" << rank << "] "
    //     << "Secondary load balancing finished " << stats.local.secondary_load_balancing.phase_time << " s";
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
        stats, node_ordering::id());
    // }
    triangle_count += triangle_count_global_phase.combine(std::plus<>{});
    stats.local.global_phase_time = timer.elapsed_time();
    if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
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
    }
    LOG << "[R" << rank << "] "
        << "Global phase finished " << stats.local.global_phase_time << " s";
    timer.restart();
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    stats.local.reduce_time = timer.elapsed_time();
    return stats.counted_triangles;
}
}  // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
