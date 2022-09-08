#ifndef CETRIC_H_1MZUS6LP
#define CETRIC_H_1MZUS6LP

#include <cstddef>
#include <limits>
#include <numeric>
#include <sparsehash/dense_hash_set>
#include <sstream>
#include <utility>

#include <debug_assert.hpp>
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

template <typename NodeIndexer, typename GhostSet>
inline void preprocessing(
    DistributedGraph<NodeIndexer>&              G,
    cetric::profiling::PreprocessingStatistics& stats,
    AuxiliaryNodeData<Degree>&                  ghost_degree,
    GhostSet&                                   ghosts,
    const Config&                               conf,
    Phase                                       phase
) {
    tlx::MultiTimer phase_timer;
    if (phase == Phase::Global || conf.algorithm == Algorithm::Patric) {
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

        phase_timer.start("orientation");
        auto nodes = G.local_nodes();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &ghost_degree](auto const& r) {
                for (auto node: r) {
                    G.orient(node, node_ordering::degree(G, ghost_degree));
                }
            });
        } else {
            for (auto node: nodes) {
                G.orient(node, node_ordering::degree(G, ghost_degree));
            }
        }
        phase_timer.start("sorting");
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &conf](auto const& r) {
                for (auto node: r) {
                    if (conf.algorithm == Algorithm::Patric) {
                        G.sort_neighborhoods(node, node_ordering::id());
                    } else {
                        G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
                    }
                }
            });
        } else {
            for (auto node: nodes) {
                if (conf.algorithm == Algorithm::Patric) {
                    G.sort_neighborhoods(node, node_ordering::id());
                } else {
                    G.sort_neighborhoods(node, node_ordering::id_outward(G.rank()));
                }
            }
        }
    } else {
        phase_timer.start("orientation");
        auto nodes = G.local_nodes();
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
                for (auto node: r) {
                    G.orient(node, node_ordering::degree_outward(G));
                }
            });
        } else {
            for (auto node: nodes) {
                G.orient(node, node_ordering::degree_outward(G));
            }
        }
        phase_timer.start("sorting");
        if (conf.num_threads > 1) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
                for (auto node: r) {
                    G.sort_neighborhoods(node, node_ordering::degree_outward(G));
                }
            });
        } else {
            for (auto node: nodes) {
                G.sort_neighborhoods(node, node_ordering::degree_outward(G));
            }
        }
    }
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
            G.find_ghost_ranks<true>(execution_policy::parallel{});
        } else {
            G.find_ghost_ranks<false>(execution_policy::parallel{});
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
                    execution_policy::parallel{}
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
                G.find_ghost_ranks<true>(execution_policy::parallel{});
            } else {
                G.find_ghost_ranks<false>(execution_policy::parallel{});
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
    preprocessing(G, stats.local.preprocessing_local_phase, ghost_degrees, ghosts, conf, Phase::Local);
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
        node_ordering::id()
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
inline size_t
run_cetric(DistributedGraph<>& G, cetric::profiling::Statistics& stats, const Config& conf, PEID rank, PEID size, CommunicationPolicy&&) {
    tlx::MultiTimer phase_timer;
    phase_timer.start("ghost_ranks");
    size_t threshold                   = get_threshold(G, conf);
    stats.local.global_phase_threshold = threshold;
    bool debug                         = false;
    if (conf.num_threads > 1) {
        if (conf.binary_rank_search) {
            G.find_ghost_ranks<true>(execution_policy::parallel{});
        } else {
            G.find_ghost_ranks<false>(execution_policy::parallel{});
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
                    execution_policy::parallel{}
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
                G.find_ghost_ranks<true>(execution_policy::parallel{});
            } else {
                G.find_ghost_ranks<false>(execution_policy::parallel{});
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
    preprocessing(G, stats.local.preprocessing_local_phase, ghost_degrees, ghosts, conf, Phase::Local);
    LOG << "[R" << rank << "] "
        << "Preprocessing finished ";
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
            // atomic_debug(omp_get_thread_num());
            triangle_count_local_phase.local()++;
        },
        stats,
        node_ordering::degree_outward(G)
    );
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    LOG << "[R" << rank << "] "
        << "Local phase finished ";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.start("contraction");
    auto nodes = G.local_nodes();
    if (conf.local_parallel) {
        tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
            for (auto node: r) {
                G.remove_internal_edges(node);
            }
        });
    } else {
        for (auto node: nodes) {
            G.remove_internal_edges(node);
        }
    }
    auto G_compact = G.compact();
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
                    execution_policy::parallel{}
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
                G_global_phase.find_ghost_ranks<true>(execution_policy::parallel{});
            } else {
                G_global_phase.find_ghost_ranks<false>(execution_policy::parallel{});
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
        phase_timer.start("preprocessing");
        preprocessing(
            G_global_phase,
            stats.local.preprocessing_global_phase,
            ghost_degrees,
            ghosts,
            conf,
            Phase::Global
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
            node_ordering::id_outward(rank)
        );
        // atomic_debug(
        //     fmt::format("Found {} triangles in global phase 1", triangle_count_global_phase.combine(std::plus<>{})));
        auto nodes = G_global_phase.local_nodes();
        if (conf.local_parallel) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G_global_phase](auto const& r) {
                for (auto node: r) {
                    G_global_phase.remove_internal_edges(node);
                }
            });
        } else {
            for (auto node: nodes) {
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
            stats,
            node_ordering::id() // even though the neighborhoods are sorted using id_outward, after removing the
                                // internal edges, this is the same as id
        );
        // atomic_debug(
        // fmt::format("Found {} triangles in global phase 2", triangle_count_global_phase.combine(std::plus<>{})));
    } else {
        ConditionalBarrier(conf.global_synchronization);
        phase_timer.start("preprocessing");
        preprocessing(G_compact, stats.local.preprocessing_global_phase, ghost_degrees, ghosts, conf, Phase::Global);
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
} // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
