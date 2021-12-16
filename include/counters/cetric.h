#ifndef CETRIC_H_1MZUS6LP
#define CETRIC_H_1MZUS6LP

#include <config.h>
#include <counters/cetric_edge_iterator.h>
#include <datastructures/distributed/graph_communicator.h>
#include <load_balancing.h>
#include <statistics.h>
#include <timer.h>
#include <util.h>
#include "cost_function.h"
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/local_graph_view.h"
#include "datastructures/graph_definitions.h"
#include "debug_assert.hpp"
#include "tlx/algorithm/multiway_merge.hpp"
#include "tlx/logger.hpp"
#include "tlx/multi_timer.hpp"

namespace cetric {
enum class Phase { Local, Global };

inline void preprocessing(DistributedGraph<>& G,
                          cetric::profiling::Statistics& stats,
                          const Config& conf,
                          Phase phase) {
    cetric::profiling::Timer timer;
    if (!conf.orient_locally || phase == Phase::Global) {
        DegreeCommunicator comm(G, conf.rank, conf.PEs, as_int(MessageTag::Orientation));
        comm.get_ghost_degree(
            [&](NodeId global_id, Degree degree) { G.get_ghost_payload(G.to_local_id(global_id)).degree = degree; },
            stats.local.preprocessing.message_statistics);
        G.get_graph_payload().ghost_degree_available = true;

        timer.restart();
        G.orient([&](NodeId a, NodeId b) {
            return std::make_tuple(G.degree(a), G.to_global_id(a)) < std::make_tuple(G.degree(b), G.to_global_id(b));
        });
        stats.local.preprocessing.orientation_time += timer.elapsed_time();
    } else {
        G.orient([&](NodeId a, NodeId b) {
            return std::make_tuple(G.local_degree(a), G.to_global_id(a)) <
                   std::make_tuple(G.local_degree(b), G.to_global_id(b));
        });
        stats.local.preprocessing.orientation_time += timer.elapsed_time();
    }
    timer.restart();
    G.sort_neighborhoods();

    stats.local.preprocessing.sorting_time += timer.elapsed_time();
}

inline size_t run_patric(DistributedGraph<>& G,
                         cetric::profiling::Statistics& stats,
                         const Config& conf,
                         PEID rank,
                         PEID size) {
    bool debug = false;
    G.find_ghost_ranks();
    cetric::profiling::Timer timer;
    if (conf.primary_cost_function != "N") {
        auto cost_function = CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
                                                                           stats.local.primary_load_balancing);
        LocalGraphView tmp = G.to_local_graph_view(true, false);
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
                                                        stats.local.primary_load_balancing);
        G = DistributedGraph(std::move(tmp), rank, size);
        G.find_ghost_ranks();
    }
    stats.local.primary_load_balancing.phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished " << stats.local.primary_load_balancing.phase_time << " s";
    if (conf.skip_local_neighborhood) {
        G.expand_ghosts();
    }
    preprocessing(G, stats, conf, Phase::Global);
    LOG << "[R" << rank << "] "
        << "Preprocessing finished";
    timer.restart();
    cetric::CetricEdgeIterator<DistributedGraph<>> ctr(G, conf, rank, size);
    size_t triangle_count = 0;
    ctr.run_local(
        [&](Triangle t) {
            (void)t;
            triangle_count++;
        },
        stats);
    stats.local.local_phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Local phase finished " << stats.local.local_phase_time << " s";
    LOG << "[R" << rank << "] "
        << "Contraction finished " << stats.local.contraction_time << " s";
    timer.restart();
    ctr.run_distributed(
        [&](Triangle t) {
            (void)t;
            triangle_count++;
        },
        stats);
    stats.local.global_phase_time = timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Global phase finished " << stats.local.global_phase_time << " s";
    timer.restart();
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    stats.local.reduce_time = timer.elapsed_time();
    return stats.counted_triangles;
}

inline size_t run_cetric(DistributedGraph<>& G,
                         cetric::profiling::Statistics& stats,
                         const Config& conf,
                         PEID rank,
                         PEID size) {
    bool debug = false;
    G.find_ghost_ranks();
    cetric::profiling::Timer timer;
    if (conf.primary_cost_function != "N") {
        auto cost_function = CostFunctionRegistry<DistributedGraph<>>::get(conf.primary_cost_function, G, conf,
                                                                           stats.local.primary_load_balancing);
        LocalGraphView tmp = G.to_local_graph_view(true, false);
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
                                                        stats.local.primary_load_balancing);
        G = DistributedGraph(std::move(tmp), rank, size);
        G.find_ghost_ranks();
    }
    stats.local.primary_load_balancing.phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Primary load balancing finished " << stats.local.primary_load_balancing.phase_time << " s";
    G.expand_ghosts();
    preprocessing(G, stats, conf, Phase::Local);
    LOG << "[R" << rank << "] "
        << "Preprocessing finished";
    timer.restart();
    cetric::CetricEdgeIterator<DistributedGraph<>> ctr(G, conf, rank, size);
    size_t triangle_count = 0;
    ctr.run_local(
        [&](Triangle t) {
            (void)t;
            // atomic_debug(t);
            triangle_count++;
        },
        stats);
    stats.local.local_phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Local phase finished " << stats.local.local_phase_time << " s";
    timer.restart();
    G.remove_internal_edges();
    stats.local.contraction_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Contraction finished " << stats.local.contraction_time << " s";
    timer.restart();
    if (conf.secondary_cost_function != "none") {
        auto cost_function = CostFunctionRegistry<DistributedGraph<>>::get(conf.secondary_cost_function, G, conf,
                                                                           stats.local.secondary_load_balancing);
        auto tmp = G.to_local_graph_view(true, false);
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp), cost_function, conf,
                                                        stats.local.secondary_load_balancing);
        G = DistributedGraph(std::move(tmp), rank, size);
        G.expand_ghosts();
        G.find_ghost_ranks();
        preprocessing(G, stats, conf, Phase::Global);
    } else {
        if (conf.orient_locally) {
            preprocessing(G, stats, conf, Phase::Global);
        }
    }
    stats.local.secondary_load_balancing.phase_time += timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Secondary load balancing finished " << stats.local.secondary_load_balancing.phase_time << " s";
    timer.restart();
    if (!conf.secondary_cost_function.empty()) {
        cetric::CetricEdgeIterator<DistributedGraph<>> ctr_dist(G, conf, rank, size);
        ctr_dist.run(
            [&](Triangle t) {
                (void)t;
                // atomic_debug(t);
                triangle_count++;
            },
            stats);
    } else {
        ctr.run_distributed(
            [&](Triangle t) {
                (void)t;
                // atomic_debug(t);
                triangle_count++;
            },
            stats);
    }
    stats.local.global_phase_time = timer.elapsed_time();
    LOG << "[R" << rank << "] "
        << "Global phase finished " << stats.local.global_phase_time << " s";
    timer.restart();
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    stats.local.reduce_time = timer.elapsed_time();
    return stats.counted_triangles;
}
}  // namespace cetric

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
