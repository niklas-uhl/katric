#ifndef CETRIC_H_1MZUS6LP
#define CETRIC_H_1MZUS6LP

#include "cost_function.h"
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/local_graph_view.h"
#include "datastructures/graph_definitions.h"
#include "tlx/algorithm/multiway_merge.hpp"
#include "tlx/multi_timer.hpp"
#include <config.h>
#include <counters/cetric_edge_iterator.h>
#include <datastructures/distributed/graph_communicator.h>
#include <load_balancing.h>
#include <statistics.h>
#include <timer.h>
#include <util.h>

enum class Phase {
    Local,
    Global
};

template<Phase phase>
inline void preprocessing(DistributedGraph<> &G,
                          cetric::profiling::Statistics &stats, const Config& conf) {
    cetric::profiling::Timer timer;
    if constexpr (phase == Phase::Local) {
            if (!conf.orient_locally) {
                GraphCommunicator comm(G, conf.rank, conf.PEs,
                                       as_int(MessageTag::Orientation));
                comm.get_ghost_degree(
                    [&](NodeId global_id, Degree degree) {
                        G.get_ghost_payload(G.to_local_id(global_id)).degree = degree;
                    },
                    stats.local.preprocessing.message_statistics);

                timer.restart();
                G.orient([&](NodeId a, NodeId b) {
                        return std::make_tuple(G.degree(a), G.to_global_id(a)) <
                            std::make_tuple(G.degree(b), G.to_global_id(b));
                    });
                stats.local.preprocessing.orientation_time += timer.elapsed_time();
            } else {
                G.orient([&](NodeId a, NodeId b) {
                        return std::make_tuple(G.local_degree(a), G.to_global_id(a)) <
                            std::make_tuple(G.local_degree(b), G.to_global_id(b));
                    });
                stats.local.preprocessing.orientation_time += timer.elapsed_time();
            }
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

inline void run_cetric(DistributedGraph<> &G,
                cetric::profiling::Statistics &stats, const Config &conf,
                PEID rank, PEID size) {

    G.find_ghost_ranks();
    if (conf.primary_cost_function != "N") {
        auto cost_function = get_cost_function_by_name(
            conf.primary_cost_function, G, rank, size);
        LocalGraphView tmp = G.to_local_graph_view(false, false);
        tmp = cetric::load_balancing::LoadBalancer::run(
            std::move(tmp), *cost_function, conf);
        G = DistributedGraph(std::move(tmp), rank, size);
        G.find_ghost_ranks();
        //TODO this is weird
        preprocessing<Phase::Local>(G, stats, conf);
    }
    G.expand_ghosts();
    preprocessing<Phase::Local>(G, stats, conf);
    cetric::CetricEdgeIterator<DistributedGraph<>, true, true>
        ctr(G, conf, rank, size);
    size_t triangle_count = 0;
    ctr.run_local(
        [&](Triangle t) {
          (void)t;
          // atomic_debug(t);
          triangle_count++;
        },
        stats);
    G.remove_internal_edges();
    if (!conf.secondary_cost_function.empty()) {
        auto cost_function = get_cost_function_by_name(
            conf.secondary_cost_function, G, rank, size);
        auto tmp =
            G.to_local_graph_view(true, false);
        tmp = cetric::load_balancing::LoadBalancer::run(std::move(tmp),
                                                        *cost_function, conf);
        G = DistributedGraph(std::move(tmp), rank, size);
        G.expand_ghosts();
        G.find_ghost_ranks();
        preprocessing<Phase::Global>(G, stats, conf);
    }
    cetric::CetricEdgeIterator<DistributedGraph<>, true, true>
        ctr_dist(G, conf, rank, size);
    ctr_dist.run(
        [&](Triangle t) {
          (void)t;
          // atomic_debug(t);
          triangle_count++;
        },
        stats);
    // tlx::MultiTimer dummy;
    // ctr.get_triangle_count(stats);
    /* size_t triangle_count = 0; */
    /* ctr.run_local([&](Triangle) { */
    /*     triangle_count++; */
    /* }, stats, dummy); */
    /* cetric::profiling::Timer phase_time; */
    /* G.remove_internal_edges(); */
    /* stats.local.contraction_time = phase_time.elapsed_time(); */
    /* ctr.run_distributed([&](Triangle) { */
    /*     triangle_count++; */
    /* }, stats, dummy); */
    MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        atomic_debug("total number of triangles: " + std::to_string(stats.counted_triangles));
    }
}

#endif /* end of include guard: CETRIC_H_1MZUS6LP */
