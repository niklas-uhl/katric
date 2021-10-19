#ifndef CETRIC_H_1MZUS6LP
#define CETRIC_H_1MZUS6LP

#include "cost_function.h"
#include <counters/cetric_edge_iterator.h>
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/graph_definitions.h"
#include "tlx/algorithm/multiway_merge.hpp"
#include "tlx/multi_timer.hpp"
#include <util.h>
#include <datastructures/distributed/graph_communicator.h>
#include <config.h>
#include <counters/cetric_edge_iterator.h>
#include <statistics.h>
#include <timer.h>
#include <load_balancing.h>

inline void preprocessing(DistributedGraph& G, cetric::profiling::Statistics& stats, PEID rank, PEID size) {
    GraphCommunicator comm(G, rank, size, MessageTag::Orientation);
    comm.distribute_degree(stats.local.preprocessing.message_statistics);
    auto degree = [&](NodeId node) {
        if (G.is_ghost(node)) {
            return comm.get_ghost_degree(node);
        } else {
            return G.initial_degree(node);
        }
    };

    cetric::profiling::Timer timer;
    G.orient([&](NodeId a, NodeId b) {
            return std::make_tuple(degree(a), G.to_global_id(a)) < std::make_tuple(degree(b), G.to_global_id(b));
    });
    G.for_each_ghost_node([&](NodeId node) {
        G.set_ghost_info(node, degree(node));
    });
    stats.local.preprocessing.orientation_time += timer.elapsed_time();

    timer.restart();
    G.sort_neighborhoods();

    stats.local.preprocessing.sorting_time += timer.elapsed_time();
}

template<class Graph>
void run_cetric(Graph& G, cetric::profiling::Statistics& stats, const Config& conf, PEID rank, PEID size) {
    preprocessing(G, stats, rank, size);

    /* if (conf.cost_function != "N") { */
    /*     auto G_simple = make_simple(std::move(G)); */
    /*     auto cost_function = get_cost_function_by_name(conf.cost_function, G_simple, rank, size); */
    /*     LoadBalancer load_balancer(G_simple, *cost_function, rank, size); */
    /*     load_balancer.relocate_graph(); */
    /*     G = DistributedGraph(std::move(G_simple)); */
    /*     preprocessing(G, stats, rank, size); */
    /* } */
    /* atomic_debug(G); */
    //atomic_debug(G_compact);
    auto cost = [](LocalGraphView& G_local, NodeId node) {
        return G_local.node_info[node].degree;
    };
    G = cetric::load_balancing::LoadBalancer::run(G.to_local_graph_view(), cost, conf);
    //atomic_debug(G_compact);
    //cetric::CetricEdgeIterator<CompactGraph, true, true> ctr(G, conf, rank, size);
    //tlx::MultiTimer dummy;
    //ctr.get_triangle_count(stats);
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
    /* MPI_Reduce(&triangle_count, &stats.counted_triangles, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD); */
}


#endif /* end of include guard: CETRIC_H_1MZUS6LP */
