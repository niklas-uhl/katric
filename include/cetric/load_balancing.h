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

#ifndef NEW_LOAD_BALANCING_H_CYE8RYEL
#define NEW_LOAD_BALANCING_H_CYE8RYEL

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <sparsehash/dense_hash_map>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <graph-io/local_graph_view.h>
#include <mpi.h>

#include "cetric/config.h"
#include "cetric/cost_function.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/timer.h"
#include "cetric/util.h"

namespace cetric {
namespace load_balancing {

class LoadBalancer {
public:
    template <typename CostFunction>
    static graph::LocalGraphView
    run(graph::LocalGraphView&&             G,
        CostFunction&                       cost_function,
        const Config&                       conf,
        profiling::LoadBalancingStatistics& stats) {
        cetric::profiling::Timer timer;
        auto                     to_send    = reassign_nodes(G, cost_function, conf);
        auto                     G_balanced = GraphCommunicator::relocate(
            std::move(G),
            to_send,
            stats.message_statistics,
            conf.rank,
            conf.PEs,
            !conf.dense_load_balancing
        );
        stats.redistribution_time = timer.elapsed_time();
        return G_balanced;
    }

private:
    template <typename CostFunction>
    static google::dense_hash_map<PEID, GraphCommunicator::NodeRange>
    reassign_nodes(const graph::LocalGraphView& G, CostFunction& cost_function, const Config& conf) {
        using namespace cetric::graph;
        std::vector<size_t> cost(G.node_info.size());
        size_t              prefix_sum = 0;
        for (size_t node = 0; node < G.node_info.size(); ++node) {
            cost[node]       = prefix_sum;
            size_t node_cost = cost_function(G, node);
            prefix_sum += node_cost;
        }

        size_t global_prefix;
        MPI_Exscan(&prefix_sum, &global_prefix, 1, MPI_NODE, MPI_SUM, MPI_COMM_WORLD);
        if (conf.rank == 0) {
            global_prefix = 0;
        }
        size_t total_cost;
        if (conf.rank == conf.PEs - 1) {
            total_cost = global_prefix + prefix_sum;
        }
        MPI_Bcast(&total_cost, 1, MPI_NODE, conf.PEs - 1, MPI_COMM_WORLD);
        size_t                                                     per_pe_cost = (total_cost + conf.PEs - 1) / conf.PEs;
        google::dense_hash_map<PEID, GraphCommunicator::NodeRange> to_send(conf.PEs);
        to_send.set_empty_key(conf.PEs);
        for (size_t node = 0; node < cost.size(); ++node) {
            cost[node] += global_prefix;
            PEID new_pe = std::min(static_cast<int>(cost[node] / per_pe_cost), conf.PEs - 1);
            if (to_send.find(new_pe) == to_send.end()) {
                to_send[new_pe] = GraphCommunicator::NodeRange{node, node};
            } else {
                to_send[new_pe].to = node;
            }
        }

        cost.resize(0);
        cost.shrink_to_fit();
        return to_send;
    }
};

} // namespace load_balancing
} // namespace cetric

#endif /* cargend of include guard: NEW_LOAD_BALANCING_H_CYE8RYEL */
