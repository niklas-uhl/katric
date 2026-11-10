// Copyright (c) 2020-2023 Tim Niklas Uhl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstddef>

#include <gtest/gtest.h>
#include "cetric/load_balancing.h"
#include <kamping/types/mpi_type_traits.hpp>
#include <mpi.h>
#include "cetric/util.h"

#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph_definitions.h"
#include "test_graphs.h"

namespace {
using namespace cetric;
using namespace cetric::graph;
class LoadBalancing : public ::testing::TestWithParam<const char*> {
protected:
    void SetUp() override {
        auto input = GetParam();
        // GENERATE("examples/metis-sample.metis", "examples/triangle.metis");
        //  auto input = GENERATE("examples/triangle.metis");
        SCOPED_TRACE(input);

        // the MPI stuff
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // the whole graph, read on every rank
        G_full = cetric::testing::read_full_graph(input);

        conf.PEs  = size;
        conf.rank = rank;

        // read it distributed
        G_view = cetric::testing::read_local_graph(input).G;
    }

    std::vector<std::vector<Edge>>            G_full;
    LocalGraphView                            G_view;
    Config                                    conf;
    PEID                                      rank, size;
};

TEST_P(LoadBalancing, BalancingWithConstCost) {
    auto G_view_old = G_view;
    auto cost       = [](const LocalGraphView&, NodeId) {
        return 1;
    };
    size_t per_pe_cost = (G_full.size() + size - 1) / size;
    G_view             = load_balancing::LoadBalancer::run(std::move(G_view), cost, conf);
    for (size_t node = 0; node < G_view.node_info.size(); ++node) {
        auto expected_degree = G_full.at(G_view.node_info.at(node).global_id).size();
        auto actual_degree   = G_view.node_info.at(node).degree;
        EXPECT_EQ(G_view.node_info.at(node).global_id / per_pe_cost, rank);
        EXPECT_EQ(expected_degree, actual_degree);
    }
    NodeId local_node_count = G_view.node_info.size();
    NodeId total_node_count = 0;
    MPI_Reduce(&local_node_count, &total_node_count, 1, kamping::types::mpi_type_traits<NodeId>::data_type(), MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        ASSERT_EQ(total_node_count, G_full.size());
    }
}

TEST_P(LoadBalancing, BalancingWithDegree) {
    std::vector<NodeId> actual_cost(G_full.size());
    auto                running_sum = 0;
    for (size_t i = 0; i < G_full.size(); ++i) {
        actual_cost[i] = running_sum;
        running_sum += G_full.at(i).size();
    }
    size_t per_pe_cost = (running_sum + size - 1) / size;
    auto   cost        = [](const LocalGraphView& G, NodeId node) {
        return G.node_info[node].degree;
    };
    G_view = load_balancing::LoadBalancer::run(std::move(G_view), cost, conf);
    for (size_t node = 0; node < G_view.node_info.size(); ++node) {
        auto expected_degree = G_full.at(G_view.node_info.at(node).global_id).size();
        auto actual_degree   = G_view.node_info.at(node).degree;
        EXPECT_EQ(expected_degree, actual_degree);
        EXPECT_EQ(actual_cost.at(G_view.node_info.at(node).global_id) / per_pe_cost, rank);
    }
    NodeId local_node_count = G_view.node_info.size();
    NodeId total_node_count = 0;
    MPI_Reduce(&local_node_count, &total_node_count, 1, kamping::types::mpi_type_traits<NodeId>::data_type(), MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        ASSERT_EQ(total_node_count, G_full.size());
    }
}

const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis", "examples/com-amazon.metis"};
INSTANTIATE_TEST_SUITE_P(SimpleGraphs, LoadBalancing, ::testing::ValuesIn(graphs));
} // namespace
