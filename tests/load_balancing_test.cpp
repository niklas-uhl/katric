#include <gtest/gtest.h>
#include <mpi.h>
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/local_graph_view.h"
#include "datastructures/graph_definitions.h"
#include <cstddef>
#include <load_balancing.h>
#include <io/distributed_graph_io.h>
#include <util.h>

namespace {
    using namespace cetric;
    class LoadBalancing : public ::testing::TestWithParam<const char*> {
    protected:
        void SetUp() override {
            auto input = GetParam();
            //GENERATE("examples/metis-sample.metis", "examples/triangle.metis");
            // auto input = GENERATE("examples/triangle.metis");
            SCOPED_TRACE(input);

            // the MPI stuff
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            DEBUG_BARRIER(rank);

            // our metis parser should work sequentially, so we read the whole graph
            EdgeId total_number_of_edges = 0;
            read_metis(
                input,
                [&](NodeId node_count, EdgeId edge_count) {
                    G_full.resize(node_count);
                    total_number_of_edges = edge_count;
                },
                [](NodeId) {}, [&](Edge edge) { G_full[edge.tail].push_back(edge);
                });

            conf.PEs = size;
            conf.rank = rank;

            // read it distributed
            G_view =
                cetric::read_local_graph(input, InputFormat::metis, rank, size);
        }

        std::vector<std::vector<Edge>> G_full;
        LocalGraphView G_view;
        Config conf;
        PEID rank, size;
    };

    TEST_P(LoadBalancing, BalancingWithConstCost) {
        DEBUG_BARRIER(rank);
        auto G_view_old = G_view;
        auto cost = [](const LocalGraphView&, NodeId) { return 1; };
        G_view = load_balancing::LoadBalancer::run(std::move(G_view), cost, conf);
        EXPECT_EQ(G_view_old.node_info.size(), G_view.node_info.size());
        for (size_t node = 0; node < G_view.node_info.size(); ++node) {
            auto expected_degree = G_full.at(G_view.node_info.at(node).global_id).size();
            auto actual_degree = G_view.node_info.at(node).degree;
            EXPECT_EQ(expected_degree, actual_degree);
        }
    }

    TEST_P(LoadBalancing, BalancingWithDegree) {
        auto G_view_old = G_view;
        std::vector<NodeId> actual_cost(G_full.size());
        auto running_sum = 0;
        for (size_t i = 0; i < G_full.size(); ++i) {
            actual_cost[i] = running_sum;
            running_sum += G_full.at(i).size();
        }
        size_t per_pe_cost = (running_sum + size - 1) / size;
        auto cost = [](const LocalGraphView& G, NodeId node) { return G.node_info[node].degree; };
        G_view = load_balancing::LoadBalancer::run(std::move(G_view), cost, conf);
        for (size_t node = 0; node < G_view.node_info.size(); ++node) {
            auto expected_degree =
                G_full.at(G_view.node_info.at(node).global_id).size();
            auto actual_degree = G_view.node_info.at(node).degree;
            EXPECT_EQ(expected_degree, actual_degree);
            EXPECT_EQ(actual_cost.at(G_view.node_info.at(node).global_id) / per_pe_cost, rank);
        }
        NodeId local_node_count = G_view.node_info.size();
        NodeId total_node_count = 0;
        MPI_Reduce(&local_node_count, &total_node_count, 1, MPI_NODE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            ASSERT_EQ(total_node_count, G_full.size());
        }
    }

    const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis", "examples/com-amazon.metis"};
    INSTANTIATE_TEST_SUITE_P(
        SimpleGraphs, LoadBalancing,
        ::testing::ValuesIn(graphs));
}
