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

        // our metis parser should work sequentially, so we read the whole graph
        EdgeId total_number_of_edges = 0; read_metis(
            input,
            [&](NodeId node_count, EdgeId edge_count) {
                G_full.resize(node_count);
                total_number_of_edges = edge_count;
            },
            [](NodeId) {}, [&](Edge edge) { G_full[edge.tail].push_back(edge);
            });

        // the MPI stuff
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        DEBUG_BARRIER(rank);

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
    auto G_view_old = G_view;
    auto cost = [](LocalGraphView, NodeId) { return 1; };
    G_view = load_balancing::LoadBalancer::run(std::move(G_view), cost, conf);
    atomic_debug(G_view.node_info);
    EXPECT_EQ(G_view_old.node_info.size(), G_view.node_info.size());
    for (size_t node = 0; node < G_view.node_info.size(); ++node) {
        auto expected_degree = G_full.at(G_view.node_info.at(node).global_id).size();
        auto actual_degree = G_view.node_info.at(node).degree;
        EXPECT_EQ(expected_degree, actual_degree);
    }
}

    const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis"};
    INSTANTIATE_TEST_SUITE_P(
        Foo, LoadBalancing,
        ::testing::ValuesIn(graphs));
}
