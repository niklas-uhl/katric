#include <datastructures/distributed/distributed_graph.h>
#include <gmock/gmock-matchers.h>
#include <graph-io/distributed_graph_io.h>
#include <graph-io/graph_io.h>
#include <graph-io/local_graph_view.h>
#include <graph-io/parsing.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <numeric>
#include <vector>
#include "datastructures/graph_definitions.h"

TEST(DistributedGraphTest, loading_works) {
    using namespace graphio;
    std::vector<std::vector<Edge<>>> G_full;
    auto input = "examples/hiv.metis";
    // auto input = GENERATE("examples/triangle.metis");

    // our metis parser should work sequentially, so we read the whole graph
    EdgeId total_number_of_edges = 0;
    graphio::internal::read_metis(
        input,
        [&](NodeId node_count, EdgeId edge_count) {
            G_full.resize(node_count);
            total_number_of_edges = edge_count;
        },
        [](NodeId) {}, [&](Edge<> edge) { G_full[edge.tail].push_back(edge); });

    // the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // read it distributed
    auto result = graphio::read_local_graph(input, InputFormat::metis, rank, size);

    auto G_view = result.G;
    // SECTION( "view is correct" ) {
    std::vector<EdgeId> index;
    std::exclusive_scan(G_view.node_info.begin(), G_view.node_info.end(), std::back_inserter(index), 0,
                        [](EdgeId acc, const auto& node_info) {
                            return acc + node_info.degree;
                            ;
                        });
    for (size_t i = 0; i < G_view.node_info.size(); ++i) {
        std::vector<Edge<>> edges;
        NodeId tail = G_view.node_info[i].global_id;
        for (size_t edge_id = index[i]; edge_id < index[i] + G_view.node_info[i].degree; edge_id++) {
            NodeId head = G_view.edge_heads[edge_id];
            edges.emplace_back(tail, head);
        }
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[tail]));
    }
    // }

    // construct the "real" datastructure
    cetric::graph::DistributedGraph G =
        cetric::graph::DistributedGraph<Degree>(std::move(result.G), {result.info.local_from, result.info.local_to}, rank, size);
    // SECTION( "global and local ids are correct" ) {
    //     std::vector<NodeId> local_nodes;
    //     G.for_each_local_node([&](NodeId node) {
    //         NodeId global_id = G.to_global_id(node);
    //         CHECK(G.to_local_id(global_id) == node);
    //         local_nodes.emplace_back(global_id);
    //     });
    //     std::sort(local_nodes.begin(), local_nodes.end());
    //     for(NodeId node = 0; node < G_full.size(); ++node) {
    //         INFO(node);
    //         if (std::binary_search(local_nodes.begin(), local_nodes.end(), node)) {
    //             CHECK(G.is_local(node));
    //             CHECK_FALSE(G.is_ghost_from_global(node));
    //         } else {
    //             CHECK_FALSE(G.is_local(node));
    //         }
    //     }
    // }
    // SECTION( "all nodes are present" ) {
    std::vector<NodeId> nodes;
    for (auto node : G.local_nodes()) {
        nodes.emplace_back(node.id());
    }

    std::vector<int> counts(size);
    int local_count = nodes.size();
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs;
    std::exclusive_scan(counts.begin(), counts.end(), std::back_inserter(displs), 0);
    std::vector<NodeId> all_nodes;
    if (rank == 0) {
        all_nodes.resize(std::accumulate(counts.begin(), counts.end(), 0));
    }
    MPI_Gatherv(nodes.data(), local_count, MPI_NODE, all_nodes.data(), counts.data(), displs.data(), MPI_NODE, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
        auto expected_nodes = std::vector<NodeId>(G_full.size());
        std::iota(expected_nodes.begin(), expected_nodes.end(), 0);
        EXPECT_THAT(all_nodes, testing::UnorderedElementsAreArray(expected_nodes));
    }
    // }
    // SECTION( "node knows all edges" ) {
    for (auto node : G.local_nodes()) {
        std::vector<Edge<>> edges;
        EXPECT_EQ(G.degree(node), G_full[node.id()].size());
        for (cetric::graph::RankEncodedNodeId head : G.adj(node).neighbors()) {
            edges.push_back(Edge<>{node.id(), head.id()});
        }
        EXPECT_EQ(G.degree(node), edges.size());
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[node.id()]));
        edges.clear();
        for (auto edge : G.adj(node).edges()) {
            edges.push_back(Edge<>{edge.tail.id(), edge.head.id()});
        }
        EXPECT_EQ(G.degree(node), edges.size());
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[node.id()]));
    }
    // }
    // SECTION( "out and in edges well defined" ) {
    //     G.for_each_local_node([&](NodeId node) {
    //         Degree out_degree = 0;
    //         G.for_each_local_out_edge(node, [&](Edge) {
    //             out_degree++;
    //         });
    //         CHECK(G.degree(node) == out_degree);
    //     });
    //     G.for_each_local_node([&](NodeId node) {
    //         NodeId in_degree = 0;
    //         G.for_each_local_in_edge(node, [&](Edge) {
    //             in_degree++;
    //         });
    //         CHECK(in_degree == 0);
    //     });
    // }
    // SECTION("orientation works") {
    //     G.orient([&](NodeId x, NodeId y) {
    //         return G.to_global_id(x) < G.to_global_id(y);
    //     });
    //     G.for_each_local_node([&](NodeId node) {
    //       Degree degree = 0;
    //       G.for_each_local_out_edge(node, [&](Edge edge) {
    //         edge =
    //             edge.map([&](NodeId local) { return G.to_global_id(local); });
    //         CHECK(edge.tail < edge.head);
    //         degree++;
    //       });
    //       G.for_each_local_in_edge(node, [&](Edge edge) {
    //         edge =
    //             edge.map([&](NodeId local) { return G.to_global_id(local); });
    //         CHECK(edge.tail < edge.head);
    //         degree++;
    //       });
    //       REQUIRE(degree == G.degree(node));
    //     });
    // }
    // SECTION("Graph expansion works") {
    //     auto to_global = [&](NodeId node_id) { return G.to_global_id(node_id); };
    //     G.expand_ghosts();
    //     G.for_each_ghost_node([&](NodeId node) {
    //         std::vector<Edge> edges;
    //         G.for_each_edge(node, [&](Edge edge) {
    //             NodeId global_id = G.to_global_id(edge.head);
    //             INFO("global_id " << global_id);
    //             CHECK(G.to_local_id(global_id) == edge.head);
    //             // edges.emplace_back(to_global(edge.tail), to_global(edge.head));
    //             edges.push_back(edge.map(to_global));
    //         });
    //         CHECK_THAT(G_full[to_global(node)], Catch::Contains(edges));
    //     });
    // }
}
