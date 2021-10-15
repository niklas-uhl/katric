#include <algorithm>
#include <catch2/catch.hpp>
#include <datastructures/distributed/distributed_graph.h>
#include <datastructures/distributed/local_graph_view.h>
#include <filesystem>
#include <io/distributed_graph_io.h>
#include <iterator>
#include <mpi.h>
#include <numeric>
#include <string>

TEST_CASE("Construct graph from input file", "[io][datastructure]") {
    using namespace cetric;
    std::vector<std::vector<Edge>> G_full;
    auto input = GENERATE("examples/metis-sample.metis", "examples/triangle.metis");
    //auto input = GENERATE("examples/triangle.metis");
    INFO(input);

    //our metis parser should work sequentially, so we read the whole graph
    EdgeId total_number_of_edges = 0;
    read_metis(input, [&](NodeId node_count, EdgeId edge_count) {
        G_full.resize(node_count);
        total_number_of_edges = edge_count;
    }, [](NodeId) {}, [&](Edge edge) {
        G_full[edge.tail].push_back(edge);
    });

    // the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //read it distributed
    LocalGraphView G_view = cetric::read_local_graph(input, InputFormat::metis, rank, size);

    SECTION( "view is correct" ) {
        std::vector<EdgeId> index;
        std::exclusive_scan(G_view.node_info.begin(), G_view.node_info.end(), std::back_inserter(index), 0, [](EdgeId acc, const auto& node_info) {
            return acc + node_info.degree;;
        });
        for(size_t i = 0; i < G_view.node_info.size(); ++i) {
            std::vector<Edge> edges;
            NodeId tail = G_view.node_info[i].global_id;
            for(size_t edge_id = index[i]; edge_id < index[i] + G_view.node_info[i].degree; edge_id++) {
                NodeId head = G_view.edge_heads[edge_id];
                edges.emplace_back(tail, head);
            }
            CHECK_THAT(edges, Catch::UnorderedEquals<Edge>(G_full[tail]));
        }
    }

    //construct the "real" datastructure
    DistributedGraph G = std::move(G_view);
    SECTION( "global and local ids are correct" ) {
        std::vector<NodeId> local_nodes;
        G.for_each_local_node([&](NodeId node) {
            NodeId global_id = G.to_global_id(node);
            CHECK(G.to_local_id(global_id) == node);
            local_nodes.emplace_back(global_id);
        });
        std::sort(local_nodes.begin(), local_nodes.end());
        for(NodeId node = 0; node < G_full.size(); ++node) {
            INFO(node);
            if (std::binary_search(local_nodes.begin(), local_nodes.end(), node)) {
                CHECK(G.is_local(node));
                CHECK_FALSE(G.is_ghost_from_global(node));
            } else {
                CHECK_FALSE(G.is_local(node));
            }
        }
    }
    SECTION( "all nodes are present" ) {
        std::vector<NodeId> nodes;
        G.for_each_local_node([&](NodeId node) {
            nodes.emplace_back(G.to_global_id(node));
        });

        std::vector<int> counts(size);
        int local_count = nodes.size();
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> displs;
        std::exclusive_scan(counts.begin(), counts.end(), std::back_inserter(displs), 0);
        std::vector<NodeId> all_nodes;
        if (rank == 0) {
            all_nodes.resize(std::accumulate(counts.begin(), counts.end(), 0));
        }
        MPI_Gatherv(nodes.data(), local_count, MPI_NODE, all_nodes.data(), counts.data(), displs.data(), MPI_NODE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            auto expected_nodes = std::vector<NodeId>(G_full.size());
            std::iota(expected_nodes.begin(), expected_nodes.end(), 0);
            REQUIRE_THAT(all_nodes, Catch::UnorderedEquals<NodeId>(expected_nodes));
        }
    }
    SECTION( "node knows all edges" ) {
        G.for_each_local_node([&](NodeId node) {
            std::vector<Edge> edges;
            auto to_global = [&](NodeId node_id) {
                return G.to_global_id(node_id);
            };
            CHECK(G.degree(node) == G_full[G.to_global_id(node)].size());
            G.for_each_edge(node, [&](Edge edge) {
                NodeId global_id = G.to_global_id(edge.head);
                INFO("global_id " << global_id);
                CHECK(G.to_local_id(global_id) == edge.head);
                //edges.emplace_back(to_global(edge.tail), to_global(edge.head));
                edges.push_back(edge.map(to_global));
            });
            CHECK(G.degree(node) == edges.size());
            CHECK_THAT(edges, Catch::UnorderedEquals<Edge>(G_full[to_global(node)]));
        });
    }
    SECTION( "out and in edges well defined" ) {
        G.for_each_local_node([&](NodeId node) {
            Degree out_degree = 0;
            G.for_each_local_out_edge(node, [&](Edge) {
                out_degree++;
            });
            CHECK(G.degree(node) == out_degree);
        });
        G.for_each_local_node([&](NodeId node) {
            NodeId in_degree = 0;
            G.for_each_local_in_edge(node, [&](Edge) {
                in_degree++;
            });
            CHECK(in_degree == 0);
        });
    }
}
