#include <gmock/gmock-matchers.h>
#include <graph-io/distributed_graph_io.h>
#include <graph-io/parsing.h>
#include <gtest/gtest.h>
#include <load_balancing.h>
#include <mpi.h>
#include <util.h>
#include <cstddef>
#include <unordered_map>
#include "counters/cetric_edge_iterator.h"
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/helpers.h"
#include "datastructures/graph_definitions.h"
#include "message_statistics.h"

namespace {
class Orientation : public ::testing::TestWithParam<const char*> {
protected:
    void SetUp() override {
        auto input = GetParam();
        // GENERATE("examples/metis-sample.metis", "examples/triangle.metis");
        // auto input = GENERATE("examples/triangle.metis");
        SCOPED_TRACE(input);

        // the MPI stuff
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // our metis parser should work sequentially, so we read the whole graph
        EdgeId total_number_of_edges = 0;
        graphio::internal::read_metis(
            input,
            [&](NodeId node_count, EdgeId edge_count) {
                G_full.resize(node_count);
                total_number_of_edges = edge_count;
            },
            [](NodeId) {}, [&](Edge edge) { G_full[edge.tail].push_back(edge); });

        conf.PEs = size;
        conf.rank = rank;

        // read it distributed
        graphio::IOResult result = graphio::read_local_graph(input, graphio::InputFormat::metis, rank, size);
        G = cetric::graph::DistributedGraph<>(std::move(result.G), {result.info.local_from, result.info.local_to}, rank,
                                              size);
    }

    std::vector<std::vector<cetric::graph::Edge>> G_full;
    cetric::graph::DistributedGraph<> G;
    cetric::Config conf;
    PEID rank, size;
};

TEST_P(Orientation, GlobalDegree) {
    cetric::profiling::MessageStatistics dummy;
    G.find_ghost_ranks();
    std::unordered_set<cetric::graph::RankEncodedNodeId> ghosts;
    find_ghosts(G, ghosts);
    auto ghost_degree = cetric::AuxiliaryNodeData<Degree>{ghosts.begin(), ghosts.end()};
    DegreeCommunicator comm(G, conf.rank, conf.PEs, as_int(MessageTag::Orientation));
    comm.get_ghost_degree([&](RankEncodedNodeId node, Degree degree) { ghost_degree[node] = degree; }, dummy, true);
    cetric::node_ordering::degree ord(G, ghost_degree);
    for (auto node : G.local_nodes()) {
        G.orient(node, ord);
        std::vector<Edge> edges;
        for (auto e : G.out_adj(node).edges()) {
            edges.emplace_back(e.tail.id(), e.head.id());
        }
        for (auto e : G.in_adj(node).edges()) {
            edges.emplace_back(e.tail.id(), e.head.id());
        }
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[node.id()]));
        edges.clear();
        for (auto e : G.adj(node).edges()) {
            edges.emplace_back(e.tail.id(), e.head.id());
        }
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[node.id()]));
    }
}

TEST_P(Orientation, LocalDegree) {
    cetric::profiling::MessageStatistics dummy;
    G.find_ghost_ranks();
    cetric::node_ordering::degree_outward ord(G);
    for (auto node : G.local_nodes()) {
        G.orient(node, ord);
        std::vector<Edge> edges;
        for (auto e : G.out_adj(node).edges()) {
            edges.emplace_back(e.tail.id(), e.head.id());
        }
        for (auto e : G.in_adj(node).edges()) {
            edges.emplace_back(e.tail.id(), e.head.id());
        }
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[node.id()]));
        edges.clear();
        for (auto e : G.adj(node).edges()) {
            edges.emplace_back(e.tail.id(), e.head.id());
        }
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[node.id()]));
    }
}

// TEST_P(Orientation, OrientTwice) {
//     profiling::MessageStatistics dummy;
//     G.find_ghost_ranks();
//     G.expand_ghosts();
//     DegreeCommunicator comm(G, conf.rank, conf.PEs, as_int(MessageTag::Orientation));
//     comm.get_ghost_degree(
//         [&](NodeId global_id, Degree degree) { G.get_ghost_payload(G.to_local_id(global_id)).degree = degree; }, dummy);
//     G.get_graph_payload().ghost_degree_available = true;

//     G.orient([&](NodeId a, NodeId b) {
//         bool cmp = std::make_tuple(G.local_degree(a), G.to_global_id(a)) <
//                    std::make_tuple(G.local_degree(b), G.to_global_id(b));
//         return cmp;
//     });
//     G.sort_neighborhoods();
//     G.for_each_local_node([&](NodeId node) {
//         std::vector<Edge> edges;
//         G.for_each_local_out_edge(node,
//                                   [&](Edge e) { edges.emplace_back(G.to_global_id(e.tail), G.to_global_id(e.head)); });
//         G.for_each_local_in_edge(node,
//                                  [&](Edge e) { edges.emplace_back(G.to_global_id(e.head), G.to_global_id(e.tail)); });
//         EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[G.to_global_id(node)]));
//         edges.clear();
//         G.for_each_edge(node, [&](Edge e) { edges.emplace_back(G.to_global_id(e.tail), G.to_global_id(e.head)); });
//         EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[G.to_global_id(node)]));
//     });
//     G.orient([&](NodeId a, NodeId b) {
//         bool cmp = std::make_tuple(G.degree(a), G.to_global_id(a)) < std::make_tuple(G.degree(b), G.to_global_id(b));
//         return cmp;
//     });
//     G.sort_neighborhoods();
//     G.for_each_local_node([&](NodeId node) {
//         std::vector<Edge> edges;
//         G.for_each_local_out_edge(node,
//                                   [&](Edge e) { edges.emplace_back(G.to_global_id(e.tail), G.to_global_id(e.head)); });
//         G.for_each_local_in_edge(node,
//                                  [&](Edge e) { edges.emplace_back(G.to_global_id(e.head), G.to_global_id(e.tail)); });
//         EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[G.to_global_id(node)]));
//         edges.clear();
//         G.for_each_edge(node, [&](Edge e) { edges.emplace_back(G.to_global_id(e.tail), G.to_global_id(e.head)); });
//         EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[G.to_global_id(node)]));
//     });
// }

const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis" /*, "examples/com-amazon.metis"*/};
INSTANTIATE_TEST_SUITE_P(SimpleGraphs, Orientation, ::testing::ValuesIn(graphs));
}  // namespace
