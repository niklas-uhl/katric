#include <catch2/catch.hpp>
#include <datastructures/distributed/distributed_graph.h>
#include <datastructures/distributed/local_graph_view.h>
#include <filesystem>
#include <io/distributed_graph_io.h>
#include <iterator>
#include <mpi.h>
#include <numeric>

TEST_CASE("Construct graph from input file", "[io][datastructure]") {
    using namespace cetric;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    DistributedGraph G = cetric::read_local_graph("examples/metis-sample.metis", InputFormat::metis, rank, size);
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
            REQUIRE_THAT(all_nodes, Catch::UnorderedEquals<NodeId>({0, 1, 2, 3, 4, 5, 6}));
        }
    }
}
