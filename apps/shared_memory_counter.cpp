#include "CLI/CLI.hpp"
#include "counters/shared_memory_edge_iterator.h"
#include <graph-io/graph_io.h>
#include "datastructures/graph.h"
#include "graph-io/graph_definitions.h"
#include <atomic>

int main(int argc, char* argv[]) {
    CLI::App app("Parallel Triangle Counter");
    std::string input_file;
    app.add_option("input", input_file, "The input graph");
    graphio::InputFormat input_format;
    app.add_option("--input-format", input_format)
        ->transform(CLI::CheckedTransformer(graphio::input_types, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);
    auto G = cetric::graph::AdjacencyGraph(graphio::read_graph(input_file, input_format));
    G.orient([&](const auto& lhs, const auto& rhs) {
        return std::pair(G.local_degree(lhs), lhs) < std::pair(G.local_degree(rhs), rhs);
    });
    G.sort_neighborhoods();
    cetric::SharedMemoryEdgeIterator ctr(G);
    std::atomic<size_t> number_of_triangles = 0;
    ctr.run([&](auto){
        number_of_triangles++;
    });
    std::cout << "triangles: " << number_of_triangles << std::endl;
    return 0;
}
