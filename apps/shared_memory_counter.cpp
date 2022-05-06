#include "CLI/CLI.hpp"
#include "counters/shared_memory_edge_iterator.h"
#include "datastructures/distributed/local_graph_view.h"
#include "datastructures/graph.h"
#include "io/definitions.h"

int main(int argc, char* argv[]) {
    CLI::App app("Parallel Triangle Counter");
    std::string input_file;
    app.add_option("input", input_file, "The input graph");
    InputFormat input_format;
    app.add_option("--input-format", input_format)
        ->transform(CLI::CheckedTransformer(input_types, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);
    auto G = cetric::graph::AdjacencyGraph(cetric::read_local_graph(input_file, input_format, 0, 1));
    SharedMemoryEdgeIterator ctr(G);
    return 0;
}
