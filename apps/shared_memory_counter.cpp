#include <graph-io/graph_io.h>
#include <tbb/global_control.h>
#include <atomic>
#include <chrono>
#include <map>
#include "CLI/CLI.hpp"
#include "CLI/Validators.hpp"
#include "counters/shared_memory_edge_iterator.h"
#include "datastructures/graph.h"
#include "graph-io/graph_definitions.h"

int main(int argc, char* argv[]) {
    CLI::App app("Parallel Triangle Counter");
    std::string input_file;
    app.add_option("input", input_file, "The input graph");
    graphio::InputFormat input_format;
    app.add_option("--input-format", input_format)
        ->transform(CLI::CheckedTransformer(graphio::input_types, CLI::ignore_case));
    size_t num_threads = 0;
    app.add_option("--num_threads", num_threads);
    size_t iterations = 1;
    app.add_option("--iterations", iterations);

    cetric::shared_memory::Config conf;
    app.add_option("--partition", conf.partition)
        ->transform(CLI::CheckedTransformer(
            std::map<std::string, cetric::shared_memory::Partition>{
                {"1D", cetric::shared_memory::Partition::one_dimensional},
                {"2D", cetric::shared_memory::Partition::two_dimensional}},
            CLI::ignore_case));

    app.add_option("--intersection_method", conf.intersection_method)
        ->transform(CLI::CheckedTransformer(
            std::map<std::string, cetric::shared_memory::IntersectionMethod>{
                {"merge", cetric::shared_memory::IntersectionMethod::merge},
                {"binary_search", cetric::shared_memory::IntersectionMethod::binary_search},
                {"hybrid", cetric::shared_memory::IntersectionMethod::hybrid}},
            CLI::ignore_case));
    app.add_option("--grainsize", conf.grainsize);
    app.add_option("--partitioner", conf.partitioner)
        ->transform(CLI::CheckedTransformer(
            std::map<std::string, cetric::shared_memory::Partitioner>{
                {"auto", cetric::shared_memory::Partitioner::auto_partitioner},
                {"static", cetric::shared_memory::Partitioner::static_partitioner},
                {"simple", cetric::shared_memory::Partitioner::simple_partitioner},
                {"affinity", cetric::shared_memory::Partitioner::affinity_partitioner}},
            CLI::ignore_case));

    CLI11_PARSE(app, argc, argv)

    // set the number of threads
    if (num_threads != 0) {
        tbb::global_control global_control(tbb::global_control::max_allowed_parallelism, num_threads);
    }

    auto G = cetric::graph::AdjacencyGraph(graphio::read_graph(input_file, input_format));
    G.orient([&](const auto& lhs, const auto& rhs) {
        return std::pair(G.local_degree(lhs), lhs) < std::pair(G.local_degree(rhs), rhs);
    });
    G.sort_neighborhoods();
    cetric::shared_memory::SharedMemoryEdgeIterator ctr(G);
    for (size_t i = 0; i < iterations; ++i) {
        std::atomic<size_t> number_of_triangles = 0;
        auto start = std::chrono::high_resolution_clock::now();
        ctr.run([&](auto) { number_of_triangles++; }, conf);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;
        std::cout << "RESULT"
                  << " input=" << input_file                              //
                  << " num_threads=" << num_threads                       //
                  << " iteration=" << i                                   //
                  << " partition=" << conf.partition                      //
                  << " intersection_method=" << conf.intersection_method  //
                  << " grainsize=" << conf.grainsize                      //
                  << " partitioner=" << conf.partitioner                  //
                  << " triangles=" << number_of_triangles                 //
                  << " time=" << time.count()                             //
                  << std::endl;
    }
    return 0;
}
