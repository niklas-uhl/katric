#include <atomic>
#include <chrono>
#include <map>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <backward.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <graph-io/graph_definitions.h>
#include <graph-io/graph_io.h>
#include <tbb/global_control.h>

#include "cetric/counters/shared_memory_edge_iterator.h"
#include "cetric/datastructures/graph.h"

int main(int argc, char* argv[]) {
    backward::SignalHandling sh;
    CLI::App                 app("Parallel Triangle Counter");
    std::string              input_file;
    app.add_option("input", input_file, "The input graph");
    graphio::InputFormat input_format;
    app.add_option("--input-format", input_format)
        ->transform(CLI::CheckedTransformer(graphio::input_types, CLI::ignore_case));
    size_t num_threads = 0;
    app.add_option("--num_threads", num_threads);
    bool pinning = false;
    app.add_flag("--pinning", pinning);

    size_t iterations = 1;
    app.add_option("--iterations", iterations);

    cetric::shared_memory::Config conf;
    app.add_option("--partition", conf.partition)
        ->transform(CLI::CheckedTransformer(
            std::map<std::string, cetric::shared_memory::Partition>{
                {"1D", cetric::shared_memory::Partition::one_dimensional},
                {"2D", cetric::shared_memory::Partition::two_dimensional}},
            CLI::ignore_case
        ));

    app.add_option("--intersection_method", conf.intersection_method)
        ->transform(CLI::CheckedTransformer(
            std::map<std::string, cetric::shared_memory::IntersectionMethod>{
                {"merge", cetric::shared_memory::IntersectionMethod::merge},
                {"binary_search", cetric::shared_memory::IntersectionMethod::binary_search},
                {"hybrid", cetric::shared_memory::IntersectionMethod::hybrid}},
            CLI::ignore_case
        ));
    app.add_option("--grainsize", conf.grainsize);
    app.add_option("--partitioner", conf.partitioner)
        ->transform(CLI::CheckedTransformer(
            std::map<std::string, cetric::shared_memory::Partitioner>{
                {"auto", cetric::shared_memory::Partitioner::auto_partitioner},
                {"static", cetric::shared_memory::Partitioner::static_partitioner},
                {"simple", cetric::shared_memory::Partitioner::simple_partitioner},
                {"affinity", cetric::shared_memory::Partitioner::affinity_partitioner}},
            CLI::ignore_case
        ));
    app.add_flag("--skip_previous_edges", conf.skip_previous_edges);
    bool degree_reordering = false;
    app.add_flag("--degree_reordering", degree_reordering);

    CLI11_PARSE(app, argc, argv)

    // set the number of threads

    size_t default_num_threads = tbb::this_task_arena::max_concurrency();
    if (num_threads == 0) {
        num_threads = default_num_threads;
    }
    tbb::global_control global_control(tbb::global_control::max_allowed_parallelism, num_threads);
    conf.num_threads = num_threads;

    auto G = cetric::graph::AdjacencyGraph(graphio::read_graph(input_file, input_format));
    // auto perm = cetric::graph::ordering_permutation(G, [&](graphio::NodeId lhs, graphio::NodeId rhs) {
    //     return std::pair(G.local_degree(lhs), lhs) < std::pair(G.local_degree(rhs), rhs);
    // });
    // // auto perm = cetric::graph::ordering_permutation(G);
    // G.permutate(perm);
    auto node_ordering = [&](const auto& lhs, const auto& rhs) {
        return std::pair(G.local_degree(lhs), lhs) < std::pair(G.local_degree(rhs), rhs);
    };
    if (degree_reordering) {
        auto perm = cetric::graph::ordering_permutation(G, node_ordering);
        G.permutate(perm);
    }
    G.orient(node_ordering);
    if (!degree_reordering && conf.skip_previous_edges) {
        G.sort_neighborhoods(node_ordering);
    } else {
        G.sort_neighborhoods(std::less<>{});
    }
    cetric::shared_memory::SharedMemoryEdgeIterator ctr(G);
    for (size_t i = 0; i < iterations; ++i) {
        std::atomic<size_t> number_of_triangles = 0;
        auto                start               = std::chrono::high_resolution_clock::now();
        if (!degree_reordering && conf.skip_previous_edges) {
            ctr.run([&](auto) { number_of_triangles++; }, conf, node_ordering);
        } else {
            ctr.run([&](auto) { number_of_triangles++; }, conf, std::less<>{});
        }
        auto                          end  = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time = end - start;
        std::cout << "RESULT" << std::boolalpha << " input=" << input_file //
                  << " num_threads=" << num_threads                        //
                  << " pinning=" << pinning                                //
                  << " iteration=" << i                                    //
                  << " partition=" << conf.partition                       //
                  << " intersection_method=" << conf.intersection_method   //
                  << " grainsize=" << conf.grainsize                       //
                  << " degree_reordering=" << degree_reordering            //
                  << " skip_previous_edges=" << conf.skip_previous_edges   //
                  << " partitioner=" << conf.partitioner                   //
                  << " triangles=" << number_of_triangles                  //
                  << " time=" << time.count()                              //
                  << std::endl;
    }
    return 0;
}
