#include <atomic>
#include <chrono>
#include <map>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <backward.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <graph-io/graph_definitions.h>
#include <graph-io/graph_io.h>
#include <magic_enum.hpp>
#include <omp.h>
#include <tbb/global_control.h>

#include "cetric/counters/shared_memory_cetric.h"
#include "cetric/counters/shared_memory_edge_iterator.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph.h"
#include "cetric/statistics.h"
#include "git_hash.h"
#include "parse_parameters.h"

void print_summary(
    const cetric::Config&                       conf,
    std::vector<cetric::profiling::Statistics>& all_stats,
    std::optional<double>                       io_time = std::nullopt
) {
    if (!conf.json_output.empty()) {
        if (conf.rank == 0) {
            KASSERT(all_stats[0].triangles == all_stats[0].counted_triangles);
            auto write_json_to_stream = [&](auto& stream) {
                cereal::JSONOutputArchive ar(stream);
                ar(cereal::make_nvp("stats", all_stats));
                if (io_time.has_value()) {
                    ar(cereal::make_nvp("io_time", io_time.value()));
                }
                ar(cereal::make_nvp("config", conf));
            };
            if (conf.json_output == "stdout") {
                std::stringstream out;
                write_json_to_stream(out);
                std::cout << out.str();
            } else {
                std::ofstream out(conf.json_output);
                write_json_to_stream(out);
            }
        }
    } else {
        if (conf.rank == 0) {
            std::cout << std::left << std::setw(15) << "triangles: " << std::right << std::setw(10)
                      << all_stats[0].counted_triangles << std::endl;
            double min_time = std::numeric_limits<double>::max();
            for (size_t i = 0; i < all_stats.size(); ++i) {
                std::cout << std::left << std::setw(15) << fmt::format("[{}] time: ", i + 1) << std::right
                          << std::setw(10) << all_stats[i].global_wall_time << " s" << std::endl;
                min_time = std::min(all_stats[i].global_wall_time, min_time);
            }
            std::cout << std::left << std::setw(15) << "min time: " << std::right << std::setw(10) << min_time << " s"
                      << std::endl;
        }
    }
}
cetric::Config parse_config(int argc, char* argv[]) {
    CLI::App app("Parallel Triangle Counter");
    app.option_defaults()->always_capture_default();
    cetric::Config conf;
    conf.git_commit = cetric::git_hash;

    // conf.hostname = std::getenv("HOST");
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    conf.hostname = hostname;

    app.add_option("input", conf.input_file, "The input graph");

    app.add_option("--input-format", conf.input_format)
        ->transform(CLI::CheckedTransformer(graphio::input_types, CLI::ignore_case));
    app.add_option("--partitioning", conf.partitioning);

    app.add_option("--iterations", conf.iterations);

    app.add_flag("-v", conf.verbosity_level, "verbosity level");

    app.add_option("--json-output", conf.json_output);

    app.add_flag("--degree-filtering", conf.degree_filtering);

    app.add_option("--num-threads", conf.num_threads);
    app.add_option("--grainsize", conf.grainsize);
    app.add_option("--local-degree-of-parallelism", conf.local_degree_of_parallelism);

    app.add_flag("--pseudo2core", conf.pseudo2core);

    app.add_flag("--id-node-ordering", conf.id_node_ordering);

    app.add_flag("--flag-intersection", conf.flag_intersection);

    app.add_flag("--skip-local-neighborhood", conf.skip_local_neighborhood);

    conf.local_parallel = true;
    // app.add_flag("--local-parallel", conf.local_parallel);

    app.add_option("--parallelization-method", conf.parallelization_method)
        ->transform(CLI::CheckedTransformer(enum_name_to_value_map<cetric::ParallelizationMethod>(), CLI::ignore_case));
    app.add_flag("--edge-partitioning", conf.edge_partitioning);
    app.add_flag("--edge-partitioning-static", conf.edge_partitioning_static);
    app.add_option("--omp-schedule", conf.omp_schedule)
        ->transform(CLI::CheckedTransformer(enum_name_to_value_map<cetric::OMPSchedule>(), CLI::ignore_case));
    app.add_option("--omp-chunksize", conf.omp_chunksize);

    app.add_option("--tbb-partitioner", conf.tbb_partitioner)
        ->transform(CLI::CheckedTransformer(enum_name_to_value_map<cetric::TBBPartitioner>(), CLI::ignore_case));

    app.add_option("--high-degree-threshold-scale", conf.high_degree_threshold_scale);

    app.add_option("--intersection-method", conf.intersection_method)
        ->transform(CLI::CheckedTransformer(cetric::intersection_method_map, CLI::ignore_case));
    app.add_option("--binary-intersection-cutoff", conf.binary_intersection_cutoff);
    app.add_option("--hybrid-cutoff-scale", conf.hybrid_cutoff_scale);

    parse_gen_parameters(app, conf);

    CLI::Option* input_option = app.get_option("input");
    CLI::Option* gen_option   = app.get_option("--gen");
    input_option->excludes(gen_option);
    gen_option->excludes(input_option);

    try {
        app.parse(argc, argv);
    } catch (const CLI ::ParseError& e) {
        app.exit(e);
    }
    if (conf.input_file.empty() && conf.gen.generator.empty()) {
        CLI::RequiredError e("Provide an input file via 'input' or using the "
                             "generator via '--gen'");
        exit(app.exit(e));
    }
    conf.rank = 0;
    conf.PEs  = 1;
    return conf;
}
int main(int argc, char* argv[]) {
    backward::SignalHandling sh;
    auto                     conf            = parse_config(argc, argv);
    size_t                   max_concurrency = tbb::this_task_arena::max_concurrency();
    if (conf.num_threads == 0) {
        conf.num_threads = max_concurrency;
    }
    if (conf.num_threads > max_concurrency) {
        std::cout
            << fmt::format("Warning, TBB uses only {} instead of {} threads!\n", max_concurrency, conf.num_threads);
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, conf.num_threads + 1);

    omp_set_num_threads(conf.num_threads);
    if (conf.omp_chunksize == 0) {
        omp_sched_t type;
        int         chunksize;
        omp_get_schedule(&type, &chunksize);
        conf.omp_chunksize = chunksize;
    }
    switch (conf.omp_schedule) {
        using cetric::OMPSchedule;
        case OMPSchedule::stat:
            omp_set_schedule(omp_sched_static, conf.omp_chunksize);
            break;
        case OMPSchedule::dynamic:
            omp_set_schedule(omp_sched_dynamic, conf.omp_chunksize);
            break;
        case OMPSchedule::guided:
            omp_set_schedule(omp_sched_guided, conf.omp_chunksize);
            break;
        case OMPSchedule::standard:
            omp_set_schedule(omp_sched_auto, conf.omp_chunksize);
            break;
    }

    std::optional<double>                      io_time;
    cetric::profiling::Timer                   t;
    cetric::graph::DistributedGraph<>          G{graphio::read_graph(conf.input_file, conf.input_format)};
    std::vector<cetric::profiling::Statistics> all_stats;
    for (size_t iter = 0; iter < conf.iterations; ++iter) {
        cetric::profiling::Statistics stats(0, 1);
        cetric::profiling::Timer      timer;
        // LOG << "[R" << rank << "] "
        //     << "Loading from cache";
        // graphio::IOResult input = input_cache.get();
        // LOG << "[R" << rank << "] "
        //     << "Finished loading from cache";
        // // atomic_debug(G_local.node_info);
        //  atomic_debug(G_local.edge_heads);
        // G = cetric::graph::DistributedGraph<>(
        //     std::move(input.G),
        //     {input.info.local_from, input.info.local_to},
        //     rank,
        //     size
        // );
        tbb::enumerable_thread_specific<size_t> wedges = 0;
        tbb::parallel_for(tbb::blocked_range(G.local_nodes().begin(), G.local_nodes().end()), [&](auto const& r) {
            for (auto node: r) {
                auto degree = G.degree(node);
                wedges.local() += (degree * (degree - 1)) / 2;
            }
        });
        stats.local.wedges = wedges.combine(std::plus<>{});
        // atomic_debug(G);
        stats.local.io_time = timer.elapsed_time();
        cetric::profiling::Timer global_time;

        cetric::run_shmem(G, stats, conf);

        // MPI_Barrier(MPI_COMM_WORLD);
        stats.local.local_wall_time = global_time.elapsed_time();

        stats.collapse();
        all_stats.emplace_back(std::move(stats));
    }
    print_summary(conf, all_stats, io_time);
}
