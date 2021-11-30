#include <config.h>
#include <counters/cetric.h>
#include <io/distributed_graph_io.h>
#include <mpi.h>
#include <unistd.h>
#include <util.h>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cereal/archives/json.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include "CLI/Validators.hpp"
#include "backward.hpp"
#include "cereal/cereal.hpp"
#include "datastructures/distributed/distributed_graph.h"
#include "datastructures/distributed/local_graph_view.h"
#include "parse_parameters.h"
#include "statistics.h"

Config parse_config(int argc, char* argv[], PEID rank, PEID size) {
    (void)size;

    CLI::App app("Parallel Triangle Counter");
    Config conf;

    // conf.hostname = std::getenv("HOST");
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    conf.hostname = hostname;

    app.add_option("input", conf.input_file, "The input graph");

    app.add_option("--input-format", conf.input_format)
        ->transform(CLI::CheckedTransformer(input_types, CLI::ignore_case));

    app.add_option("--iterations", conf.iterations);

    app.add_set("--primary-cost-function", conf.primary_cost_function,
                {"N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"});
    app.add_set("--secondary-cost-function", conf.secondary_cost_function,
                {"none", "N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"});

    app.add_flag("-v", conf.verbosity_level, "verbosity level");

    std::map<std::string, CacheInput> map{
        {"none", CacheInput::None}, {"fs", CacheInput::Filesystem}, {"mem", CacheInput::InMemory}};
    app.add_option("--cache-input", conf.cache_input)->transform(CLI::CheckedTransformer(map, CLI::ignore_case));

    app.add_flag("--rhg-fix", conf.rhg_fix);

    app.add_option("--json-output", conf.json_output);

    app.add_flag("--degree-filtering", conf.degree_filtering);

    app.add_flag("--orient-locally", conf.orient_locally);

    app.add_flag("--pseudo2core", conf.pseudo2core);

    app.add_flag("--dense-load-balancing", conf.dense_load_balancing);

    parse_gen_parameters(app, conf);

    CLI::Option* input_option = app.get_option("input");
    CLI::Option* gen_option = app.get_option("--gen");
    input_option->excludes(gen_option);
    gen_option->excludes(input_option);

    CLI11_PARSE_MPI(app, argc, argv, rank, size);
    if (conf.input_file.empty() && conf.gen.empty()) {
        int retval;
        if (rank == 0) {
            CLI::RequiredError e("Providing an input file via 'input' or using the generator via '--gen'");
            retval = app.exit(e);
        } else {
            retval = 1;
        }
        MPI_Finalize();
        exit(retval);
    }
    conf.rank = rank;
    conf.PEs = size;
    return conf;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    PEID rank;
    PEID size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    DEBUG_BARRIER(rank);
    backward::SignalHandling sh;
    Config conf = parse_config(argc, argv, rank, size);

    auto load_graph = [&]() {
        LocalGraphView G;
        if (conf.gen == "") {
            G = cetric::read_local_graph(conf.input_file, conf.input_format, rank, size);
        } else {
            G = cetric::gen_local_graph(conf, rank, size);
        }
        return G;
    };
    std::optional<LocalGraphView> input_cache;
    if (conf.cache_input != CacheInput::None) {
        LocalGraphView G = load_graph();
        if (conf.cache_input == CacheInput::Filesystem) {
            auto tmp_file = cetric::dump_to_tmp(G, rank, size);
            conf.cache_file = tmp_file;
        } else if (conf.cache_input == CacheInput::InMemory) {
            input_cache = G;
        }
    }
    std::vector<cetric::profiling::Statistics> all_stats;
    for (size_t iter = 0; iter < conf.iterations; ++iter) {
        MPI_Barrier(MPI_COMM_WORLD);

        DistributedGraph<> G;
        cetric::profiling::Statistics stats(rank, size);
        cetric::profiling::Timer timer;
        switch (conf.cache_input) {
            case CacheInput::None:
                G = DistributedGraph<>(load_graph(), rank, size);
                break;
            case CacheInput::Filesystem:
                G = DistributedGraph<>(cetric::read_graph_view(conf.cache_file, rank, size), rank, size);
                break;
            case CacheInput::InMemory:
                LocalGraphView G_local = input_cache.value();
                G = DistributedGraph<>(std::move(G_local), rank, size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        stats.local.io_time = timer.elapsed_time();
        cetric::profiling::Timer global_time;

        run_cetric(G, stats, conf, rank, size);

        MPI_Barrier(MPI_COMM_WORLD);
        stats.global_wall_time = global_time.elapsed_time();

        stats.reduce();
        all_stats.emplace_back(std::move(stats));
    }
    if (conf.cache_input == CacheInput::Filesystem) {
        std::filesystem::remove(conf.cache_file);
    }
    if (!conf.json_output.empty()) {
        if (rank == 0) {
            assert(all_stats[0].triangles == all_stats[0].counted_triangles);
            if (conf.json_output == "stdout") {
                std::stringstream out;
                {
                    cereal::JSONOutputArchive ar(out);
                    ar(cereal::make_nvp("stats", all_stats));
                    ar(cereal::make_nvp("config", conf));
                }

                std::cout << out.str();
            } else {
                std::ofstream out(conf.json_output);
                {
                    cereal::JSONOutputArchive ar(out);
                    ar(cereal::make_nvp("stats", all_stats));
                    ar(cereal::make_nvp("config", conf));
                }
            }
        }
    } else {
        if (rank == 0) {
            std::cout << std::left << std::setw(15) << "triangles: " << std::right << std::setw(10)
                      << all_stats[0].counted_triangles << std::endl;
            std::cout << std::left << std::setw(15) << "time: " << std::right << std::setw(10)
                      << all_stats[0].global_wall_time << " s" << std::endl;
        }
    }
    return MPI_Finalize();
}
