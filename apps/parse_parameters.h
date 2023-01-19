#pragma once

#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>

#include "cetric/config.h"
#include "git_hash.h"
#include "magic_enum.hpp"

inline std::shared_ptr<CLI::App> parse_parameters(const std::string& app_name, cetric::Config& conf) {
    auto app = std::make_shared<CLI::App>(app_name);

    app->add_option("input", conf.input_file, "The input graph")->required()->check(CLI::ExistingFile);
    return app;
}

template <typename E>
constexpr auto enum_name_to_value_map() {
    std::map<std::string, E> mapping;
    constexpr auto           values = magic_enum::enum_values<E>();
    constexpr auto           names  = magic_enum::enum_names<E>();
    for (unsigned i = 0; i < magic_enum::enum_count<E>(); ++i) {
        mapping[std::string(names[i])] = values[i];
    }
    return mapping;
}

#ifndef CLI11_PARSE_MPI
    #define CLI11_PARSE_MPI(app, argc, argv, rank, size) \
        int retval = -1;                                 \
        try {                                            \
            (app).parse((argc), (argv));                 \
        } catch (const CLI::ParseError& e) {             \
            if (rank == 0) {                             \
                retval = (app).exit(e);                  \
            } else {                                     \
                retval = 1;                              \
            }                                            \
        }                                                \
        if (retval > -1) {                               \
            MPI_Finalize();                              \
            exit(retval);                                \
        }
#endif

inline void parse_gen_parameters(CLI::App& app, cetric::Config& conf) {
    app.add_option("--seed", conf.gen.seed);
    app.add_option("--gen", conf.gen.generator)
        ->transform(CLI::IsMember({"gnm", "rdg_2d", "rdg_3d", "rgg_2d", "rgg_3d", "rhg", "ba", "grid_2d", "rmat"}));
    app.add_option("--gen_n", conf.gen.n);
    app.add_option("--gen_m", conf.gen.m);
    // app.add_option("--gen_r", conf.gen_r);
    app.add_option("--gen_p", conf.gen.p);
    app.add_flag("--gen_periodic", conf.gen.periodic);
    app.add_option("--gen_gamma", conf.gen.gamma);
    app.add_option("--gen_d", conf.gen.d);
    app.add_option("--gen_a", conf.gen.a);
    app.add_option("--gen_b", conf.gen.b);
    app.add_option("--gen_c", conf.gen.c);
    app.add_flag("--gen_verify_graph", conf.gen.verify_graph);
    app.add_flag("--gen_statistics", conf.gen.statistics);
}

namespace shmetric {
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
} // namespace shmetric
