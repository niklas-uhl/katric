//
// Created by Tim Niklas Uhl on 17.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARSE_PARAMETERS_H
#define PARALLEL_TRIANGLE_COUNTER_PARSE_PARAMETERS_H

#include <CLI/CLI.hpp>
#include <memory>
#include "config.h"

std::shared_ptr<CLI::App> parse_parameters(const std::string& app_name, cetric::Config& conf) {
    auto app = std::make_shared<CLI::App>(app_name);

    app->add_option("input", conf.input_file, "The input graph")->required()->check(CLI::ExistingFile);
    app->add_option("--lcc", conf.output_file, "Output local clustering coefficients to file");
    if (!conf.output_file.empty()) {
        conf.lcc = true;
    }
    return app;
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
    app.add_option("--seed", conf.seed);
    app.add_set("--gen", conf.gen, {"gnm_undirected", "rdg_2d", "rdg_3d", "rgg_2d", "rgg_3d", "rhg", "ba", "grid_2d"});
    app.add_option("--gen_n", conf.gen_n);
    app.add_option("--gen_m", conf.gen_m);
    // app.add_option("--gen_r", conf.gen_r);
    app.add_option("--gen_r_coeff", conf.gen_r_coeff);
    app.add_flag("--gen_scale_weak", conf.gen_scale_weak);
    app.add_option("--gen_p", conf.gen_p);
    app.add_flag("--gen_periodic", conf.gen_periodic);
    app.add_option("--gen_k", conf.gen_k);
    app.add_option("--gen_gamma", conf.gen_gamma);
    app.add_option("--gen_d", conf.gen_d);
}

#endif  // PARALLEL_TRIANGLE_COUNTER_PARSE_PARAMETERS_H
