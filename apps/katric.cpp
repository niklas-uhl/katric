// Copyright (c) 2020-2023 Tim Niklas Uhl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <istream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <adiak.hpp>
#include <caliper/ConfigManager.h>
#include <caliper/cali.h>
#include <kamping/spdlog_adapter/logging.hpp>
#include <kassert/kassert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <mpi.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include "cetric/counters/cetric.h"
#ifdef CETRIC_HAS_OPENMP
    #include <omp.h>
#endif
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "./algorithm.h"
#include "./parse_parameters.h"
#include "./verify.h"
#include "CLI/Error.hpp"
#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/input.h"
#include "cetric/counters/cetric_edge_iterator.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/timer.h"
#include "cetric/util.h"
#include "git_hash.h"

cetric::Config parse_config(int argc, char* argv[], cetric::PEID rank, cetric::PEID size) {
    (void)size;

    CLI::App app("Parallel Triangle Counter");
    app.option_defaults()->always_capture_default();
    app.set_config("--config")->expected(0, 3);
    // unknown keys in a --config file are errors instead of being ignored (e.g. options that were removed)
    app.allow_config_extras(CLI::config_extras_mode::error);

    cetric::Config conf;
    conf.git_commit = cetric::git_hash;

    // conf.hostname = std::getenv("HOST");
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    conf.hostname = hostname;

    app.add_option(
           "--kagen_option_string",
           conf.kagen_option_string,
           "KaGen generator or file input, e.g. \"rgg2d;n=16384;m=131072;seed=7\" or "
           "\"file;filename=graph.metis;input_format=metis;distribution=balance-edges\""
    )
        ->required();

    app.add_option("--iterations", conf.iterations);
    app.add_flag(
        "--verify",
        conf.verify,
        "Check the triangle count of every iteration against a sequential count on the graph gathered to rank 0 "
        "(small inputs only); exit with 1 on a mismatch"
    );

    app.add_option("--primary-cost-function", conf.primary_cost_function)
        ->transform(CLI::IsMember({"none", "N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"}));
    app.add_option("--secondary-cost-function", conf.secondary_cost_function)
        ->transform(CLI::IsMember({"none", "N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"}));

    app.add_flag("-v", conf.verbosity_level, "verbosity level");

    app.add_option(
        "--caliper-config",
        conf.caliper_config,
        "Caliper config, e.g. \"katric-timers,katric-counters\" (JSON), \"spot,katric-counters\" or "
        "\"runtime-report\". "
        "Caliper's CALI_CONFIG environment variable works independently of this option."
    );
    app.add_option(
        "--caliper-output",
        conf.caliper_output,
        "Output prefix: <prefix>.cali (spot, for Thicket), <prefix>.timers.json and <prefix>.counters.json. Enables "
        "spot,katric-timers,katric-counters unless --caliper-config is given."
    );

    app.add_flag("--degree-filtering", conf.degree_filtering);

    app.add_flag("--orient-locally", conf.orient_locally);

    app.add_option("--num-threads", conf.num_threads);
    app.add_option("--grainsize", conf.grainsize);
    app.add_option("--global-degree-of-parallelism", conf.global_degree_of_parallelism);
    app.add_option("--local-degree-of-parallelism", conf.local_degree_of_parallelism);

    app.add_flag("--pseudo2core", conf.pseudo2core);

    app.add_flag("--dense-load-balancing", conf.dense_load_balancing);
    app.add_flag("--dense-degree-exchange", conf.dense_degree_exchange);
    app.add_flag("--compact-degree-exchange", conf.compact_degree_exchange);
    app.add_flag("--global-synchronization", conf.global_synchronization);
    app.add_flag("--binary-rank-search", conf.binary_rank_search);
    app.add_flag("--id-node-ordering", conf.id_node_ordering);

    add_enum_option(app, "--algorithm", conf.algorithm);

    app.add_flag("--flag-intersection", conf.flag_intersection);

    app.add_flag("--skip-local-neighborhood", conf.skip_local_neighborhood);

    app.add_option("--communication-policy", conf.communication_policy)->transform(CLI::IsMember({"new", "grid"}));

    app.add_flag("--local-parallel", conf.local_parallel);
    app.add_flag("--global-parallel", conf.global_parallel);
    app.add_flag("--parallel-compact", conf.parallel_compact);

    add_enum_option(app, "--parallelization-method", conf.parallelization_method);
    app.add_flag("--edge-partitioning", conf.edge_partitioning);
    app.add_flag("--edge-partitioning-static", conf.edge_partitioning_static);
    add_enum_option(app, "--omp-schedule", conf.omp_schedule);
    app.add_option("--omp-chunksize", conf.omp_chunksize);

    add_enum_option(app, "--tbb-partitioner", conf.tbb_partitioner);
    add_enum_option(app, "--task-pool-type", conf.task_pool_type);
    app.add_flag("--task-priorities", conf.task_priorities);

    add_enum_option(app, "--threshold", conf.threshold);
    app.add_option("--threshold-scale", conf.threshold_scale);
    app.add_option("--high-degree-threshold-scale", conf.high_degree_threshold_scale);

    add_enum_option(app, "--intersection-method", conf.intersection_method);
    app.add_option("--binary-intersection-cutoff", conf.binary_intersection_cutoff);
    app.add_option("--hybrid-cutoff-scale", conf.hybrid_cutoff_scale);

    CLI11_PARSE_MPI(app, argc, argv, rank, size);
#ifndef CETRIC_HAS_OPENMP
    if (conf.parallelization_method == cetric::ParallelizationMethod::omp_for) {
        int retval;
        if (rank == 0) {
            CLI::ValidationError e("--parallelization-method", "omp_for requires building with CETRIC_USE_OPENMP=ON");
            retval = app.exit(e);
        } else {
            retval = 1;
        }
        MPI_Finalize();
        exit(retval);
    }
#endif
    if (conf.num_threads == 0) {
        conf.num_threads = tbb::this_task_arena::max_concurrency();
    }
    if (conf.caliper_config.empty() && !conf.caliper_output.empty()) {
        conf.caliper_config = "spot,katric-timers,katric-counters";
    }
    conf.rank    = rank;
    conf.PEs     = size;
    conf.options = option_values(app);
    return conf;
}

int main(int argc, char* argv[]) {
    int thread_support_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support_level);
    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();
    if (thread_support_level < MPI_THREAD_FUNNELED) {
        SPDLOG_CRITICAL("The MPI implementation must support MPI_THREAD_FUNNELED");
        std::exit(1);
    }
    cetric::PEID rank;
    cetric::PEID size;
    int          exit_code = 0;
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        DEBUG_BARRIER(rank);
        cetric::Config      conf = parse_config(argc, argv, rank, size);
        size_t              expected_triangles = 0;
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, conf.num_threads);

#ifdef CETRIC_HAS_OPENMP
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
#endif

        // Caliper: region times and counters aggregated across ranks, iterations annotated as a loop; Adiak holds the
        // run metadata
        cali::ConfigManager caliper;
        if (!conf.caliper_config.empty()) {
            cetric::profiling::add_katric_configs(caliper, conf.caliper_output);
            caliper.add(conf.caliper_config.c_str());
            if (caliper.error()) {
                SPDLOG_CRITICAL("Invalid Caliper config: {}", caliper.error_msg());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Comm adiak_comm = MPI_COMM_WORLD;
            adiak::init(&adiak_comm);
            for (auto const& [name, value]: conf.options) {
                adiak::value("config." + name, value);
            }
            adiak::value("git_commit", conf.git_commit);
            adiak::collect_all(); // user, launch date, command line, hosts, job size, MPI version, ...
            caliper.start();
        }

        // The input is generated (or read) once and ingested; every iteration resets the algorithm to that state.
        auto const   algorithm = cetric::make_algorithm(conf);
        kagen::Graph graph;
        {
            CALI_CXX_MARK_SCOPE("input");
            try {
                graph = cetric::generate_input(conf.kagen_option_string, MPI_COMM_WORLD);
            } catch (std::exception const& e) {
                SPDLOG_CRITICAL("KaGen failed for --kagen_option_string '{}': {}", conf.kagen_option_string, e.what());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (conf.verify) {
            CALI_CXX_MARK_SCOPE("verify");
            expected_triangles = cetric::reference_triangle_count(graph, MPI_COMM_WORLD);
            if (rank == 0) {
                SPDLOG_INFO("reference: {} triangles", expected_triangles);
            }
        }
        {
            CALI_CXX_MARK_SCOPE("ingest");
            algorithm->ingest(std::move(graph));
        }

        size_t           triangles = 0;
        bool             mismatch  = false;
        double           min_time  = std::numeric_limits<double>::max();
        cali::Annotation iteration_annotation("iteration", CALI_ATTR_NESTED);
        for (size_t iter = 0; iter < conf.iterations; ++iter) {
            iteration_annotation.begin(static_cast<int>(iter));
            MPI_Barrier(MPI_COMM_WORLD);
            {
                CALI_CXX_MARK_SCOPE("reset");
                algorithm->reset();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double local_time = 0;
            {
                // the algorithm's run time; its maximum across ranks in the timers output is the iteration's run time
                CALI_CXX_MARK_SCOPE("total");
                cetric::profiling::Timer timer;
                algorithm->run();
                MPI_Barrier(MPI_COMM_WORLD);
                local_time = timer.elapsed_time();
            }
            iteration_annotation.end();
            triangles   = algorithm->triangle_count();
            double time = 0; // only for the log, Caliper's results are available after the run
            MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            min_time = std::min(min_time, time);
            if (rank == 0) {
                SPDLOG_INFO("iteration {}/{}: {:.6f} s, {} triangles", iter + 1, conf.iterations, time, triangles);
                if (conf.verify && triangles != expected_triangles) {
                    SPDLOG_ERROR(
                        "iteration {}: {} triangles, expected {} (reference)",
                        iter + 1,
                        triangles,
                        expected_triangles
                    );
                    mismatch = true;
                }
            }
        }
        if (!conf.caliper_config.empty()) {
            adiak::value("triangles", static_cast<long>(triangles));
            caliper.flush(); // collective, has to happen before MPI_Finalize
            adiak::fini();
        }
        if (rank == 0 && conf.iterations > 1) {
            SPDLOG_INFO("min time: {:.6f} s", min_time);
        }
        MPI_Bcast(&mismatch, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        exit_code = mismatch ? 1 : 0;
    }
    MPI_Finalize();
    return exit_code;
}
