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

#include "cetric/counters/cetric.h"

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
#include <caliper/ConfigManager.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <graph-io/definitions.h>
#include <graph-io/distributed_graph_io.h>
#include <graph-io/local_graph_view.h>
#include <kamping/spdlog_adapter/logging.hpp>
#include <kassert/kassert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <mpi.h>
#ifdef CETRIC_HAS_OPENMP
    #include <omp.h>
#endif
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "./parse_parameters.h"
#include "CLI/Error.hpp"
#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/counters/cetric_edge_iterator.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/timer.h"
#include "cetric/util.h"
#include "git_hash.h"
#include "graph-io/graph_definitions.h"

cetric::Config parse_config(int argc, char* argv[], cetric::PEID rank, cetric::PEID size) {
    (void)size;

    CLI::App app("Parallel Triangle Counter");
    app.option_defaults()->always_capture_default();
    app.set_config("--config")
        ->expected(0, 3);
    
    cetric::Config conf;
    conf.git_commit = cetric::git_hash;

    // conf.hostname = std::getenv("HOST");
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    conf.hostname = hostname;

    app.add_option("input", conf.input_file, "The input graph");

    add_enum_option(app, "--input-format", conf.input_format, graphio::input_types);
    app.add_option("--partitioning", conf.partitioning);

    app.add_option("--iterations", conf.iterations);

    app.add_option("--primary-cost-function", conf.primary_cost_function)
        ->transform(CLI::IsMember({"none", "N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"}));
    app.add_option("--secondary-cost-function", conf.secondary_cost_function)
        ->transform(CLI::IsMember({"none", "N", "D", "DH", "DDH", "DH2", "DPD", "IDPD", "D*"}));

    app.add_flag("-v", conf.verbosity_level, "verbosity level");

    add_enum_option(app, "--cache-input", conf.cache_input);
    app.add_flag("--read-edge-partitioned", conf.read_edge_partitioned);

    app.add_option(
        "--caliper-config",
        conf.caliper_config,
        "Caliper config, e.g. \"katric-timers,katric-counters\" (JSON), \"spot,katric-counters\" or \"runtime-report\". "
        "Caliper's CALI_CONFIG environment variable works independently of this option."
    );
    app.add_option(
        "--caliper-output",
        conf.caliper_output,
        "Output prefix: <prefix>.timers.json, <prefix>.counters.json and <prefix>.cali (spot). Enables "
        "katric-timers,katric-counters unless --caliper-config is given."
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

    parse_gen_parameters(app, conf);

    CLI::Option* input_option = app.get_option("input");
    CLI::Option* gen_option   = app.get_option("--gen");
    input_option->excludes(gen_option);
    gen_option->excludes(input_option);

    CLI11_PARSE_MPI(app, argc, argv, rank, size);
    if (conf.input_file.empty() && conf.gen.generator.empty()) {
        int retval;
        if (rank == 0) {
            CLI::RequiredError e("Provide an input file via 'input' or using the generator via '--gen'");
            retval = app.exit(e);
        } else {
            retval = 1;
        }
        MPI_Finalize();
        exit(retval);
    }
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
    if (!conf.partitioning.empty()) {
        conf.partitioned_input = true;
    }
    // if (conf.primary_cost_function == "none" && conf.gen.generator.empty() && !conf.partitioned_input) {
    //     int retval;
    //     if (rank == 0) {
    //         CLI::RequiredError e("Primary cost function 'none' is only allowed for generated or prepartitioned
    //         graphs."
    //         );
    //         retval = app.exit(e);
    //     } else {
    //         retval = 1;
    //     }
    //     MPI_Finalize();
    //     exit(retval);
    // }
    if (conf.num_threads == 0) {
        conf.num_threads = tbb::this_task_arena::max_concurrency();
    }
    if (conf.caliper_config.empty() && !conf.caliper_output.empty()) {
        conf.caliper_config = "katric-timers,katric-counters";
    }
    conf.rank    = rank;
    conf.PEs     = size;
    conf.options = option_values(app);
    return conf;
}

class InputCache {
public:
    InputCache(const cetric::Config& conf) : conf_(conf), cache_file_(), G_() {
        if (conf_.cache_input != cetric::CacheInput::none) {
            auto [G, info] = load_graph();
            G_             = std::move(G);
            info_          = std::move(info);
        }
        if (conf_.cache_input == cetric::CacheInput::filesystem) {
            auto tmp_file = graphio::dump_to_tmp(G_.value(), conf_.rank, conf_.PEs);
            cache_file_   = tmp_file;
            G_            = std::nullopt;
        }
    }
    graphio::IOResult get() {
        switch (conf_.cache_input) {
            case cetric::CacheInput::none:
                return load_graph();
            case cetric::CacheInput::filesystem:
                return graphio::read_graph_view(cache_file_, conf_.rank, conf_.PEs);
            case cetric::CacheInput::in_memory:
                return {G_.value(), info_};
            default:
                // unreachable
                return {};
        }
    }
    virtual ~InputCache() {
        if (conf_.cache_input == cetric::CacheInput::filesystem) {
            std::filesystem::remove(cache_file_);
        }
    }

private:
    graphio::IOResult load_graph() {
        if (conf_.gen.generator == "") {
            auto G = graphio::read_local_graph(
                conf_.input_file,
                conf_.input_format,
                conf_.rank,
                conf_.PEs,
                conf_.read_edge_partitioned
            );
            if (conf_.partitioned_input) {
                auto partitioning = graphio::read_local_partition(
                    conf_.partitioning,
                    G.info.local_from,
                    G.info.local_to,
                    conf_.rank,
                    conf_.PEs
                );
                G.G = graphio::apply_partition(std::move(G.G), partitioning, MPI_COMM_WORLD);
                graphio::relabel_consecutively(G.G, MPI_COMM_WORLD);
                if (G.G.local_node_count() != 0) {
                    G.info.local_from = G.G.node_info.front().global_id;
                    G.info.local_to   = G.G.node_info.back().global_id + 1;
                } else {
                    G.info.local_from = std::numeric_limits<cetric::NodeId>::max();
                    G.info.local_to   = std::numeric_limits<cetric::NodeId>::max();
                }
            }
            return G;
        } else {
            return graphio::gen_local_graph(conf_.gen, conf_.rank, conf_.PEs);
        }
    }
    const cetric::Config&                  conf_;
    std::string                            cache_file_;
    std::optional<graphio::LocalGraphView> G_;
    graphio::internal::GraphInfo           info_;
};

int main(int argc, char* argv[]) {
    int thread_support_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support_level);
    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();
    if (thread_support_level < MPI_THREAD_FUNNELED) {
        SPDLOG_CRITICAL("The MPI implementation must support MPI_THREAD_FUNNELED");
        std::exit(1);
    }
    cetric::PEID             rank;
    cetric::PEID             size;
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        DEBUG_BARRIER(rank);
        cetric::Config conf = parse_config(argc, argv, rank, size);
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

        std::optional<double>    io_time;
        cetric::profiling::Timer t;
        InputCache               input_cache(conf);
        if (conf.cache_input != cetric::CacheInput::none) {
            io_time = t.elapsed_time();
        }
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
            adiak::cmdline();
            adiak::hostname();
            adiak::jobsize();
            adiak::launchdate();
            if (io_time.has_value()) {
                adiak::value("io_time", io_time.value());
            }
            caliper.start();
        }
        size_t              triangles = 0;
        std::vector<double> wall_times;
        cali::Annotation    iteration_annotation("iteration", CALI_ATTR_NESTED);
        for (size_t iter = 0; iter < conf.iterations; ++iter) {
            iteration_annotation.begin(static_cast<int>(iter));
            MPI_Barrier(MPI_COMM_WORLD);

            cetric::graph::DistributedGraph<> G;
            SPDLOG_DEBUG("Loading from cache");
            graphio::IOResult input = input_cache.get();
            SPDLOG_DEBUG("Finished loading from cache");
            G = cetric::graph::DistributedGraph<>(
                std::move(input.G),
                {input.info.local_from, input.info.local_to},
                rank,
                size
            );
            tbb::enumerable_thread_specific<size_t> wedges = 0;
            tbb::parallel_for(tbb::blocked_range(G.local_nodes().begin(), G.local_nodes().end()), [&](auto const& r) {
                for (auto node: r) {
                    auto degree = G.degree(node);
                    wedges.local() += (degree * (degree - 1)) / 2;
                }
            });
            cetric::profiling::counter_add("wedges", wedges.combine(std::plus<>{}));
            SPDLOG_DEBUG("Finished conversion");
            MPI_Barrier(MPI_COMM_WORLD);
            cetric::profiling::Timer global_time;

            if (conf.algorithm == cetric::Algorithm::cetric) {
                if (conf.communication_policy == "new") {
                    triangles = run_cetric(G, conf, rank, size, cetric::MessageQueuePolicy{}, conf.id_node_ordering);
                } else if (conf.communication_policy == "grid") {
                    triangles = run_cetric(G, conf, rank, size, cetric::GridPolicy{}, conf.id_node_ordering);
                }
            } else if (conf.algorithm == cetric::Algorithm::patric) {
                if (conf.communication_policy == "new") {
                    triangles = run_patric(G, conf, rank, size, cetric::MessageQueuePolicy{});
                } else if (conf.communication_policy == "grid") {
                    triangles = run_patric(G, conf, rank, size, cetric::GridPolicy{});
                }
            } else if (conf.algorithm == cetric::Algorithm::cetric_x) {
                if (conf.communication_policy == "new") {
                    triangles = run_cetric_new(G, conf, rank, size, cetric::MessageQueuePolicy{});
                } else if (conf.communication_policy == "grid") {
                    triangles = run_cetric_new(G, conf, rank, size, cetric::GridPolicy{});
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double local_wall_time = global_time.elapsed_time();
            iteration_annotation.end();
            double wall_time = 0;
            MPI_Reduce(&local_wall_time, &wall_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            wall_times.push_back(wall_time);
            if (rank == 0) {
                SPDLOG_INFO("iteration {}/{}: {:.6f} s, {} triangles", iter + 1, conf.iterations, wall_time, triangles);
            }
        }
        if (!conf.caliper_config.empty()) {
            adiak::value("triangles", static_cast<long>(triangles));
            adiak::value("wall_times", wall_times);
            caliper.flush(); // collective, has to happen before MPI_Finalize
            adiak::fini();
        }
        if (rank == 0 && wall_times.size() > 1) {
            SPDLOG_INFO("min time: {:.6f} s", *std::min_element(wall_times.begin(), wall_times.end()));
        }
    }
    return MPI_Finalize();
}
