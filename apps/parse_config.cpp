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

#include "./parse_config.h"

#include <cstdlib>

#include <CLI/CLI.hpp>
#include <CLI/Error.hpp>
#include <CLI/Validators.hpp>
#include <mpi.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "./parse_parameters.h"
#include "cetric/config.h"
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

    app.add_option("--num-threads", conf.num_threads);
    app.add_option("--grainsize", conf.grainsize);
    app.add_option("--global-degree-of-parallelism", conf.global_degree_of_parallelism);
    app.add_option("--local-degree-of-parallelism", conf.local_degree_of_parallelism);

    app.add_flag("--pseudo2core", conf.pseudo2core);

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

    app.add_flag("--edge-partitioning", conf.edge_partitioning);

    add_enum_option(app, "--task-pool-type", conf.task_pool_type);
    app.add_flag("--task-priorities", conf.task_priorities);

    add_enum_option(app, "--threshold", conf.threshold);
    app.add_option("--threshold-scale", conf.threshold_scale);

    add_enum_option(app, "--intersection-method", conf.intersection_method);
    app.add_option("--binary-intersection-cutoff", conf.binary_intersection_cutoff);
    app.add_option("--hybrid-cutoff-scale", conf.hybrid_cutoff_scale);

    CLI11_PARSE_MPI(app, argc, argv, rank, size);
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
