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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <CLI/Error.hpp>
#include <CLI/Validators.hpp>
#include <mpi.h>
#include <tbb/task_arena.h>

#include "./algorithm.h"
#include "./parse_parameters.h"
#include "katric/config.h"
#include "git_hash.h"

katric::Config parse_config(int argc, char* argv[], int rank, int size) {
    (void)size;

    CLI::App app("Parallel Triangle Counter");
    app.option_defaults()->always_capture_default();
    app.set_config("--config")->expected(0, 3);
    // unknown keys in a --config file are errors instead of being ignored (e.g. options that were removed)
    app.allow_config_extras(CLI::config_extras_mode::error);

    katric::Config conf;
    conf.git_commit = katric::git_hash;

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

    add_enum_option(app, "--primary-cost-function", conf.primary_cost_function);
    app.add_option(
           "--rebalance-threshold",
           conf.rebalance_threshold,
           "Redistribute in the primary load balancing only if the cost's max/avg over the ranks exceeds this "
           "(1 = whenever it is not perfectly balanced); otherwise the input graph is kept, with its exchanged degrees"
    )
        ->capture_default_str();

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
    app.add_option("--global-degree-of-parallelism", conf.global_degree_of_parallelism);
    app.add_option("--local-degree-of-parallelism", conf.local_degree_of_parallelism);

    app.add_flag("--pseudo2core", conf.pseudo2core);
    app.add_flag(
        "--message-trimming",
        conf.message_trimming,
        "Global phase: send each rank only the part of N+(v) from its first receiver on (in orientation order), and "
        "nothing if no entry follows it"
    );

    add_enum_option(app, "--degree-exchange", conf.degree_exchange);
    // Removed in 401dcb9 (there are no barriers between the phases any more); still accepted, and ignored, so that job
    // files generated before still run. Remove once they have.
    app.add_flag("--global-synchronization", "Ignored (deprecated)");

    app.add_flag(
        "--contraction",
        conf.contraction,
        "CETRIC: the local phase also finds the triangles with two local vertices (through the ghosts' out-edges to "
        "local vertices), the global phase sends only the cut edges. Without: DiTriC"
    );
    // Before the rename (2026-11-19) the variants were --algorithm patric (DiTriC), cetric-x (CETRIC) and cetric
    // (pre-CETRIC, removed). Still accepted so that job files generated before still run; remove once they have.
    std::vector<std::string> algorithms{"patric", "cetric-x", "cetric_x", "cetric"};
    for (katric::Algorithm value: magic_enum::enum_values<katric::Algorithm>()) {
        algorithms.push_back(katric::cli_name(value));
    }
    std::string algorithm = katric::cli_name(conf.algorithm);
    app.add_option(
           "--algorithm",
           algorithm,
           "katric, or a competitor (doc/competitors.md; each needs its CMake option KATRIC_COMPETITOR_<NAME>). "
           "Deprecated: patric = katric, cetric-x = katric --contraction"
    )
        ->transform(CLI::IsMember(algorithms, CLI::ignore_case));

    add_enum_option(app, "--tric-variant", conf.tric_variant)->group("TriC (--algorithm tric)");
    app.add_option(
           "--tric-buffer-size",
           conf.tric_buffer_size,
           "Per-rank buffer size of the buffered variants (TriC's -s)"
    )
        ->check(CLI::Range(std::size_t{100}, std::numeric_limits<std::size_t>::max()))
        ->group("TriC (--algorithm tric)");
    app.add_flag(
           "--tric-sparse-ranges",
           conf.tric_sparse_ranges,
           "Buffered variants: fetch the neighbour id ranges of the ghosts from their owners instead of TriC's "
           "allreduce of all 2n ranges onto every rank (O(n) memory per rank); the pruning stays the same"
    )
        ->group("TriC (--algorithm tric)");
    app.add_option(
           "--havoqgt-delegate-threshold",
           conf.havoqgt_delegate_threshold,
           "Vertices with a higher degree become delegates (split over all ranks)"
    )
        ->capture_default_str()
        ->group("HavoqGT (--algorithm havoqgt)");
    app.add_option(
           "--havoqgt-partition-passes",
           conf.havoqgt_partition_passes,
           "Edge partitioning passes during graph construction (at most the ranks per node)"
    )
        ->capture_default_str()
        ->check(CLI::PositiveNumber)
        ->group("HavoqGT (--algorithm havoqgt)");
    app.add_option("--havoqgt-chunk-size", conf.havoqgt_chunk_size, "Edges per chunk during graph construction")
        ->capture_default_str()
        ->check(CLI::PositiveNumber)
        ->group("HavoqGT (--algorithm havoqgt)");
    app.add_flag(
           "--havoqgt-stats,!--no-havoqgt-stats",
           conf.havoqgt_stats,
           "Compute and print HavoqGT's diagnostics (allreduces, degree histograms) between orientation and count, "
           "as its own driver does"
    )
        ->group("HavoqGT (--algorithm havoqgt)");

    // `new` was the name of `direct` until 2026-11-20 (job files and configs before); remove once they have run
    add_enum_option(
        app, "--communication-policy", conf.communication_policy, {{"new", katric::CommunicationPolicy::direct}}
    );

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
    if (algorithm == "cetric") {
        if (rank == 0) {
            std::cerr << "--algorithm cetric (pre-CETRIC) was removed; run it from the tag pre-refactor. CETRIC is "
                         "--contraction (formerly --algorithm cetric-x)\n";
        }
        MPI_Finalize();
        std::exit(1);
    }
    if (algorithm == "patric" || algorithm == "cetric-x" || algorithm == "cetric_x") {
        conf.contraction = algorithm != "patric";
        algorithm        = katric::cli_name(katric::Algorithm::katric); // recorded as katric (variant: --contraction)
    }
    conf.algorithm = *magic_enum::enum_cast<katric::Algorithm>(algorithm);
    if (!katric::is_compiled_in(conf.algorithm)) {
        if (rank == 0) {
            std::string name(magic_enum::enum_name(conf.algorithm));
            std::ranges::transform(name, name.begin(), [](unsigned char c) { return std::toupper(c); });
            std::cerr << "--algorithm " << algorithm << " is not compiled in: configure with -DKATRIC_COMPETITOR_"
                      << name << "=ON\n";
        }
        MPI_Finalize();
        std::exit(1);
    }
    if (conf.tric_sparse_ranges && conf.tric_variant != katric::TricVariant::buffered
        && conf.tric_variant != katric::TricVariant::hash2) {
        if (rank == 0) {
            std::cerr << "--tric-sparse-ranges is implemented only for --tric-variant buffered and hash2\n";
        }
        MPI_Finalize();
        std::exit(1);
    }
    if (conf.num_threads == 0) {
        conf.num_threads = tbb::this_task_arena::max_concurrency();
    }
    if (conf.num_threads > 1) {
        // the phases are sequential; the options of the hybrid path (--local-parallel, --global-parallel,
        // --edge-partitioning, --task-pool-type, --task-priorities, --*-degree-of-parallelism) stay accepted so that
        // the paper configurations parse, and come back into use when it is re-ported (doc/refactoring-plan.md)
        if (rank == 0) {
            std::cerr << "--num-threads > 1 is not supported: the hybrid (MPI + TBB) path is being re-ported\n";
        }
        MPI_Finalize();
        std::exit(1);
    }
    if (conf.caliper_config.empty() && !conf.caliper_output.empty()) {
        conf.caliper_config = "spot,katric-timers,katric-counters";
    }
    conf.options = option_values(app);
    return conf;
}
