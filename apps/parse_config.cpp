// Copyright (c) 2020-2023 Tim Niklas Uhl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "./parse_config.h"

#include <cstdlib>
#include <iostream>

#include <mpi.h>
#include <tbb/task_arena.h>
#include <CLI/CLI.hpp>
#include <CLI/Error.hpp>
#include <CLI/Validators.hpp>
#include <kamping/v2/comm.hpp>
#include <span>

#include "./parse_parameters.h"
#include "git_hash.h"
#include "katric/config.h"

auto parse_config(std::span<char*> argv, kamping::v2::comm_view comm)
    -> katric::Config {
  CLI::App app("Parallel Triangle Counter");
  app.option_defaults()->always_capture_default();
  app.set_config("--config")->expected(0, 3);
  // unknown keys in a --config file are errors instead of being ignored (e.g.
  // options that were removed)
  app.allow_config_extras(CLI::config_extras_mode::error);

  katric::Config conf;
  conf.git_commit = katric::git_hash;

  app.add_option("--kagen_option_string", conf.kagen_option_string,
                 "KaGen generator or file input, e.g. "
                 "\"rgg2d;n=16384;m=131072;seed=7\" or "
                 "\"file;filename=graph.metis;input_format=metis;distribution="
                 "balance-edges\"")
      ->required();

  app.add_option("--iterations", conf.iterations);
  app.add_flag("--verify", conf.verify,
               "Check the triangle count of every iteration against a "
               "sequential count on the graph gathered to rank 0 "
               "(small inputs only); exit with 1 on a mismatch");

  add_enum_option(app, "--primary-cost-function", conf.primary_cost_function);
  app.add_option("--rebalance-threshold", conf.rebalance_threshold,
                 "Redistribute in the primary load balancing only if the "
                 "cost's max over the ranks exceeds this times the "
                 "achievable, max(avg, the largest vertex cost) "
                 "(1 = whenever it can be improved at all); otherwise the "
                 "input graph is kept, with its exchanged degrees and "
                 "orientation")
      ->capture_default_str();
  add_enum_option(app, "--rebalance-precheck", conf.rebalance_precheck)
      ->description(
          "degree-bound: first the cost with each ghost's out-degree bounded "
          "by its degree (IDD for IDPD; needs only the degree exchange), and "
          "keep the input graph if its max/avg does not exceed "
          "--rebalance-threshold; none: exchange the ghosts' out-degrees and "
          "compute the cost always");

  app.add_option("--caliper-config", conf.caliper_config,
                 "Caliper config, e.g. \"katric-timers,katric-counters\" "
                 "(JSON), \"spot,katric-counters\" or "
                 "\"runtime-report\". "
                 "Caliper's CALI_CONFIG environment variable works "
                 "independently of this option.");
  app.add_option(
      "--caliper-output", conf.caliper_output,
      "Output prefix: <prefix>.cali (spot, for Thicket), <prefix>.timers.json "
      "and <prefix>.counters.json. Enables "
      "spot,katric-timers,katric-counters unless --caliper-config is given.");

  app.add_option("--num-threads", conf.num_threads);
  app.add_option("--global-degree-of-parallelism",
                 conf.global_degree_of_parallelism);
  app.add_option("--local-degree-of-parallelism",
                 conf.local_degree_of_parallelism);

  app.add_flag("--pseudo2core", conf.pseudo2core);
  // on by default; kaval omits false flags, so suites turn it off with
  // `no-message-trimming: true`
  app.add_flag("--message-trimming,!--no-message-trimming",
               conf.message_trimming,
               "Global phase: send each rank only the part of N+(v) from its "
               "first receiver on (in orientation order), and "
               "nothing if no entry follows it")
      ->default_str("true");

  add_enum_option(app, "--degree-exchange", conf.degree_exchange);
  // Removed in 401dcb9 (there are no barriers between the phases any more);
  // still accepted, and ignored, so that job files generated before still run.
  // Remove once they have.
  app.add_flag("--global-synchronization", "Ignored (deprecated)");

  app.add_flag("--contraction", conf.contraction,
               "CETRIC: the local phase also finds the triangles with two "
               "local vertices (through the ghosts' out-edges to "
               "local vertices), the global phase sends only the cut edges. "
               "Without: DiTriC");
  // Before the rename (2026-11-19) the variants were --algorithm patric
  // (DiTriC), cetric-x (CETRIC) and cetric (pre-CETRIC, removed). Still
  // accepted so that job files generated before still run; remove once they
  // have.
  std::string algorithm;
  app.add_option("--algorithm", algorithm,
                 "Deprecated: patric = DiTriC, cetric-x = --contraction")
      ->transform(CLI::IsMember({"patric", "cetric-x", "cetric", "cetric_x"},
                                CLI::ignore_case));

  // `new` was the name of `direct` until 2026-11-20 (job files and configs
  // before); remove once they have run
  add_enum_option(app, "--communication-policy", conf.communication_policy,
                  {{"new", katric::CommunicationPolicy::direct}});

  app.add_flag("--local-parallel", conf.local_parallel);
  app.add_flag("--global-parallel", conf.global_parallel);
  app.add_flag("--parallel-compact", conf.parallel_compact);

  app.add_flag("--edge-partitioning", conf.edge_partitioning);

  add_enum_option(app, "--task-pool-type", conf.task_pool_type);
  app.add_flag("--task-priorities", conf.task_priorities);

  add_enum_option(app, "--message-queue", conf.message_queue);
  // legacy queue only
  add_enum_option(app, "--threshold", conf.threshold);
  app.add_option("--threshold-scale", conf.threshold_scale);
  // BriefKAsten only
  app.add_option("--buffer-size", conf.buffer_size,
                 "BriefKAsten: bytes per destination buffer")
      ->check(CLI::PositiveNumber);
  app.add_option("--request-slots", conf.request_slots,
                 "BriefKAsten: sends in flight at once")
      ->check(CLI::PositiveNumber);
  app.add_flag("--flow-control,!--no-flow-control", conf.flow_control,
               "BriefKAsten: credit-based flow control")
      ->default_str("true");
  app.add_flag("--legacy-pool", conf.legacy_pool,
               "BriefKAsten experiment: the pool handling from before the "
               "pool fix (A/B only)");
  app.add_flag("--drop-shells", conf.drop_shells,
               "BriefKAsten experiment: keep the entry on overflow and drop "
               "the moved-from shell instead of erasing (the pool fix without "
               "#29's erase; A/B only)");
  add_enum_option(app, "--receiver", conf.receiver);
  app.add_option("--poll-skip", conf.poll_skip,
                 "BriefKAsten: poll after every n-th sender vertex only")
      ->check(CLI::PositiveNumber);

  add_enum_option(app, "--intersection-method", conf.intersection_method);
  app.add_option("--binary-intersection-cutoff",
                 conf.binary_intersection_cutoff);
  app.add_option("--hybrid-cutoff-scale", conf.hybrid_cutoff_scale);

  CLI11_PARSE_MPI(app, argv.size(), argv.data(), comm.rank(), comm.size());
  if (!algorithm.empty()) {
    if (algorithm == "cetric") {
      if (comm.rank() == 0) {
        std::cerr << "--algorithm cetric (pre-CETRIC) was removed; run it from "
                     "the tag pre-refactor. CETRIC is "
                     "--contraction (formerly --algorithm cetric-x)\n";
      }
      MPI_Finalize();
      std::exit(1);
    }
    conf.contraction = algorithm != "patric";
  }
  // BriefKAsten's grid proxies forward from inside the receive handler and
  // block for a send slot; without credit-based flow control two proxies can
  // wait for each other (doc/briefkasten-migration.md, step 5)
  if (conf.message_queue == katric::MessageQueue::briefkasten &&
      conf.communication_policy == katric::CommunicationPolicy::grid &&
      !conf.flow_control) {
    if (comm.rank() == 0) {
      std::cerr << "--communication-policy grid needs flow control with "
                   "--message-queue briefkasten (drop --no-flow-control)\n";
    }
    MPI_Finalize();
    std::exit(1);
  }
  if (conf.num_threads == 0) {
    conf.num_threads = tbb::this_task_arena::max_concurrency();
  }
  if (conf.num_threads > 1) {
    // the phases are sequential; the options of the hybrid path
    // (--local-parallel, --global-parallel,
    // --edge-partitioning, --task-pool-type, --task-priorities,
    // --*-degree-of-parallelism) stay accepted so that the paper configurations
    // parse, and come back into use when it is re-ported
    // (doc/refactoring-plan.md)
    if (comm.rank() == 0) {
      std::cerr << "--num-threads > 1 is not supported: the hybrid (MPI + TBB) "
                   "path is being re-ported\n";
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
