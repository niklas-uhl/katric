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

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <kamping/v2/collectives/bcast.hpp>
#include <kamping/v2/collectives/reduce.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/ops.hpp>
#include <kamping/v2/sentinels.hpp>
#include <kamping/v2/views/ref_single_view.hpp>
#include <memory>

#include <caliper/ConfigManager.h>
#include <caliper/cali.h>
#include <fmt/chrono.h>
#include <mpi.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <adiak.hpp>
#include <dstl/debugging.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <kamping/v2/collectives/barrier.hpp>
#include <kamping/v2/environment.hpp>
#include <kamping/v2/time.hpp>
#include <kassert/kassert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <mpi/thread_level.hpp>

#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include "./algorithm.h"
#include "./memory_environment.h"
#include "./build_info.h"
#include "./parse_config.h"
#include "./verify.h"
#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/input.h"

auto main(int argc, char* argv[]) -> int {
  kamping::v2::environment env(kamping::v2::thread_level::funneled);
  spdlog::cfg::load_env_levels();
  kamping::logging::setup_logging();
  if (env.provided_thread_level() <  // NOLINT(*-static-accessed-through-instance)
      mpi::experimental::thread_level::funneled) {
    SPDLOG_LOGGER_CRITICAL(
        spdlog::get("root"),
        "The MPI implementation must support MPI_THREAD_FUNNELED");
    std::exit(1);
  }
  int exit_code = 0;
  // the communicator katric runs on (the global phase's legacy message queue is
  // fixed to MPI_COMM_WORLD)
  auto const comm = kamping::v2::comm_world();
  {
    dstl::wait_for_debugger_if_requested();
    katric::Config conf =
        parse_config(std::span{argv, static_cast<std::size_t>(argc)}, comm);
    size_t expected_triangles = 0;
    tbb::global_control global_limit(
        tbb::global_control::max_allowed_parallelism, conf.num_threads);

    // Caliper: region times and counters aggregated across ranks, iterations
    // annotated as a loop; Adiak holds the run metadata
    cali::ConfigManager caliper;
    if (!conf.caliper_config.empty()) {
      katric::profiling::add_katric_configs(caliper, conf.caliper_output);
      caliper.add(conf.caliper_config.c_str());
      if (caliper.error()) {
        SPDLOG_LOGGER_CRITICAL(spdlog::get("root"),
                               "Invalid Caliper config: {}",
                               caliper.error_msg());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      // adiak takes a pointer to the MPI_Comm (void*, so passing the handle
      // itself compiles and crashes in adiak_collect_all)
      MPI_Comm adiak_comm = comm.mpi_handle();
      adiak::init(static_cast<void*>(&adiak_comm));
      for (auto const& [name, value] : conf.options) {
        adiak::value("config." + name, value);
      }
      adiak::value("git_commit", conf.git_commit);
      // How the binary was built (apps/build_info.cpp.in)
      adiak::value("build.type", std::string(katric::build_info::build_type));
      adiak::value("build.compiler",
                   std::string(katric::build_info::cxx_compiler));
      adiak::value("build.cxx_flags",
                   std::string(katric::build_info::cxx_flags));
      adiak::value("build.cxx_definitions",
                   std::string(katric::build_info::cxx_definitions));
      record_memory_environment();
      adiak::collect_all();  // user, launch date, command line, hosts, job
                             // size, MPI version, ...
      caliper.start();
    }

    // The input is generated (or read) once and ingested; every iteration
    // resets the algorithm to that state.
    auto const algorithm = katric::make_algorithm(conf, comm);
    kagen::Graph graph;
    {
      CALI_CXX_MARK_SCOPE("input");
      try {
        graph = katric::generate_input(conf.kagen_option_string, comm);
      } catch (std::exception const& e) {
        SPDLOG_LOGGER_CRITICAL(
            spdlog::get("root"),
            "KaGen failed for --kagen_option_string '{}': {}",
            conf.kagen_option_string, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    if (conf.verify) {
      CALI_CXX_MARK_SCOPE("verify");
      expected_triangles = katric::reference_triangle_count(graph, comm);
      SPDLOG_LOGGER_INFO(spdlog::get("root"), "reference: {} triangles",
                         expected_triangles);
    }
    {
      CALI_CXX_MARK_SCOPE("ingest");
      algorithm->ingest(std::move(graph));
    }

    size_t triangles = 0;
    bool mismatch = false;
    auto min_time = kamping::v2::clock::duration::max();
    cali::Annotation iteration_annotation("iteration", CALI_ATTR_NESTED);
    for (size_t iter = 0; iter < conf.iterations; ++iter) {
      iteration_annotation.begin(static_cast<int>(iter));
      kamping::v2::barrier(comm);
      {
        CALI_CXX_MARK_SCOPE("reset");
        algorithm->reset();
      }
      kamping::v2::barrier(comm);
      kamping::v2::clock::duration local_time{};
      {
        // the algorithm's run time; its maximum across ranks in the timers
        // output is the iteration's run time
        CALI_CXX_MARK_SCOPE("total");
        auto const start = kamping::v2::clock::now();
        algorithm->run();
        kamping::v2::barrier(comm);
        local_time = kamping::v2::clock::now() - start;
      }
      iteration_annotation.end();
      triangles = algorithm->triangle_count();
      kamping::v2::clock::duration
          time{};  // only for the log, Caliper's results are available after
                   // the run
      kamping::v2::reduce(kamping::v2::views::ref_single(local_time),
                          kamping::v2::views::ref_single(time),
                          kamping::v2::ops::max<>{}, comm);
      min_time = std::min(min_time, time);
      if (comm.rank() == 0) {
        SPDLOG_LOGGER_INFO(spdlog::get("root"),
                           "iteration {}/{}: {}, {} triangles", iter + 1,
                           conf.iterations, time, triangles);
        if (conf.verify && comm.rank() == 0 &&
            triangles != expected_triangles) {
          SPDLOG_LOGGER_ERROR(
              spdlog::get("root"),
              "iteration {}: {} triangles, expected {} (reference)", iter + 1,
              triangles, expected_triangles);
          mismatch = true;
        }
      }
    }
    if (!conf.caliper_config.empty()) {
      // long long: Caliper imports adiak_long as int (AdiakImport.cpp), which
      // wraps at 2^31
      adiak::value("triangles", static_cast<long long>(triangles));
      caliper.flush();  // collective, has to happen before MPI_Finalize
      adiak::fini();
    }
    if (conf.iterations > 1) {
      SPDLOG_LOGGER_INFO(spdlog::get("root"), "min time: {}", min_time);
    }
    kamping::v2::bcast(kamping::v2::views::ref_single(mismatch), comm);
    exit_code = mismatch ? 1 : 0;
  }
  return exit_code;
}
