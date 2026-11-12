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

#include <adiak.hpp>
#include <caliper/ConfigManager.h>
#include <caliper/cali.h>
#include <kamping/spdlog_adapter/logging.hpp>
#include <kassert/kassert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <mpi.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#ifdef CETRIC_HAS_OPENMP
    #include <omp.h>
#endif
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "./algorithm.h"
#include "./parse_config.h"
#include "./verify.h"
#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/input.h"
#include "cetric/timer.h"

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
