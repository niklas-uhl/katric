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

#include <atomic>
#include <chrono>
#include <map>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <cereal/cereal.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <graph-io/graph_definitions.h>
#include <graph-io/graph_io.h>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#ifdef CETRIC_HAS_OPENMP
    #include <omp.h>
#endif
#include <tbb/global_control.h>

#include "cetric/counters/shared_memory_cetric.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph.h"
#include "git_hash.h"
#include "parse_parameters.h"

void print_summary(size_t triangles, std::vector<double> const& wall_times) {
    SPDLOG_INFO("{:<15}{:>10}", "triangles: ", triangles);
    for (size_t i = 0; i < wall_times.size(); ++i) {
        SPDLOG_INFO("{:<15}{:>10} s", fmt::format("[{}] time: ", i + 1), wall_times[i]);
    }
    SPDLOG_INFO("{:<15}{:>10} s", "min time: ", *std::min_element(wall_times.begin(), wall_times.end()));
}

int main(int argc, char* argv[]) {
    spdlog::cfg::load_env_levels();
    spdlog::set_pattern("[%Y-%m-%d %T.%e] [%^%l%$] %v");
    auto                     conf            = shmetric::parse_config(argc, argv);
    size_t                   max_concurrency = tbb::this_task_arena::max_concurrency();
    if (conf.num_threads == 0) {
        conf.num_threads = max_concurrency;
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, conf.num_threads + 1);

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

    cetric::graph::DistributedGraph<> G{graphio::read_graph(conf.input_file, conf.input_format)};

    size_t              triangles = 0;
    std::vector<double> wall_times;
    for (size_t iter = 0; iter < conf.iterations; ++iter) {
        cetric::profiling::Timer global_time;
        triangles = cetric::run_shmem(G, conf);
        wall_times.push_back(global_time.elapsed_time());
    }
    print_summary(triangles, wall_times);
}
