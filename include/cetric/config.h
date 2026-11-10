/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

//
// Created by Tim Niklas Uhl on 26.10.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_CONFIG_H
#define PARALLEL_TRIANGLE_COUNTER_CONFIG_H

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <magic_enum/magic_enum.hpp>

#include "cetric/datastructures/graph_definitions.h"
#include "cetric/util.h"

namespace cetric {

// Enum values are named after their command line spelling: `_` is written as `-` (both are accepted).
enum class Algorithm { patric, cetric, cetric_x };
enum class Threshold { local_nodes, local_edges, none };
enum class IntersectionMethod { merge, binary_search, binary, hybrid };
enum class ParallelizationMethod { tbb, omp_for, omp_task };
enum class OMPSchedule { stat, dynamic, guided, standard };
enum class TBBPartitioner { stat, simple, standard, affinity };
enum class TaskPoolType { std_thread, task_group };

struct Config {
    Config() = default;
    std::string           kagen_option_string;
    size_t                buffer_threshold                  = std::numeric_limits<size_t>::max();
    double                max_degree_threshold_alpha        = 1.0;
    bool                  empty_pending_buffers_on_overflow = false;
    size_t                iterations                        = 1;
    std::string           primary_cost_function             = "N";
    std::string           secondary_cost_function           = "none";
    Algorithm             algorithm                         = Algorithm::cetric;
    Threshold             threshold                         = Threshold::local_nodes;
    double                threshold_scale                   = 1.0;
    double                high_degree_threshold_scale       = 1.0;
    std::string           communication_policy              = "new";
    bool                  local_parallel                    = false;
    bool                  global_parallel                   = false;
    bool                  parallel_compact                  = false;
    ParallelizationMethod parallelization_method            = ParallelizationMethod::tbb;
    TaskPoolType          task_pool_type                    = TaskPoolType::std_thread;
    bool                  task_priorities                   = false;
    bool                  edge_partitioning                 = false;
    bool                  edge_partitioning_static          = false;
    TBBPartitioner        tbb_partitioner                   = TBBPartitioner::standard;
    OMPSchedule           omp_schedule                      = OMPSchedule::standard;
    size_t                omp_chunksize                     = 0;
    size_t                local_degree_of_parallelism       = 1;
    size_t                global_degree_of_parallelism      = 1;
    IntersectionMethod    intersection_method               = IntersectionMethod::merge;
    size_t                binary_intersection_cutoff        = 1000;
    double                hybrid_cutoff_scale               = 1.0;

    bool dense_degree_exchange   = false;
    bool compact_degree_exchange = false;
    bool global_synchronization  = false;
    bool binary_rank_search      = false;
    bool id_node_ordering        = false;

    bool degree_filtering        = false;
    bool orient_locally          = false;
    bool pseudo2core             = false;
    bool dense_load_balancing    = false;
    bool flag_intersection       = false;
    bool skip_local_neighborhood = false;
    bool synchronized            = false;

    int verbosity_level = 0;
    // Caliper ConfigManager configuration (e.g. "spot"); empty disables Caliper. Output file via caliper_output.
    std::string caliper_config = "";
    std::string caliper_output = "";
    std::string hostname;
    PEID        PEs;
    PEID        rank;
    size_t      num_threads = 1;
    size_t      grainsize   = 1;

    std::string git_commit;

    // name and parsed value of every command line option, recorded as run metadata
    std::vector<std::pair<std::string, std::string>> options;
};

/// Command line spelling of an enum value: its name with `_` written as `-`.
template <typename Enum>
    requires std::is_enum_v<Enum>
std::string cli_name(Enum value) {
    std::string name(magic_enum::enum_name(value));
    std::replace(name.begin(), name.end(), '_', '-');
    return name;
}

} // namespace cetric

#endif // PARALLEL_TRIANGLE_COUNTER_CONFIG_H
