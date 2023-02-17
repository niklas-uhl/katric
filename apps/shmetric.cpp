#include <atomic>
#include <chrono>
#include <map>

#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <backward.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <graph-io/graph_definitions.h>
#include <graph-io/graph_io.h>
#include <magic_enum.hpp>
#include <omp.h>
#include <tbb/global_control.h>

#include "cetric/counters/shared_memory_cetric.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph.h"
#include "cetric/statistics.h"
#include "git_hash.h"
#include "parse_parameters.h"

void print_summary(
    const cetric::Config&                       conf,
    std::vector<cetric::profiling::Statistics>& all_stats,
    std::optional<double>                       io_time = std::nullopt
) {
    if (!conf.json_output.empty()) {
        if (conf.rank == 0) {
            KASSERT(all_stats[0].triangles == all_stats[0].counted_triangles);
            auto write_json_to_stream = [&](auto& stream) {
                cereal::JSONOutputArchive ar(stream);
                ar(cereal::make_nvp("stats", all_stats));
                if (io_time.has_value()) {
                    ar(cereal::make_nvp("io_time", io_time.value()));
                }
                ar(cereal::make_nvp("config", conf));
            };
            if (conf.json_output == "stdout") {
                std::stringstream out;
                write_json_to_stream(out);
                std::cout << out.str();
            } else {
                std::ofstream out(conf.json_output);
                write_json_to_stream(out);
            }
        }
    } else {
        if (conf.rank == 0) {
            std::cout << std::left << std::setw(15) << "triangles: " << std::right << std::setw(10)
                      << all_stats[0].counted_triangles << std::endl;
            double min_time = std::numeric_limits<double>::max();
            for (size_t i = 0; i < all_stats.size(); ++i) {
                std::cout << std::left << std::setw(15) << fmt::format("[{}] time: ", i + 1) << std::right
                          << std::setw(10) << all_stats[i].global_wall_time << " s" << std::endl;
                min_time = std::min(all_stats[i].global_wall_time, min_time);
            }
            std::cout << std::left << std::setw(15) << "min time: " << std::right << std::setw(10) << min_time << " s"
                      << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    backward::SignalHandling sh;
    auto                     conf            = shmetric::parse_config(argc, argv);
    size_t                   max_concurrency = tbb::this_task_arena::max_concurrency();
    if (conf.num_threads == 0) {
        conf.num_threads = max_concurrency;
    }
    if (conf.num_threads > max_concurrency) {
        std::cout
            << fmt::format("Warning, TBB uses only {} instead of {} threads!\n", max_concurrency, conf.num_threads);
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, conf.num_threads + 1);

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

    std::optional<double>             io_time;
    cetric::profiling::Timer          t;
    cetric::graph::DistributedGraph<> G{graphio::read_graph(conf.input_file, conf.input_format)};
    // compute the number of wedges
    tbb::enumerable_thread_specific<size_t> thread_local_wedges = 0;
    tbb::parallel_for(tbb::blocked_range(G.local_nodes().begin(), G.local_nodes().end()), [&](auto const& r) {
        auto& wedges_local = thread_local_wedges.local();
        for (auto node: r) {
            auto degree = G.degree(node);
            wedges_local += (degree * (degree - 1)) / 2;
        }
    });
    size_t wedges = thread_local_wedges.combine(std::plus<>{});

    std::vector<cetric::profiling::Statistics> all_stats;
    for (size_t iter = 0; iter < conf.iterations; ++iter) {
        cetric::profiling::Statistics stats(0, 1);
        cetric::profiling::Timer      timer;
        stats.local.wedges = wedges;
        // atomic_debug(G);
        stats.local.io_time = timer.elapsed_time();
        cetric::profiling::Timer global_time;

        cetric::run_shmem(G, stats, conf);

        stats.local.local_wall_time = global_time.elapsed_time();

        stats.collapse();
        all_stats.emplace_back(std::move(stats));
    }
    print_summary(conf, all_stats, io_time);
}
