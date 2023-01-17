#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>

#include <boost/iterator/function_output_iterator.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/range/join.hpp>
#include <mpi.h>
#include <omp.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tlx/meta/has_member.hpp>
#include <tlx/meta/has_method.hpp>
#include <tlx/thread_barrier_mutex.hpp>

#include "cetric/atomic_debug.h"
#include "cetric/config.h"
#include "cetric/counters/cetric_edge_iterator.h"
#include "cetric/counters/intersection.h"
#include "cetric/datastructures/auxiliary_node_data.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/distributed/helpers.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/datastructures/span.h"
#include "cetric/indirect_message_queue.h"
#include "cetric/statistics.h"
#include "cetric/thread_pool.h"
#include "cetric/timer.h"
#include "cetric/util.h"
#include "kassert/kassert.hpp"

namespace cetric {
template <typename NodeIndexer>
inline void
preprocessing(DistributedGraph<NodeIndexer>& G, cetric::profiling::PreprocessingStatistics& stats, const Config& conf) {
    tlx::MultiTimer phase_timer;
    phase_timer.start("orientation");
    auto nodes = G.local_nodes();
    if (conf.num_threads > 1) {
        tbb::task_arena arena(conf.num_threads, 0);
        arena.execute([&] {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G](auto const& r) {
                for (auto node: r) {
                    G.orient(node, node_ordering::degree_outward(G));
                }
            });
            phase_timer.start("sorting");
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&G, &conf](auto const& r) {
                for (auto node: r) {
                    G.sort_neighborhoods(node, node_ordering::id());
                }
            });
        });
    } else {
        for (auto node: nodes) {
            G.orient(node, node_ordering::degree_outward(G));
        }
        phase_timer.start("sorting");
        for (auto node: nodes) {
            G.sort_neighborhoods(node, node_ordering::id_outward(0));
        }
    }
    phase_timer.stop();
    stats.ingest(phase_timer);
}

inline size_t run_shmem(DistributedGraph<>& G, cetric::profiling::Statistics& stats, const Config& conf) {
    tlx::MultiTimer phase_timer;
    bool            debug = false;
    // if (conf.num_threads > 1) {
    //     if (conf.binary_rank_search) {
    //         G.find_ghost_ranks<true>(execution_policy::parallel{conf.num_threads});
    //     } else {
    //         G.find_ghost_ranks<false>(execution_policy::parallel{conf.num_threads});
    //     }
    // } else {
    //     if (conf.binary_rank_search) {
    //         G.find_ghost_ranks<true>(execution_policy::sequential{});
    //     } else {
    //         G.find_ghost_ranks<false>(execution_policy::sequential{});
    //     }
    // }
    node_set ghosts;
    phase_timer.start("preprocessing");
    preprocessing(G, stats.local.preprocessing_local_phase, conf);
    LOG << "Preprocessing finished";
    phase_timer.start("local_phase");
    auto   ctr            = cetric::CetricEdgeIterator(G, conf, 0, 1, MessageQueuePolicy{});
    size_t triangle_count = 0;
    tbb::concurrent_vector<Triangle<RankEncodedNodeId>> triangles;
    tbb::combinable<size_t>                             triangle_count_local_phase{0};
    ctr.run_local(
        [&](Triangle<RankEncodedNodeId> t) {
            (void)t;
            // atomic_debug(t);

            if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                t.normalize();
                triangles.push_back(t);
            }
            // atomic_debug(tbb::this_task_arena::current_thread_index());
            // triangle_count++;
            triangle_count_local_phase.local()++;
        },
        stats,
        node_ordering::id_outward(0),
        node_ordering::degree_outward(G)
    );
    triangle_count += triangle_count_local_phase.combine(std::plus<>{});
    LOG << "Local phase finished ";
    ConditionalBarrier(conf.global_synchronization);
    phase_timer.stop();
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
        LOG << "Verification started";
        KASSERT(triangles.size() == triangle_count);
        std::sort(triangles.begin(), triangles.end(), [](auto const& lhs, auto const& rhs) {
            return std::tuple(lhs.x.id(), lhs.y.id(), lhs.z.id()) < std::tuple(rhs.x.id(), rhs.y.id(), rhs.z.id());
        });
        auto current = triangles.begin();
        while (true) {
            current = std::adjacent_find(current, triangles.end());
            if (current == triangles.end()) {
                break;
            }
            KASSERT(false, "Duplicate triangle " << *current);
            current++;
        }
        LOG << "Verification finished";
    }
    stats.local.ingest(phase_timer);
    return triangle_count;
}
} // namespace cetric
