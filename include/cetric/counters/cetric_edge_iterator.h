//
// Created by Tim Niklas Uhl on 22.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H
#define PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H

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
#include <gmpxx.h>
#include <message-queue/buffered_queue.h>
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
#include "cetric/concurrent_buffered_queue.h"
#include "cetric/config.h"
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
using namespace graph;

struct MessageQueuePolicy {};
struct GridPolicy {};

TLX_MAKE_HAS_METHOD(grow_by);

template <typename VectorType>
static constexpr bool has_grow_by_v =
    has_method_grow_by<VectorType, typename VectorType::iterator(typename VectorType::size_type)>::value;
static_assert(has_grow_by_v<tbb::concurrent_vector<RankEncodedNodeId>>);

struct Merger {
    template <template <typename...> typename VectorType>
    size_t
    operator()(VectorType<RankEncodedNodeId>& buffer, std::vector<RankEncodedNodeId> msg, int tag [[maybe_unused]]) {
        if constexpr (has_grow_by_v<VectorType<RankEncodedNodeId>>) {
            // atomic_debug(fmt::format("grow by {} with {}", msg.size(), msg));
            auto insert_position = buffer.grow_by(msg.size() + 1);
            std::copy(msg.begin(), msg.end(), insert_position);
            *(insert_position + msg.size()) = RankEncodedNodeId::sentinel();
            // atomic_debug(fmt::format(
            //     "done with copy of {} -> {}",
            //     msg,
            //     boost::make_iterator_range(insert_position, insert_position + msg.size())
            // ));
        } else {
            for (auto elem: msg) {
                buffer.push_back(elem);
            }
            buffer.push_back(RankEncodedNodeId::sentinel());
        }
        return msg.size() + 1;
    }
};

struct Splitter {
    template <typename MessageFunc, template <typename...> typename VectorType>
    void operator()(VectorType<RankEncodedNodeId> buffer, MessageFunc&& on_message, PEID sender) {
        auto buffer_ptr = std::make_shared<VectorType<RankEncodedNodeId>>(std::move(buffer));
        std::vector<RankEncodedNodeId>::iterator slice_begin = buffer_ptr->begin();
        while (slice_begin != buffer_ptr->end()) {
            auto slice_end = std::find(slice_begin, buffer_ptr->end(), RankEncodedNodeId::sentinel());
            on_message(
                SharedVectorSpan(
                    buffer_ptr,
                    std::distance(buffer_ptr->begin(), slice_begin),
                    std::distance(buffer_ptr->begin(), slice_end)
                ),
                sender
            );
            slice_begin = slice_end + 1;
        }
    }
};
template <typename GraphType>
size_t get_threshold(const GraphType& G, Config const& conf) {
    switch (conf.threshold) {
        case Threshold::local_nodes:
            return conf.threshold_scale * G.local_node_count();
            break;
        case Threshold::local_edges:
            return conf.threshold_scale * G.local_edge_count();
            break;
        case Threshold::none:
            return std::numeric_limits<size_t>::max();
            break;
    }
    return 0;
}
template <class GraphType, class CommunicationPolicy>
class CetricEdgeIterator {
public:
    CetricEdgeIterator(
        GraphType& G, const Config& conf, PEID rank, PEID size, CommunicationPolicy&& = CommunicationPolicy{}
    )
        : G(G),
          conf_(conf),
          rank_(rank),
          size_(size),
          last_proc_(G.local_node_count(), -1),
          is_v_neighbor_(/*G.local_node_count() + G.ghost_count(), false*/),
          interface_nodes_(),
          pe_min_degree(),
          threshold_(std::numeric_limits<size_t>::max()) {
        EdgeId total_edge_count = G.total_edge_count();
        high_degree_threshold_  = conf_.high_degree_threshold_scale * sqrt(total_edge_count / 2.0);
    }

    void set_threshold(size_t threshold) {
        threshold_ = threshold;
    }

    template <typename TriangleFunc, typename NodeOrdering, typename EdgeOrientationOrdering>
    inline void
    run_local(TriangleFunc emit, cetric::profiling::Statistics& stats, NodeOrdering&& node_ordering, EdgeOrientationOrdering&&) {
        if (conf_.local_parallel && conf_.num_threads > 1) {
            if (conf_.algorithm == Algorithm::CetricX) {
                if (conf_.edge_partitioning) {
                    run_local_parallel_edge_partitioned_with_ghosts<EdgeOrientationOrdering>(
                        emit,
                        stats,
                        interface_nodes_,
                        std::forward<NodeOrdering>(node_ordering)
                    );
                } else {
                    run_local_parallel_with_ghosts<EdgeOrientationOrdering>(
                        emit,
                        stats,
                        interface_nodes_,
                        std::forward<NodeOrdering>(node_ordering)
                    );
                }
                return;
            }
            if (conf_.edge_partitioning) {
                run_local_parallel_edge_partitioned<EdgeOrientationOrdering>(
                    emit,
                    stats,
                    interface_nodes_,
                    std::forward<NodeOrdering>(node_ordering)
                );
                return;
            }
            run_local_parallel<EdgeOrientationOrdering>(
                emit,
                stats,
                interface_nodes_,
                std::forward<NodeOrdering>(node_ordering)
            );
        } else {
            if (conf_.algorithm == Algorithm::CetricX) {
                run_local_sequential_with_ghosts<EdgeOrientationOrdering>(
                    emit,
                    stats,
                    interface_nodes_.local(),
                    std::forward<NodeOrdering>(node_ordering)
                );
            } else {
                run_local_sequential<EdgeOrientationOrdering>(
                    emit,
                    stats,
                    interface_nodes_.local(),
                    std::forward<NodeOrdering>(node_ordering)
                );
            }
        }
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_sequential(
        TriangleFunc                    emit,
        cetric::profiling::Statistics&  stats,
        std::vector<RankEncodedNodeId>& interface_nodes,
        NodeOrdering&&                  node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        for (auto v: G.local_nodes()) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                stats.local.skipped_nodes++;
                continue;
            }
            bool   interface_node_pushed = false;
            size_t neighbor_index        = 0;
            for (RankEncodedNodeId u: G.out_adj(v).neighbors()) {
                KASSERT(*(G.out_adj(v).neighbors().begin() + neighbor_index) == u);
                if (u.rank() != rank_) {
                    if (!interface_node_pushed) {
                        interface_nodes.emplace_back(v);
                        interface_node_pushed = true;
                    }
                    if constexpr (!std::is_same_v<NodeOrdering, node_ordering::id_outward> && !std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>) {
                        neighbor_index++;
                        continue;
                    }
                    // if (conf_.algorithm == Algorithm::Patric || conf_.algorithm == Algorithm::CetricX) {
                    //     neighbor_index++;
                    //     continue;
                    // }
                    // atomic_debug(
                    //     fmt::format("Remaining neighbors {}",
                    //                 boost::make_iterator_range(G.out_adj(v).neighbors().begin() + neighbor_index,
                    //                                            G.out_adj(v).neighbors().end())));
                    // TODO: we should be able to break here when the edges are properly sorted
                    KASSERT(std::all_of(
                        G.out_adj(v).neighbors().begin() + neighbor_index,
                        G.out_adj(v).neighbors().end(),
                        [rank = this->rank_](RankEncodedNodeId node) { return node.rank() != rank; }
                    ));
                    break;
                }
                auto   v_neighbors = G.out_adj(v).neighbors();
                auto   u_neighbors = G.out_adj(u).neighbors();
                size_t offset      = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = neighbor_index;
                }
                // for each edge (v, u) we check the open wedge (v, u, w) for w in N(u)+
                stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                stats.local.intersection_size_local +=
                    v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
                neighbor_index++;
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_sequential_with_ghosts(
        TriangleFunc                    emit,
        cetric::profiling::Statistics&  stats,
        std::vector<RankEncodedNodeId>& interface_nodes,
        NodeOrdering&&                  node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        for (auto v: G.local_nodes()) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                stats.local.skipped_nodes++;
                continue;
            }
            bool   interface_node_pushed = false;
            size_t neighbor_index        = 0;
            for (RankEncodedNodeId u: G.out_adj(v).neighbors()) {
                KASSERT(*(G.out_adj(v).neighbors().begin() + neighbor_index) == u);
                auto v_neighbors = G.out_adj(v).neighbors();
                EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable>::neighbor_range_type u_neighbors;
                if (u.rank() != rank_) {
                    if (!interface_node_pushed) {
                        interface_nodes.emplace_back(v);
                        interface_node_pushed = true;
                    }
                    u_neighbors = G.ghost_adj(u).neighbors();
                } else {
                    u_neighbors = G.out_adj(u).neighbors();
                }
                // for each edge (v, u) we check the open wedge (v, u, w) for
                // w in N(u)+
                size_t offset = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = neighbor_index;
                }
                stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                stats.local.intersection_size_local +=
                    v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
                neighbor_index++;
            }
        }
        for (auto v: G.ghosts()) {
            if (conf_.pseudo2core && G.ghost_degree(v) < 2) {
                stats.local.skipped_nodes++;
                continue;
            }
            size_t neighbor_index = 0;
            for (RankEncodedNodeId u: G.ghost_adj(v).neighbors()) {
                KASSERT(*(G.ghost_adj(v).neighbors().begin() + neighbor_index) == u);
                auto v_neighbors = G.ghost_adj(v).neighbors();
                auto u_neighbors = G.out_adj(u).neighbors();
                // for each edge (v, u) we check the open wedge (v, u, w) for
                // w in N(u)+
                stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                size_t offset = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = neighbor_index;
                }
                stats.local.intersection_size_local +=
                    v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
                neighbor_index++;
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_edge_partitioned_with_ghosts(
        TriangleFunc                                                     emit,
        cetric::profiling::Statistics&                                   stats,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();

        auto edge_locator = G.build_ghost_edge_locator();

        tbb::task_arena            arena(conf_.num_threads, 0);
        tbb::blocked_range<EdgeId> edge_ids(EdgeId{0}, G.local_edge_count_with_ghost_edges(), conf_.grainsize);
        std::atomic<size_t>        skipped_nodes = 0;

        std::vector<std::atomic<bool>> is_interface;
        constexpr bool fast_is_interface = std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                                           || std::is_same_v<NodeOrdering, node_ordering::id>
                                           || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
        if constexpr (!fast_is_interface) {
            is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
        }
        auto body = [&](auto const& edge_id_range) {
            auto tail_idx        = edge_locator.edge_tail_idx(edge_id_range.begin());
            auto tail            = edge_locator.get_node(tail_idx);
            auto tail_first_edge = edge_locator.first_edge_id_for_idx(tail_idx);
            auto tail_last_edge  = edge_locator.last_edge_id_for_idx(tail_idx);
            // atomic_debug(fmt::format(
            //     "[t{}] processing edges {} to {}",
            //     tbb::this_task_arena::current_thread_index(),
            //     std::max(edge_id_range.begin(), tail_first_edge),
            //     edge_id_range.end()
            // ));
            auto degree = [this](RankEncodedNodeId node) {
                if (node.rank() == rank_) {
                    return G.outdegree(node);
                } else {
                    return G.ghost_degree(node);
                }
            };
            for (auto edge_id = std::max(edge_id_range.begin(), tail_first_edge); edge_id < edge_id_range.end();
                 edge_id++) {
                while (edge_id >= tail_last_edge || (conf_.pseudo2core && degree(tail) < 2)) {
                    if (conf_.pseudo2core && G.degree(tail) < 2) {
                        skipped_nodes++;
                    }
                    tail_idx++;
                    tail_first_edge = edge_locator.first_edge_id_for_idx(tail_idx);
                    edge_id         = tail_first_edge;
                    if (edge_id >= edge_id_range.end()) {
                        goto finish;
                    }
                    tail           = edge_locator.get_node(tail_idx);
                    tail_last_edge = edge_locator.last_edge_id_for_idx(tail_idx);
                }
                KASSERT(tail_first_edge <= edge_id);
                KASSERT(edge_id < tail_last_edge);
                auto v = tail;
                auto u = edge_locator.get_edge_head(edge_id);
                if constexpr (fast_is_interface) {
                    if (tail.rank() == rank_ && edge_id == tail_first_edge) {
                        // we are the first thread to examine this node
                        // we check if it an interface node
                        if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(tail)) {
                            interface_nodes.local().emplace_back(tail);
                        }
                    }
                } else {
                    if (u.rank() != rank_) {
                        bool was_set = is_interface[tail_idx].exchange(true);
                        if (!was_set) {
                            interface_nodes.local().emplace_back(tail);
                        }
                    }
                }
                // processed_edges[tail_idx].push_back(RankEncodedEdge{u, v});
                auto   v_neighbors = v.rank() == rank_ ? G.out_adj(v).neighbors() : G.ghost_adj(v).neighbors();
                auto   u_neighbors = u.rank() == rank_ ? G.out_adj(u).neighbors() : G.ghost_adj(u).neighbors();
                size_t offset      = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = edge_id - tail_first_edge;
                }
                // auto offset      = edge_id - tail_first_edge;
                // atomic_debug(
                //     fmt::format("[t{}] intersecting {} {}",
                //     tbb::this_task_arena::current_thread_index(), v, u)
                // );
                // for each edge (v, u) we check the open wedge (v, u, w)
                // for w in N(u)+
                stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                stats.local.intersection_size_local +=
                    v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
            }
        finish:
            return;
        };

        switch (conf_.tbb_partitioner) {
            case TBBPartitioner::stat:
                arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::static_partitioner{}); });
                break;
            case TBBPartitioner::simple:
                arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::simple_partitioner{}); });
                break;
            case TBBPartitioner::standard:
                arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::auto_partitioner{}); });
                break;
            case TBBPartitioner::affinity:
                tbb::affinity_partitioner partitioner;
                arena.execute([&] { tbb::parallel_for(edge_ids, body, partitioner); });
                break;
        }
        // for (size_t i = 0; i < edges.size(); ++i) {
        //     KASSERT(edges[i].size() == processed_edges[i].size(), "Failed for "
        //     << i);
        // }

        stats.local.local_phase_time += phase_time.elapsed_time();
        stats.local.skipped_nodes += skipped_nodes;
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_edge_partitioned(
        TriangleFunc                                                     emit,
        cetric::profiling::Statistics&                                   stats,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();

        auto        edge_locator = G.build_edge_locator();
        auto const& node_indexer = G.node_indexer();

        std::atomic<size_t> skipped_nodes = 0;
        if (conf_.edge_partitioning_static) {
            std::vector<std::thread> threads(conf_.num_threads);
            tlx::ThreadBarrierMutex  barrier(conf_.num_threads);
            std::vector<size_t>      thread_local_cost(conf_.num_threads + 1);
            size_t                   per_thread_cost;
            auto                edges_per_thread = (G.local_edge_count() + conf_.num_threads - 1) / conf_.num_threads;
            std::vector<EdgeId> first_edge(conf_.num_threads + 1);
            for (size_t i = 0; i < conf_.num_threads; i++) {
                threads[i] = std::thread([&, i = i] {
                    EdgeId edge_begin = i * edges_per_thread;
                    EdgeId edge_end   = std::min((i + 1) * edges_per_thread, G.local_edge_count());
                    // std::cout << fmt::format("[t{}] procesing edges {} to {}\n", i,
                    // edge_begin, edge_end);

                    auto first_tail_idx  = edge_locator.edge_tail_idx(edge_begin);
                    auto tail_idx        = first_tail_idx;
                    auto tail            = node_indexer.get_node(tail_idx);
                    auto tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    auto tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);

                    for (auto edge_id = edge_begin; edge_id < edge_end; edge_id++) {
                        if (edge_id >= tail_last_edge) {
                            tail_idx++;
                            tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                            tail            = node_indexer.get_node(tail_idx);
                            tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                        }
                        auto v    = tail;
                        auto u    = edge_locator.get_edge_head(edge_id);
                        auto cost = [&](RankEncodedNodeId const& v, RankEncodedNodeId const& u) -> size_t {
                            if (edge_id < tail_first_edge) {
                                return 0;
                            }
                            if (u.rank() != rank_) {
                                return 1;
                            }
                            auto offset = edge_id - tail_first_edge;
                            return 1 + G.outdegree(v) - offset + G.outdegree(u);
                        };
                        thread_local_cost[i] += cost(v, u);
                    }
                    barrier.wait([&] {
                        // std::cout << fmt::format("thread local cost {}\n",
                        // thread_local_cost);
                        std::exclusive_scan(
                            thread_local_cost.begin(),
                            thread_local_cost.end(),
                            thread_local_cost.begin(),
                            size_t{0}
                        );
                        auto total_cost = thread_local_cost.back();
                        per_thread_cost = total_cost / conf_.num_threads;
                        // std::cout << fmt::format("thread local cost {}\n",
                        // thread_local_cost); std::cout << fmt::format("per thread cost
                        // {}\n", per_thread_cost);
                    });
                    tail_idx        = first_tail_idx;
                    tail            = node_indexer.get_node(tail_idx);
                    tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);

                    size_t running_sum      = thread_local_cost[i];
                    auto   running_sum_prev = running_sum;
                    for (auto edge_id = edge_begin; edge_id < edge_end; edge_id++) {
                        if (edge_id >= tail_last_edge) {
                            tail_idx++;
                            tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                            tail            = node_indexer.get_node(tail_idx);
                            tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                        }
                        auto v    = tail;
                        auto u    = edge_locator.get_edge_head(edge_id);
                        auto cost = [&](RankEncodedNodeId const& v, RankEncodedNodeId const& u) -> size_t {
                            if (edge_id < tail_first_edge) {
                                return 0;
                            }
                            if (u.rank() != rank_) {
                                return 1;
                            }
                            auto offset = edge_id - tail_first_edge;
                            return 1 + G.outdegree(v) - offset + G.outdegree(u);
                        };
                        running_sum_prev = running_sum;
                        running_sum += cost(v, u);
                        if (running_sum_prev / per_thread_cost != running_sum / per_thread_cost) {
                            // we found a splitter
                            first_edge[running_sum / per_thread_cost] = edge_id;
                        }
                    }
                    barrier.wait([&] {
                        // std::cout << fmt::format("first edge {}\n", first_edge);
                        first_edge[0]     = 0;
                        first_edge.back() = G.local_edge_count();
                        // std::cout << fmt::format("first edge {}\n", first_edge);
                    });
                    edge_begin = first_edge[i];
                    edge_end   = first_edge[i + 1];
                    // std::cout << fmt::format("[t{}] procesing edges {} to {}\n", i,
                    // edge_begin, edge_end);
                    tail_idx        = edge_locator.edge_tail_idx(edge_begin);
                    tail            = node_indexer.get_node(tail_idx);
                    tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    // atomic_debug(fmt::format(
                    //     "[t{}] processing edges {} to {}",
                    //     tbb::this_task_arena::current_thread_index(),
                    //     std::max(edge_id_range.begin(), tail_first_edge),
                    //     edge_id_range.end()
                    // ));
                    std::vector<std::atomic<bool>> is_interface;
                    constexpr bool                 fast_is_interface =
                        std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                        || std::is_same_v<NodeOrdering, node_ordering::id>
                        || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
                    if constexpr (!fast_is_interface) {
                        is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
                    }
                    for (auto edge_id = std::max(edge_begin, tail_first_edge); edge_id < edge_end; edge_id++) {
                        while (edge_id >= tail_last_edge || (conf_.pseudo2core && G.outdegree(tail) < 2)) {
                            if (conf_.pseudo2core && G.outdegree(tail) < 2) {
                                skipped_nodes++;
                            }
                            // if we leave the edge range of the current tail, we
                            // switch to the next tail node
                            // this is wrapped in a loop, because this vertex may
                            // not have any outgoing edges
                            tail_idx++;
                            tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                            edge_id         = tail_first_edge;
                            if (edge_id >= edge_end) {
                                goto finish;
                            }
                            tail           = node_indexer.get_node(tail_idx);
                            tail_last_edge = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                        }
                        KASSERT(tail_first_edge <= edge_id);
                        KASSERT(edge_id < tail_last_edge);
                        auto v = tail;
                        auto u = edge_locator.get_edge_head(edge_id);
                        if constexpr (fast_is_interface) {
                            if (tail.rank() == rank_ && edge_id == tail_first_edge) {
                                // we are the first thread to examine this node
                                // we check if it an interface node
                                if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(tail)) {
                                    interface_nodes.local().emplace_back(tail);
                                }
                            }
                        } else {
                            if (u.rank() != rank_) {
                                bool was_set = is_interface[tail_idx].exchange(true);
                                if (!was_set) {
                                    interface_nodes.local().emplace_back(tail);
                                }
                            }
                        }
                        // processed_edges[tail_idx].push_back(RankEncodedEdge{u, v});
                        if (u.rank() != rank_) {
                            continue;
                        }
                        auto v_neighbors = G.out_adj(v).neighbors();
                        auto u_neighbors = G.out_adj(u).neighbors();
                        // auto offset      = edge_id - tail_first_edge;
                        size_t offset = 0;
                        if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                            offset = edge_id - tail_first_edge;
                        }
                        // atomic_debug(
                        //     fmt::format("[t{}] intersecting {} {}",
                        //     tbb::this_task_arena::current_thread_index(), v, u)
                        // );
                        // for each edge (v, u) we check the open wedge (v, u,
                        // w) for w in N(u)+
                        stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                        stats.local.intersection_size_local +=
                            v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                        cetric::intersection(
                            v_neighbors.begin() + offset,
                            v_neighbors.end(),
                            u_neighbors.begin(),
                            u_neighbors.end(),
                            [&](RankEncodedNodeId w) {
                                stats.local.local_triangles++;
                                emit(Triangle<RankEncodedNodeId>{v, u, w});
                            },
                            node_ordering,
                            conf_
                        );
                    }
                finish:
                    return;
                });
            }
            std::for_each(threads.begin(), threads.end(), [](auto& t) { t.join(); });
        } else {
            tbb::task_arena            arena(conf_.num_threads, 0);
            tbb::blocked_range<EdgeId> edge_ids(EdgeId{0}, G.local_edge_count(), conf_.grainsize);

            std::vector<std::atomic<bool>> is_interface;
            constexpr bool fast_is_interface = std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                                               || std::is_same_v<NodeOrdering, node_ordering::id>
                                               || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
            if constexpr (!fast_is_interface) {
                is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
            }
            auto body = [&](auto const& edge_id_range) {
                auto tail_idx        = edge_locator.edge_tail_idx(edge_id_range.begin());
                auto tail            = node_indexer.get_node(tail_idx);
                auto tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                auto tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                // atomic_debug(fmt::format(
                //     "[t{}] processing edges {} to {}",
                //     tbb::this_task_arena::current_thread_index(),
                //     std::max(edge_id_range.begin(), tail_first_edge),
                //     edge_id_range.end()
                // ));
                for (auto edge_id = std::max(edge_id_range.begin(), tail_first_edge); edge_id < edge_id_range.end();
                     edge_id++) {
                    while (edge_id >= tail_last_edge || (conf_.pseudo2core && G.outdegree(tail) < 2)) {
                        if (conf_.pseudo2core && G.outdegree(tail) < 2) {
                            skipped_nodes++;
                        }
                        tail_idx++;
                        tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                        edge_id         = tail_first_edge;
                        if (edge_id >= edge_id_range.end()) {
                            goto finish;
                        }
                        tail           = node_indexer.get_node(tail_idx);
                        tail_last_edge = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    }
                    KASSERT(tail_first_edge <= edge_id);
                    KASSERT(edge_id < tail_last_edge);
                    auto v = tail;
                    auto u = edge_locator.get_edge_head(edge_id);
                    if constexpr (fast_is_interface) {
                        if (tail.rank() == rank_ && edge_id == tail_first_edge) {
                            // we are the first thread to examine this node
                            // we check if it an interface node
                            if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(tail)) {
                                interface_nodes.local().emplace_back(tail);
                            }
                        }
                    } else {
                        if (u.rank() != rank_) {
                            bool was_set = is_interface[tail_idx].exchange(true);
                            if (!was_set) {
                                interface_nodes.local().emplace_back(tail);
                            }
                        }
                    }
                    // processed_edges[tail_idx].push_back(RankEncodedEdge{u,
                    // v});
                    if (u.rank() != rank_) {
                        continue;
                    }
                    auto   v_neighbors = G.out_adj(v).neighbors();
                    auto   u_neighbors = G.out_adj(u).neighbors();
                    size_t offset      = 0;
                    if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                        offset = edge_id - tail_first_edge;
                    }
                    // auto offset      = edge_id - tail_first_edge;
                    // atomic_debug(
                    //     fmt::format("[t{}] intersecting {} {}",
                    //     tbb::this_task_arena::current_thread_index(), v, u)
                    // );
                    // for each edge (v, u) we check the open wedge (v, u, w)
                    // for w in N(u)+
                    stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                    stats.local.intersection_size_local +=
                        v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                    cetric::intersection(
                        v_neighbors.begin() + offset,
                        v_neighbors.end(),
                        u_neighbors.begin(),
                        u_neighbors.end(),
                        [&](RankEncodedNodeId w) {
                            stats.local.local_triangles++;
                            emit(Triangle<RankEncodedNodeId>{v, u, w});
                        },
                        node_ordering,
                        conf_
                    );
                }
            finish:
                return;
            };

            switch (conf_.tbb_partitioner) {
                case TBBPartitioner::stat:
                    arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::static_partitioner{}); });
                    break;
                case TBBPartitioner::simple:
                    arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::simple_partitioner{}); });
                    break;
                case TBBPartitioner::standard:
                    arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::auto_partitioner{}); });
                    break;
                case TBBPartitioner::affinity:
                    tbb::affinity_partitioner partitioner;
                    arena.execute([&] { tbb::parallel_for(edge_ids, body, partitioner); });
                    break;
            }
        }
        // for (size_t i = 0; i < edges.size(); ++i) {
        //     KASSERT(edges[i].size() == processed_edges[i].size(), "Failed for " << i);
        // }

        stats.local.local_phase_time += phase_time.elapsed_time();
        stats.local.skipped_nodes += skipped_nodes;
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel(
        TriangleFunc                                                     emit,
        cetric::profiling::Statistics&                                   stats,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        auto                    nodes = G.local_nodes();
        tbb::combinable<size_t> local_triangles_stats{0};

        auto partition_neighborhood_2d = [this](RankEncodedNodeId node [[maybe_unused]]) {
            if (conf_.parallelization_method == ParallelizationMethod::omp_for) {
                return false;
            }
            if (conf_.local_degree_of_parallelism > 2) {
                return true;
            } else if (conf_.local_degree_of_parallelism > 1) {
                return G.outdegree(node) > high_degree_threshold_;
            } else {
                return false;
            }
        };

        auto handle_neighbor_range = [&node_ordering, this, &emit, &stats, &local_triangles_stats](
                                         RankEncodedNodeId                 v,
                                         tbb::blocked_range<size_t> const& neighbor_range,
                                         bool                              spawn [[maybe_unused]] = false
                                     ) {
            for (size_t neighbor_index = neighbor_range.begin(); neighbor_index != neighbor_range.end();
                 neighbor_index++) {
                auto u = *(G.out_adj(v).neighbors().begin() + neighbor_index);
                KASSERT(*(G.out_adj(v).neighbors().begin() + neighbor_index) == u);
                if (u.rank() != rank_) {
                    if constexpr (!std::is_same_v<NodeOrdering, node_ordering::id_outward> && !std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>) {
                        continue;
                    }
                    // if (conf_.algorithm == Algorithm::Patric) {
                    //     // neighbor_index++;
                    // }
                    // atomic_debug(
                    //     fmt::format("Remaining neighbors {}",
                    //                 boost::make_iterator_range(G.out_adj(v).neighbors().begin() +
                    //                 neighbor_index,
                    //                                            G.out_adj(v).neighbors().end())));
                    // TODO: we should be able to break here when the edges are properly sorted
                    KASSERT(std::all_of(
                        G.out_adj(v).neighbors().begin() + neighbor_index,
                        G.out_adj(v).neighbors().end(),
                        [rank = this->rank_](RankEncodedNodeId node) { return node.rank() != rank; }
                    ));
                    break;
                }
                // #pragma omp task if (spawn)
                {
                    auto   v_neighbors = G.out_adj(v).neighbors();
                    auto   u_neighbors = G.out_adj(u).neighbors();
                    size_t offset      = 0;
                    if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                        offset = neighbor_index;
                    }
                    // for each edge (v, u) we check the open wedge (v, u, w)
                    // for w in N(u)+
                    stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                    stats.local.intersection_size_local +=
                        v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                    cetric::intersection(
                        v_neighbors.begin() + offset,
                        v_neighbors.end(),
                        u_neighbors.begin(),
                        u_neighbors.end(),
                        [&](RankEncodedNodeId w) {
                            local_triangles_stats.local()++;
                            // stats.local.local_triangles++;
                            emit(Triangle<RankEncodedNodeId>{v, u, w});
                        },
                        node_ordering,
                        conf_
                    );
                }
            }
        };
        std::atomic<size_t> skipped_nodes = 0;
        // tbb::parallel_for(
        //     tbb::blocked_range(nodes.begin(), nodes.end()),
        //     [this,
        //      &interface_nodes,
        //      &node_ordering,
        //      &emit,
        //      &stats,
        //      handle_neighbor_range,
        //      partition_neighborhood_2d,
        //      &skipped_nodes](auto const& nodes_r) {
        // #define CETRIC_OMP_TASK_PARALLEL 1
        // #if !defined(CETRIC_OMP_TASK_PARALLEL)
        //     #pragma omp for schedule(runtime) i
        // #else
        //     #pragma omp          single
        //     #pragma omp taskloop grainsize(conf_.grainsize)
        // #endif
        auto node_loop_body = [&](RankEncodedNodeId const& v) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                skipped_nodes++;
                // continue;
            } else {
                if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                    interface_nodes.local().emplace_back(v);
                }
                auto v_neighbors_size = std::distance(G.out_adj(v).neighbors().begin(), G.out_adj(v).neighbors().end());
                if (conf_.parallelization_method == ParallelizationMethod::tbb && partition_neighborhood_2d(v)) {
                    stats.local.nodes_parallel2d++;
                    tbb::parallel_for(
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        [&handle_neighbor_range, v = v](auto const& neighbor_range) {
                            handle_neighbor_range(v, neighbor_range);
                        }
                    );
                } else {
                    if (partition_neighborhood_2d(v)) {
                        stats.local.nodes_parallel2d++;
                    }
                    handle_neighbor_range(
                        v,
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        partition_neighborhood_2d(v)
                    );
                }
            }
        };
        switch (conf_.parallelization_method) {
            case ParallelizationMethod::tbb: {
                tbb::task_arena arena(conf_.num_threads, 0);
                arena.execute([&] {
                    tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
                        for (auto v: r) {
                            node_loop_body(v);
                        }
                    });
                });
                break;
            }
            case ParallelizationMethod::omp_for: {
// clang-format off
                #pragma omp parallel for schedule(runtime)
                // clang-format on
                for (auto v: nodes) {
                    node_loop_body(v);
                }
                break;
            }
            case ParallelizationMethod::omp_task: {
                throw "currently unsupported";
                //                 int grainsize = conf_.grainsize;
                // #pragma omp parallel
                //                 {
                //                     // clang-format off
                //                 #pragma omp          single
                //                 #pragma omp taskloop grainsize(1)
                //                     // clang-format on
                //                     for (auto v: nodes) {
                //                         node_loop_body(v);
                //                     }
                //                 }
                //                 break;
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
        stats.local.skipped_nodes += skipped_nodes;
        stats.local.local_triangles += local_triangles_stats.combine(std::plus<>{});
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_with_ghosts(
        TriangleFunc                                                     emit,
        cetric::profiling::Statistics&                                   stats,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();

        auto partition_neighborhood_2d = [this](RankEncodedNodeId node [[maybe_unused]]) {
            if (conf_.parallelization_method == ParallelizationMethod::omp_for) {
                return false;
            }
            if (conf_.local_degree_of_parallelism > 2) {
                return true;
            } else if (conf_.local_degree_of_parallelism > 1) {
                if (node.rank() == rank_) {
                    return G.outdegree(node) > high_degree_threshold_;
                } else {
                    return G.ghost_degree(node) > high_degree_threshold_;
                }
            } else {
                return false;
            }
        };
        std::vector<std::atomic<bool>> is_interface;
        constexpr bool fast_is_interface = std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                                           || std::is_same_v<NodeOrdering, node_ordering::id>
                                           || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
        if constexpr (!fast_is_interface) {
            is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
        }

        auto handle_neighbor_range = [&](RankEncodedNodeId                 v,
                                         tbb::blocked_range<size_t> const& neighbor_range,
                                         bool                              spawn [[maybe_unused]] = false) {
            for (size_t neighbor_index = neighbor_range.begin(); neighbor_index != neighbor_range.end();
                 neighbor_index++) {
                EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable>::neighbor_range_type v_neighbors;
                if (v.rank() != rank_) {
                    v_neighbors = G.ghost_adj(v).neighbors();
                } else {
                    v_neighbors = G.out_adj(v).neighbors();
                }
                auto u = *(v_neighbors.begin() + neighbor_index);
                KASSERT(*(v_neighbors.begin() + neighbor_index) == u);
                EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable>::neighbor_range_type u_neighbors;
                if (u.rank() != rank_) {
                    if constexpr (!fast_is_interface) {
                        bool was_set = is_interface[G.to_local_idx(v)].exchange(true);
                        if (!was_set) {
                            interface_nodes.local().emplace_back(v);
                        }
                    }
                    u_neighbors = G.ghost_adj(u).neighbors();
                } else {
                    u_neighbors = G.out_adj(u).neighbors();
                }
                size_t offset = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = neighbor_index;
                }
                // for each edge (v, u) we check the open wedge (v, u, w)
                // for w in N(u)+
                stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
                stats.local.intersection_size_local +=
                    v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
            }
        };
        std::atomic<size_t> skipped_nodes  = 0;
        auto                node_loop_body = [&](RankEncodedNodeId const& v) {
            Degree v_degree;
            if (v.rank() == rank_) {
                v_degree = G.outdegree(v);
            } else {
                v_degree = G.ghost_degree(v);
            }
            if (conf_.pseudo2core && v_degree < 2) {
                skipped_nodes++;
                // continue;
            } else {
                if constexpr (fast_is_interface) {
                    if (v.rank() == rank_ && G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                        interface_nodes.local().emplace_back(v);
                    }
                }
                size_t v_neighbors_size;
                if (v.rank() == rank_) {
                    v_neighbors_size = v_degree;
                } else {
                    v_neighbors_size = v_degree;
                }
                if (conf_.parallelization_method == ParallelizationMethod::tbb && partition_neighborhood_2d(v)) {
                    stats.local.nodes_parallel2d++;
                    tbb::parallel_for(
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        [&handle_neighbor_range, v = v](auto const& neighbor_range) {
                            handle_neighbor_range(v, neighbor_range);
                        }
                    );
                } else {
                    if (partition_neighborhood_2d(v)) {
                        stats.local.nodes_parallel2d++;
                    }
                    handle_neighbor_range(
                        v,
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        partition_neighborhood_2d(v)
                    );
                }
            }
        };
        auto nodes = boost::join(G.local_nodes(), G.ghosts());
        switch (conf_.parallelization_method) {
            case ParallelizationMethod::tbb: {
                tbb::task_arena arena(conf_.num_threads, 0);
                arena.execute([&] {
                    tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
                        for (auto v: r) {
                            node_loop_body(v);
                        }
                    });
                });
                break;
            }
            case ParallelizationMethod::omp_for: {
// clang-format off
                #pragma omp parallel for schedule(runtime)
                // clang-format on
                for (auto v: nodes) {
                    node_loop_body(v);
                }
                break;
            }
            case ParallelizationMethod::omp_task: {
                throw "currently unsupported";
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
        stats.local.skipped_nodes += skipped_nodes;
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_distributed(TriangleFunc emit, cetric::profiling::Statistics& stats, NodeOrdering&& node_ordering) {
        std::set<RankEncodedNodeId> ghosts;
        run_distributed(emit, stats, std::forward<NodeOrdering>(node_ordering), ghosts);
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering>
    inline void run_distributed(
        TriangleFunc                   emit,
        cetric::profiling::Statistics& stats,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_ndoes_end,
        NodeOrdering&&                 node_ordering
    ) {
        std::set<RankEncodedNodeId> ghosts;
        run_distributed(
            emit,
            stats,
            interface_nodes_begin,
            interface_ndoes_end,
            std::forward<NodeOrdering>(node_ordering),
            ghosts
        );
    }

    template <typename TriangleFunc, typename NodeOrdering, typename GhostSet>
    inline void run_distributed(
        TriangleFunc emit, cetric::profiling::Statistics& stats, NodeOrdering&& node_ordering, GhostSet const& ghosts
    ) {
        auto all_interface_nodes = tbb::flatten2d(interface_nodes_);
        run_distributed(
            emit,
            stats,
            all_interface_nodes.begin(),
            all_interface_nodes.end(),
            std::forward<NodeOrdering>(node_ordering),
            ghosts
        );
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering, typename GhostSet>
    inline void run_distributed(
        TriangleFunc                   emit,
        cetric::profiling::Statistics& stats,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_nodes_end,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        if (conf_.global_parallel && conf_.num_threads > 1) {
            run_distributed_parallel(
                emit,
                stats,
                interface_nodes_begin,
                interface_nodes_end,
                std::forward<NodeOrdering>(node_ordering),
                ghosts
            );
        } else {
            run_distributed_sequential(
                emit,
                stats,
                interface_nodes_begin,
                interface_nodes_end,
                std::forward<NodeOrdering>(node_ordering),
                ghosts
            );
        }
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering, typename GhostSet>
    inline void run_distributed_sequential(
        TriangleFunc                   emit,
        cetric::profiling::Statistics& stats,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_nodes_end,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        auto base_queue = message_queue::make_buffered_queue<RankEncodedNodeId>(Merger{}, Splitter{});
        auto queue      = [&]() {
            if constexpr (std::is_same_v<CommunicationPolicy, cetric::GridPolicy>) {
                return IndirectMessageQueueAdaptor(base_queue);
            } else {
                return DirectMessageQueueAdaptor(base_queue);
            }
        }();
        queue.set_threshold(threshold_);
        cetric::profiling::Timer phase_time;
        for (auto current = interface_nodes_begin; current != interface_nodes_end; current++) {
            RankEncodedNodeId v = *current;
            // iterate over neighborhood and delegate to other PEs if necessary
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                stats.local.skipped_nodes++;
            } else {
                for (RankEncodedEdge edge: G.out_adj(v).edges()) {
                    RankEncodedNodeId u = edge.head;
                    if (/*conf_.algorithm == Algorithm::Cetric || */ u.rank() != rank_) {
                        // assert(G.is_local_from_local(v));
                        // assert(G.is_local(G.to_global_id(v)));
                        // assert(!G.is_local(G.to_global_id(u)));
                        // assert(!G.is_local_from_local(u));
                        PEID u_rank = u.rank();
                        assert(u_rank != rank_);
                        if (last_proc_[G.to_local_idx(v)] != u_rank) {
                            enqueue_for_sending(queue, v, u_rank, emit, stats);
                        }
                    }
                }
            }
            // timer.start("Communication");
            queue.poll([&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
                handle_buffer(span.begin(), span.end(), emit, stats, node_ordering, ghosts);
            });
        }
        queue.terminate([&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
            handle_buffer(span.begin(), span.end(), emit, stats, node_ordering, ghosts);
        });
        stats.local.global_phase_time += phase_time.elapsed_time();
        stats.local.message_statistics.add(queue.stats());
        queue.reset();
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering, typename GhostSet>
    inline void run_distributed_parallel(
        TriangleFunc                   emit,
        cetric::profiling::Statistics& stats,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_nodes_end,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        KASSERT(conf_.num_threads > 1ul);
        std::thread::id master_thread_id = std::this_thread::get_id();
        auto            base_queue       = cetric::make_concurrent_buffered_queue<RankEncodedNodeId>(
            conf_.num_threads - 1,
            master_thread_id,
            Merger{},
            Splitter{}
        );
        auto queue = [&]() {
            if constexpr (std::is_same_v<CommunicationPolicy, cetric::GridPolicy>) {
                return IndirectMessageQueueAdaptor(base_queue);
            } else {
                return DirectMessageQueueAdaptor(base_queue);
            }
        }();
        queue.set_threshold(threshold_);
        cetric::profiling::Timer phase_time;
        std::atomic<size_t>      nodes_queued = 0;
        std::atomic<size_t>      write_jobs   = 0;
        ThreadPool               pool(conf_.num_threads - 1);

        std::atomic<size_t> skipped_nodes = 0;
        for (auto current = interface_nodes_begin; current != interface_nodes_end; current++) {
            RankEncodedNodeId v = *current;
            nodes_queued++;
            auto task = [v, this, &stats, emit, &queue, &pool, &nodes_queued, &write_jobs, &skipped_nodes]() {
                if (conf_.pseudo2core && G.outdegree(v) < 2) {
                    skipped_nodes++;
                    nodes_queued--;
                    return;
                }
                for (RankEncodedEdge edge: G.out_adj(v).edges()) {
                    RankEncodedNodeId u = edge.head;
                    if (/*conf_.algorithm == Algorithm::Cetric || */ u.rank() != rank_) {
                        // assert(G.is_local_from_local(v));
                        // assert(G.is_local(G.to_global_id(v)));
                        // assert(!G.is_local(G.to_global_id(u)));
                        // assert(!G.is_local_from_local(u));
                        PEID u_rank = u.rank();
                        assert(u_rank != rank_);
                        if (last_proc_[G.to_local_idx(v)] != u_rank) {
                            // atomic_debug(fmt::format("Send N({}) to {}",
                            // G.to_global_id(v), u_rank));
                            enqueue_for_sending_async(queue, v, u_rank, pool, write_jobs, emit, stats);
                        }
                    }
                }
                nodes_queued--;
            };
            if (conf_.global_degree_of_parallelism > 1) {
                pool.enqueue(task);
            } else {
                task();
            }
        }
        while (true) {
            if (write_jobs == 0 && nodes_queued == 0) {
                // atomic_debug(fmt::format("No more polling, enqueued: {}, done: {}", pool.enqueued(), pool.done()));
                break;
            }
            queue.check_for_overflow_and_flush();
            queue.poll([&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
                handle_buffer_hybrid(std::move(span), pool, emit, stats, node_ordering, ghosts);
            });
        }
        queue.terminate([&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
            handle_buffer_hybrid(std::move(span), pool, emit, stats, node_ordering, ghosts);
        });
        pool.loop_until_empty();
        pool.terminate();
        // atomic_debug(fmt::format("Finished Pool, enqueued: {}, done: {}", pool.enqueued(), pool.done()));
        stats.local.global_phase_time += phase_time.elapsed_time();
        stats.local.message_statistics.add(queue.stats());
        stats.local.skipped_nodes += skipped_nodes;
        queue.reset();
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run(TriangleFunc emit, cetric::profiling::Statistics& stats, NodeOrdering&& node_ordering) {
        std::set<RankEncodedNodeId> ghosts;
        run(emit, stats, std::forward<NodeOrdering>(node_ordering), ghosts);
    }
    template <typename TriangleFunc, typename NodeOrdering, typename GhostSet>
    inline void
    run(TriangleFunc emit, cetric::profiling::Statistics& stats, NodeOrdering&& node_ordering, GhostSet const& ghosts) {
        run_local(emit, stats, node_ordering);
        cetric::profiling::Timer phase_time;
        for (auto node: G.local_nodes()) {
            G.remove_internal_edges(node);
        }
        stats.local.contraction_time += phase_time.elapsed_time();
        run_distributed(emit, stats, node_ordering, ghosts);
    }

    inline size_t get_triangle_count(cetric::profiling::Statistics& stats) {
        size_t triangle_count = 0;
        run([&](auto) { triangle_count++; }, stats);
        return triangle_count;
    }

private:
    void pre_intersection(RankEncodedNodeId v) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(v, [&](RankEncodedEdge edge) {
                assert(is_v_neighbor_[edge.head.id()] == false);
                is_v_neighbor_[edge.head.id()] = true;
            });
        }
    }

    void post_intersection(RankEncodedNodeId v) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(v, [&](RankEncodedEdge edge) {
                assert(is_v_neighbor_[edge.head.id()] == true);
                is_v_neighbor_[edge.head.id()] = false;
            });
        }
    }

    template <typename IntersectFunc>
    void intersect(RankEncodedNodeId v, RankEncodedNodeId u, IntersectFunc on_intersection) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(u, [&](RankEncodedEdge uw) {
                NodeId w = uw.head.id();
                if (is_v_neighbor_[w]) {
                    on_intersection(w);
                }
            });
        } else {
            G.intersect_neighborhoods(v, u, on_intersection);
        }
    }

    bool send_neighbor(RankEncodedNodeId u, PEID rank) {
        if (conf_.skip_local_neighborhood) {
            if (conf_.algorithm == Algorithm::Cetric) {
                // we omit all vertices located on the receiving PE (all vertices are ghosts)
                return u.rank() != rank;
            } else {
                // we send all vertices, but omit those already located on the receiving PE
                return u.rank() == rank_ || u.rank() != rank;
            }
        } else {
            return true;
        }
    }

    template <typename MessageQueue, typename TriangleFunc>
    void enqueue_for_sending(
        MessageQueue&                  queue,
        RankEncodedNodeId              v,
        PEID                           u_rank,
        TriangleFunc                   emit [[maybe_unused]],
        cetric::profiling::Statistics& stats [[maybe_unused]]
    ) {
        std::vector<RankEncodedNodeId> buffer;
        buffer.emplace_back(v);
        // size_t send_count = 0;
        for (RankEncodedEdge e: G.out_adj(v).edges()) {
            assert(conf_.algorithm == Algorithm::Patric || e.head.rank() != rank_);
            // using payload_type = typename GraphType::payload_type;
            // if constexpr (std::is_convertible_v<decltype(payload_type{}.degree), Degree>) {
            //     if (conf_.degree_filtering) {
            //         const auto& ghost_data = G.get_ghost_data(e.head.id());
            //         if (ghost_data.payload.degree < pe_min_degree[u_rank]) {
            //             return;
            //         }
            //     }
            // }
            if (send_neighbor(e.head, u_rank)) {
                buffer.emplace_back(e.head);
            }
        }
        // atomic_debug(fmt::format("Sending {} to {}", buffer, u_rank));
        queue.post_message(std::move(buffer), u_rank);
        last_proc_[G.to_local_idx(v)] = u_rank;
    }

    template <typename MessageQueue, typename TriangleFunc>
    void enqueue_for_sending_async(
        MessageQueue&                  queue,
        RankEncodedNodeId              v,
        PEID                           u_rank,
        ThreadPool&                    thread_pool [[maybe_unused]],
        std::atomic<size_t>&           write_jobs,
        TriangleFunc                   emit [[maybe_unused]],
        cetric::profiling::Statistics& stats [[maybe_unused]]
    ) {
        // atomic_debug(fmt::format("rank {}", u_rank));
        // atomic_debug(u_rank);
        write_jobs++;
        thread_pool.enqueue([&queue, &write_jobs, this, v, u_rank] {
            // tg.run([&stats, emit, this, v, u_rank]() {
            std::vector<RankEncodedNodeId> buffer;
            buffer.emplace_back(v);
            // size_t send_count = 0;
            // atomic_debug(u_rank);
            for (RankEncodedNodeId node: G.out_adj(v).neighbors()) {
                assert(conf_.algorithm == Algorithm::Patric || node.rank() != rank_);
                if (send_neighbor(node, u_rank)) {
                    buffer.emplace_back(node);
                }
            }
            // atomic_debug(u_rank);
            queue.post_message(std::move(buffer), u_rank);
            write_jobs--;
        });
        last_proc_[G.to_local_idx(v)] = u_rank;
    }

    template <typename IterType, typename TriangleFunc, typename NodeOrdering, typename GhostSet>
    void handle_buffer(
        IterType                       begin,
        IterType                       end,
        TriangleFunc                   emit,
        cetric::profiling::Statistics& stats,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        RankEncodedNodeId v = *begin;
        process_neighborhood(v, begin + 1, end, emit, stats, std::forward<NodeOrdering>(node_ordering), ghosts);
    }

    template <typename TriangleFunc, typename NodeOrdering, typename GhostSet>
    void handle_buffer_hybrid(
        SharedVectorSpan<RankEncodedNodeId> span,
        ThreadPool&                         thread_pool,
        TriangleFunc                        emit,
        cetric::profiling::Statistics&      stats,
        NodeOrdering&&                      node_ordering,
        GhostSet const&                     ghosts
    ) {
        thread_pool.enqueue(
            [this, emit, span = std::move(span), &stats, &node_ordering, &ghosts] {
                RankEncodedNodeId v = *span.begin();
                // atomic_debug(
                //     fmt::format("Spawn task for {} on thread {}", v,
                //     tbb::this_task_arena::current_thread_index()));
                process_neighborhood(
                    v,
                    span.begin() + 1,
                    span.end(),
                    emit,
                    stats,
                    std::forward<NodeOrdering>(node_ordering),
                    ghosts
                );
            },
            ThreadPool::Priority::high
        );
    }
    // template <typename NodeBufferIter>
    // void distributed_pre_intersect(RankEncodedNodeId v, NodeBufferIter begin, NodeBufferIter end) {
    //     if (conf_.flag_intersection) {
    //         for (auto it = begin; it != end; it++) {
    //             RankEncodedNodeId node = *it;
    //             if (G.is_local(node) || G.is_ghost_from_global(node)) {
    //                 is_v_neighbor_[G.to_local_id(node)] = true;
    //             }
    //         }
    //         // for CETRIC we don't have to consider the local neighbors of v, because these are no ghost
    //         // vertices
    //         if (conf_.algorithm == Algorithm::Patric && conf_.skip_local_neighborhood) {
    //             G.for_each_local_out_edge(G.to_local_id(v),
    //                                       [&](RankEncodedEdge edge) { is_v_neighbor_[edge.head.id()] = true; });
    //         }
    //     }
    // }

    // template <typename NodeBufferIter>
    // void distributed_post_intersect(RankEncodedNodeId v, NodeBufferIter begin, NodeBufferIter end) {
    //     if (conf_.flag_intersection) {
    //         for (auto it = begin; it != end; it++) {
    //             RankEncodedNodeId node = *it;
    //             if (G.is_local(node) || G.is_ghost_from_global(node)) {
    //                 is_v_neighbor_[G.to_local_id(node)] = false;
    //             }
    //         }
    //         // for CETRIC we don't have to consider the local neighbors of v, because these are no ghost
    //         // vertices
    //         if (conf_.algorithm == Algorithm::Patric && conf_.skip_local_neighborhood) {
    //             G.for_each_local_out_edge(G.to_local_id(v),
    //                                       [&](RankEncodedEdge edge) { is_v_neighbor_[edge.head.id()] = false; });
    //         }
    //     }
    // }

    template <typename IntersectFunc, typename NodeBufferIter, typename NodeOrdering, typename GhostSet>
    void intersect_from_message(
        RankEncodedNodeId              u,
        RankEncodedNodeId              v,
        NodeBufferIter                 begin,
        NodeBufferIter                 end,
        IntersectFunc                  on_intersection,
        cetric::profiling::Statistics& stats,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        // if (conf_.flag_intersection) {
        //     G.for_each_local_out_edge(u_local, [&](RankEncodedEdge uw) {
        //         RankEncodedNodeId w = uw.head.id();
        //         if (is_v_neighbor_[w]) {
        //             on_intersection(w);
        //         }
        //     });
        // } else {
        if (conf_.algorithm == Algorithm::CetricX) {
            // this node may have no out edges and was therefore removed
            if (!G.is_local(u)) {
                return;
            }
        }
        auto u_neighbors = G.out_adj(u).neighbors();
        if (!ghosts.empty()) {
            auto filtered_neighbors_it =
                boost::adaptors::filter(boost::make_iterator_range(begin, end), [&ghosts](RankEncodedNodeId node) {
                    return ghosts.find(node) != ghosts.end();
                });
            std::vector<RankEncodedNodeId> filtered_neighbors(
                filtered_neighbors_it.begin(),
                filtered_neighbors_it.end()
            );
            // atomic_debug(fmt::format("intersecting N({})={} and N({}){}", u, u_neighbors, v, filtered_neighbors));

            // for each edge (v, u) we check the open wedge
            //  (v, u, w) for w in N(u)+
            stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
            stats.local.intersection_size_global +=
                filtered_neighbors.end() - filtered_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
            cetric::intersection(
                u_neighbors.begin(),
                u_neighbors.end(),
                filtered_neighbors.begin(),
                filtered_neighbors.end(),
                on_intersection,
                node_ordering,
                conf_
            );
        } else {
            // for each edge (v, u) we check the open wedge (v, u, w) for w in
            // N(u)+
            stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();

            stats.local.intersection_size_global += end - begin + u_neighbors.end() - u_neighbors.begin();
            cetric::intersection(
                u_neighbors.begin(),
                u_neighbors.end(),
                begin,
                end,
                on_intersection,
                node_ordering,
                conf_
            );
        }
        if (conf_.skip_local_neighborhood && conf_.algorithm == Algorithm::Patric) {
            auto v_neighbors = G.out_adj(v).neighbors();
            // for each edge (v, u) we check the open wedge (v, u, w) for w in
            // N(u)+
            stats.local.wedge_checks += u_neighbors.end() - u_neighbors.begin();
            stats.local.intersection_size_global +=
                v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
            cetric::intersection(
                u_neighbors.begin(),
                u_neighbors.end(),
                v_neighbors.begin(),
                v_neighbors.end(),
                on_intersection,
                node_ordering,
                conf_
            );
        }
        // }
    }

    template <typename TriangleFunc, typename NodeBufferIter, typename NodeOrdering, typename GhostSet>
    void process_neighborhood(
        RankEncodedNodeId              v,
        NodeBufferIter                 begin,
        NodeBufferIter                 end,
        TriangleFunc                   emit,
        cetric::profiling::Statistics& stats,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        std::vector<RankEncodedNodeId> buffer{begin, end};
        // atomic_debug(fmt::format("Handling message {}", buffer));
        assert(v.rank() != rank_);
        // distributed_pre_intersect(v, begin, end);
        auto for_each_local_receiver = [&](auto on_node) {
            if (conf_.skip_local_neighborhood) {
                auto neighbors = G.out_adj(v).neighbors();
                if (false) {
                    tbb::parallel_for(
                        tbb::blocked_range(neighbors.begin(), neighbors.end()),
                        [&on_node](auto const& r) {
                            for (auto node: r) {
                                on_node(node);
                            }
                        }
                    );
                } else {
                    for (auto node: G.out_adj(v).neighbors()) {
                        on_node(node);
                    }
                }
            } else {
                if (false) {
                    tbb::parallel_for(tbb::blocked_range(begin, end), [&on_node, this](auto const& r) {
                        for (auto u: r) {
                            if (u.rank() == rank_) {
                                on_node(u);
                            }
                        }
                    });
                } else {
                    for (auto current = begin; current != end; current++) {
                        RankEncodedNodeId u = *current;
                        if (u.rank() == rank_) {
                            on_node(u);
                        }
                    }
                }
            }
        };
        for_each_local_receiver([&](RankEncodedNodeId u) {
            assert(u.rank() == rank_);
            intersect_from_message(
                u,
                v,
                begin,
                end,
                [&](RankEncodedNodeId local_intersection) {
                    emit(Triangle<RankEncodedNodeId>{v, u, local_intersection});
                    stats.local.type3_triangles++;
                },
                stats,
                node_ordering,
                ghosts
            );
        });
        // distributed_post_intersect(v, begin, end);
    }
    GraphType&                                                      G;
    const Config&                                                   conf_;
    PEID                                                            rank_;
    PEID                                                            size_;
    std::vector<PEID>                                               last_proc_;
    std::vector<bool>                                               is_v_neighbor_;
    tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>> interface_nodes_;
    std::vector<Degree>                                             pe_min_degree;
    size_t                                                          threshold_;
    size_t                                                          high_degree_threshold_;
}; // namespace cetric

} // namespace cetric
#endif // PARALLEL_TRIANGLE_COUNTER_PARALLEL_NODE_ITERATOR_H
