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
// Created by Tim Niklas Uhl on 22.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H
#define PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include <ranges>
#include <message-queue/buffered_queue.h>
#include <mpi.h>
#ifdef CETRIC_HAS_OPENMP
    #include <omp.h>
#endif
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_queue.h>
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

#include <spdlog/spdlog.h>
#include "cetric/caliper.h"
#include "cetric/concurrent_buffered_queue.h"
#include "cetric/config.h"
#include "cetric/counters/intersection.h"
#include "cetric/datastructures/auxiliary_node_data.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/distributed/helpers.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/datastructures/span.h"
#include "cetric/indirect_message_queue.h"
#include "cetric/task_pool.h"
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
            SPDLOG_DEBUG("grow by {} with {}", msg.size(), msg);
            auto insert_position = buffer.grow_by(msg.size() + 1);
            std::copy(msg.begin(), msg.end(), insert_position);
            *(insert_position + msg.size()) = RankEncodedNodeId::sentinel();
            SPDLOG_DEBUG(
                "done with copy of {} -> {}",
                msg,
                std::ranges::subrange(insert_position, insert_position + msg.size())
            );
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
    run_local(TriangleFunc emit, NodeOrdering&& node_ordering, EdgeOrientationOrdering&&) {
        if (conf_.local_parallel && conf_.num_threads > 1) {
            if (conf_.algorithm == Algorithm::cetric_x) {
                if (conf_.edge_partitioning) {
                    run_local_parallel_edge_partitioned_with_ghosts<EdgeOrientationOrdering>(
                        emit,
                        interface_nodes_,
                        std::forward<NodeOrdering>(node_ordering)
                    );
                } else {
                    run_local_parallel_with_ghosts<EdgeOrientationOrdering>(
                        emit,
                        interface_nodes_,
                        std::forward<NodeOrdering>(node_ordering)
                    );
                }
                return;
            }
            if (conf_.edge_partitioning) {
                run_local_parallel_edge_partitioned<EdgeOrientationOrdering>(
                    emit,
                    interface_nodes_,
                    std::forward<NodeOrdering>(node_ordering)
                );
                return;
            }
            run_local_parallel<EdgeOrientationOrdering>(
                emit,
                interface_nodes_,
                std::forward<NodeOrdering>(node_ordering)
            );
        } else {
            if (conf_.algorithm == Algorithm::cetric_x) {
                run_local_sequential_with_ghosts<EdgeOrientationOrdering>(
                    emit,
                    interface_nodes_.local(),
                    std::forward<NodeOrdering>(node_ordering)
                );
            } else {
                run_local_sequential<EdgeOrientationOrdering>(
                    emit,
                    interface_nodes_.local(),
                    std::forward<NodeOrdering>(node_ordering)
                );
            }
        }
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_sequential(
        TriangleFunc                    emit,
        std::vector<RankEncodedNodeId>& interface_nodes,
        NodeOrdering&&                  node_ordering
    ) {
        interface_nodes.clear();
        size_t skipped_nodes           = 0;
        size_t wedge_checks            = 0;
        size_t intersection_size_local = 0;
        size_t local_triangles         = 0;
        for (auto v: G.local_nodes()) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                skipped_nodes++;
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
                    // if (conf_.algorithm == Algorithm::patric || conf_.algorithm == Algorithm::cetric_x) {
                    //     neighbor_index++;
                    //     continue;
                    // }
                    SPDLOG_DEBUG(
                        "Remaining neighbors {}",
                        std::ranges::subrange(
                            G.out_adj(v).neighbors().begin() + neighbor_index,
                            G.out_adj(v).neighbors().end()
                        )
                    );
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
                wedge_checks += u_neighbors.end() - u_neighbors.begin();
                intersection_size_local +=
                    v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
                neighbor_index++;
            }
        }
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_local", intersection_size_local);
        profiling::counter_add("local_triangles", local_triangles);
    }

    /// End of the local prefix of a neighborhood in the ghost-expanded graph (cetric-x). A ghost's adjacency holds only
    /// local vertices, so when one side of an intersection is a ghost, only the local part of the other side can
    /// match. With `id_outward` the local vertices come first, so the other side is clipped there. This bounds the
    /// merge and keeps the work counters exact. Other orderings do not sort local vertices first, so the range stays
    /// as it is.
    template <typename NodeOrdering, typename It>
    [[nodiscard]] It local_prefix_end(It first, It last) const {
        if constexpr (std::is_same_v<std::remove_cvref_t<NodeOrdering>, node_ordering::id_outward>) {
            return std::partition_point(first, last, [this](RankEncodedNodeId w) { return w.rank() == rank_; });
        } else {
            return last;
        }
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_sequential_with_ghosts(
        TriangleFunc                    emit,
        std::vector<RankEncodedNodeId>& interface_nodes,
        NodeOrdering&&                  node_ordering
    ) {
        interface_nodes.clear();
        size_t skipped_nodes           = 0;
        size_t wedge_checks            = 0;
        size_t intersection_size_local = 0;
        size_t local_triangles         = 0;
        for (auto v: G.local_nodes()) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                skipped_nodes++;
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
                auto v_first = v_neighbors.begin() + offset;
                auto v_last  = v_neighbors.end();
                if (u.rank() != rank_) {
                    v_last = local_prefix_end<NodeOrdering>(v_first, v_last);
                }
                wedge_checks += u_neighbors.end() - u_neighbors.begin();
                intersection_size_local += (v_last - v_first) + (u_neighbors.end() - u_neighbors.begin());
                cetric::intersection(
                    v_first,
                    v_last,
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    [&](RankEncodedNodeId w) {
                        local_triangles++;
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
                skipped_nodes++;
                continue;
            }
            size_t neighbor_index = 0;
            for (RankEncodedNodeId u: G.ghost_adj(v).neighbors()) {
                KASSERT(*(G.ghost_adj(v).neighbors().begin() + neighbor_index) == u);
                auto v_neighbors = G.ghost_adj(v).neighbors();
                auto u_neighbors = G.out_adj(u).neighbors();
                // for each edge (v, u) we check the open wedge (v, u, w) for
                // w in N(u)+; v is a ghost, so only the local part of N(u)+ can match
                auto u_first = u_neighbors.begin();
                auto u_last  = local_prefix_end<NodeOrdering>(u_first, u_neighbors.end());
                wedge_checks += u_last - u_first;
                size_t offset = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = neighbor_index;
                }
                intersection_size_local += (v_neighbors.end() - (v_neighbors.begin() + offset)) + (u_last - u_first);
                cetric::intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_first,
                    u_last,
                    [&](RankEncodedNodeId w) {
                        local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
                neighbor_index++;
            }
        }
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_local", intersection_size_local);
        profiling::counter_add("local_triangles", local_triangles);
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_edge_partitioned_with_ghosts(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        interface_nodes.clear();

        auto edge_locator = G.build_ghost_edge_locator();

        tbb::task_arena            arena(conf_.num_threads);
        tbb::blocked_range<EdgeId> edge_ids(EdgeId{0}, G.local_edge_count_with_ghost_edges(), conf_.grainsize);

        std::vector<std::atomic<bool>> is_interface;
        constexpr bool fast_is_interface = std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                                           || std::is_same_v<NodeOrdering, node_ordering::id>
                                           || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
        if constexpr (!fast_is_interface) {
            is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
        }
        tbb::combinable<size_t> thread_local_skipped_nodes{0};
        tbb::combinable<size_t> thread_local_wedge_checks{0};
        tbb::combinable<size_t> thread_local_intersection_size_local{0};
        tbb::combinable<size_t> thread_local_local_triangles{0};
        auto                    body = [&](auto const& edge_id_range) {
            auto& skipped_nodes           = thread_local_skipped_nodes.local();
            auto& wedge_checks            = thread_local_wedge_checks.local();
            auto& intersection_size_local = thread_local_intersection_size_local.local();
            auto& local_triangles         = thread_local_local_triangles.local();
            auto& local_interface_nodes   = interface_nodes.local();

            auto tail_idx        = edge_locator.edge_tail_idx(edge_id_range.begin());
            auto tail            = edge_locator.get_node(tail_idx);
            auto tail_first_edge = edge_locator.first_edge_id_for_idx(tail_idx);
            auto tail_last_edge  = edge_locator.last_edge_id_for_idx(tail_idx);
            SPDLOG_DEBUG(
                "[t{}] processing edges {} to {}",
                tbb::this_task_arena::current_thread_index(),
                std::max(edge_id_range.begin(), tail_first_edge),
                edge_id_range.end()
            );
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
                            local_interface_nodes.emplace_back(tail);
                        }
                    }
                } else {
                    if (u.rank() != rank_) {
                        bool was_set = is_interface[tail_idx].exchange(true);
                        if (!was_set) {
                            local_interface_nodes.emplace_back(tail);
                        }
                    }
                }
                // processed_edges[tail_idx].push_back(RankEncodedEdge{u, v});
                auto   v_neighbors = v.rank() == rank_ ? G.out_adj(v).neighbors() : G.ghost_adj(v).neighbors();
                auto   u_neighbors = u.rank() == rank_ ? G.out_adj(u).neighbors() : G.ghost_adj(u).neighbors();
                size_t offset = 0;
                if constexpr (std::is_same_v<NodeOrdering, EdgeOrientationOrdering>) {
                    offset = edge_id - tail_first_edge;
                }
                // auto offset      = edge_id - tail_first_edge;
                SPDLOG_DEBUG("[t{}] intersecting {} {}", tbb::this_task_arena::current_thread_index(), v, u);
                // for each edge (v, u) we check the open wedge (v, u, w)
                // for w in N(u)+; if either endpoint is a ghost, only local vertices can close it
                auto v_first = v_neighbors.begin() + offset;
                auto v_last  = v_neighbors.end();
                auto u_first = u_neighbors.begin();
                auto u_last  = u_neighbors.end();
                if (u.rank() != rank_) {
                    v_last = local_prefix_end<NodeOrdering>(v_first, v_last);
                } else if (v.rank() != rank_) {
                    u_last = local_prefix_end<NodeOrdering>(u_first, u_last);
                }
                wedge_checks += u_last - u_first;
                intersection_size_local += (v_last - v_first) + (u_last - u_first);
                cetric::intersection(
                    v_first,
                    v_last,
                    u_first,
                    u_last,
                    [&](RankEncodedNodeId w) {
                        local_triangles++;
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

        profiling::counter_add("skipped_nodes", thread_local_skipped_nodes.combine(std::plus<>{}));
        profiling::counter_add("wedge_checks", thread_local_wedge_checks.combine(std::plus<>{}));
        profiling::counter_add("intersection_size_local", thread_local_intersection_size_local.combine(std::plus<>{}));
        profiling::counter_add("local_triangles", thread_local_local_triangles.combine(std::plus<>{}));
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_edge_partitioned(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        interface_nodes.clear();

        auto        edge_locator = G.build_edge_locator();
        auto const& node_indexer = G.node_indexer();

        if (conf_.edge_partitioning_static) {
            std::vector<std::thread> threads(conf_.num_threads);
            tlx::ThreadBarrierMutex  barrier(conf_.num_threads);
            std::vector<size_t>      thread_local_cost(conf_.num_threads + 1);
            size_t                   per_thread_cost;
            auto                edges_per_thread = (G.local_edge_count() + conf_.num_threads - 1) / conf_.num_threads;
            std::vector<EdgeId> first_edge(conf_.num_threads + 1);

            std::vector<size_t> thread_local_skipped_nodes(conf_.num_threads);
            std::vector<size_t> thread_local_wedge_checks(conf_.num_threads);
            std::vector<size_t> thread_local_intersection_size_local(conf_.num_threads);
            std::vector<size_t> thread_local_local_triangles(conf_.num_threads);

            for (size_t i = 0; i < conf_.num_threads; i++) {
                threads[i] = std::thread([&, i = i] {
                    EdgeId edge_begin = i * edges_per_thread;
                    EdgeId edge_end   = std::min((i + 1) * edges_per_thread, G.local_edge_count());
                    SPDLOG_DEBUG("[t{}] procesing edges {} to {}", i, edge_begin, edge_end);

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
                        SPDLOG_DEBUG("thread local cost {}", thread_local_cost);
                        std::exclusive_scan(
                            thread_local_cost.begin(),
                            thread_local_cost.end(),
                            thread_local_cost.begin(),
                            size_t{0}
                        );
                        auto total_cost = thread_local_cost.back();
                        per_thread_cost = total_cost / conf_.num_threads;
                        SPDLOG_DEBUG("thread local cost {}", thread_local_cost);
                        SPDLOG_DEBUG("per thread cost {}", per_thread_cost);
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
                        SPDLOG_DEBUG("first edge {}", first_edge);
                        first_edge[0]     = 0;
                        first_edge.back() = G.local_edge_count();
                        SPDLOG_DEBUG("first edge {}", first_edge);
                    });
                    edge_begin = first_edge[i];
                    edge_end   = first_edge[i + 1];
                    SPDLOG_DEBUG("[t{}] procesing edges {} to {}", i, edge_begin, edge_end);
                    tail_idx        = edge_locator.edge_tail_idx(edge_begin);
                    tail            = node_indexer.get_node(tail_idx);
                    tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    SPDLOG_DEBUG(
                        "[t{}] processing edges {} to {}",
                        tbb::this_task_arena::current_thread_index(),
                        std::max(edge_begin, tail_first_edge),
                        edge_end
                    );
                    std::vector<std::atomic<bool>> is_interface;
                    constexpr bool                 fast_is_interface =
                        std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                        || std::is_same_v<NodeOrdering, node_ordering::id>
                        || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
                    if constexpr (!fast_is_interface) {
                        is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
                    }
                    auto& local_interface_nodes = interface_nodes.local();
                    for (auto edge_id = std::max(edge_begin, tail_first_edge); edge_id < edge_end; edge_id++) {
                        while (edge_id >= tail_last_edge || (conf_.pseudo2core && G.outdegree(tail) < 2)) {
                            if (conf_.pseudo2core && G.outdegree(tail) < 2) {
                                thread_local_skipped_nodes[i]++;
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
                                    local_interface_nodes.emplace_back(tail);
                                }
                            }
                        } else {
                            if (u.rank() != rank_) {
                                bool was_set = is_interface[tail_idx].exchange(true);
                                if (!was_set) {
                                    local_interface_nodes.emplace_back(tail);
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
                        SPDLOG_DEBUG("[t{}] intersecting {} {}", tbb::this_task_arena::current_thread_index(), v, u);
                        // for each edge (v, u) we check the open wedge (v, u,
                        // w) for w in N(u)+
                        thread_local_wedge_checks[i] += u_neighbors.end() - u_neighbors.begin();
                        thread_local_intersection_size_local[i] +=
                            v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                        cetric::intersection(
                            v_neighbors.begin() + offset,
                            v_neighbors.end(),
                            u_neighbors.begin(),
                            u_neighbors.end(),
                            [&](RankEncodedNodeId w) {
                                thread_local_local_triangles[i]++;
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
            profiling::counter_add("skipped_nodes", std::reduce(thread_local_skipped_nodes.begin(), thread_local_skipped_nodes.end()));
            profiling::counter_add("intersection_size_local", std::reduce(thread_local_intersection_size_local.begin(), thread_local_intersection_size_local.end()));
            profiling::counter_add("wedge_checks", std::reduce(thread_local_wedge_checks.begin(), thread_local_wedge_checks.end()));
            profiling::counter_add("local_triangles", std::reduce(thread_local_local_triangles.begin(), thread_local_local_triangles.end()));
        } else {
            tbb::task_arena            arena(conf_.num_threads);
            tbb::blocked_range<EdgeId> edge_ids(EdgeId{0}, G.local_edge_count(), conf_.grainsize);

            std::vector<std::atomic<bool>> is_interface;
            constexpr bool fast_is_interface = std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>
                                               || std::is_same_v<NodeOrdering, node_ordering::id>
                                               || std::is_same_v<NodeOrdering, node_ordering::id_outward>;
            if constexpr (!fast_is_interface) {
                is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
            }
            tbb::combinable<size_t> thread_local_skipped_nodes{0};
            tbb::combinable<size_t> thread_local_wedge_checks{0};
            tbb::combinable<size_t> thread_local_intersection_size_local{0};
            tbb::combinable<size_t> thread_local_local_triangles{0};
            auto                    body = [&](auto const& edge_id_range) {
                auto& skipped_nodes           = thread_local_skipped_nodes.local();
                auto& wedge_checks            = thread_local_wedge_checks.local();
                auto& intersection_size_local = thread_local_intersection_size_local.local();
                auto& local_triangles         = thread_local_local_triangles.local();
                auto& local_interface_nodes   = interface_nodes.local();

                auto tail_idx = edge_locator.edge_tail_idx(edge_id_range.begin());
                auto tail     = node_indexer.get_node(tail_idx);
                auto tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                auto tail_last_edge = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                SPDLOG_DEBUG(
                    "[t{}] processing edges {} to {}",
                    tbb::this_task_arena::current_thread_index(),
                    std::max(edge_id_range.begin(), tail_first_edge),
                    edge_id_range.end()
                );
                for (auto edge_id = std::max(edge_id_range.begin(), tail_first_edge); edge_id < edge_id_range.end();
                     edge_id++) {
                    while (edge_id >= tail_last_edge || (conf_.pseudo2core && G.outdegree(tail) < 2)) {
                        if (conf_.pseudo2core && G.outdegree(tail) < 2) {
                            skipped_nodes++;
                        }
                        tail_idx++;
                        tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                        edge_id = tail_first_edge;
                        if (edge_id >= edge_id_range.end()) {
                            goto finish;
                        }
                        tail = node_indexer.get_node(tail_idx);
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
                                local_interface_nodes.emplace_back(tail);
                            }
                        }
                    } else {
                        if (u.rank() != rank_) {
                            bool was_set = is_interface[tail_idx].exchange(true);
                            if (!was_set) {
                                local_interface_nodes.emplace_back(tail);
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
                    SPDLOG_DEBUG("[t{}] intersecting {} {}", tbb::this_task_arena::current_thread_index(), v, u);
                    // for each edge (v, u) we check the open wedge (v, u, w)
                    // for w in N(u)+
                    wedge_checks += u_neighbors.end() - u_neighbors.begin();
                    intersection_size_local +=
                        v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                    cetric::intersection(
                        v_neighbors.begin() + offset,
                        v_neighbors.end(),
                        u_neighbors.begin(),
                        u_neighbors.end(),
                        [&](RankEncodedNodeId w) {
                            local_triangles++;
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
            profiling::counter_add("skipped_nodes", thread_local_skipped_nodes.combine(std::plus<>{}));
            profiling::counter_add("wedge_checks", thread_local_wedge_checks.combine(std::plus<>{}));
            profiling::counter_add("intersection_size_local", thread_local_intersection_size_local.combine(std::plus<>{}));
            profiling::counter_add("local_triangles", thread_local_local_triangles.combine(std::plus<>{}));
        }
        // for (size_t i = 0; i < edges.size(); ++i) {
        //     KASSERT(edges[i].size() == processed_edges[i].size(), "Failed for " << i);
        // }

    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        interface_nodes.clear();
        auto                    nodes = G.local_nodes();
        tbb::combinable<size_t> thread_local_skipped_nodes{0};
        tbb::combinable<size_t> thread_local_wedge_checks{0};
        tbb::combinable<size_t> thread_local_intersection_size_local{0};
        tbb::combinable<size_t> thread_local_local_triangles{0};

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

        auto handle_neighbor_range = [&](RankEncodedNodeId                 v,
                                         tbb::blocked_range<size_t> const& neighbor_range,
                                         bool                              spawn [[maybe_unused]] = false) {
            auto& wedge_checks            = thread_local_wedge_checks.local();
            auto& intersection_size_local = thread_local_intersection_size_local.local();
            auto& local_triangles         = thread_local_local_triangles.local();
            for (size_t neighbor_index = neighbor_range.begin(); neighbor_index != neighbor_range.end();
                 neighbor_index++) {
                auto u = *(G.out_adj(v).neighbors().begin() + neighbor_index);
                KASSERT(*(G.out_adj(v).neighbors().begin() + neighbor_index) == u);
                if (u.rank() != rank_) {
                    if constexpr (!std::is_same_v<NodeOrdering, node_ordering::id_outward> && !std::is_same_v<NodeOrdering, node_ordering::degree_outward<GraphType>>) {
                        continue;
                    }
                    // if (conf_.algorithm == Algorithm::patric) {
                    //     // neighbor_index++;
                    // }
                    SPDLOG_DEBUG(
                        "Remaining neighbors {}",
                        std::ranges::subrange(
                            G.out_adj(v).neighbors().begin() + neighbor_index,
                            G.out_adj(v).neighbors().end()
                        )
                    );
                    // TODO: we should be able to break here when the edges are
                    // properly sorted
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
                    wedge_checks += u_neighbors.end() - u_neighbors.begin();
                    intersection_size_local +=
                        v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin();
                    cetric::intersection(
                        v_neighbors.begin() + offset,
                        v_neighbors.end(),
                        u_neighbors.begin(),
                        u_neighbors.end(),
                        [&](RankEncodedNodeId w) {
                            local_triangles++;
                            // local_triangles++;
                            emit(Triangle<RankEncodedNodeId>{v, u, w});
                        },
                        node_ordering,
                        conf_
                    );
                }
            }
        };
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
            auto& skipped_nodes         = thread_local_skipped_nodes.local();
            auto& local_interface_nodes = interface_nodes.local();
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                skipped_nodes++;
                // continue;
            } else {
                if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                    local_interface_nodes.emplace_back(v);
                }
                auto v_neighbors_size = std::distance(G.out_adj(v).neighbors().begin(), G.out_adj(v).neighbors().end());
                if (conf_.parallelization_method == ParallelizationMethod::tbb && partition_neighborhood_2d(v)) {
                    // stats.local.nodes_parallel2d++;
                    tbb::parallel_for(
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        [&handle_neighbor_range, v = v](auto const& neighbor_range) {
                            handle_neighbor_range(v, neighbor_range);
                        }
                    );
                } else {
                    if (partition_neighborhood_2d(v)) {
                        // stats.local.nodes_parallel2d++;
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
                tbb::task_arena arena(conf_.num_threads);
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
#ifdef CETRIC_HAS_OPENMP
                // clang-format off
                #pragma omp parallel for schedule(runtime)
                // clang-format on
#else
                throw std::runtime_error("parallelization method omp_for requires building with CETRIC_USE_OPENMP=ON");
#endif
                for (auto v: nodes) {
                    node_loop_body(v);
                }
                break;
            }
            case ParallelizationMethod::omp_task: {
                throw std::runtime_error("parallelization method omp_task is currently unsupported");
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
        profiling::counter_add("skipped_nodes", thread_local_skipped_nodes.combine(std::plus<>{}));
        profiling::counter_add("wedge_checks", thread_local_wedge_checks.combine(std::plus<>{}));
        profiling::counter_add("intersection_size_local", thread_local_intersection_size_local.combine(std::plus<>{}));
        profiling::counter_add("local_triangles", thread_local_local_triangles.combine(std::plus<>{}));
    }

    template <typename EdgeOrientationOrdering, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_with_ghosts(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        interface_nodes.clear();

        tbb::combinable<size_t> thread_local_skipped_nodes{0};
        tbb::combinable<size_t> thread_local_wedge_checks{0};
        tbb::combinable<size_t> thread_local_intersection_size_local{0};
        tbb::combinable<size_t> thread_local_local_triangles{0};

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
            auto& wedge_checks            = thread_local_wedge_checks.local();
            auto& intersection_size_local = thread_local_intersection_size_local.local();
            auto& local_triangles         = thread_local_local_triangles.local();
            auto& local_interface_nodes   = interface_nodes.local();

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
                            local_interface_nodes.emplace_back(v);
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
                // for w in N(u)+; if either endpoint is a ghost, only local vertices can close it
                auto v_first = v_neighbors.begin() + offset;
                auto v_last  = v_neighbors.end();
                auto u_first = u_neighbors.begin();
                auto u_last  = u_neighbors.end();
                if (u.rank() != rank_) {
                    v_last = local_prefix_end<NodeOrdering>(v_first, v_last);
                } else if (v.rank() != rank_) {
                    u_last = local_prefix_end<NodeOrdering>(u_first, u_last);
                }
                wedge_checks += u_last - u_first;
                intersection_size_local += (v_last - v_first) + (u_last - u_first);
                cetric::intersection(
                    v_first,
                    v_last,
                    u_first,
                    u_last,
                    [&](RankEncodedNodeId w) {
                        local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    },
                    node_ordering,
                    conf_
                );
            }
        };
        auto node_loop_body = [&](RankEncodedNodeId const& v) {
            auto&  skipped_nodes = thread_local_skipped_nodes.local();
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
                size_t v_neighbors_size = v_degree;
                if (conf_.parallelization_method == ParallelizationMethod::tbb && partition_neighborhood_2d(v)) {
                    // stats.local.nodes_parallel2d++;
                    tbb::parallel_for(
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        [&handle_neighbor_range, v = v](auto const& neighbor_range) {
                            handle_neighbor_range(v, neighbor_range);
                        }
                    );
                } else {
                    if (partition_neighborhood_2d(v)) {
                        // stats.local.nodes_parallel2d++;
                    }
                    handle_neighbor_range(
                        v,
                        tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                        partition_neighborhood_2d(v)
                    );
                }
            }
        };
        // Local nodes followed by ghosts, addressed by index (C++20 has no random-access concat view).
        auto const local_nodes = G.local_nodes();
        auto const ghosts      = G.ghosts();
        auto const local_count = static_cast<size_t>(std::ranges::size(local_nodes));
        auto const total_count = local_count + static_cast<size_t>(std::ranges::size(ghosts));
        auto       node_at     = [&](size_t i) { return i < local_count ? local_nodes[i] : ghosts[i - local_count]; };
        switch (conf_.parallelization_method) {
            case ParallelizationMethod::tbb: {
                tbb::task_arena arena(conf_.num_threads);
                arena.execute([&] {
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, total_count), [&](auto const& r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                            node_loop_body(node_at(i));
                        }
                    });
                });
                break;
            }
            case ParallelizationMethod::omp_for: {
#ifdef CETRIC_HAS_OPENMP
                // clang-format off
                #pragma omp parallel for schedule(runtime)
                // clang-format on
#else
                throw std::runtime_error("parallelization method omp_for requires building with CETRIC_USE_OPENMP=ON");
#endif
                for (size_t i = 0; i < total_count; ++i) {
                    node_loop_body(node_at(i));
                }
                break;
            }
            case ParallelizationMethod::omp_task: {
                throw std::runtime_error("parallelization method omp_task is currently unsupported");
            }
        }
        profiling::counter_add("skipped_nodes", thread_local_skipped_nodes.combine(std::plus<>{}));
        profiling::counter_add("wedge_checks", thread_local_wedge_checks.combine(std::plus<>{}));
        profiling::counter_add("intersection_size_local", thread_local_intersection_size_local.combine(std::plus<>{}));
        profiling::counter_add("local_triangles", thread_local_local_triangles.combine(std::plus<>{}));
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_distributed(TriangleFunc emit, NodeOrdering&& node_ordering) {
        std::set<RankEncodedNodeId> ghosts;
        run_distributed(emit, std::forward<NodeOrdering>(node_ordering), ghosts);
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering>
    inline void run_distributed(
        TriangleFunc                   emit,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_ndoes_end,
        NodeOrdering&&                 node_ordering
    ) {
        std::set<RankEncodedNodeId> ghosts;
        run_distributed(
            emit,
            interface_nodes_begin,
            interface_ndoes_end,
            std::forward<NodeOrdering>(node_ordering),
            ghosts
        );
    }

    template <typename TriangleFunc, typename NodeOrdering, typename GhostSet>
    inline void run_distributed(
        TriangleFunc emit, NodeOrdering&& node_ordering, GhostSet const& ghosts
    ) {
        auto all_interface_nodes = tbb::flatten2d(interface_nodes_);
        run_distributed(
            emit,
            all_interface_nodes.begin(),
            all_interface_nodes.end(),
            std::forward<NodeOrdering>(node_ordering),
            ghosts
        );
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering, typename GhostSet>
    inline void run_distributed(
        TriangleFunc                   emit,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_nodes_end,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        if (conf_.global_parallel && conf_.num_threads > 1) {
            auto do_run = [&](auto& pool) {
                run_distributed_parallel(
                    emit,
                    interface_nodes_begin,
                    interface_nodes_end,
                    std::forward<NodeOrdering>(node_ordering),
                    ghosts,
                    pool
                );
            };
            switch (conf_.task_pool_type) {
                case TaskPoolType::std_thread: {
                    ThreadTaskPool pool(conf_.num_threads);
                    do_run(pool);
                    break;
                }
                case TaskPoolType::task_group: {
                   if (conf_.task_priorities) {
                       TbbTaskPool<true> pool(conf_.num_threads);
                       do_run(pool);
                   } else {
                       TbbTaskPool<false> pool(conf_.num_threads);
                       do_run(pool);
                   }
                   break;
               }
            }

        } else {
            run_distributed_sequential(
                emit,
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
        size_t skipped_nodes   = 0;
        size_t interface_nodes = 0;
        // Time spent building and posting messages and handling received ones (the intersections), accumulated per
        // call because a Caliper region per message would cost more than the work it measures. The rest of the
        // phase's exclusive time is polling; the region `termination` is the wait for the other ranks.
        using clock = std::chrono::steady_clock;
        clock::duration send_time{};
        clock::duration handle_time{};
        auto            on_message = [&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
            auto const start = clock::now();
            handle_buffer(span.begin(), span.end(), emit, node_ordering, ghosts);
            handle_time += clock::now() - start;
        };
        for (auto current = interface_nodes_begin; current != interface_nodes_end; current++) {
            RankEncodedNodeId v = *current;
            interface_nodes++;
            // iterate over neighborhood and delegate to other PEs if necessary
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                skipped_nodes++;
            } else {
                auto const start = clock::now();
                for (RankEncodedEdge edge: G.out_adj(v).edges()) {
                    RankEncodedNodeId u = edge.head;
                    if (/*conf_.algorithm == Algorithm::cetric || */ u.rank() != rank_) {
                        PEID u_rank = u.rank();
                        assert(u_rank != rank_);
                        if (last_proc_[G.to_local_idx(v)] != u_rank) {
                            enqueue_for_sending(queue, v, u_rank, emit);
                        }
                    }
                }
                send_time += clock::now() - start;
            }
            queue.poll(on_message);
        }
        {
            CALI_CXX_MARK_SCOPE("termination");
            queue.terminate(on_message);
        }
        report_distributed_counters(queue.stats());
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("interface_nodes", interface_nodes);
        profiling::counter_add("send_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(send_time).count());
        profiling::counter_add("handle_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(handle_time).count());
        queue.reset();
    }

    template <typename TriangleFunc, typename NodeIterator, typename NodeOrdering, typename GhostSet, typename Pool>
    inline void run_distributed_parallel(
        TriangleFunc                   emit,
        NodeIterator                   interface_nodes_begin,
        NodeIterator                   interface_nodes_end,
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts,
        Pool&                          pool
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
        std::atomic<size_t>      nodes_queued = 0;
        std::atomic<size_t>      write_jobs   = 0;
        tbb::task_arena          arena(conf_.num_threads);
        // TbbTaskPool              pool(conf_.num_threads);
        //  ThreadPool               pool(conf_.num_threads - 1);
        tbb::concurrent_queue<std::function<void()>> high_priority_tasks;

        // ConcurrencyTracker  tracker(conf_.num_threads);
        tbb::combinable<size_t> skipped_nodes {0};
        auto                task          = [&](const RankEncodedNodeId& v) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                skipped_nodes.local()++;
                nodes_queued--;
                return;
            }
            for (RankEncodedEdge edge: G.out_adj(v).edges()) {
                RankEncodedNodeId u = edge.head;
                if (/*conf_.algorithm == Algorithm::cetric || */ u.rank() != rank_) {
                    // assert(G.is_local_from_local(v));
                    // assert(G.is_local(G.to_global_id(v)));
                    // assert(!G.is_local(G.to_global_id(u)));
                    // assert(!G.is_local_from_local(u));
                    PEID u_rank = u.rank();
                    assert(u_rank != rank_);
                    if (last_proc_[G.to_local_idx(v)] != u_rank) {
                        SPDLOG_DEBUG("Send N({}) to {}", v, u_rank);
                        enqueue_for_sending_async(queue, v, u_rank, pool, write_jobs, emit);
                    }
                }
            }
            // tracker.track("enqueue");
            nodes_queued--;
        };
        // pool.submit_work([&] {
        //     std::for_each(
        //         interface_nodes_begin,
        //         interface_nodes_end,
        //         [&](RankEncodedNodeId v /*, tbb::feeder<RankEncodedNodeId> &
        //         feeder*/) {
        //             nodes_queued++;
        //             if (conf_.global_degree_of_parallelism > 1) {
        //                 pool.submit_work([v = v, task] { task(v); });
        //                 // task_group.run(task);
        //                 //  pool.enqueue(task);
        //             } else {
        //                 task(v);
        //             }
        //         }
        //     );
        // });
        nodes_queued = std::distance(interface_nodes_begin, interface_nodes_end);
        tbb::enumerable_thread_specific<size_t> priorized_nodes = 0;
        pool.submit_work([&] {
            // tbb::parallel_for_each(interface_nodes_begin, interface_nodes_end,
            // [&](const RankEncodedNodeId& v) {
            if constexpr (Pool::has_tbb) {
                tbb::parallel_for_each(
                    interface_nodes_begin,
                    interface_nodes_end,
                    [&](const RankEncodedNodeId& v, tbb::feeder<RankEncodedNodeId>& feeder) {
                        std::function<void()> high_priority_task;
                        if constexpr (Pool::use_priorities) {
                            if (high_priority_tasks.try_pop(high_priority_task)) {
                                priorized_nodes.local()++;
                                feeder.add(v);
                                high_priority_task();
                                return;
                            }
                        }
                        //if (conf_.global_degree_of_parallelism > 1) {
                        pool.submit_work([v = v, task] { task(v); });
                            // task_group.run(task);
                            //  pool.enqueue(task);
                        //} else {
                         //   task(v);
                    }
                );
            } else {
                std::for_each(interface_nodes_begin, interface_nodes_end, [&](const RankEncodedNodeId& v) {
                    pool.submit_work([v = v, task] { task(v); });
                });
            }
            // for (auto current = interface_nodes_begin;
            //      current != interface_nodes_end; current++) {
            //   RankEncodedNodeId v = *current;
            //   // nodes_queued++;
            // }
        });
        auto master_task = [&] {
            while (true) {
                if (write_jobs == 0 && nodes_queued == 0) {
                    SPDLOG_DEBUG("No more polling");
                    break;
                }
                queue.check_for_overflow_and_flush();
                queue.poll([&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
                    handle_buffer_hybrid(
                        std::move(span),
                        pool,
                        high_priority_tasks,
                        emit,
                        std::forward<NodeOrdering>(node_ordering),
                        ghosts
                    );
                });
                if constexpr (Pool::use_priorities) {
                    std::function<void()> intersect_task;
                    while (high_priority_tasks.try_pop(intersect_task)) {
                        intersect_task();
                        //pool.submit_work(std::move(intersect_task));
                    }
                }
            }
            SPDLOG_DEBUG("remaining high priority tasks: {}", high_priority_tasks.unsafe_size());
            queue.terminate([&](SharedVectorSpan<RankEncodedNodeId> span, PEID sender [[maybe_unused]]) {
                handle_buffer_hybrid(
                    std::move(span),
                    pool,
                    high_priority_tasks,
                    emit,
                    std::forward<NodeOrdering>(node_ordering),
                    ghosts
                );
            });
            SPDLOG_DEBUG("priorized nodes: {}", priorized_nodes.combine(std::plus<>{}));
            std::function<void()> intersect_task;
            if constexpr (Pool::use_priorities) {
                std::function<void()> intersect_task;
                while (high_priority_tasks.try_pop(intersect_task)) {
                    pool.submit_work(std::move(intersect_task));
                }
            }
            while (high_priority_tasks.try_pop(intersect_task)) {
                pool.submit_work(std::move(intersect_task));
            }
            // tracker.track("master");
        };
        pool.run_and_wait(master_task);
        // pool.loop_until_empty();
        // pool.terminate();
        SPDLOG_DEBUG("Finished pool");
        report_distributed_counters(queue.stats());
        profiling::counter_add("skipped_nodes", skipped_nodes.combine(std::plus<>{}));
        queue.reset();
    }

private:
    /// Reports the counts of a distributed run (made on the pool threads) and the message statistics of its queue
    /// to the open Caliper region, and resets them for the next run.
    template <typename MessageStatistics>
    void report_distributed_counters(MessageStatistics const& messages) {
        profiling::counter_add("wedge_checks", distributed_wedge_checks_.exchange(0));
        profiling::counter_add("intersection_size_global", distributed_intersection_size_.exchange(0));
        profiling::counter_add("type3_triangles", distributed_triangles_.exchange(0));
        profiling::counter_add("sent_messages", messages.sent_messages);
        profiling::counter_add("send_volume", messages.send_volume);
        profiling::counter_add("received_messages", messages.received_messages);
        profiling::counter_add("receive_volume", messages.receive_volume);
    }

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
            if (conf_.algorithm == Algorithm::cetric) {
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
        TriangleFunc                   emit [[maybe_unused]]
    ) {
        std::vector<RankEncodedNodeId> buffer;
        buffer.emplace_back(v);
        // size_t send_count = 0;
        for (RankEncodedEdge e: G.out_adj(v).edges()) {
            assert(conf_.algorithm == Algorithm::patric || e.head.rank() != rank_);
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
        SPDLOG_DEBUG("Sending {} to {}", buffer, u_rank);
        queue.post_message(std::move(buffer), u_rank);
        last_proc_[G.to_local_idx(v)] = u_rank;
    }

    template <typename MessageQueue, typename TriangleFunc, typename Pool>
    void enqueue_for_sending_async(
        MessageQueue&                  queue,
        RankEncodedNodeId              v,
        PEID                           u_rank,
        Pool&                          pool,
        std::atomic<size_t>&           write_jobs,
        TriangleFunc                   emit [[maybe_unused]]
    ) {
        SPDLOG_DEBUG("rank {}", u_rank);
        SPDLOG_DEBUG("{}", u_rank);
        write_jobs++;
        // thread_pool.enqueue([&queue, &write_jobs, this, v, u_rank] {
        pool.submit_work([&queue, &write_jobs, this, v, u_rank]() {
            std::vector<RankEncodedNodeId> buffer;
            buffer.emplace_back(v);
            // size_t send_count = 0;
            SPDLOG_DEBUG("{}", u_rank);
            for (RankEncodedNodeId node: G.out_adj(v).neighbors()) {
                assert(conf_.algorithm == Algorithm::patric || node.rank() != rank_);
                if (send_neighbor(node, u_rank)) {
                    buffer.emplace_back(node);
                }
            }
            SPDLOG_DEBUG("{}", u_rank);
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
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        RankEncodedNodeId v = *begin;
        process_neighborhood(v, begin + 1, end, emit, std::forward<NodeOrdering>(node_ordering), ghosts);
    }

    template <typename TriangleFunc, typename NodeOrdering, typename GhostSet, typename Pool>
    void handle_buffer_hybrid(
        SharedVectorSpan<RankEncodedNodeId>           span,
        Pool&                                         pool [[maybe_unused]],
        tbb::concurrent_queue<std::function<void()>>& high_priority_tasks,
        TriangleFunc                                  emit,
        NodeOrdering&&                                node_ordering,
        GhostSet const&                               ghosts
    ) {
        auto task = [this, emit, span = std::move(span), &node_ordering, &ghosts] {
            RankEncodedNodeId v = *span.begin();
            SPDLOG_DEBUG("Spawn task for {} on thread {}", v, tbb::this_task_arena::current_thread_index());
            process_neighborhood(
                v,
                span.begin() + 1,
                span.end(),
                emit,
                std::forward<NodeOrdering>(node_ordering),
                ghosts
            );
        };
        if constexpr (Pool::use_priorities) {
            high_priority_tasks.emplace(std::move(task));
        } else {
            pool.submit_work(std::move(task));
        }

        // pool.submit_work(task);
        // pool.submit_work();
        // thread_pool.enqueue(
        //     [this, emit, span = std::move(span), &node_ordering,
        //     &ghosts] {
        //         RankEncodedNodeId v = *span.begin();
        //         // atomic_debug(
        //         //     fmt::format("Spawn task for {} on thread {}", v,
        //         //     tbb::this_task_arena::current_thread_index()));
        //         process_neighborhood(
        //             v,
        //             span.begin() + 1,
        //             span.end(),
        //             emit,
        //             std::forward<NodeOrdering>(node_ordering),
        //             ghosts
        //         );
        //     },
        //     TaskPriority::high
        // );
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
    //         if (conf_.algorithm == Algorithm::patric && conf_.skip_local_neighborhood) {
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
    //         if (conf_.algorithm == Algorithm::patric && conf_.skip_local_neighborhood) {
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
        if (conf_.algorithm == Algorithm::cetric_x) {
            // this node may have no out edges and was therefore removed
            if (!G.is_local(u)) {
                return;
            }
        }
        auto u_neighbors = G.out_adj(u).neighbors();
        if (!ghosts.empty()) {
            auto filtered_neighbors_it =
                std::views::filter(std::ranges::subrange(begin, end), [&ghosts](RankEncodedNodeId node) {
                    return ghosts.find(node) != ghosts.end();
                });
            std::vector<RankEncodedNodeId> filtered_neighbors(
                filtered_neighbors_it.begin(),
                filtered_neighbors_it.end()
            );
            SPDLOG_DEBUG("intersecting N({})={} and N({}){}", u, u_neighbors, v, filtered_neighbors);

            // for each edge (v, u) we check the open wedge
            //  (v, u, w) for w in N(u)+
            distributed_wedge_checks_.fetch_add(u_neighbors.end() - u_neighbors.begin(), std::memory_order_relaxed);
            distributed_intersection_size_.fetch_add(
                filtered_neighbors.end() - filtered_neighbors.begin() + u_neighbors.end() - u_neighbors.begin(),
                std::memory_order_relaxed
            );
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
            distributed_wedge_checks_.fetch_add(u_neighbors.end() - u_neighbors.begin(), std::memory_order_relaxed);

            distributed_intersection_size_.fetch_add(
                end - begin + u_neighbors.end() - u_neighbors.begin(),
                std::memory_order_relaxed
            );
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
        if (conf_.skip_local_neighborhood && conf_.algorithm == Algorithm::patric) {
            auto v_neighbors = G.out_adj(v).neighbors();
            // for each edge (v, u) we check the open wedge (v, u, w) for w in
            // N(u)+
            distributed_wedge_checks_.fetch_add(u_neighbors.end() - u_neighbors.begin(), std::memory_order_relaxed);
            distributed_intersection_size_.fetch_add(
                v_neighbors.end() - v_neighbors.begin() + u_neighbors.end() - u_neighbors.begin(),
                std::memory_order_relaxed
            );
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
        NodeOrdering&&                 node_ordering,
        GhostSet const&                ghosts
    ) {
        SPDLOG_DEBUG("Handling message {}", std::ranges::subrange(begin, end));
        assert(v.rank() != rank_);
        // distributed_pre_intersect(v, begin, end);
        auto for_each_local_receiver = [&](auto on_node) {
            if (conf_.skip_local_neighborhood) {
                auto neighbors = G.out_adj(v).neighbors();
                if (conf_.global_degree_of_parallelism > 2) {
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
                if (conf_.global_degree_of_parallelism > 2) {
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
                    distributed_triangles_.fetch_add(1, std::memory_order_relaxed);
                },
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
    // counted concurrently by the tasks of run_distributed_*(), see report_distributed_counters()
    std::atomic<size_t> distributed_wedge_checks_{0};
    std::atomic<size_t> distributed_intersection_size_{0};
    std::atomic<size_t> distributed_triangles_{0};
}; // namespace cetric

} // namespace cetric
#endif // PARALLEL_TRIANGLE_COUNTER_PARALLEL_NODE_ITERATOR_H
