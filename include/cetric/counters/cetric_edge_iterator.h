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
#include "cetric/communication_policy.h"
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
/// Work counters of the local phase, one instance per thread in the parallel variants.
struct LocalPhaseCounters {
    size_t skipped_nodes           = 0;
    size_t wedge_checks            = 0;
    size_t intersection_size_local = 0;
    size_t local_triangles         = 0;

    friend LocalPhaseCounters operator+(LocalPhaseCounters lhs, LocalPhaseCounters const& rhs) {
        lhs.skipped_nodes += rhs.skipped_nodes;
        lhs.wedge_checks += rhs.wedge_checks;
        lhs.intersection_size_local += rhs.intersection_size_local;
        lhs.local_triangles += rhs.local_triangles;
        return lhs;
    }

    /// Adds the counts to the open Caliper region.
    void report() const {
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_local", intersection_size_local);
        profiling::counter_add("local_triangles", local_triangles);
    }
};

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
          interface_nodes_(),
          pe_min_degree(),
          threshold_(std::numeric_limits<size_t>::max()) {
        EdgeId total_edge_count = G.total_edge_count();
        high_degree_threshold_  = sqrt(total_edge_count / 2.0);
    }

    void set_threshold(size_t threshold) {
        threshold_ = threshold;
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local(TriangleFunc emit, NodeOrdering&& node_ordering) {
        bool const with_ghosts = conf_.algorithm == Algorithm::cetric_x;
        if (conf_.local_parallel && conf_.num_threads > 1) {
            // hybrid path, frozen during the refactoring (doc/refactoring-plan.md)
            if (with_ghosts) {
                if (conf_.edge_partitioning) {
                    run_local_parallel_edge_partitioned_with_ghosts(emit, interface_nodes_, node_ordering);
                } else {
                    run_local_parallel_with_ghosts(emit, interface_nodes_, node_ordering);
                }
            } else if (conf_.edge_partitioning) {
                run_local_parallel_edge_partitioned(emit, interface_nodes_, node_ordering);
            } else {
                run_local_parallel(emit, interface_nodes_, node_ordering);
            }
            return;
        }
        if (with_ghosts) {
            run_local_sequential<true>(emit, interface_nodes_.local(), node_ordering);
        } else {
            run_local_sequential<false>(emit, interface_nodes_.local(), node_ordering);
        }
    }

    /// The local phase on one thread: the intersection for every edge v -> u with both ends local and, `with_ghosts`
    /// (cetric-x), also for the edges with a ghost end, through the ghosts' adjacencies. Collects the interface
    /// vertices (local vertices with a cut out-edge) for the global phase.
    template <bool with_ghosts, typename TriangleFunc, typename NodeOrdering>
    inline void run_local_sequential(
        TriangleFunc emit, std::vector<RankEncodedNodeId>& interface_nodes, NodeOrdering const& node_ordering
    ) {
        interface_nodes.clear();
        LocalPhaseCounters counters;
        auto               for_each_out_edge = [&](RankEncodedNodeId v, Degree degree) {
            if (conf_.pseudo2core && degree < 2) {
                counters.skipped_nodes++;
                return;
            }
            bool is_interface = false;
            for (RankEncodedNodeId u: neighbors_with_ghosts(v)) {
                if (u.rank() != rank_) {
                    if (v.rank() == rank_ && !is_interface) {
                        interface_nodes.emplace_back(v);
                        is_interface = true;
                    }
                    if constexpr (!with_ghosts) {
                        // without ghost adjacencies only local heads are intersected; with the local vertices
                        // first, the first cut edge ends them
                        if constexpr (local_first<NodeOrdering>) {
                            break;
                        } else {
                            continue;
                        }
                    }
                }
                intersect_local_edge(v, u, counters, emit, node_ordering);
            }
        };
        for (auto v: G.local_nodes()) {
            for_each_out_edge(v, G.outdegree(v));
        }
        if constexpr (with_ghosts) {
            for (auto v: G.ghosts()) {
                for_each_out_edge(v, G.ghost_degree(v));
            }
        }
        counters.report();
    }

    /// Whether neighborhoods sorted by `NodeOrdering` have the local vertices first.
    template <typename NodeOrdering>
    static constexpr bool local_first = std::is_same_v<std::remove_cvref_t<NodeOrdering>, node_ordering::id_outward>;

    /// End of the local prefix of a neighborhood in the ghost-expanded graph (cetric-x). A ghost's adjacency holds only
    /// local vertices, so when one side of an intersection is a ghost, only the local part of the other side can
    /// match. With `id_outward` the local vertices come first, so the other side is clipped there. This bounds the
    /// merge and keeps the work counters exact. Other orderings do not sort local vertices first, so the range stays
    /// as it is.
    template <typename NodeOrdering, typename It>
    [[nodiscard]] It local_prefix_end(It first, It last) const {
        if constexpr (local_first<NodeOrdering>) {
            return std::partition_point(first, last, [this](RankEncodedNodeId w) { return w.rank() == rank_; });
        } else {
            return last;
        }
    }

    /// N+(v) for a local vertex, the ghost adjacency (cetric-x: its out-edges to local vertices) for a ghost.
    [[nodiscard]] auto neighbors_with_ghosts(RankEncodedNodeId v) const {
        return v.rank() == rank_ ? G.out_adj(v).neighbors() : G.ghost_adj(v).neighbors();
    }

    /// The local phase's work for one edge v -> u: emits the triangles (v, u, w) for w in N+(v) ∩ N+(u), where a
    /// ghost's neighborhood is its ghost adjacency. Every variant of the local phase calls this.
    template <typename TriangleFunc, typename NodeOrdering>
    inline void intersect_local_edge(
        RankEncodedNodeId   v,
        RankEncodedNodeId   u,
        LocalPhaseCounters& counters,
        TriangleFunc&       emit,
        NodeOrdering const& node_ordering
    ) {
        auto const v_neighbors = neighbors_with_ghosts(v);
        auto const u_neighbors = neighbors_with_ghosts(u);
        auto       v_first     = v_neighbors.begin();
        auto       v_last      = v_neighbors.end();
        auto       u_first     = u_neighbors.begin();
        auto       u_last      = u_neighbors.end();
        if (u.rank() != rank_) {
            v_last = local_prefix_end<NodeOrdering>(v_first, v_last);
        } else if (v.rank() != rank_) {
            u_last = local_prefix_end<NodeOrdering>(u_first, u_last);
        }
        // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
        counters.wedge_checks += u_last - u_first;
        counters.intersection_size_local += (v_last - v_first) + (u_last - u_first);
        cetric::intersection(
            v_first,
            v_last,
            u_first,
            u_last,
            [&](RankEncodedNodeId w) {
                counters.local_triangles++;
                emit(Triangle<RankEncodedNodeId>{v, u, w});
            },
            node_ordering,
            conf_
        );
    }

    // The hybrid (MPI + TBB) variants of the local phase, frozen during the refactoring: node- or edge-partitioned,
    // with or without ghosts. Ported onto run_local_sequential's structure afterwards.

    /// Whether a vertex's interface status follows from its neighborhood order alone.
    template <typename NodeOrdering>
    static constexpr bool fast_is_interface =
        std::is_same_v<std::remove_cvref_t<NodeOrdering>, node_ordering::degree_outward<GraphType>>
        || std::is_same_v<std::remove_cvref_t<NodeOrdering>, node_ordering::id>
        || std::is_same_v<std::remove_cvref_t<NodeOrdering>, node_ordering::id_outward>;

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_edge_partitioned_with_ghosts(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering const&                                              node_ordering
    ) {
        interface_nodes.clear();

        auto edge_locator = G.build_ghost_edge_locator();

        tbb::task_arena            arena(conf_.num_threads);
        tbb::blocked_range<EdgeId> edge_ids(EdgeId{0}, G.local_edge_count_with_ghost_edges(), conf_.grainsize);

        std::vector<std::atomic<bool>> is_interface;
        if constexpr (!fast_is_interface<NodeOrdering>) {
            is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
        }
        tbb::combinable<LocalPhaseCounters> thread_counters;
        auto                                body = [&](auto const& edge_id_range) {
            auto& counters              = thread_counters.local();
            auto& local_interface_nodes = interface_nodes.local();

            auto tail_idx        = edge_locator.edge_tail_idx(edge_id_range.begin());
            auto tail            = edge_locator.get_node(tail_idx);
            auto tail_first_edge = edge_locator.first_edge_id_for_idx(tail_idx);
            auto tail_last_edge  = edge_locator.last_edge_id_for_idx(tail_idx);
            auto degree          = [this](RankEncodedNodeId node) {
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
                        counters.skipped_nodes++;
                    }
                    tail_idx++;
                    tail_first_edge = edge_locator.first_edge_id_for_idx(tail_idx);
                    edge_id         = tail_first_edge;
                    if (edge_id >= edge_id_range.end()) {
                        return;
                    }
                    tail           = edge_locator.get_node(tail_idx);
                    tail_last_edge = edge_locator.last_edge_id_for_idx(tail_idx);
                }
                KASSERT(tail_first_edge <= edge_id);
                KASSERT(edge_id < tail_last_edge);
                auto v = tail;
                auto u = edge_locator.get_edge_head(edge_id);
                if constexpr (fast_is_interface<NodeOrdering>) {
                    // the first thread to examine this node checks whether it is an interface node
                    if (tail.rank() == rank_ && edge_id == tail_first_edge
                        && G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(tail)) {
                        local_interface_nodes.emplace_back(tail);
                    }
                } else if (u.rank() != rank_ && !is_interface[tail_idx].exchange(true)) {
                    local_interface_nodes.emplace_back(tail);
                }
                intersect_local_edge(v, u, counters, emit, node_ordering);
            }
        };

        arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::auto_partitioner{}); });
        thread_counters.combine(std::plus<>{}).report();
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_edge_partitioned(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering const&                                              node_ordering
    ) {
        interface_nodes.clear();

        auto        edge_locator = G.build_edge_locator();
        auto const& node_indexer = G.node_indexer();

        tbb::task_arena            arena(conf_.num_threads);
        tbb::blocked_range<EdgeId> edge_ids(EdgeId{0}, G.local_edge_count(), conf_.grainsize);

        std::vector<std::atomic<bool>> is_interface;
        if constexpr (!fast_is_interface<NodeOrdering>) {
            is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
        }
        tbb::combinable<LocalPhaseCounters> thread_counters;
        auto                                body = [&](auto const& edge_id_range) {
            auto& counters              = thread_counters.local();
            auto& local_interface_nodes = interface_nodes.local();

            auto tail_idx        = edge_locator.edge_tail_idx(edge_id_range.begin());
            auto tail            = node_indexer.get_node(tail_idx);
            auto tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
            auto tail_last_edge  = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
            for (auto edge_id = std::max(edge_id_range.begin(), tail_first_edge); edge_id < edge_id_range.end();
                 edge_id++) {
                while (edge_id >= tail_last_edge || (conf_.pseudo2core && G.outdegree(tail) < 2)) {
                    if (conf_.pseudo2core && G.outdegree(tail) < 2) {
                        counters.skipped_nodes++;
                    }
                    tail_idx++;
                    tail_first_edge = edge_locator.template first_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                    edge_id         = tail_first_edge;
                    if (edge_id >= edge_id_range.end()) {
                        return;
                    }
                    tail           = node_indexer.get_node(tail_idx);
                    tail_last_edge = edge_locator.template last_edge_id_for_idx<AdjacencyType::out>(tail_idx);
                }
                KASSERT(tail_first_edge <= edge_id);
                KASSERT(edge_id < tail_last_edge);
                auto v = tail;
                auto u = edge_locator.get_edge_head(edge_id);
                if constexpr (fast_is_interface<NodeOrdering>) {
                    // the first thread to examine this node checks whether it is an interface node
                    if (tail.rank() == rank_ && edge_id == tail_first_edge
                        && G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(tail)) {
                        local_interface_nodes.emplace_back(tail);
                    }
                } else if (u.rank() != rank_ && !is_interface[tail_idx].exchange(true)) {
                    local_interface_nodes.emplace_back(tail);
                }
                if (u.rank() != rank_) {
                    continue;
                }
                intersect_local_edge(v, u, counters, emit, node_ordering);
            }
        };

        arena.execute([&] { tbb::parallel_for(edge_ids, body, tbb::auto_partitioner{}); });
        thread_counters.combine(std::plus<>{}).report();
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering const&                                              node_ordering
    ) {
        interface_nodes.clear();
        auto                                nodes = G.local_nodes();
        tbb::combinable<LocalPhaseCounters> thread_counters;

        auto partition_neighborhood_2d = [this](RankEncodedNodeId node) {
            if (conf_.local_degree_of_parallelism > 2) {
                return true;
            } else if (conf_.local_degree_of_parallelism > 1) {
                return G.outdegree(node) > high_degree_threshold_;
            } else {
                return false;
            }
        };

        auto handle_neighbor_range = [&](RankEncodedNodeId v, tbb::blocked_range<size_t> const& neighbor_range) {
            auto& counters = thread_counters.local();
            for (size_t neighbor_index = neighbor_range.begin(); neighbor_index != neighbor_range.end();
                 neighbor_index++) {
                auto u = *(G.out_adj(v).neighbors().begin() + neighbor_index);
                if (u.rank() != rank_) {
                    if constexpr (local_first<NodeOrdering>) {
                        break;
                    } else {
                        continue;
                    }
                }
                intersect_local_edge(v, u, counters, emit, node_ordering);
            }
        };
        auto node_loop_body = [&](RankEncodedNodeId const& v) {
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                thread_counters.local().skipped_nodes++;
                return;
            }
            if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                interface_nodes.local().emplace_back(v);
            }
            auto const all_neighbors = tbb::blocked_range(size_t{0}, static_cast<size_t>(G.outdegree(v)));
            if (partition_neighborhood_2d(v)) {
                tbb::parallel_for(all_neighbors, [&handle_neighbor_range, v](auto const& neighbor_range) {
                    handle_neighbor_range(v, neighbor_range);
                });
            } else {
                handle_neighbor_range(v, all_neighbors);
            }
        };
        tbb::task_arena arena(conf_.num_threads);
        arena.execute([&] {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
                for (auto v: r) {
                    node_loop_body(v);
                }
            });
        });
        thread_counters.combine(std::plus<>{}).report();
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel_with_ghosts(
        TriangleFunc                                                     emit,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering const&                                              node_ordering
    ) {
        interface_nodes.clear();
        tbb::combinable<LocalPhaseCounters> thread_counters;

        auto degree = [this](RankEncodedNodeId node) {
            return node.rank() == rank_ ? G.outdegree(node) : G.ghost_degree(node);
        };
        auto partition_neighborhood_2d = [&](RankEncodedNodeId node) {
            if (conf_.local_degree_of_parallelism > 2) {
                return true;
            } else if (conf_.local_degree_of_parallelism > 1) {
                return degree(node) > high_degree_threshold_;
            } else {
                return false;
            }
        };
        std::vector<std::atomic<bool>> is_interface;
        if constexpr (!fast_is_interface<NodeOrdering>) {
            is_interface = std::vector<std::atomic<bool>>(G.local_node_count());
        }

        auto handle_neighbor_range = [&](RankEncodedNodeId v, tbb::blocked_range<size_t> const& neighbor_range) {
            auto& counters = thread_counters.local();
            for (size_t neighbor_index = neighbor_range.begin(); neighbor_index != neighbor_range.end();
                 neighbor_index++) {
                auto u = *(neighbors_with_ghosts(v).begin() + neighbor_index);
                if constexpr (!fast_is_interface<NodeOrdering>) {
                    if (u.rank() != rank_ && !is_interface[G.to_local_idx(v)].exchange(true)) {
                        interface_nodes.local().emplace_back(v);
                    }
                }
                intersect_local_edge(v, u, counters, emit, node_ordering);
            }
        };
        auto node_loop_body = [&](RankEncodedNodeId const& v) {
            Degree const v_degree = degree(v);
            if (conf_.pseudo2core && v_degree < 2) {
                thread_counters.local().skipped_nodes++;
                return;
            }
            if constexpr (fast_is_interface<NodeOrdering>) {
                if (v.rank() == rank_ && G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                    interface_nodes.local().emplace_back(v);
                }
            }
            auto const all_neighbors = tbb::blocked_range(size_t{0}, static_cast<size_t>(v_degree));
            if (partition_neighborhood_2d(v)) {
                tbb::parallel_for(all_neighbors, [&handle_neighbor_range, v](auto const& neighbor_range) {
                    handle_neighbor_range(v, neighbor_range);
                });
            } else {
                handle_neighbor_range(v, all_neighbors);
            }
        };
        // Local nodes followed by ghosts, addressed by index (C++20 has no random-access concat view).
        auto const local_nodes = G.local_nodes();
        auto const ghosts      = G.ghosts();
        auto const local_count = static_cast<size_t>(std::ranges::size(local_nodes));
        auto const total_count = local_count + static_cast<size_t>(std::ranges::size(ghosts));
        auto       node_at     = [&](size_t i) { return i < local_count ? local_nodes[i] : ghosts[i - local_count]; };
        tbb::task_arena arena(conf_.num_threads);
        arena.execute([&] {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, total_count), [&](auto const& r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    node_loop_body(node_at(i));
                }
            });
        });
        thread_counters.combine(std::plus<>{}).report();
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
        tbb::concurrent_queue<std::function<void()>> high_priority_tasks;

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
                    PEID u_rank = u.rank();
                    assert(u_rank != rank_);
                    if (last_proc_[G.to_local_idx(v)] != u_rank) {
                        SPDLOG_DEBUG("Send N({}) to {}", v, u_rank);
                        enqueue_for_sending_async(queue, v, u_rank, pool, write_jobs, emit);
                    }
                }
            }
            nodes_queued--;
        };
        nodes_queued = std::distance(interface_nodes_begin, interface_nodes_end);
        tbb::enumerable_thread_specific<size_t> priorized_nodes = 0;
        pool.submit_work([&] {
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
                        pool.submit_work([v = v, task] { task(v); });
                    }
                );
            } else {
                std::for_each(interface_nodes_begin, interface_nodes_end, [&](const RankEncodedNodeId& v) {
                    pool.submit_work([v = v, task] { task(v); });
                });
            }
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
        };
        pool.run_and_wait(master_task);
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

    template <typename MessageQueue, typename TriangleFunc>
    void enqueue_for_sending(
        MessageQueue&                  queue,
        RankEncodedNodeId              v,
        PEID                           u_rank,
        TriangleFunc                   emit [[maybe_unused]]
    ) {
        std::vector<RankEncodedNodeId> buffer;
        buffer.emplace_back(v);
        for (RankEncodedEdge e: G.out_adj(v).edges()) {
            assert(conf_.algorithm == Algorithm::patric || e.head.rank() != rank_);
            buffer.emplace_back(e.head);
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
        pool.submit_work([&queue, &write_jobs, this, v, u_rank]() {
            std::vector<RankEncodedNodeId> buffer;
            buffer.emplace_back(v);
            SPDLOG_DEBUG("{}", u_rank);
            for (RankEncodedNodeId node: G.out_adj(v).neighbors()) {
                assert(conf_.algorithm == Algorithm::patric || node.rank() != rank_);
                buffer.emplace_back(node);
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
        auto for_each_local_receiver = [&](auto on_node) {
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
    }
    GraphType&                                                      G;
    const Config&                                                   conf_;
    PEID                                                            rank_;
    PEID                                                            size_;
    std::vector<PEID>                                               last_proc_;
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
