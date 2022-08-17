//
// Created by Tim Niklas Uhl on 22.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H
#define PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>

#include <boost/iterator/function_output_iterator.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <gmpxx.h>
#include <message-queue/buffered_queue.h>
#include <mpi.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_group.h>
#include <tlx/meta/has_member.hpp>
#include <tlx/meta/has_method.hpp>

#include "cetric/concurrent_buffered_queue.h"
#include "cetric/config.h"
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

namespace cetric {
using namespace graph;

namespace node_ordering {
struct id {
    explicit id(){};
    inline bool operator()(RankEncodedNodeId const& lhs, RankEncodedNodeId const& rhs) const {
        // return lhs < rhs;
        return lhs.id() < rhs.id();
    }

private:
};
struct id_outward {
    explicit id_outward(PEID rank) : rank_(rank) {}
    inline bool operator()(RankEncodedNodeId const& lhs, RankEncodedNodeId const& rhs) const {
        return std::pair(lhs.rank() != rank_, lhs.id()) < std::pair(rhs.rank() != rank_, rhs.id());
    }

private:
    PEID rank_;
};
template <typename GraphType>
struct degree {
    explicit degree(GraphType const& G, AuxiliaryNodeData<Degree> const& ghost_degree)
        : G_(G),
          ghost_degree_(ghost_degree) {
        // KASSERT(ghost_degree_.size() > 0ul);
    }
    inline bool operator()(RankEncodedNodeId const& lhs, RankEncodedNodeId const& rhs) const {
        auto degree = [this](RankEncodedNodeId const& node) {
            if (node.rank() != G_.rank()) {
                return ghost_degree_[node];
            }
            return G_.degree(node);
        };
        return std::tuple(degree(lhs), lhs.id()) < std::tuple(degree(rhs), rhs.id());
    }

private:
    GraphType const&                 G_;
    AuxiliaryNodeData<Degree> const& ghost_degree_;
};
template <typename GraphType>
struct degree_outward {
    explicit degree_outward(GraphType const& G) : G_(G) {}
    inline bool operator()(RankEncodedNodeId const& lhs, RankEncodedNodeId const& rhs) const {
        auto degree = [this](RankEncodedNodeId const& node) {
            if (node.rank() != G_.rank()) {
                return std::numeric_limits<Degree>::max();
            }
            return G_.degree(node);
        };
        return std::tuple(degree(lhs), lhs.id()) < std::tuple(degree(rhs), rhs.id());
    }

private:
    GraphType const& G_;
};
} // namespace node_ordering

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
            auto insert_position = buffer.grow_by(msg.size() + 1);
            std::copy(msg.begin(), msg.end(), insert_position);
            *(insert_position + msg.size()) = RankEncodedNodeId::sentinel();
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
        EdgeId total_edge_count = G.local_edge_count();
        MPI_Allreduce(MPI_IN_PLACE, &total_edge_count, 1, MPI_NODE, MPI_SUM, MPI_COMM_WORLD);
        high_degree_threshold_ = conf_.high_degree_threshold_scale * sqrt(total_edge_count / 2.0);
    }

    void set_threshold(size_t threshold) {
        threshold_ = threshold;
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local(TriangleFunc emit, cetric::profiling::Statistics& stats, NodeOrdering&& node_ordering) {
        if (conf_.local_parallel && conf_.num_threads > 1) {
            run_local_parallel(emit, stats, interface_nodes_, std::forward<NodeOrdering>(node_ordering));
        } else {
            run_local_sequential(emit, stats, interface_nodes_.local(), std::forward<NodeOrdering>(node_ordering));
        }
    }

    template <typename TriangleFunc, typename NodeOrdering>
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
            if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                interface_nodes.emplace_back(v);
            }
            size_t neighbor_index = 0;
            for (RankEncodedNodeId u: G.out_adj(v).neighbors()) {
                KASSERT(*(G.out_adj(v).neighbors().begin() + neighbor_index) == u);
                if (u.rank() != rank_) {
                    if (conf_.algorithm == Algorithm::Patric) {
                        neighbor_index++;
                        continue;
                    }
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
                auto v_neighbors = G.out_adj(v).neighbors();
                auto u_neighbors = G.out_adj(u).neighbors();
                auto offset      = (conf_.algorithm == Algorithm::Cetric) * neighbor_index;
                std::set_intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    boost::function_output_iterator([&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    }),
                    node_ordering
                );
                neighbor_index++;
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename TriangleFunc, typename NodeOrdering>
    inline void run_local_parallel(
        TriangleFunc                                                     emit,
        cetric::profiling::Statistics&                                   stats,
        tbb::enumerable_thread_specific<std::vector<RankEncodedNodeId>>& interface_nodes,
        NodeOrdering&&                                                   node_ordering
    ) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        auto nodes = G.local_nodes();

        auto partition_neighborhood_2d = [this](RankEncodedNodeId node [[maybe_unused]]) {
            if (conf_.local_degree_of_parallelism > 2) {
                return true;
            } else if (conf_.local_degree_of_parallelism > 1) {
                bool partition = G.outdegree(node) > high_degree_threshold_;
                if (partition) {
                    atomic_debug("High degree node");
                }
                return partition;
            } else {
                return false;
            }
        };

        auto handle_neighbor_range = [&node_ordering,
                                      this,
                                      &emit,
                                      &stats](RankEncodedNodeId v, tbb::blocked_range<size_t> const& neighbor_range) {
            for (size_t neighbor_index = neighbor_range.begin(); neighbor_index != neighbor_range.end();
                 neighbor_index++) {
                auto u = *(G.out_adj(v).neighbors().begin() + neighbor_index);
                KASSERT(*(G.out_adj(v).neighbors().begin() + neighbor_index) == u);
                if (u.rank() != rank_) {
                    if (conf_.algorithm == Algorithm::Patric) {
                        // neighbor_index++;
                        continue;
                    }
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
                auto v_neighbors = G.out_adj(v).neighbors();
                auto u_neighbors = G.out_adj(u).neighbors();
                auto offset      = (conf_.algorithm == Algorithm::Cetric) * neighbor_index;
                std::set_intersection(
                    v_neighbors.begin() + offset,
                    v_neighbors.end(),
                    u_neighbors.begin(),
                    u_neighbors.end(),
                    boost::function_output_iterator([&](RankEncodedNodeId w) {
                        stats.local.local_triangles++;
                        emit(Triangle<RankEncodedNodeId>{v, u, w});
                    }),
                    node_ordering
                );
            }
        };
        std::atomic<size_t> skipped_nodes = 0;
        tbb::parallel_for(
            tbb::blocked_range(nodes.begin(), nodes.end()),
            [this,
             &interface_nodes,
             &node_ordering,
             &emit,
             &stats,
             handle_neighbor_range,
             partition_neighborhood_2d,
             &skipped_nodes](auto const& nodes_r) {
                for (auto v: nodes_r) {
                    if (conf_.pseudo2core && G.outdegree(v) < 2) {
                        skipped_nodes++;
                        continue;
                    }
                    if (G.template is_interface_node_if_sorted_by_rank<AdjacencyType::out>(v)) {
                        interface_nodes.local().emplace_back(v);
                    }
                    auto v_neighbors_size =
                        std::distance(G.out_adj(v).neighbors().begin(), G.out_adj(v).neighbors().end());
                    if (partition_neighborhood_2d(v)) {
                        stats.local.nodes_parallel2d++;
                        tbb::parallel_for(
                            tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)),
                            [&handle_neighbor_range, v = v](auto const& neighbor_range) {
                                handle_neighbor_range(v, neighbor_range);
                            }
                        );
                    } else {
                        handle_neighbor_range(v, tbb::blocked_range(size_t{0}, static_cast<size_t>(v_neighbors_size)));
                    }
                }
            }
        );
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
        auto queue = []() {
            if constexpr (std::is_same_v<CommunicationPolicy, cetric::GridPolicy>) {
                return cetric::make_indirect_queue<RankEncodedNodeId>(Merger{}, Splitter{});
            } else {
                return message_queue::make_buffered_queue<RankEncodedNodeId>(Merger{}, Splitter{});
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
        auto queue = cetric::make_concurrent_buffered_queue<RankEncodedNodeId>(conf_.num_threads, Merger{}, Splitter{});
        queue.set_threshold(threshold_);
        cetric::profiling::Timer phase_time;
        std::atomic<size_t>      nodes_queued = 0;
        std::atomic<size_t>      write_jobs   = 0;
        ThreadPool               pool(conf_.num_threads - 1);
        std::atomic<size_t>      skipped_nodes = 0;
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
                            // atomic_debug(fmt::format("Send N({}) to {}", G.to_global_id(v), u_rank));
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
                // atomic_debug(fmt::format("No more polling, enqueued: {}, done: {}", pool.enqueued(),
                // pool.done()));
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
        RankEncodedNodeId u,
        RankEncodedNodeId v,
        NodeBufferIter    begin,
        NodeBufferIter    end,
        IntersectFunc     on_intersection,
        NodeOrdering&&    node_ordering,
        GhostSet const&   ghosts
    ) {
        // if (conf_.flag_intersection) {
        //     G.for_each_local_out_edge(u_local, [&](RankEncodedEdge uw) {
        //         RankEncodedNodeId w = uw.head.id();
        //         if (is_v_neighbor_[w]) {
        //             on_intersection(w);
        //         }
        //     });
        // } else {
        auto u_neighbors = G.out_adj(u).neighbors();
        if (!ghosts.empty()) {
            auto filtered_neighbors =
                boost::adaptors::filter(boost::make_iterator_range(begin, end), [&ghosts](RankEncodedNodeId node) {
                    return ghosts.find(node) != ghosts.end();
                });
            // atomic_debug(fmt::format("intersecting {} and {}", u_neighbors, filtered_neighbors));
            std::set_intersection(
                u_neighbors.begin(),
                u_neighbors.end(),
                filtered_neighbors.begin(),
                filtered_neighbors.end(),
                boost::make_function_output_iterator(on_intersection),
                node_ordering
            );
        } else {
            std::set_intersection(
                u_neighbors.begin(),
                u_neighbors.end(),
                begin,
                end,
                boost::make_function_output_iterator(on_intersection),
                node_ordering
            );
        }
        if (conf_.skip_local_neighborhood && conf_.algorithm == Algorithm::Patric) {
            auto v_neighbors = G.out_adj(v).neighbors();
            std::set_intersection(
                u_neighbors.begin(),
                u_neighbors.end(),
                v_neighbors.begin(),
                v_neighbors.end(),
                boost::make_function_output_iterator(on_intersection),
                node_ordering
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
                if (conf_.num_threads > 1 && conf_.global_parallel && conf_.global_degree_of_parallelism > 2) {
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
                if (conf_.num_threads > 1 && conf_.global_parallel && conf_.global_degree_of_parallelism > 2) {
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
};

} // namespace cetric
#endif // PARALLEL_TRIANGLE_COUNTER_PARALLEL_NODE_ITERATOR_H
