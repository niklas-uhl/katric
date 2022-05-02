//
// Created by Tim Niklas Uhl on 22.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H
#define PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H

#include <config.h>
#include <datastructures/graph_definitions.h>
#include <statistics.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_group.h>
#include <timer.h>
#include <cstddef>
#include <memory>
#include <thread>
#include <tlx/meta/has_member.hpp>
#include <type_traits>
#include "concurrent_buffered_queue.h"
#include "datastructures/distributed/distributed_graph.h"
#include "indirect_message_queue.h"
#include "message-queue/buffered_queue.h"
#include "thread_pool.h"
#include "tlx/meta/has_method.hpp"
#include "util.h"

namespace cetric {
using namespace graph;
static const NodeId sentinel_node = -1;

struct MessageQueuePolicy {};
struct GridPolicy {};

TLX_MAKE_HAS_METHOD(grow_by);

template <typename VectorType>
static constexpr bool has_grow_by_v =
    has_method_grow_by<VectorType, typename VectorType::iterator(typename VectorType::size_type)>::value;
static_assert(has_grow_by_v<tbb::concurrent_vector<NodeId>>);

template <typename T>
class SharedVectorSpan {
public:
    using iterator = typename std::vector<T>::iterator;
    SharedVectorSpan(std::shared_ptr<std::vector<T>> ptr, size_t begin, size_t end)
        : ptr_(std::move(ptr)), begin_(begin), end_(end) {}

    iterator begin() const {
        return ptr_->begin() + begin_;
    }

    iterator end() const {
        return ptr_->begin() + end_;
    }

private:
    std::shared_ptr<std::vector<T>> ptr_;
    size_t begin_;
    size_t end_;
};

struct Merger {
    template <template <typename> typename VectorType>
    size_t operator()(VectorType<NodeId>& buffer, std::vector<NodeId> msg, int tag [[maybe_unused]]) {
        if constexpr (has_grow_by_v<VectorType<NodeId>>) {
            auto insert_position = buffer.grow_by(msg.size() + 1);
            std::copy(msg.begin(), msg.end(), insert_position);
            *(insert_position + msg.size()) = sentinel_node;
        } else {
            for (auto elem : msg) {
                buffer.push_back(elem);
            }
            buffer.push_back(sentinel_node);
        }
        return msg.size() + 1;
    }
};

struct Splitter {
    template <typename MessageFunc, template <typename> typename VectorType>
    void operator()(VectorType<NodeId> buffer, MessageFunc&& on_message, PEID sender) {
        auto buffer_ptr = std::make_shared<VectorType<NodeId>>(std::move(buffer));
        std::vector<NodeId>::iterator slice_begin = buffer_ptr->begin();
        while (slice_begin != buffer_ptr->end()) {
            auto slice_end = std::find(slice_begin, buffer_ptr->end(), sentinel_node);
            on_message(SharedVectorSpan(buffer_ptr, std::distance(buffer_ptr->begin(), slice_begin),
                                        std::distance(buffer_ptr->begin(), slice_end)),
                       sender);
            slice_begin = slice_end + 1;
        }
    }
};
template <class GraphType, class CommunicationPolicy>
class CetricEdgeIterator {
public:
    CetricEdgeIterator(GraphType& G,
                       const Config& conf,
                       PEID rank,
                       PEID size,
                       CommunicationPolicy&& = CommunicationPolicy{})
        : G(G),
          conf_(conf),
          rank_(rank),
          size_(size),
          last_proc_(G.local_node_count(), -1),
          is_v_neighbor_(G.local_node_count() + G.ghost_count(), false),
          interface_nodes_(),
          pe_min_degree(),
          threshold_() {
        if constexpr (payload_has_degree<typename GraphType::payload_type>::value) {
            if (conf_.degree_filtering) {
                pe_min_degree.resize(size);
                G.for_each_ghost_node([&](NodeId node) {
                    auto ghost_data = G.get_ghost_data(node);
                    pe_min_degree[ghost_data.rank] =
                        std::min(pe_min_degree[ghost_data.rank], ghost_data.payload.degree);
                });
            }
        }
        switch(conf.threshold) {
            case Threshold::local_nodes:
                threshold_ = conf.threshold_scale * G.local_node_count();
                break;
            case Threshold::local_edges:
                threshold_ = conf.threshold_scale * G.local_edge_count();
                break;
            case Threshold::none:
                threshold_ = std::numeric_limits<size_t>::max();
                break;
        }
    }

    template <typename TriangleFunc>
    inline void run_local(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        if (conf_.local_parallel && conf_.num_threads > 1) {
            run_local_parallel(emit, stats, interface_nodes_);
        } else {
            run_local_sequential(emit, stats, interface_nodes_.local());
        }
    }

    template <typename TriangleFunc>
    inline void run_local_sequential(TriangleFunc emit,
                          cetric::profiling::Statistics& stats,
                          std::vector<NodeId>& interface_nodes) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        // std::vector<NodeId> interface_nodes;
        auto find_intersections = [&](NodeId v) {
            if (conf_.pseudo2core && G.local_outdegree(v) < 2) {
                stats.local.skipped_nodes++;
                return;
            }
            if (G.is_local_from_local(v) && G.get_local_data(v).is_interface) {
                interface_nodes.emplace_back(v);
            }
            pre_intersection(v);
            G.for_each_local_out_edge(v, [&](Edge edge) {
                NodeId u = edge.head;
                auto on_intersection = [&](NodeId node) {
                    stats.local.local_triangles++;
                    emit(Triangle{G.to_global_id(v), G.to_global_id(u), G.to_global_id(node)});
                };
                if (conf_.algorithm == Algorithm::Patric && G.is_ghost(u)) {
                    return;
                }
                intersect(v, u, on_intersection);
            });
            post_intersection(v);
        };
        switch (conf_.algorithm) {
            case Algorithm::Cetric:
                G.for_each_local_node_and_ghost(find_intersections);
                break;
            case Algorithm::Patric:
                G.for_each_local_node(find_intersections);
                break;
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename TriangleFunc>
    inline void run_local_parallel(TriangleFunc emit,
                                   cetric::profiling::Statistics& stats,
                                   tbb::enumerable_thread_specific<std::vector<NodeId>>& interface_nodes) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        // std::vector<NodeId> interface_nodes;
        auto find_intersections = [&](NodeId v) {
            if (conf_.pseudo2core && G.local_outdegree(v) < 2) {
                stats.local.skipped_nodes++;
                return;
            }
            if (G.is_local_from_local(v) && G.get_local_data(v).is_interface) {
                interface_nodes.local().emplace_back(v);
            }
            pre_intersection(v);
            G.parallel_for_each_local_out_edge(v, [&stats, this, emit](Edge edge) {
                NodeId v = edge.tail;
                NodeId u = edge.head;
                auto on_intersection = [&](NodeId node) {
                    stats.local.local_triangles++;
                    emit(Triangle{G.to_global_id(v), G.to_global_id(u), G.to_global_id(node)});
                };
                if (conf_.algorithm != Algorithm::Patric || !G.is_ghost(u)) {
                    intersect(v, u, on_intersection);
                }
                // }
            });
            post_intersection(v);
        };
        switch (conf_.algorithm) {
            case Algorithm::Cetric:
                G.parallel_for_each_local_node_and_ghost(find_intersections);
                break;
            case Algorithm::Patric: {
                G.parallel_for_each_local_node(find_intersections);
                tbb::parallel_for_each(G.local_nodes(), find_intersections);
                break;
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename TriangleFunc>
    inline void run_distributed(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        auto all_interface_nodes = tbb::flatten2d(interface_nodes_);
        if (conf_.global_parallel && conf_.num_threads > 1) {
            run_distributed_parallel(emit, stats, all_interface_nodes.begin(), all_interface_nodes.end());
        } else {
            run_distributed_sequential(emit, stats, all_interface_nodes.begin(), all_interface_nodes.end());
        }
    }

    template <typename TriangleFunc, typename NodeIterator>
    inline void run_distributed_sequential(TriangleFunc emit,
                                           cetric::profiling::Statistics& stats,
                                           NodeIterator interface_nodes_begin,
                                           NodeIterator interface_nodes_end) {
        auto queue = message_queue::make_buffered_queue<NodeId>(Merger{}, Splitter{});
        queue.set_threshold(threshold_);
        cetric::profiling::Timer phase_time;
        for (auto current = interface_nodes_begin; current != interface_nodes_end; current++) {
            NodeId v = *current;
            // iterate over neighborhood and delegate to other PEs if necessary
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                stats.local.skipped_nodes++;
            } else {
                G.for_each_local_out_edge(v, [&](Edge edge) {
                    NodeId u = edge.head;
                    if (conf_.algorithm == Algorithm::Cetric || G.is_ghost(u)) {
                        assert(G.is_local_from_local(v));
                        assert(G.is_local(G.to_global_id(v)));
                        assert(!G.is_local(G.to_global_id(u)));
                        assert(!G.is_local_from_local(u));
                        PEID u_rank = G.get_ghost_data(u).rank;
                        assert(u_rank != rank_);
                        if (last_proc_[v] != u_rank) {
                            enqueue_for_sending(queue, v, u_rank, emit, stats);
                        }
                    }
                });
            }
            // timer.start("Communication");
            queue.poll([&](SharedVectorSpan<NodeId> span, PEID sender [[maybe_unused]]) {
                handle_buffer(span.begin(), span.end(), emit, stats);
            });
        }
        queue.terminate([&](SharedVectorSpan<NodeId> span, PEID sender [[maybe_unused]]) {
            handle_buffer(span.begin(), span.end(), emit, stats);
        });
        stats.local.global_phase_time += phase_time.elapsed_time();
        stats.local.message_statistics.add(queue.stats());
        queue.reset();
    }

    template <typename TriangleFunc, typename NodeIterator>
    inline void run_distributed_parallel(TriangleFunc emit,
                                         cetric::profiling::Statistics& stats,
                                         NodeIterator interface_nodes_begin,
                                         NodeIterator interface_nodes_end) {
        assert(conf_.num_threads > 1);
        auto queue = message_queue::make_concurrent_buffered_queue<NodeId>(conf_.num_threads, Merger{}, Splitter{});
        queue.set_threshold(threshold_);
        cetric::profiling::Timer phase_time;
        std::atomic<size_t> nodes_queued = 0;
        std::atomic<size_t> write_jobs = 0;
        ThreadPool pool(conf_.num_threads - 1);
        for (auto current = interface_nodes_begin; current != interface_nodes_end; current++) {
            NodeId v = *current;
            nodes_queued++;
            pool.enqueue([v, this, &stats, emit, &queue, &pool, &nodes_queued, &write_jobs]() {
                if (conf_.pseudo2core && G.outdegree(v) < 2) {
                    stats.local.skipped_nodes++;
                    return;
                }
                G.for_each_local_out_edge(v, [&](Edge edge) {
                    NodeId u = edge.head;
                    if (conf_.algorithm == Algorithm::Cetric || G.is_ghost(u)) {
                        assert(G.is_local_from_local(v));
                        assert(G.is_local(G.to_global_id(v)));
                        assert(!G.is_local(G.to_global_id(u)));
                        assert(!G.is_local_from_local(u));
                        PEID u_rank = G.get_ghost_data(u).rank;
                        assert(u_rank != rank_);
                        if (last_proc_[v] != u_rank) {
                            // atomic_debug(fmt::format("Send N({}) to {}", G.to_global_id(v), u_rank));
                            enqueue_for_sending_async(queue, v, u_rank, pool, write_jobs, emit, stats);
                        }
                    }
                });
                nodes_queued--;
            });
        }
        while (true) {
            if (write_jobs == 0 && nodes_queued == 0) {
                //atomic_debug(fmt::format("No more polling, enqueued: {}, done: {}", pool.enqueued(), pool.done()));
                break;
            }
            queue.check_for_overflow_and_flush();
            queue.poll([&](SharedVectorSpan<NodeId> span, PEID sender [[maybe_unused]]) {
                handle_buffer_hybrid(std::move(span), pool, emit, stats);
            });
        }
        queue.terminate([&](SharedVectorSpan<NodeId> span, PEID sender [[maybe_unused]]) {
            handle_buffer_hybrid(std::move(span), pool, emit, stats);
        });
        pool.loop_until_empty();
        pool.terminate();
        //atomic_debug(fmt::format("Finished Pool, enqueued: {}, done: {}", pool.enqueued(), pool.done()));
        stats.local.global_phase_time += phase_time.elapsed_time();
        stats.local.message_statistics.add(queue.stats());
        queue.reset();
    }

    template <typename TriangleFunc>
    inline void run(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        run_local(emit, stats);
        cetric::profiling::Timer phase_time;
        G.remove_internal_edges();
        stats.local.contraction_time += phase_time.elapsed_time();
        run_distributed(emit, stats);
    }

    inline size_t get_triangle_count(cetric::profiling::Statistics& stats) {
        size_t triangle_count = 0;
        run([&](Triangle) { triangle_count++; }, stats);
        return triangle_count;
    }

private:
    void pre_intersection(NodeId v) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(v, [&](Edge edge) {
                assert(is_v_neighbor_[edge.head] == false);
                is_v_neighbor_[edge.head] = true;
            });
        }
    }

    void post_intersection(NodeId v) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(v, [&](Edge edge) {
                assert(is_v_neighbor_[edge.head] == true);
                is_v_neighbor_[edge.head] = false;
            });
        }
    }

    template <typename IntersectFunc>
    void intersect(NodeId v, NodeId u, IntersectFunc on_intersection) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(u, [&](Edge uw) {
                NodeId w = uw.head;
                if (is_v_neighbor_[w]) {
                    on_intersection(w);
                }
            });
        } else {
            G.intersect_neighborhoods(v, u, on_intersection);
        }
    }

    bool send_neighbor(NodeId u, PEID rank) {
        if (conf_.skip_local_neighborhood) {
            if (conf_.algorithm == Algorithm::Cetric) {
                // we omit all vertices located on the receiving PE (all vertices are ghosts)
                return G.get_ghost_data(u).rank != rank;
            } else {
                // we send all vertices, but omit those already located on the receiving PE
                return !G.is_ghost(u) || G.get_ghost_data(u).rank != rank;
            }
        } else {
            return true;
        }
    }

    template <typename MessageQueue, typename TriangleFunc>
    void enqueue_for_sending(MessageQueue& queue,
                             NodeId v,
                             PEID u_rank,
                             TriangleFunc emit [[maybe_unused]],
                             cetric::profiling::Statistics& stats [[maybe_unused]]) {
        std::vector<NodeId> buffer;
        buffer.emplace_back(G.to_global_id(v));
        // size_t send_count = 0;
        G.for_each_local_out_edge(v, [&](Edge e) {
            assert(conf_.algorithm == Algorithm::Patric || G.is_ghost(e.head));
            using payload_type = typename GraphType::payload_type;
            if constexpr (std::is_convertible_v<decltype(payload_type{}.degree), Degree>) {
                if (conf_.degree_filtering) {
                    const auto& ghost_data = G.get_ghost_data(e.head);
                    if (ghost_data.payload.degree < pe_min_degree[u_rank]) {
                        return;
                    }
                }
            }
            if (send_neighbor(e.head, u_rank)) {
                buffer.emplace_back(G.to_global_id(e.head));
            }
        });
        queue.post_message(std::move(buffer), u_rank);
        last_proc_[v] = u_rank;
    }

    template <typename MessageQueue, typename TriangleFunc>
    void enqueue_for_sending_async(MessageQueue& queue,
                                   NodeId v,
                                   PEID u_rank,
                                   ThreadPool& thread_pool [[maybe_unused]],
                                   std::atomic<size_t>& write_jobs,
                                   TriangleFunc emit [[maybe_unused]],
                                   cetric::profiling::Statistics& stats [[maybe_unused]]) {
        // atomic_debug(fmt::format("rank {}", u_rank));
        // atomic_debug(u_rank);
        write_jobs++;
        thread_pool.enqueue([&queue, &stats, &write_jobs, emit, this, v, u_rank] {
            // tg.run([&stats, emit, this, v, u_rank]() {
            std::vector<NodeId> buffer;
            buffer.emplace_back(G.to_global_id(v));
            // size_t send_count = 0;
            // atomic_debug(u_rank);
            G.for_each_local_out_edge(v, [&](Edge e) {
                assert(conf_.algorithm == Algorithm::Patric || G.is_ghost(e.head));
                using payload_type = typename GraphType::payload_type;
                if constexpr (std::is_convertible_v<decltype(payload_type{}.degree), Degree>) {
                    if (conf_.degree_filtering) {
                        const auto& ghost_data = G.get_ghost_data(e.head);
                        if (ghost_data.payload.degree < pe_min_degree[u_rank]) {
                            return;
                        }
                    }
                }
                if (send_neighbor(e.head, u_rank)) {
                    buffer.emplace_back(G.to_global_id(e.head));
                }
            });
            // atomic_debug(u_rank);
            queue.post_message(std::move(buffer), u_rank);
            write_jobs--;
        });
        last_proc_[v] = u_rank;
    }

    template <typename IterType, typename TriangleFunc>
    void handle_buffer(IterType begin, IterType end, TriangleFunc emit, cetric::profiling::Statistics& stats) {
        NodeId v = *begin;
        process_neighborhood(v, begin + 1, end, emit, stats);
    }

    template <typename TriangleFunc>
    void handle_buffer_hybrid(SharedVectorSpan<NodeId> span,
                              ThreadPool& thread_pool,
                              TriangleFunc emit,
                              cetric::profiling::Statistics& stats) {
        thread_pool.enqueue(
            [this, emit, span = std::move(span), &stats] {
                NodeId v = *span.begin();
                // atomic_debug(
                //     fmt::format("Spawn task for {} on thread {}", v,
                //     tbb::this_task_arena::current_thread_index()));
                process_neighborhood(v, span.begin() + 1, span.end(), emit, stats);
            },
            ThreadPool::Priority::high);
    }
    template <typename NodeBufferIter>
    void distributed_pre_intersect(NodeId v, NodeBufferIter begin, NodeBufferIter end) {
        if (conf_.flag_intersection) {
            for (auto it = begin; it != end; it++) {
                NodeId node = *it;
                if (G.is_local(node) || G.is_ghost_from_global(node)) {
                    is_v_neighbor_[G.to_local_id(node)] = true;
                }
            }
            // for CETRIC we don't have to consider the local neighbors of v, because these are no ghost
            // vertices
            if (conf_.algorithm == Algorithm::Patric && conf_.skip_local_neighborhood) {
                G.for_each_local_out_edge(G.to_local_id(v), [&](Edge edge) { is_v_neighbor_[edge.head] = true; });
            }
        }
    }

    template <typename NodeBufferIter>
    void distributed_post_intersect(NodeId v, NodeBufferIter begin, NodeBufferIter end) {
        if (conf_.flag_intersection) {
            for (auto it = begin; it != end; it++) {
                NodeId node = *it;
                if (G.is_local(node) || G.is_ghost_from_global(node)) {
                    is_v_neighbor_[G.to_local_id(node)] = false;
                }
            }
            // for CETRIC we don't have to consider the local neighbors of v, because these are no ghost
            // vertices
            if (conf_.algorithm == Algorithm::Patric && conf_.skip_local_neighborhood) {
                G.for_each_local_out_edge(G.to_local_id(v), [&](Edge edge) { is_v_neighbor_[edge.head] = false; });
            }
        }
    }

    template <typename IntersectFunc, typename NodeBufferIter>
    void intersect_from_message(NodeId u_local,
                                NodeId v_local,
                                NodeBufferIter begin,
                                NodeBufferIter end,
                                IntersectFunc on_intersection) {
        if (conf_.flag_intersection) {
            G.for_each_local_out_edge(u_local, [&](Edge uw) {
                NodeId w = uw.head;
                if (is_v_neighbor_[w]) {
                    on_intersection(w);
                }
            });
        } else {
            G.intersect_neighborhoods(u_local, begin, end, [&](NodeId x) { on_intersection(G.to_local_id(x)); });
            if (conf_.skip_local_neighborhood && conf_.algorithm == Algorithm::Patric) {
                G.intersect_neighborhoods(u_local, v_local, on_intersection);
            }
        }
    }

    template <typename TriangleFunc, typename NodeBufferIter>
    void process_neighborhood(NodeId v,
                              NodeBufferIter begin,
                              NodeBufferIter end,
                              TriangleFunc emit,
                              cetric::profiling::Statistics& stats) {
        std::vector<NodeId> buffer{begin, end};
        // atomic_debug(fmt::format("Handling message {}", buffer));
        assert(!G.is_local(v));
        distributed_pre_intersect(v, begin, end);
        auto for_each_local_receiver = [&](auto on_node) {
            if (conf_.skip_local_neighborhood) {
                G.for_each_local_out_edge(G.to_local_id(v), [&](Edge edge) { on_node(edge.head); });
            } else {
                for (auto current = begin; current != end; current++) {
                    NodeId u = *current;
                    if (G.is_local(u)) {
                        on_node(G.to_local_id(u));
                    }
                }
            }
        };
        for_each_local_receiver([&](NodeId u_local) {
            NodeId u = G.to_global_id(u_local);
            assert(G.is_local_from_local(u_local));
            intersect_from_message(u_local, G.to_local_id(v), begin, end, [&](NodeId local_intersection) {
                emit(Triangle{v, u, G.to_global_id(local_intersection)});
                stats.local.type3_triangles++;
            });
        });
        distributed_post_intersect(v, begin, end);
    }
    using NodeBuffer = google::dense_hash_map<PEID, std::vector<NodeId>>;
    GraphType& G;
    const Config& conf_;
    PEID rank_;
    PEID size_;
    std::vector<PEID> last_proc_;
    std::vector<bool> is_v_neighbor_;
    tbb::enumerable_thread_specific<std::vector<NodeId>> interface_nodes_;
    std::vector<Degree> pe_min_degree;
    size_t threshold_;
};

}  // namespace cetric
#endif  // PARALLEL_TRIANGLE_COUNTER_PARALLEL_NODE_ITERATOR_H
