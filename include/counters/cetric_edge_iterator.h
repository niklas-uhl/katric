//
// Created by Tim Niklas Uhl on 22.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H
#define PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H

#include <config.h>
#include <datastructures/graph_definitions.h>
#include <omp.h>
#include <statistics.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_group.h>
#include <timer.h>
#include <thread>
#include <tlx/meta/has_member.hpp>
#include <type_traits>
#include "concurrent_buffered_queue.h"
#include "datastructures/distributed/distributed_graph.h"
#include "indirect_message_queue.h"
#include "tlx/meta/has_method.hpp"
#include "util.h"

namespace cetric {
using namespace graph;
static const NodeId sentinel_node = -1;

TLX_MAKE_HAS_METHOD(grow_by);

template <typename VectorType>
static constexpr bool has_grow_by_v =
    has_method_grow_by<VectorType, typename VectorType::iterator(typename VectorType::size_type)>::value;
static_assert(has_grow_by_v<tbb::concurrent_vector<NodeId>>);

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
    void operator()(VectorType<NodeId>& buffer, MessageFunc&& on_message, PEID sender) {
        std::vector<NodeId>::iterator slice_begin = buffer.begin();
        while (slice_begin != buffer.end()) {
            auto slice_end = std::find(slice_begin, buffer.end(), sentinel_node);
            on_message(slice_begin, slice_end, sender);
            slice_begin = slice_end + 1;
        }
    }
};
struct OldCommBase {
protected:
    OldCommBase(const Config& conf, PEID rank, PEID size)
        : comm_(conf.buffer_threshold,
                MPI_NODE,
                rank,
                size,
                as_int(MessageTag::Neighborhood),
                conf.empty_pending_buffers_on_overflow) {}
    BufferedCommunicator<NodeId> comm_;
};
template <template <typename, typename, typename> class MessageQueueType>
struct MessageQueueBase {
protected:
    MessageQueueBase(const Config&, PEID, PEID) : queue_(Merger{}, Splitter{}) {}
    MessageQueueType<NodeId, Merger, Splitter> queue_;
};
enum class InterfaceType { comm, queue };
struct BufferedCommunicatorPolicy {
    static constexpr InterfaceType interface = InterfaceType::comm;
};
struct MessageQueuePolicy {
    static constexpr InterfaceType interface = InterfaceType::queue;
};
struct GridPolicy {
    static constexpr InterfaceType interface = InterfaceType::queue;
};
template <class CommunicationPolicy>
using commnicator_base =
    std::conditional_t<std::is_same_v<CommunicationPolicy, BufferedCommunicatorPolicy>,
                       OldCommBase,
                       std::conditional_t<std::is_same_v<CommunicationPolicy, MessageQueuePolicy>,
                                          MessageQueueBase<message_queue::ConcurrentBufferedMessageQueue>,
                                          MessageQueueBase<message_queue::ConcurrentBufferedMessageQueue>>>;
template <class GraphType, class CommunicationPolicy>
class CetricEdgeIterator : commnicator_base<CommunicationPolicy> {
public:
    using base_type = commnicator_base<CommunicationPolicy>;
    CetricEdgeIterator(GraphType& G,
                       const Config& conf,
                       PEID rank,
                       PEID size,
                       CommunicationPolicy&& = CommunicationPolicy{})
        : base_type(conf, rank, size),
          G(G),
          conf_(conf),
          rank_(rank),
          size_(size),
          last_proc_(G.local_node_count(), -1),
          is_v_neighbor_(G.local_node_count() + G.ghost_count(), false),
          interface_nodes_(),
          pe_min_degree() {
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
        if constexpr (std::is_same_v<CommunicationPolicy, MessageQueuePolicy>) {
            this->queue_.set_threshold(10);
            // this->queue_.set_threshold(G.local_node_count());
        }
    }

    template <typename TriangleFunc>
    inline void run_local(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        run_local(emit, stats, interface_nodes_);
    }

    template <typename TriangleFunc>
    inline void run_local(TriangleFunc emit,
                          cetric::profiling::Statistics& stats,
                          tbb::concurrent_vector<NodeId>& interface_nodes) {
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
            // #pragma omp parallel for
            //             for (Edge edge : G.out_edges(v)) {
            tbb::parallel_for_each(G.out_edges(v), [&](Edge edge) {
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
                // #pragma omp parallel for
                //                 for (NodeId node : G.local_and_ghost_nodes()) {
                //                     find_intersections(node);
                //                 }
                tbb::parallel_for_each(G.local_and_ghost_nodes(), find_intersections);
                break;
            case Algorithm::Patric: {
                // #pragma omp parallel for
                //                 for (NodeId node : G.local_nodes()) {
                //                     find_intersections(node);
                //                 }
                tbb::parallel_for_each(G.local_nodes(), find_intersections);
                break;
            }
        }
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename TriangleFunc>
    inline void run_distributed(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        run_distributed(emit, stats, interface_nodes_);
    }

    template <typename TriangleFunc>
    inline void run_distributed(TriangleFunc emit,
                                cetric::profiling::Statistics& stats,
                                tbb::concurrent_vector<NodeId>& interface_nodes) {
        cetric::profiling::Timer phase_time;
        bool finished = false;
        tbb::task_group tg;
        tg.run([&]() {
            // std::thread message_aggregation([&]() {
            //#pragma omp parallel for num_threads(conf_.num_threads)
            tbb::parallel_for_each(interface_nodes, [&](NodeId v) {
                // for (NodeId v : interface_nodes) {
                // atomic_debug(omp_get_num_threads());
                // atomic_debug(tbb::this_task_arena::max_concurrency());
                // atomic_debug(tbb::this_task_arena::current_thread_index());
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
                                enqueue_for_sending(v, u_rank, emit, stats);
                            }
                        }
                    });
                }
                // timer.start("Communication");
                // if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
                //     // this->queue_.poll([&](auto begin, auto end, PEID sender [[maybe_unused]]) {
                //     //     handle_buffer(begin, end, emit, stats);
                //     // });
                // } else {
                //     this->comm_.check_for_message(
                //         [&](PEID, const std::vector<NodeId>& message) {
                //             handle_buffer(message.begin(), message.end(), emit, stats);
                //         },
                //         stats.local.message_statistics);
                // }
            });
            finished = true;
        });
        // tbb::task_group tg;
        while (!finished) {
            if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
                this->queue_.check_for_overflow_and_flush();
                // atomic_debug(fmt::format("Polling on {}", tbb::this_task_arena::current_thread_index()));
                this->queue_.poll([&](auto begin, auto end, PEID sender [[maybe_unused]]) {
                    // atomic_debug(fmt::format("Got something on {}",
                    // tbb::this_task_arena::current_thread_index()));
                    std::vector<NodeId> buffer{begin, end};
                    // atomic_debug(fmt::format("Delegating {}", buffer));
                    handle_buffer_hybrid(begin, end, tg, emit, stats);
                    // tg.wait();
                });
            } else {
                this->comm_.check_for_message(
                    [&](PEID, const std::vector<NodeId>& message) {
                        handle_buffer(message.begin(), message.end(), emit, stats);
                    },
                    stats.local.message_statistics);
            }
        }
        tg.wait();
        // message_aggregation.join();
        //  atomic_debug("Wait 1");
        //  g.wait();
        if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
            this->queue_.terminate([&](auto begin, auto end, PEID sender [[maybe_unused]]) {
                std::vector<NodeId> buffer{begin, end};
                // atomic_debug(fmt::format("Delegating {}", buffer));
                handle_buffer_hybrid(begin, end, tg, emit, stats);
            });
        } else {
            this->comm_.all_to_all(
                [&](PEID, const std::vector<NodeId>& message) {
                    handle_buffer(message.begin(), message.end(), emit, stats);
                },
                stats.local.message_statistics, conf_.full_all_to_all);
        }
        tg.wait();
        // g2.wait();
        stats.local.global_phase_time += phase_time.elapsed_time();
        if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
            stats.local.message_statistics.add(this->queue_.stats());
            this->queue_.reset();
        }
        // atomic_debug("Wait 2");
        // g.wait();
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

    template <typename TriangleFunc>
    void enqueue_for_sending(NodeId v,
                             PEID u_rank,
                             // tbb::task_group& tg,
                             TriangleFunc emit [[maybe_unused]],
                             cetric::profiling::Statistics& stats [[maybe_unused]]) {
        // atomic_debug(fmt::format("rank {}", u_rank));
        // atomic_debug(u_rank);
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
        if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
            // atomic_debug(fmt::format("Posting message to {} from thread {}", u_rank,
            //                          tbb::this_task_arena::current_thread_index()));
            this->queue_.post_message(std::move(buffer), u_rank);
        } else {
            if (!buffer.empty()) {
                buffer.emplace_back(sentinel_node);
                this->comm_.add_message(
                    buffer, u_rank,
                    [&](PEID, const std::vector<NodeId>& message) {
                        handle_buffer(message.begin(), message.end(), emit, stats);
                    },
                    stats.local.message_statistics);
            }
        }
        //});
        last_proc_[v] = u_rank;
    }

    template <typename IterType, typename TriangleFunc>
    void handle_buffer(IterType begin, IterType end, TriangleFunc emit, cetric::profiling::Statistics& stats) {
        if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
            NodeId v = *begin;
            process_neighborhood(v, begin + 1, end, emit, stats);
        } else {
            auto current = begin;
            while (current != end) {
                auto chunk_begin = current;
                while (*current != sentinel_node) {
                    current++;
                }
                auto chunk_end = current;
                NodeId v = *chunk_begin;
                chunk_begin++;
                process_neighborhood(v, chunk_begin, chunk_end, emit, stats);
                current++;
            }
        }
    }

    template <typename IterType, typename TriangleFunc>
    void handle_buffer_hybrid(IterType begin,
                              IterType end,
                              tbb::task_group& tg,
                              TriangleFunc emit,
                              cetric::profiling::Statistics& stats) {
        if constexpr (CommunicationPolicy::interface == InterfaceType::queue) {
            NodeId v = *begin;
            std::vector<NodeId> neighborhood{begin + 1, end};
            // atomic_debug(fmt::format("Submit task for {} on thread {}", v,
            // tbb::this_task_arena::current_thread_index()));
            tg.run([&, v, emit, neighborhood = std::move(neighborhood)] {
                // atomic_debug(
                //     fmt::format("Spawn task for {} on thread {}", v, tbb::this_task_arena::current_thread_index()));
                process_neighborhood(v, neighborhood.begin(), neighborhood.end(), emit, stats);
            });
            // tg.wait();
        }
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
    // message_queue::BufferedMessageQueue<NodeId, Merger, Splitter> queue_;
    tbb::concurrent_vector<NodeId> interface_nodes_;
    std::vector<Degree> pe_min_degree;
};

}  // namespace cetric
#endif  // PARALLEL_TRIANGLE_COUNTER_PARALLEL_NODE_ITERATOR_H
