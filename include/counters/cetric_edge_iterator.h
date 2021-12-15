//
// Created by Tim Niklas Uhl on 22.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H
#define PARALLEL_TRIANGLE_COUNTER_PARALLEL_GHOST_NODE_ITERATOR_H

#include <communicator.h>
#include <config.h>
#include <datastructures/graph_definitions.h>
#include <statistics.h>
#include <timer.h>
#include <type_traits>
#include "datastructures/distributed/distributed_graph.h"

namespace cetric {
using namespace graph;
template <class GraphType, bool compress_more, bool use_flags = false>
class CetricEdgeIterator {
public:
    CetricEdgeIterator(GraphType& G, const Config& conf, PEID rank, PEID size)
        : G(G),
          conf_(conf),
          rank_(rank),
          size_(size),
          last_proc_(G.local_node_count(), -1),
          is_v_neighbor_(G.local_node_count() + G.ghost_count(), false),
          comm(conf.buffer_threshold,
               MPI_NODE,
               rank_,
               size_,
               as_int(MessageTag::Neighborhood),
               conf.empty_pending_buffers_on_overflow),
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
    }

    template <typename TriangleFunc>
    inline void run_plain_local(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        run_plain_local(emit, stats, interface_nodes_);
    }

    template <typename TriangleFunc>
    inline void run_plain_local(TriangleFunc emit,
                                cetric::profiling::Statistics& stats,
                                std::vector<NodeId>& interface_nodes) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        // std::vector<NodeId> interface_nodes;
        G.for_each_local_node([&](NodeId v) {
            if (conf_.pseudo2core && G.local_outdegree(v) < 2) {
                stats.local.skipped_nodes++;
                return;
            }
            if (G.get_local_data(v).is_interface) {
                interface_nodes.emplace_back(v);
            }
            if constexpr (use_flags) {
                G.for_each_local_out_edge(v, [&](Edge edge) { is_v_neighbor_[edge.head] = true; });
            }
            G.for_each_local_out_edge(v, [&](Edge edge) {
                NodeId u = edge.head;
                auto on_intersection = [&](NodeId node) {
                    stats.local.local_triangles++;
                    emit(Triangle{G.to_global_id(v), G.to_global_id(u), G.to_global_id(node)});
                };
                if (G.is_local_from_local(u)) {
                    if constexpr (use_flags) {
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
            });
            if constexpr (use_flags) {
                G.for_each_local_out_edge(v, [&](Edge edge) { is_v_neighbor_[edge.head] = false; });
            }
        });
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename TriangleFunc>
    inline void run_local(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        run_local(emit, stats, interface_nodes_);
    }

    template <typename TriangleFunc>
    inline void run_local(TriangleFunc emit,
                          cetric::profiling::Statistics& stats,
                          std::vector<NodeId>& interface_nodes) {
        cetric::profiling::Timer phase_time;
        interface_nodes.clear();
        // std::vector<NodeId> interface_nodes;
        G.for_each_local_node_and_ghost([&](NodeId v) {
            if (conf_.pseudo2core && G.local_outdegree(v) < 2) {
                stats.local.skipped_nodes++;
                return;
            }
            if (G.is_local_from_local(v) && G.get_local_data(v).is_interface) {
                interface_nodes.emplace_back(v);
            }
            if constexpr (use_flags) {
                G.for_each_local_out_edge(v, [&](Edge edge) { is_v_neighbor_[edge.head] = true; });
            }
            G.for_each_local_out_edge(v, [&](Edge edge) {
                NodeId u = edge.head;
                auto on_intersection = [&](NodeId node) {
                    stats.local.local_triangles++;
                    emit(Triangle{G.to_global_id(v), G.to_global_id(u), G.to_global_id(node)});
                };
                if constexpr (use_flags) {
                    G.for_each_local_out_edge(u, [&](Edge uw) {
                        NodeId w = uw.head;
                        if (is_v_neighbor_[w]) {
                            on_intersection(w);
                        }
                    });
                } else {
                    G.intersect_neighborhoods(v, u, on_intersection);
                }
            });
            if constexpr (use_flags) {
                G.for_each_local_out_edge(v, [&](Edge edge) { is_v_neighbor_[edge.head] = false; });
            }
        });
        stats.local.local_phase_time += phase_time.elapsed_time();
    }

    template <typename TriangleFunc>
    inline void run_distributed(TriangleFunc emit, cetric::profiling::Statistics& stats) {
        run_distributed(emit, stats, interface_nodes_);
    }

    template <typename TriangleFunc>
    inline void run_distributed(TriangleFunc emit,
                                cetric::profiling::Statistics& stats,
                                std::vector<NodeId>& interface_nodes) {
        cetric::profiling::Timer phase_time;
        for (NodeId v : interface_nodes) {
            // iterate over neighborhood and delegate to other PEs if necessary
            if (conf_.pseudo2core && G.outdegree(v) < 2) {
                stats.local.skipped_nodes++;
            } else {
                G.for_each_local_out_edge(v, [&](Edge edge) {
                    NodeId u = edge.head;
                    if (G.is_ghost(u)) { // TODO: for CETRIC this is not needed ..
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
            comm.check_for_message(
                [&](PEID, const std::vector<NodeId>& message) { handle_buffer(message, emit, stats); },
                stats.local.message_statistics);
        }
        comm.all_to_all([&](PEID, const std::vector<NodeId>& message) { handle_buffer(message, emit, stats); },
                        stats.local.message_statistics, conf_.full_all_to_all);
        stats.local.global_phase_time += phase_time.elapsed_time();
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
    template <typename TriangleFunc>
    void enqueue_for_sending(NodeId v, PEID u_rank, TriangleFunc emit, cetric::profiling::Statistics& stats) {
        std::vector<NodeId> buffer;
        buffer.emplace_back(G.to_global_id(v));
        // size_t send_count = 0;
        G.for_each_local_out_edge(v, [&](Edge e) {
            assert(conf_.algorithm == "patric" || G.is_ghost(e.head));
            using payload_type = typename GraphType::payload_type;
            if constexpr (std::is_convertible_v<decltype(payload_type{}.degree), Degree>) {
                if (conf_.degree_filtering) {
                    const auto& ghost_data = G.get_ghost_data(e.head);
                    if (ghost_data.payload.degree < pe_min_degree[u_rank]) {
                        return;
                    }
                }
            }
            if constexpr (compress_more) {
                if (conf_.algorithm == "cetric") {
                    if (G.get_ghost_data(e.head).rank != u_rank) {
                        buffer.emplace_back(G.to_global_id(e.head));
                    }
                } else {
                    if (!G.is_ghost(e.head) || G.get_ghost_data(e.head).rank != u_rank) {
                        buffer.emplace_back(G.to_global_id(e.head));
                    }
                }
            } else {
                buffer.emplace_back(G.to_global_id(e.head));
                // send_count++;
            }
        });
        if (!buffer.empty()) {
            buffer.emplace_back(sentinel_node);
            comm.add_message(
                buffer, u_rank, [&](PEID, const std::vector<NodeId>& message) { handle_buffer(message, emit, stats); },
                stats.local.message_statistics);
        }
        last_proc_[v] = u_rank;
    }

    template <typename TriangleFunc>
    void handle_buffer(const std::vector<NodeId>& buffer, TriangleFunc emit, cetric::profiling::Statistics& stats) {
        size_t i = 0;
        while (i < buffer.size()) {
            int begin = i;
            while (buffer[i] != sentinel_node) {
                i++;
            }
            int end = i;
            NodeId v = buffer[begin];
            begin++;
            process_neighborhood(v, buffer.begin() + begin, buffer.begin() + end, emit, stats);
            i++;
        }
    }

    template <typename TriangleFunc, typename NodeBufferIter>
    void process_neighborhood(NodeId v,
                              NodeBufferIter begin,
                              NodeBufferIter end,
                              TriangleFunc emit,
                              cetric::profiling::Statistics& stats) {
        assert(!G.is_local(v));
        if constexpr (use_flags) {
            for (auto it = begin; it != end; it++) {
                NodeId node = *it;
                if (G.is_local(node) || G.is_ghost_from_global(node)) {
                    is_v_neighbor_[G.to_local_id(node)] = true;
                }
            }
            // for CETRIC we don't have to consider the local neighbors of v, because these are no ghost vertices
            if (conf_.algorithm == "patric" && compress_more) {
                G.for_each_local_out_edge(G.to_local_id(v), [&](Edge edge) {
                    is_v_neighbor_[edge.head] = true;
                });
            }
        }
        if constexpr (compress_more) {
            G.for_each_local_out_edge(G.to_local_id(v), [&](Edge edge) {
                NodeId u_local = edge.head;
                NodeId u = G.to_global_id(u_local);
                assert(G.is_local_from_local(u_local));
                if constexpr (use_flags) {
                    G.for_each_local_out_edge(u_local, [&](Edge uw) {
                        NodeId w = uw.head;
                        if (is_v_neighbor_[w]) {
                            emit(Triangle{v, u, G.to_global_id(w)});
                            stats.local.type3_triangles++;
                        }
                    });
                } else {
                    G.intersect_neighborhoods(u_local, begin, end, [&](NodeId x) {
                        emit(Triangle{v, u, x});
                        stats.local.type3_triangles++;
                    });
                    if (conf_.algorithm == "patric") {
                        G.intersect_neighborhoods(G.to_local_id(u), G.to_local_id(v), [&](NodeId x) {
                            emit(Triangle{v, u, G.to_global_id(x)});
                            stats.local.type3_triangles++;
                        });
                    }
                }
            });
        } else {
            for (auto current = begin; current != end; current++) {
                NodeId u = *current;
                if (G.is_local(u)) {
                    if constexpr (use_flags) {
                        G.for_each_local_out_edge(G.to_local_id(u), [&](Edge uw) {
                            NodeId w = uw.head;
                            if (is_v_neighbor_[w]) {
                                emit(Triangle{v, u, G.to_global_id(w)});
                                stats.local.type3_triangles++;
                            }
                        });
                    } else {
                        G.intersect_neighborhoods(G.to_local_id(u), begin, end, [&](NodeId x) {
                            emit(Triangle{v, u, x});
                            stats.local.type3_triangles++;
                        });
                    }
                }
            }
        }
        if constexpr (use_flags) {
            for (auto it = begin; it != end; it++) {
                NodeId node = *it;
                if (G.is_local(node) || G.is_ghost_from_global(node)) {
                    is_v_neighbor_[G.to_local_id(node)] = false;
                }
            }
            // for CETRIC we don't have to consider the local neighbors of v, because these are no ghost vertices
            if (conf_.algorithm == "patric" && compress_more) {
                G.for_each_local_out_edge(G.to_local_id(v), [&](Edge edge) { is_v_neighbor_[edge.head] = false; });
            }
        }
    }

    NodeId sentinel_node = -1;
    using NodeBuffer = google::dense_hash_map<PEID, std::vector<NodeId>>;
    GraphType& G;
    const Config& conf_;
    PEID rank_;
    PEID size_;
    std::vector<PEID> last_proc_;
    std::vector<bool> is_v_neighbor_;
    BufferedCommunicator<NodeId> comm;
    std::vector<NodeId> interface_nodes_;
    std::vector<Degree> pe_min_degree;
};

}  // namespace cetric
#endif  // PARALLEL_TRIANGLE_COUNTER_PARALLEL_NODE_ITERATOR_H
