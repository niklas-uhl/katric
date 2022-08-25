//
// Created by Tim Niklas Uhl on 30.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
#define PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H

#include <cstddef>
#include <google/dense_hash_map>
#include <google/dense_hash_set>
#include <optional>
#include <tuple>
#include <vector>

#include <graph-io/local_graph_view.h>
#include <tlx/multi_timer.hpp>

#include "cetric/communicator.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/statistics.h"
#include "cetric/util.h"

namespace cetric {
using namespace cetric::graph;

template <class Graph>
class DegreeCommunicator {
public:
    DegreeCommunicator(const Graph& G, PEID rank, PEID size, int message_tag)
        : G(G),
          rank_(rank),
          size_(size),
          send_buffers(size),
          receive_buffers(size),
          neighboring_PEs(size),
          message_tag_(message_tag) {
        send_buffers.set_empty_key(-1);
        receive_buffers.set_empty_key(-1);
        neighboring_PEs.set_empty_key(-1);
    }

    template <typename DegreeFunc>
    void get_ghost_degree(
        DegreeFunc&& on_degree_receive, cetric::profiling::MessageStatistics& stats, bool sparse, bool compact
    ) {
        get_ghost_data(
            [this](auto node) { return RankEncodedNodeId{G.degree(node)}; },
            [&on_degree_receive](auto node, auto data) { on_degree_receive(node, data.id()); },
            stats,
            sparse,
            compact
        );
    }

    template <typename DataFunc, typename ReceiveFunc>
    void get_ghost_data(
        DataFunc&&                            get_data,
        ReceiveFunc&&                         on_receive,
        cetric::profiling::MessageStatistics& stats,
        bool                                  sparse,
        bool                                  compact
    ) {
        if (compact) {
            get_ghost_data_compact(
                std::forward<DataFunc>(get_data),
                std::forward<ReceiveFunc>(on_receive),
                stats,
                sparse
            );
            return;
        }
        //  assert(G.ghost_ranks_available());
        send_buffers.clear();
        receive_buffers.clear();
        for (auto node: G.local_nodes()) {
            neighboring_PEs.clear();
            if (G.is_interface_node(node)) {
                auto data = get_data(node);
                for (auto neighbor: G.adj(node).neighbors()) {
                    if (neighbor.rank() != rank_) {
                        PEID rank = neighbor.rank();
                        if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
                            // atomic_debug(fmt::format("Sending degree of {} to rank {}", node, rank));
                            send_buffers[rank].emplace_back(node);
                            send_buffers[rank].emplace_back(data);
                            neighboring_PEs.insert(rank);
                        }
                    }
                }
            }
        }
        if (sparse) {
            CommunicationUtility::sparse_all_to_all(
                send_buffers,
                receive_buffers,
                MPI_NODE,
                rank_,
                size_,
                stats,
                message_tag_
            );
        } else {
            CommunicationUtility::all_to_all(
                send_buffers,
                receive_buffers,
                MPI_NODE,
                rank_,
                size_,
                stats,
                message_tag_
            );
        }
        for (const auto& elem: receive_buffers) {
            const std::vector<RankEncodedNodeId>& buffer = elem.second;
            assert(buffer.size() % 2 == 0);
            for (size_t i = 0; i < buffer.size(); i += 2) {
                RankEncodedNodeId node = buffer[i];
                on_receive(node, buffer[i + 1]);
            }
        }
    }
    template <typename DataFunc, typename ReceiveFunc>
    void get_ghost_data_compact(
        DataFunc&& get_data, ReceiveFunc&& on_receive, cetric::profiling::MessageStatistics& stats, bool sparse
    ) {
        //  assert(G.ghost_ranks_available());
        // tlx::MultiTimer timer;
        // timer.start("copy_degree");
        send_buffers.clear();
        receive_buffers.clear();
        std::vector<std::vector<RankEncodedNodeId>> send_buffers_vec;
        if (!sparse) {
            send_buffers_vec.resize(size_);
        }
        for (auto node: G.local_nodes()) {
            neighboring_PEs.clear();
            if (G.is_interface_node(node)) {
                auto data = get_data(node);
                for (auto neighbor: G.adj(node).neighbors()) {
                    if (neighbor.rank() != rank_) {
                        PEID rank = neighbor.rank();
                        if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
                            // atomic_debug(fmt::format("Sending degree of {} to rank {}", node, rank));
                            if (sparse) {
                                send_buffers[rank].emplace_back(node);
                                send_buffers[rank].emplace_back(data);
                            } else {
                                send_buffers_vec[rank].emplace_back(node);
                                send_buffers_vec[rank].emplace_back(data);
                            }
                            neighboring_PEs.insert(rank);
                        }
                    }
                }
            }
        }
        // timer.start("all2all");
        std::vector<RankEncodedNodeId> recv_vec;
        CompactBuffer                  recv_buf(recv_vec, rank_, size_);
        if (sparse) {
            CommunicationUtility::sparse_all_to_all<RankEncodedNodeId>(
                send_buffers,
                recv_buf,
                rank_,
                size_,
                message_tag_,
                stats
            );
        } else {
            CommunicationUtility::all_to_all(send_buffers_vec, recv_buf, rank_, size_, message_tag_, stats);
        }
        // timer.start("process");
        for (size_t i = 0; i < recv_vec.size(); i += 2) {
            RankEncodedNodeId node = recv_vec[i];
            on_receive(node, recv_vec[i + 1]);
        }
        // timer.stop();
        // std::stringstream out;
        // out << "[R" << rank_ << "] ";
        // timer.print("info", out);
        // std::cout << out.str();
    }

    // template <typename OnDegreeFunc>
    // void get_ghost_outdegree(OnDegreeFunc&& on_degree_receive, cetric::profiling::MessageStatistics& stats) {
    //     assert(G.oriented());
    //     get_ghost_outdegree([&](RankEncodedEdge e) { return G.is_outgoing(e); }, on_degree_receive, stats);
    // }

    // template <typename EdgePred, typename OnDegreeFunc>
    // void get_ghost_outdegree(EdgePred&& is_outgoing,
    //                          OnDegreeFunc&& on_degree_receive,
    //                          cetric::profiling::MessageStatistics& stats) {
    //     assert(G.ghost_ranks_available());
    //     auto get_out_degree = [&](NodeId local_node_id) {
    //         Degree outdegree = 0;
    //         G.for_each_edge(local_node_id, [&](RankEncodedEdge e) {
    //             if (is_outgoing(e)) {
    //                 outdegree++;
    //             }
    //         });
    //         return outdegree;
    //     };
    //     // assert(G.oriented());
    //     send_buffers.clear();
    //     receive_buffers.clear();
    //     for (NodeId node = 0; node < G.local_node_count(); ++node) {
    //         neighboring_PEs.clear();
    //         if (G.get_local_data(node).is_interface) {
    //             G.for_each_edge(node, [&](RankEncodedEdge edge) {
    //                 if (G.is_ghost(edge.head.id())) {
    //                     PEID rank = G.get_ghost_data(edge.head.id()).rank;
    //                     if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
    //                         send_buffers[rank].emplace_back(G.to_global_id(node));
    //                         send_buffers[rank].emplace_back(get_out_degree(node));
    //                         neighboring_PEs.insert(rank);
    //                     }
    //                 }
    //             });
    //         }
    //     }
    //     CommunicationUtility::sparse_all_to_all(send_buffers, receive_buffers, MPI_NODE, rank_, size_, stats,
    //                                             message_tag_ + 1);
    //     for (const auto& elem : receive_buffers) {
    //         const std::vector<NodeId>& buffer = elem.second;
    //         assert(buffer.size() % 2 == 0);
    //         for (size_t i = 0; i < buffer.size(); i += 2) {
    //             NodeId node = buffer[i];
    //             Degree degree = buffer[i + 1];
    //             on_degree_receive(node, degree);
    //         }
    //     }
    // }

private:
    const Graph&                                                 G;
    PEID                                                         rank_;
    PEID                                                         size_;
    google::dense_hash_map<PEID, std::vector<RankEncodedNodeId>> send_buffers;
    google::dense_hash_map<PEID, std::vector<RankEncodedNodeId>> receive_buffers;
    google::dense_hash_set<PEID>                                 neighboring_PEs;
    int                                                          message_tag_;
};

class GraphCommunicator {
public:
    struct NodeRange {
        NodeId from;
        NodeId to;
    };

    template <class Map>
    static LocalGraphView relocate(
        LocalGraphView&&                      G,
        const Map&                            nodes_to_send,
        cetric::profiling::MessageStatistics& stats,
        [[maybe_unused]] PEID                 rank,
        PEID                                  size,
        bool                                  sparse = true
    ) {
        if (sparse) {
            return relocate_sparse_impl(std::forward<LocalGraphView>(G), nodes_to_send, stats, rank, size);
        } else {
            return relocate_full_impl(std::forward<LocalGraphView>(G), nodes_to_send, stats, rank, size);
        }
    }

private:
    template <class Map>
    static LocalGraphView relocate_full_impl(
        LocalGraphView&&                      G,
        const Map&                            nodes_to_send,
        cetric::profiling::MessageStatistics& stats,
        [[maybe_unused]] PEID                 rank,
        PEID                                  size
    ) {
        std::vector<std::pair<NodeId, EdgeId>> to_send(size);
        for (const auto& kv: nodes_to_send) {
            PEID             pe    = kv.first;
            const NodeRange& range = kv.second;
            to_send[pe].first      = range.to - range.from + 1;
            for (NodeId node = range.from; node <= range.to; ++node) {
                to_send[pe].second += G.node_info[node].degree;
            }
        }
        std::vector<std::pair<NodeId, NodeId>> to_receive(size);
        stats.send_volume += 2;
        stats.receive_volume += 2;
        int errcode = MPI_Alltoall(to_send.data(), 2, MPI_NODE, to_receive.data(), 2, MPI_NODE, MPI_COMM_WORLD);
        check_mpi_error(errcode, __FILE__, __LINE__);

        std::vector<int> send_counts(size);
        std::vector<int> send_displs(size);
        std::vector<int> recv_counts(size);
        std::vector<int> recv_displs(size);

        int send_running_sum = 0;
        int recv_running_sum = 0;
        for (size_t i = 0; i < send_counts.size(); ++i) {
            send_counts[i] = to_send[i].first;
            send_displs[i] = send_running_sum;
            send_running_sum += send_counts[i];

            recv_counts[i] = to_receive[i].first;
            recv_displs[i] = recv_running_sum;
            recv_running_sum += recv_counts[i];
        }

        MPI_Datatype mpi_node_info;
        MPI_Type_contiguous(2, MPI_NODE, &mpi_node_info);
        MPI_Type_commit(&mpi_node_info);

        std::vector<LocalGraphView::NodeInfo> node_info_recv(recv_displs[size - 1] + recv_counts[size - 1]);

        stats.send_volume += G.node_info.size() * 2;
        stats.receive_volume += node_info_recv.size() * 2;
        errcode = MPI_Alltoallv(
            G.node_info.data(),
            send_counts.data(),
            send_displs.data(),
            mpi_node_info,
            node_info_recv.data(),
            recv_counts.data(),
            recv_displs.data(),
            mpi_node_info,
            MPI_COMM_WORLD
        );
        check_mpi_error(errcode, __FILE__, __LINE__);
        G.node_info.resize(0);
        G.node_info.shrink_to_fit();

        send_running_sum = 0;
        recv_running_sum = 0;
        for (size_t i = 0; i < send_counts.size(); ++i) {
            send_counts[i] = to_send[i].second;
            send_displs[i] = send_running_sum;
            send_running_sum += send_counts[i];

            recv_counts[i] = to_receive[i].second;
            recv_displs[i] = recv_running_sum;
            recv_running_sum += recv_counts[i];
        }
        std::vector<NodeId> head(recv_displs[size - 1] + recv_counts[size - 1]);
        stats.send_volume += G.edge_heads.size();
        stats.receive_volume += head.size();
        errcode = MPI_Alltoallv(
            G.edge_heads.data(),
            send_counts.data(),
            send_displs.data(),
            MPI_NODE,
            head.data(),
            recv_counts.data(),
            recv_displs.data(),
            MPI_NODE,
            MPI_COMM_WORLD
        );
        check_mpi_error(errcode, __FILE__, __LINE__);
        G.edge_heads.shrink_to_fit();

        LocalGraphView G_balanced;
        G_balanced.node_info  = std::move(node_info_recv);
        G_balanced.edge_heads = std::move(head);

        return G_balanced;
    }

    template <class Map>
    static LocalGraphView relocate_sparse_impl(
        LocalGraphView&&                      G,
        const Map&                            nodes_to_send,
        cetric::profiling::MessageStatistics& stats,
        [[maybe_unused]] PEID                 rank,
        PEID                                  size
    ) {
        std::vector<std::pair<PEID, VectorView<LocalGraphView::NodeInfo>>> to_send_node_info;
        std::vector<std::pair<PEID, VectorView<NodeId>>>                   to_send_head;
        auto                                                               idx = G.build_indexer();
        for (const auto& kv: nodes_to_send) {
            PEID             pe    = kv.first;
            const NodeRange& range = kv.second;
            to_send_node_info.emplace_back(pe, slice(G.node_info, range.from, range.to - range.from + 1));
            auto edge_begin = idx.neighborhood_index_range(range.from).first;
            auto edge_end   = idx.neighborhood_index_range(range.to).second;
            to_send_head.emplace_back(pe, slice(G.edge_heads, edge_begin, edge_end - edge_begin));
        }

        LocalGraphView G_balanced;
        CompactBuffer  node_recv_buffer(G_balanced.node_info, rank, size);
        CommunicationUtility::sparse_all_to_all<LocalGraphView::NodeInfo>(
            to_send_node_info,
            node_recv_buffer,
            rank,
            size,
            as_int(MessageTag::LoadBalancing),
            stats
        );
        CompactBuffer edge_recv_buffer(G_balanced.edge_heads, rank, size);
        CommunicationUtility::sparse_all_to_all<NodeId>(
            to_send_head,
            edge_recv_buffer,
            rank,
            size,
            as_int(MessageTag::LoadBalancing),
            stats
        );
        return G_balanced;
    }
};
} // namespace cetric
#endif // PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
