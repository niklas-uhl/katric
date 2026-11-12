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
// Created by Tim Niklas Uhl on 30.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
#define PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H

#include <cstddef>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <optional>
#include <tuple>
#include <vector>

#include <kamping/types/contiguous_type.hpp>
#include <kamping/types/mpi_type_traits.hpp>
#include <kamping/types/scoped_datatype.hpp>
#include <tlx/multi_timer.hpp>

#include "cetric/communicator.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/message_statistics.h"
#include "cetric/util.h"

namespace cetric {
using namespace cetric::graph;

// NodeInfo consists of two node ids, it is sent as a contiguous type of two of them
inline kamping::types::ScopedDatatype node_info_type() {
    static_assert(sizeof(LocalGraphView::NodeInfo) == 2 * sizeof(NodeId));
    return kamping::types::ScopedDatatype{kamping::types::mpi_type_traits<std::array<NodeId, 2>>::data_type()};
}

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
    }

    template <typename DegreeFunc>
    void get_ghost_degree(
        DegreeFunc&& on_degree_receive, bool sparse
    ) {
        get_ghost_data(
            [this](auto node) { return RankEncodedNodeId{G.degree(node)}; },
            [&on_degree_receive](auto node, auto data) { on_degree_receive(node, data.id()); },
            sparse
        );
    }

    template <typename DataFunc, typename ReceiveFunc>
    void get_ghost_data(
        DataFunc&&                            get_data,
        ReceiveFunc&&                         on_receive,
        bool                                  sparse
    ) {
        //  assert(G.ghost_ranks_available());
        send_buffers.clear();
        receive_buffers.clear();
        for (auto node: G.local_nodes()) {
            neighboring_PEs.clear();
            if (G.template is_interface_node<AdjacencyType::full>(node)) {
                auto data = get_data(node);
                for (auto neighbor: G.adj(node).neighbors()) {
                    if (neighbor.rank() != rank_) {
                        PEID rank = neighbor.rank();
                        KASSERT(rank < size_);
                        if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
                            SPDLOG_DEBUG("Sending degree of {} to rank {}", node, rank);
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
                kamping::types::mpi_type_traits<RankEncodedNodeId>::data_type(),
                rank_,
                size_,
                message_tag_
            );
        } else {
            CommunicationUtility::all_to_all(
                send_buffers,
                receive_buffers,
                kamping::types::mpi_type_traits<RankEncodedNodeId>::data_type(),
                rank_,
                size_,
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
    //     CommunicationUtility::sparse_all_to_all(send_buffers, receive_buffers,  rank_, size_, stats,
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
    absl::flat_hash_map<PEID, std::vector<RankEncodedNodeId>> send_buffers;
    absl::flat_hash_map<PEID, std::vector<RankEncodedNodeId>> receive_buffers;
    absl::flat_hash_set<PEID>                                 neighboring_PEs;
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
        [[maybe_unused]] PEID                 rank,
        PEID                                  size
    ) {
        return relocate_sparse_impl(std::forward<LocalGraphView>(G), nodes_to_send, rank, size);
    }

private:

    template <class Map>
    static LocalGraphView relocate_sparse_impl(
        LocalGraphView&&                      G,
        const Map&                            nodes_to_send,
        [[maybe_unused]] PEID                 rank,
        PEID                                  size
    ) {
        std::vector<std::pair<PEID, VectorView<LocalGraphView::NodeInfo>>> to_send_node_info;
        std::vector<std::pair<PEID, VectorView<NodeId>>>                   to_send_head;
        auto const                                                         first_out = G.first_out();
        for (const auto& kv: nodes_to_send) {
            PEID             pe    = kv.first;
            const NodeRange& range = kv.second;
            to_send_node_info.emplace_back(pe, slice(G.node_info, range.from, range.to - range.from + 1));
            auto edge_begin = first_out[range.from];
            auto edge_end   = first_out[range.to + 1];
            to_send_head.emplace_back(pe, slice(G.edge_heads, edge_begin, edge_end - edge_begin));
        }

        LocalGraphView G_balanced;
        CompactBuffer  node_recv_buffer(G_balanced.node_info, rank, size);
        kamping::types::ScopedDatatype node_info_mpi_type = node_info_type();
        CommunicationUtility::sparse_all_to_all<LocalGraphView::NodeInfo>(
            to_send_node_info,
            node_recv_buffer,
            node_info_mpi_type.data_type(),
            rank,
            size,
            as_int(MessageTag::LoadBalancing)
        );
        CompactBuffer edge_recv_buffer(G_balanced.edge_heads, rank, size);
        CommunicationUtility::sparse_all_to_all<NodeId>(
            to_send_head,
            edge_recv_buffer,
            kamping::types::mpi_type_traits<NodeId>::data_type(),
            rank,
            size,
            as_int(MessageTag::LoadBalancing)
        );
        return G_balanced;
    }
};
} // namespace cetric
#endif // PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
