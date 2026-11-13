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
#include <array>
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
#include "cetric/ghost_exchange.h"
#include "cetric/message_statistics.h"
#include "cetric/util.h"

namespace cetric {
using namespace cetric::graph;

// NodeInfo consists of two node ids, it is sent as a contiguous type of two of them
inline kamping::types::ScopedDatatype node_info_type() {
    static_assert(sizeof(LocalGraphView::NodeInfo) == 2 * sizeof(NodeId));
    return kamping::types::ScopedDatatype{kamping::types::mpi_type_traits<std::array<NodeId, 2>>::data_type()};
}

/// The degree (or other per-vertex data) of every ghost, from its owner.
template <class Graph>
class DegreeCommunicator {
public:
    DegreeCommunicator(const Graph& G, int message_tag) : G(G), message_tag_(message_tag) {}

    template <typename DegreeFunc>
    void get_ghost_degree(DegreeFunc&& on_degree_receive, GhostExchange exchange) {
        get_ghost_data(
            [this](auto node) { return RankEncodedNodeId{G.degree(node)}; },
            [&on_degree_receive](auto node, auto data) { on_degree_receive(node, data.id()); },
            exchange
        );
    }

    /// Sends `get_data(v)` (a RankEncodedNodeId) of every local vertex with a remote neighbor to that neighbor's rank,
    /// and calls `on_receive(ghost, data)` for every ghost of this rank. Message counters as RankEncodedNodeIds sent
    /// (two per ghost: vertex and data).
    template <typename DataFunc, typename ReceiveFunc>
    void get_ghost_data(DataFunc&& get_data, ReceiveFunc&& on_receive, GhostExchange exchange) {
        struct Entry {
            // initializers: the default constructor is explicit, but the grid exchange needs Entry{}
            RankEncodedNodeId node = RankEncodedNodeId{};
            RankEncodedNodeId data = RankEncodedNodeId{};
        };
        static_assert(sizeof(Entry) == 2 * sizeof(NodeId));
        kamping::types::ScopedDatatype const entry_type{
            kamping::types::mpi_type_traits<std::array<NodeId, 2>>::data_type()};
        auto const received = push_to_ghosts(
            G,
            G.local_nodes(),
            [&](RankEncodedNodeId node) { return Entry{node, RankEncodedNodeId{get_data(node)}}; },
            exchange,
            entry_type.data_type(),
            message_tag_,
            2
        );
        for (auto const& [node, data]: received) {
            on_receive(node, data);
        }
    }

private:
    const Graph& G;
    int          message_tag_;
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
        MPI_Comm                              comm,
        [[maybe_unused]] PEID                 rank,
        PEID                                  size
    ) {
        return relocate_sparse_impl(std::forward<LocalGraphView>(G), nodes_to_send, comm, rank, size);
    }

private:

    template <class Map>
    static LocalGraphView relocate_sparse_impl(
        LocalGraphView&&                      G,
        const Map&                            nodes_to_send,
        MPI_Comm                              comm,
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
            comm,
            rank,
            size,
            as_int(MessageTag::LoadBalancing)
        );
        CompactBuffer edge_recv_buffer(G_balanced.edge_heads, rank, size);
        CommunicationUtility::sparse_all_to_all<NodeId>(
            to_send_head,
            edge_recv_buffer,
            kamping::types::mpi_type_traits<NodeId>::data_type(),
            comm,
            rank,
            size,
            as_int(MessageTag::LoadBalancing)
        );
        return G_balanced;
    }
};
} // namespace cetric
#endif // PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
