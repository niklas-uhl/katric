// The local part of a 1D-distributed graph as a flat list: the nodes with their degrees and the concatenated
// adjacency lists. Input and load balancing (relocation) work on it before the DistributedGraph is built.
#pragma once

#include <cstddef>
#include <vector>

#include "cetric/datastructures/graph_definitions.h"

namespace cetric::graph {

struct LocalGraphView {
    struct NodeInfo {
        NodeInfo() = default;
        NodeInfo(NodeId global_id_, Degree degree_) : global_id(global_id_), degree(degree_) {}
        NodeId global_id = 0;
        Degree degree    = 0;

        friend bool operator==(NodeInfo const&, NodeInfo const&) = default;
    };
    std::vector<NodeInfo> node_info;  // sorted by global id
    std::vector<NodeId>   edge_heads; // the neighbors of node_info[0], then of node_info[1], ...

    [[nodiscard]] NodeId local_node_count() const {
        return node_info.size();
    }

    [[nodiscard]] Degree degree(NodeId local_index) const {
        return node_info[local_index].degree;
    }

    /// Offsets of each node's neighbors in edge_heads (size local_node_count() + 1).
    [[nodiscard]] std::vector<EdgeId> first_out() const {
        std::vector<EdgeId> offsets(node_info.size() + 1, 0);
        for (std::size_t i = 0; i < node_info.size(); ++i) {
            offsets[i + 1] = offsets[i] + node_info[i].degree;
        }
        return offsets;
    }
};

} // namespace cetric::graph
