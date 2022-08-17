#ifndef COMPACT_GRAPH_H_EDJPQOVG
#define COMPACT_GRAPH_H_EDJPQOVG

#include <google/dense_hash_map>
#include <sparsehash/dense_hash_map>
#include <utility>
#include <vector>

#include <datastructures/distributed/distributed_graph.h>
#include <datastructures/graph_definitions.h>
#include <sparsehash/internal/densehashtable.h>

namespace cetric {
namespace graph {

class CompactGraph {
    friend std::ostream& operator<<(std::ostream& os, CompactGraph& G);
    friend class cetric::load_balancing::LoadBalancer;

public:
    CompactGraph(cetric::graph::DistributedGraph G) {
        global_to_local_map.set_empty_key(-1);
        NodeId new_local_id         = 0;
        EdgeId edge_counter         = 0;
        NodeId new_local_node_count = 0;
        G.for_each_local_node([&](NodeId node) {
            if (G.outdegree(node) > 0) {
                new_local_node_count++;
            }
        });
        /* if (new_local_node_count * 2 > G.first_out_.size()) { */
        /*     G.first_out_.resize(new_local_node_count * 2); */
        /* } */
        edge_pointer.resize(new_local_node_count * 2 + 1);
        for (NodeId node = 0; node < G.local_node_count(); ++node) {
            if (G.outdegree(node) > 0) {
                EdgeId edge_begin = edge_counter;
                G.for_each_local_out_edge(node, [&](Edge e) {
                    G.head_[edge_counter] = G.to_global_id(e.head);
                    edge_counter++;
                });
                edge_pointer[2 * new_local_id]            = edge_begin;
                edge_pointer[2 * new_local_id + 1]        = (G.to_global_id(node));
                global_to_local_map[G.to_global_id(node)] = new_local_id;
                new_local_id++;
            }
        }
        edge_pointer[2 * new_local_id] = edge_counter;
        // edge_pointer = std::move(G.first_out_);
        head = std::move(G.head_);
        edge_pointer.resize(2 * new_local_id + 1);
        head.resize(edge_counter);
        global_to_local_map.resize(0);
    }

    NodeId local_node_count() const {
        return edge_pointer.size() / 2;
    }

    template <typename NodeFunc>
    void for_each_local_node(NodeFunc on_node) const {
        for (NodeId i = 0; i < local_node_count(); i++) {
            on_node(i);
        }
    }

    template <typename EdgeFunc>
    void for_each_edge(NodeId index, EdgeFunc on_edge) const {
        for (size_t edge_id = edge_pointer[index * 2]; edge_id < edge_pointer[(index + 1) * 2]; edge_id++) {
            on_edge(Edge(to_global_id(index), head[edge_id]));
        }
    }

    NodeId to_global_id(NodeId index) const {
        return edge_pointer[index * 2 + 1];
    }

    NodeId degree(NodeId node) const {
        auto edge_begin = edge_pointer[node * 2];
        auto edge_end   = edge_pointer[(node + 1) * 2];
        return edge_end - edge_begin;
    }

    std::vector<EdgeId>                    edge_pointer;
    std::vector<NodeId>                    head;
    google::dense_hash_map<NodeId, NodeId> global_to_local_map;
};

inline std::ostream& operator<<(std::ostream& out, CompactGraph& G) {
    G.for_each_local_node([&](NodeId index) {
        G.for_each_edge(index, [&](Edge edge) { out << "(" << edge.tail << ", " << edge.head << ")" << std::endl; });
    });
    return out;
}

} // namespace graph
} // namespace cetric

#endif /* end of include guard: COMPACT_GRAPH_H_EDJPQOVG */
