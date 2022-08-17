//
// Created by Tim Niklas Uhl on 28.02.21.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_SIMPLE_GRAPH_H
#define PARALLEL_TRIANGLE_COUNTER_SIMPLE_GRAPH_H

#include <google/dense_hash_set>
#include <string>
#include <vector>

#include <datastructures/graph.h>

class SimpleDistributedGraph {
public:
    struct GhostData {
        PEID   rank;
        NodeId global_id;
        NodeId node_info;
    };
    struct LocalData {
        LocalData(bool is_interface) : is_interface(is_interface) {}
        LocalData() : LocalData(false) {}
        bool is_interface;
    };
    friend struct LoadBalancer;
    friend class DistributedGraph;

    SimpleDistributedGraph(
        std::vector<EdgeId>                           first_out,
        std::vector<NodeId>                           head,
        NodeId                                        first_node,
        NodeId                                        total_node_count,
        const std::vector<std::pair<NodeId, NodeId>>& ranges,
        PEID                                          rank,
        PEID                                          size
    )
        : first_out_(std::move(first_out)),
          head_(std::move(head)),
          first_node_(first_node),
          total_node_count_(total_node_count),
          global_to_local_map(),
          ghost_data_(),
          local_data_(first_out_.size() - 1),
          rank_(rank),
          size_(size) {
        auto get_pe = [&](NodeId node) {
            NodeId local_from;
            NodeId local_to;
            for (size_t i = 0; i < ranges.size(); ++i) {
                std::tie(local_from, local_to) = ranges[i];
                if (local_from <= node && node < local_to) {
                    return static_cast<PEID>(i);
                }
            }
            throw std::runtime_error("This should not happen");
        };
        global_to_local_map.set_empty_key(-1);
        NodeId ghost_counter = local_node_count();
        for_each_local_node([&](NodeId node) {
            EdgeId begin = first_out_[node];
            EdgeId end   = first_out_[node + 1];
            for (EdgeId edge_id = begin; edge_id < end; edge_id++) {
                NodeId head = head_[edge_id];
                NodeId local_head;
                if (!is_local(head)) {
                    local_data_[node].is_interface = true;
                    if (global_to_local_map.find(head) == global_to_local_map.end()) {
                        global_to_local_map[head] = ghost_counter;
                        GhostData data{get_pe(head), head, 0};
                        ghost_data_.emplace_back(data);
                        ghost_counter++;
                    }
                    local_head = global_to_local_map[head];
                } else {
                    local_head = to_local_id(head);
                }
                head_[edge_id] = local_head;
            }
        });
    }

    explicit SimpleDistributedGraph(PEID rank, PEID size) : SimpleDistributedGraph({0}, {}, 0, 0, {}, rank, size) {}

    SimpleDistributedGraph(
        std::vector<EdgeId>                    first_out,
        std::vector<NodeId>                    head,
        NodeId                                 first_node,
        NodeId                                 total_node_count,
        google::dense_hash_map<NodeId, NodeId> global_to_local_map,
        std::vector<GhostData>                 ghost_data,
        std::vector<LocalData>                 local_data,
        PEID                                   rank,
        PEID                                   size
    )
        : first_out_(std::move(first_out)),
          head_(std::move(head)),
          first_node_(first_node),
          total_node_count_(total_node_count),
          global_to_local_map(std::move(global_to_local_map)),
          ghost_data_(std::move(ghost_data)),
          local_data_(std::move(local_data)),
          rank_(rank),
          size_(size) {}

    template <typename NodeFunc>
    inline void for_each_local_node(NodeFunc on_node) const {
        for (NodeId node = 0; node < local_node_count(); ++node) {
            on_node(node);
        }
    }

    template <typename EdgeFunc>
    inline void for_each_edge(NodeId node, EdgeFunc on_edge) const {
        EdgeId begin = first_out_[node];
        EdgeId end   = first_out_[node + 1];
        for (size_t edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            Edge   edge{node, head};
            on_edge(edge);
        }
    }

    [[nodiscard]] inline bool is_local(NodeId global_node_id) const {
        return global_node_id >= first_node_ && global_node_id < first_node_ + local_node_count();
    }

    [[nodiscard]] inline bool is_local_from_local(NodeId local_node_id) const {
        return local_node_id < local_node_count();
    }

    [[nodiscard]] inline bool is_ghost(NodeId local_node_id) const {
        return local_node_id >= local_node_count() && local_node_id < local_node_count() + global_to_local_map.size();
    }

    [[nodiscard]] inline bool is_ghost_from_global(NodeId global_node_id) const {
        return !is_local(global_node_id) && global_to_local_map.find(global_node_id) != global_to_local_map.end();
    }

    inline Degree degree(NodeId node) const {
        return first_out_[node + 1] - first_out_[node];
    }

    inline NodeId to_global_id(NodeId local_node_id) const {
        if (local_node_id < local_node_count()) {
            return local_node_id + first_node_;
        } else {
            assert(local_node_id < local_node_count() + ghost_count());
            return ghost_data_[local_node_id - local_node_count()].global_id;
        }
    }

    [[nodiscard]] inline const GhostData& get_ghost_data(NodeId local_node_id) const {
        assert(is_ghost(local_node_id));
        NodeId ghost_index = local_node_id - local_node_count();
        return ghost_data_[ghost_index];
    }

    [[nodiscard]] inline const LocalData& get_local_data(NodeId local_node_id) const {
        assert(is_local_from_local(local_node_id));
        return local_data_[local_node_id];
    }

    inline NodeId to_local_id(NodeId global_node_id) const {
        if (is_local(global_node_id)) {
            return global_node_id - first_node_;
        } else {
            assert(global_to_local_map.find(global_node_id) != global_to_local_map.end());
            return global_to_local_map.find(global_node_id)->second;
        }
    }

    inline NodeId local_node_count() const {
        return first_out_.size() - 1;
    }

    inline NodeId ghost_count() const {
        return global_to_local_map.size();
    }

    inline NodeId total_node_count() const {
        return total_node_count_;
    }

private:
    std::vector<EdgeId>                    first_out_;
    std::vector<NodeId>                    head_;
    NodeId                                 first_node_;
    NodeId                                 total_node_count_;
    google::dense_hash_map<NodeId, NodeId> global_to_local_map;
    std::vector<GhostData>                 ghost_data_;
    std::vector<LocalData>                 local_data_;
    PEID                                   rank_;
    PEID                                   size_;
};

#endif // PARALLEL_TRIANGLE_COUNTER_SIMPLE_GRAPH_H
