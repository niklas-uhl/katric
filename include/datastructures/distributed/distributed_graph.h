//
// Created by Tim Niklas Uhl on 19.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H
#define PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H

#include "io/distributed_graph_io.h"
#include <cstddef>
#include <limits>
#include <vector>
#include <sstream>
#include <google/dense_hash_map>
#include <google/dense_hash_set>
#include <util.h>
#include <datastructures/graph_definitions.h>
#include <datastructures/distributed/local_graph_view.h>
#include <communicator.h>


namespace cetric {
    namespace load_balancing {
        class LoadBalancer;
    }
    namespace graph {

using node_set = google::dense_hash_set<NodeId>;


class DistributedGraph {
    friend class cetric::load_balancing::LoadBalancer;
    friend class GraphBuilder;
    friend class CompactGraph;

public:
    struct GhostData {
        PEID rank;
        NodeId global_id;
        NodeId node_info;
    };
    struct LocalData {
        LocalData(NodeId global_id, bool is_interface) : global_id(global_id), is_interface(is_interface) { }
        LocalData(): LocalData(0, false) { }
        NodeId global_id;
        bool is_interface;
    };

    inline NodeId local_node_count() const {
        return local_node_count_;
    }

    inline NodeId ghost_count() const {
        return ghost_data_.size();
    }

    template<typename NodeFunc>
    inline void for_each_local_node(NodeFunc on_node) const {
        for (size_t node = 0; node < local_node_count(); ++node) {
            on_node(node);
        }
    }

    template<typename NodeFunc>
    inline void for_each_ghost_node(NodeFunc on_node) const {
        for (size_t node = local_node_count_; node < local_node_count_ + ghost_data_.size(); ++node) {
            on_node(node);
        }
    }

    template<typename NodeFunc>
    inline void for_each_local_node_and_ghost(NodeFunc on_node) const {
        for (size_t node = 0; node < first_out_.size() - 1; ++node) {
            on_node(node);
        }
    }

    template<typename NodeFunc>
    inline void intersect_neighborhoods(NodeId u, NodeId v, NodeFunc on_intersection) const {
        assert(is_local_from_local(u) || is_ghost(u));
        assert(is_local_from_local(v) || is_ghost(v));
        EdgeId u_current_edge = first_out_[u] + first_out_offset_[u];
        EdgeId u_end = first_out_[u] + degree_[u];
        EdgeId v_current_edge = first_out_[v] + first_out_offset_[v];
        EdgeId v_end = first_out_[v] + degree_[v];
        while (u_current_edge != u_end && v_current_edge != v_end) {
            // The ordering is based on global id, so we must respect this here
            NodeId u_node = head_[u_current_edge];
            NodeId v_node = head_[v_current_edge];
            NodeId u_global = to_global_id(u_node);
            NodeId v_global = to_global_id(v_node);
            if (u_global < v_global) {
                u_current_edge++;
            } else if (u_global > v_global) {
                v_current_edge++;
            } else {
                // u_node == v_node
                assert(u_global == v_global);
                on_intersection(u_node);
                u_current_edge++;
                v_current_edge++;
            }
        }
    }

    template<typename NodeFunc, typename Iter>
    inline void intersect_neighborhoods(NodeId u, Iter begin, Iter end,  NodeFunc on_intersection) const {
        assert(is_local_from_local(u) || is_ghost(u));
        EdgeId u_current_edge = first_out_[u] + first_out_offset_[u];
        EdgeId u_end = first_out_[u] + degree_[u];
        while (u_current_edge != u_end && begin != end) {
            NodeId u_node = to_global_id(head_[u_current_edge]);
            NodeId v_node = *begin;
            if (u_node < v_node) {
                u_current_edge++;
            } else if (u_node > v_node) {
                begin++;
            } else {
                // u_node == v_node
                assert(u_node == v_node);
                on_intersection(u_node);
                u_current_edge++;
                begin++;
            }
        }
    }


    template<typename EdgeFunc>
    inline void for_each_edge(NodeId node, EdgeFunc on_edge) const {
        assert(is_local_from_local(node) || is_ghost(node));
        EdgeId begin = first_out_[node];
        EdgeId end = first_out_[node] + degree_[node];
        for (size_t edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            Edge edge {node, head};
            on_edge(edge);
        }
    }

    template<typename EdgeFunc>
    inline void for_each_local_out_edge(NodeId node, EdgeFunc on_edge) const {
        assert(is_local_from_local(node) || is_ghost(node));
        auto begin = first_out_[node] + first_out_offset_[node];
        auto end = first_out_[node] + degree_[node];
        for (EdgeId edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            on_edge(Edge(node, head));
        }
    }
    template<typename EdgeFunc>
    inline void for_each_local_in_edge(NodeId node, EdgeFunc on_edge) const {
        assert(is_local_from_local(node) || is_ghost(node));
        auto begin = first_out_[node];
        auto end = first_out_[node] + first_out_offset_[node];
        for (EdgeId edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            on_edge(Edge(head, node));
        }
    }

    template<typename NodeCmp>
    inline void orient(NodeCmp node_cmp) {
        auto is_outgoing = [&](NodeId tail, NodeId head) {
                return node_cmp(tail, head);
        };
        for_each_local_node_and_ghost([&](NodeId node) {
            int64_t left = first_out_[node];
            int64_t right = first_out_[node] + degree_[node] - 1;
            while (left <= right) {
                while (left <= right && !is_outgoing(node, head_[left])) {
                    left++;
                }
                while (right >= (int64_t) first_out_[node] && is_outgoing(node, head_[right])) {
                    right--;
                }
                if (left <= right) {
                    std::iter_swap(head_.begin() + left, head_.begin() + right);
                    // left++;
                    // right--;
                }
            }
            first_out_offset_[node] = left - first_out_[node];
        });

    }

    inline void set_ghost_info(NodeId local_node_id, NodeId ghost_info) {
        assert(is_ghost(local_node_id));
        ghost_data_[local_node_id - local_node_count_].node_info = ghost_info;
    }

    inline void sort_neighborhoods() {
        for_each_local_node_and_ghost([&](NodeId node) {
            EdgeId begin = first_out_[node];
            EdgeId in_end  = first_out_[node] + first_out_offset_[node];
            EdgeId out_end  = first_out_[node] + degree_[node];
            auto node_cmp = [&](NodeId a, NodeId b) {
                if (is_ghost(a) || is_ghost(b)) {
                    NodeId global_a = to_global_id(a);
                    NodeId global_b = to_global_id(b);
                    return global_a < global_b;
                } else {
                    return a < b;
                }
            };
            std::sort(head_.begin() + begin, head_.begin() + in_end, node_cmp);
            std::sort(head_.begin() + in_end, head_.begin() + out_end, node_cmp);
        });
    }

    inline bool edge_exists(Edge edge) {
        NodeId tail = edge.tail;
        NodeId tail_local = to_local_id(tail);
        NodeId head = edge.head;
        NodeId head_local = to_local_id(head);
        assert(is_local(tail));
        Degree tail_outdeg = outdegree(tail_local);
        Degree head_indegree = degree(head_local) - outdegree(head_local);
        if (tail_outdeg < head_indegree) {
            auto begin = first_out_[tail_local] + first_out_offset_[tail_local];
            auto end = first_out_[tail_local] + degree_[tail_local];
            return std::binary_search(head_.begin() + begin, head_.begin() + end, head_local, [&](NodeId a, NodeId b) {
                return to_global_id(a) < to_global_id(b);
            });
        } else {
            auto begin = first_out_[head_local];
            auto end = first_out_[head_local] + first_out_offset_[head_local];
            return std::binary_search(head_.begin() + begin, head_.begin() + end, tail_local, [&](NodeId a, NodeId b) {
                return to_global_id(a) < to_global_id(b);
            });
        }
    }

    inline Degree degree(NodeId node) const {
        //TODO different types of degrees
        assert(is_local_from_local(node));
        return degree_[node];
    }

    inline Degree outdegree(NodeId node) const {
        //TODO different types of degrees
        assert(is_local_from_local(node));
        return degree_[node] - first_out_offset_[node];
    }

    Degree initial_degree(NodeId node) const {
        //TODO different types of degrees
        assert(is_local_from_local(node));
        return first_out_[node + 1] - first_out_[node];
    }

    [[nodiscard]] inline NodeId to_global_id(NodeId local_node_id) const {
        assert((is_local_from_local(local_node_id) || is_ghost(local_node_id)));
        if (!is_ghost(local_node_id)) {
            return node_range_.first + local_node_id;
        } else {
            return get_ghost_data(local_node_id).global_id;
        }
    }

    [[nodiscard]] inline Edge to_global_edge(const Edge& edge) const {
        return Edge {to_global_id(edge.tail), to_global_id(edge.head)};
    }

    [[nodiscard]] inline NodeId to_local_id(NodeId global_node_id) const {
        if (consecutive_vertices_ && is_local(global_node_id)) {
            NodeId local_id = global_node_id - node_range_.first;
            assert(is_local_from_local(local_id));
            return local_id;
        } else {
            assert(global_to_local_.find(global_node_id) != global_to_local_.end());
            return global_to_local_.find(global_node_id)->second;
        }
    }

    [[nodiscard]] inline bool is_local(NodeId global_node_id) const {
        return global_node_id >= node_range_.first && global_node_id <= node_range_.second;
    }

    [[nodiscard]] inline bool is_local_from_local(NodeId local_node_id) const {
        return local_node_id < local_node_count();
    }

    [[nodiscard]] inline bool is_ghost(NodeId local_node_id) const {
        return local_node_id >= local_node_count_ && local_node_id < local_node_count_ + ghost_count();
    }

    [[nodiscard]] inline bool is_ghost_from_global(NodeId global_node_id) const {
        auto it = global_to_local_.find(global_node_id);
        if (it == global_to_local_.end()) {
            return false;
        } else {
            NodeId local_id = (*it).second;
            return local_id >= local_node_count_;
        }
    }


    [[nodiscard]] inline const GhostData& get_ghost_data(NodeId local_node_id) const {
        assert(is_ghost(local_node_id));
        NodeId ghost_index = local_node_id - local_node_count_;
        return ghost_data_[ghost_index];
    }


    [[nodiscard]] inline const LocalData& get_local_data(NodeId local_node_id) const {
        assert(is_local_from_local(local_node_id));
        return local_data_[local_node_id];
    }

    DistributedGraph() {
        first_out_.push_back(0);
    };

    inline NodeId total_node_count() const {
        return total_node_count_;
    }

    void remove_internal_edges() {
        for_each_local_node([&](NodeId node) {
            if (!get_local_data(node).is_interface) {
                degree_[node] = 0;
                first_out_offset_[node] = 0;
                return;
            }
            Degree remaining_edges = 0;
            EdgeId begin = first_out_[node];
            EdgeId end = first_out_[node] + degree_[node];
            EdgeId offset = first_out_offset_[node];
            for (EdgeId edge_id = begin; edge_id < end; edge_id++) {
                NodeId head = head_[edge_id];
                if (is_ghost(head)) {
                    head_[begin + remaining_edges] = head;
                    remaining_edges++;
                }
                if (edge_id + 1 == begin + offset) {
                    first_out_offset_[node] = remaining_edges;
                }
            }
            degree_[node] = remaining_edges;
        });
    }

    void expand_ghosts() {
        std::vector<std::vector<NodeId>> ghost_neighbors(ghost_count());
        size_t ghost_edges = 0;
        for_each_local_node([&](NodeId node) {
            for_each_edge(node, [&](Edge edge) {
                if (is_ghost(edge.head)) {
                    ghost_neighbors[ghost_to_ghost_index(edge.head)].push_back(edge.tail);
                    ghost_edges++;
                }
            });
        });
        size_t edge_index = head_.size();
        head_.resize(head_.size() + ghost_edges);
        first_out_.resize(first_out_.size() + ghost_count());
        first_out_offset_.resize(first_out_offset_.size() + ghost_count(), 0);
        degree_.resize(degree_.size() + ghost_count());
        for(size_t i = 0; i < ghost_neighbors.size(); ++i) {
            first_out_[local_node_count_ + i + 1] = first_out_[local_node_count_ + i] + ghost_neighbors[i].size();
            degree_[local_node_count_ + i] = ghost_neighbors[i].size();
            for(NodeId neighbor : ghost_neighbors[i]) {
                head_[edge_index] = neighbor;
                edge_index++;
            }
        }
    }

    DistributedGraph(cetric::graph::LocalGraphView&& G, PEID rank, PEID size) {
        rank_ = rank;
        size_ = size;
        local_node_count_ = G.node_info.size();
        first_out_.resize(local_node_count_ + 1);
        first_out_offset_.resize(local_node_count_, 0);
        degree_.resize(local_node_count_);
        local_data_.resize(local_node_count_);
        auto degree_sum = 0;
        node_range_ = std::make_pair(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::min());
        global_to_local_.set_empty_key(-1);
        global_to_local_.set_deleted_key(-2);
        for (size_t i = 0; i < G.node_info.size(); ++i) {
            first_out_[i] = degree_sum;
            degree_sum += G.node_info[i].degree;
            NodeId local_id = i;
            NodeId global_id = G.node_info[i].global_id;
            global_to_local_[global_id] = local_id;
            degree_[local_id] = G.node_info[i].degree;
            local_data_[local_id].global_id = global_id;
            node_range_.first = std::min(node_range_.first, global_id);
            node_range_.second = std::max(node_range_.second, global_id);
        }
        consecutive_vertices_ = node_range_.second - node_range_.first  + 1 == local_node_count_;
        atomic_debug(consecutive_vertices_);
        if (consecutive_vertices_) {
            for (const LocalGraphView::NodeInfo& node_info : G.node_info) {
                global_to_local_.erase(node_info.global_id);
            }
        }
        first_out_[local_node_count_] = degree_sum;
        head_ = std::move(G.edge_heads);
        auto ghost_count = 0;
        for (NodeId node = 0; node < local_node_count_; ++node) {
            for (EdgeId edge_id = first_out_[node]; edge_id < first_out_[node + 1]; ++edge_id) {
                NodeId head = head_[edge_id];
                if (!is_local(head)) {
                    local_data_[node].is_interface = true;
                    if (global_to_local_.find(head) == global_to_local_.end()) {
                        NodeId local_id = local_node_count_ + ghost_count;
                        global_to_local_[head] = local_id;
                        GhostData ghost_data;
                        ghost_data.global_id = head;
                        ghost_data.rank = -1;
                        ghost_data_.emplace_back(std::move(ghost_data));
                        ghost_count++;
                    }
                }
                head_[edge_id] = to_local_id(head);
            }
        }
    }

    void find_ghost_ranks() {
        if (consecutive_vertices_) {
            std::vector<std::pair<NodeId, NodeId>> ranges(size_);
            gather_PE_ranges(node_range_.first, node_range_.second, ranges, MPI_COMM_WORLD, rank_, size_);
            for_each_ghost_node([&](NodeId node) {
                PEID rank = get_PE_from_node_ranges(to_global_id(node), ranges);
                ghost_data_[ghost_to_ghost_index(node)].rank = rank;
            });
        } else {
            std::vector<NodeId> nodes(local_node_count_);
            for_each_local_node([&](NodeId node) {
                nodes[node] = to_global_id(node);
            });
            auto [all_nodes, displs] = CommunicationUtility::all_gather(nodes, MPI_NODE, MPI_COMM_WORLD, rank_, size_);
            for (PEID rank = 0; rank < size_; rank++) {
                for (int i = displs[rank]; i < displs[rank + 1]; ++i) {
                    NodeId node = all_nodes[i];
                    if (is_ghost_from_global(node)) {
                        NodeId local_id = to_local_id(node);
                        ghost_data_[ghost_to_ghost_index(local_id)].rank = rank;
                    }
                }
            }
        }
    }

    LocalGraphView to_local_graph_view(bool remove_isolated = true, bool keep_only_out_edges = true) {
        DistributedGraph G = std::move(*this);
        std::vector<LocalGraphView::NodeInfo> node_info;
        EdgeId edge_counter = 0;
        G.for_each_local_node([&](NodeId node) {
            auto degree = G.degree(node);
            if (remove_isolated && (degree == 0 || (keep_only_out_edges && G.outdegree(node) == 0))) {
                return;
            }
            auto global_id = G.to_global_id(node);
            if (keep_only_out_edges) {
                degree = G.outdegree(node);
            }
            node_info.emplace_back(global_id, degree);
            if (keep_only_out_edges) {
                G.for_each_local_out_edge(node, [&](Edge edge) {
                    G.head_[edge_counter] = G.to_global_id(edge.head);
                    edge_counter++;
                });
            } else {
                G.for_each_edge(node, [&](Edge edge) {
                    G.head_[edge_counter] = G.to_global_id(edge.head);
                    edge_counter++;
                });
            }
        });
        G.head_.resize(edge_counter);
        G.head_.shrink_to_fit();
        auto head = std::move(G.head_);
        LocalGraphView view;
        view.node_info = std::move(node_info);
        view.edge_heads = std::move(head);
        return view;
    }

private:

    NodeId ghost_to_ghost_index(NodeId local_node_id) const {
        assert( is_ghost(local_node_id) );
        return local_node_id - local_node_count_;
    }

    using node_map = google::dense_hash_map<NodeId, NodeId>;
    std::vector<EdgeId> first_out_;
    std::vector<EdgeId> first_out_offset_;
    std::vector<Degree> degree_;
    std::vector<NodeId> head_;
    std::vector<GhostData> ghost_data_;
    std::vector<LocalData> local_data_;
    bool consecutive_vertices_;
    node_map global_to_local_;
    NodeId local_node_count_{};
    EdgeId local_edge_count_{};
    NodeId total_node_count_{};
    std::pair<NodeId, NodeId> node_range_;
    PEID rank_;
    PEID size_;
};

inline std::ostream& operator<<(std::ostream& out, DistributedGraph& G) {
    G.for_each_local_node([&](NodeId node) {
        G.for_each_local_out_edge(node, [&](Edge edge) {
            out << "(" << G.to_global_id(edge.tail) << ", " << G.to_global_id(edge.head) << ")" << std::endl;
        });
    });
    return out;
}

}
}
#endif //PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H
