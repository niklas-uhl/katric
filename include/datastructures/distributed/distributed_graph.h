//
// Created by Tim Niklas Uhl on 19.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H
#define PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H

#include <communicator.h>
#include <datastructures/distributed/local_graph_view.h>
#include <datastructures/graph_definitions.h>
#include <util.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <google/dense_hash_map>
#include <google/dense_hash_set>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>
#include "debug_assert.hpp"
#include "fmt/core.h"
#include "io/distributed_graph_io.h"

namespace cetric {
namespace load_balancing {
class LoadBalancer;
}
namespace graph {

using node_set = google::dense_hash_set<NodeId>;

template <typename PayloadType>
class payload_has_degree {
    template <typename C>
    static char test(decltype(&C::degree));

    template <typename C>
    static char test(int, typename std::enable_if<std::is_convertible_v<C, Degree>>::type* = 0);

    template <typename C>
    static int test(...);

public:
    static const bool value = (sizeof(test<PayloadType>(0)) == sizeof(char));
};

template <typename PayloadType>
class payload_has_outdegree {
    template <typename C>
    static char test(decltype(&C::outdegree));

    template <typename C>
    static int test(...);

public:
    static const bool value = (sizeof(test<PayloadType>(0)) == sizeof(char));
};

struct DegreePayload {
    Degree degree;
};

struct DegreeIndicators {
    bool ghost_degree_available = false;
};

template <typename GhostPayloadType = DegreePayload, typename GraphPayload = DegreeIndicators>
class DistributedGraph {
    friend class cetric::load_balancing::LoadBalancer;
    friend class GraphBuilder;
    friend class CompactGraph;

public:
    class NodeIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = NodeId;
        using pointer = NodeId*;
        using reference = NodeId&;
        explicit NodeIterator(NodeId node) : node_(node) {}
        bool operator==(const NodeIterator& rhs) const {
            return this->node_ == rhs.node_;
        }
        bool operator!=(const NodeIterator& rhs) const {
            return !(*this == rhs);
        }
        NodeId operator*() const {
            return node_;
        }
        NodeIterator& operator++() {
            this->node_++;
            return *this;
        }

        NodeIterator& operator+=(difference_type n) {
            this->node_ += n;
            return *this;
        }

        difference_type operator-(const NodeIterator& rhs) {
            return this->node_ - rhs.node_;
        }

    private:
        NodeId node_;
    };

    class NodeRange {
    public:
        explicit NodeRange(NodeId from, NodeId to) : from_(from), to_(to) {}
        NodeIterator begin() const {
            return NodeIterator(from_);
        };
        NodeIterator end() const {
            return NodeIterator(to_);
        };

    private:
        NodeId from_;
        NodeId to_;
    };

    class EdgeIterator {
    public:
        explicit EdgeIterator(NodeId tail, std::vector<NodeId>::const_iterator iter)
            : tail_(tail), iter_(std::move(iter)) {}
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Edge;
        using pointer = Edge*;
        using reference = Edge&;
        bool operator==(const EdgeIterator& rhs) const {
            return this->tail_ == rhs.tail_ && this->iter_ == rhs.iter_;
        }
        bool operator!=(const EdgeIterator& rhs) const {
            return !(*this == rhs);
        }
        Edge operator*() const {
            return Edge(tail_, *iter_);
        }
        EdgeIterator& operator++() {
            this->iter_++;
            return *this;
        }

        difference_type operator-(const EdgeIterator& rhs) {
            return this->iter_ - rhs.iter_;
        }

        EdgeIterator& operator+=(difference_type n) {
            this->iter_ += n;
            return *this;
        }

    private:
        NodeId tail_;
        std::vector<NodeId>::const_iterator iter_;
    };

    class EdgeRange {
    public:
        explicit EdgeRange(NodeId tail,
                           std::vector<NodeId>::const_iterator&& begin,
                           std::vector<NodeId>::const_iterator&& end)
            : tail_(tail), begin_(std::move(begin)), end_(std::move(end)) {}
        EdgeIterator begin() const {
            return EdgeIterator(tail_, begin_);
        };
        EdgeIterator end() const {
            return EdgeIterator(tail_, end_);
        };

    private:
        NodeId tail_;
        std::vector<NodeId>::const_iterator begin_;
        std::vector<NodeId>::const_iterator end_;
    };

    using payload_type = GhostPayloadType;

    template <typename Payload>
    struct GhostData {
        PEID rank;
        NodeId global_id;
        Payload payload;
    };
    struct LocalData {
        LocalData(NodeId global_id, bool is_interface) : global_id(global_id), is_interface(is_interface) {}
        LocalData() : LocalData(0, false) {}
        NodeId global_id;
        bool is_interface;
    };

    inline NodeId local_node_count() const {
        return local_node_count_;
    }

    inline NodeId ghost_count() const {
        return ghost_data_.size();
    }

    template <typename NodeFunc>
    inline void for_each_local_node(NodeFunc on_node) const {
        for (size_t node = 0; node < local_node_count(); ++node) {
            on_node(node);
        }
    }

    NodeRange local_nodes() const {
        return NodeRange(0, local_node_count());
    }

    template <typename NodeFunc>
    inline void for_each_ghost_node(NodeFunc on_node) const {
        for (size_t node = local_node_count_; node < local_node_count_ + ghost_data_.size(); ++node) {
            on_node(node);
        }
    }

    NodeRange ghost_nodes() const {
        return NodeRange(local_node_count(), local_node_count() + ghost_count());
    }

    template <typename NodeFunc>
    inline void for_each_local_node_and_ghost(NodeFunc on_node) const {
        for (size_t node = 0; node < first_out_.size() - 1; ++node) {
            on_node(node);
        }
    }

    NodeRange local_and_ghost_nodes() const {
        return NodeRange(0, local_node_count() + ghost_count());
    }

    template <typename NodeFunc>
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

    template <typename NodeFunc, typename Iter>
    inline void intersect_neighborhoods(NodeId u, Iter begin, Iter end, NodeFunc on_intersection) const {
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

    template <typename EdgeFunc>
    inline void for_each_edge(NodeId node, EdgeFunc on_edge) const {
        assert(is_local_from_local(node) || is_ghost(node));
        EdgeId begin = first_out_[node];
        // TODO change it!!!
        EdgeId end = first_out_[node] + degree_[node];
        // EdgeId end = first_out_[node] + first_out_[node + 1];
        for (size_t edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            Edge edge{node, head};
            on_edge(edge);
        }
    }

    EdgeRange edges(NodeId node) const {
        EdgeId begin = first_out_[node];
        EdgeId end = first_out_[node] + degree_[node];
        return EdgeRange(node, head_.cbegin() + begin, head_.cbegin() + end);
    }

    template <typename EdgeFunc>
    inline void for_each_local_out_edge(NodeId node, EdgeFunc on_edge) const {
        assert(is_local_from_local(node) || is_ghost(node));
        auto begin = first_out_[node] + first_out_offset_[node];
        auto end = first_out_[node] + degree_[node];
        for (EdgeId edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            on_edge(Edge(node, head));
        }
    }

    EdgeRange out_edges(NodeId node) const {
        auto begin = first_out_[node] + first_out_offset_[node];
        auto end = first_out_[node] + degree_[node];
        return EdgeRange(node, head_.cbegin() + begin, head_.cbegin() + end);
    }

    template <typename EdgeFunc>
    inline void for_each_local_in_edge(NodeId node, EdgeFunc on_edge) const {
        assert(is_local_from_local(node) || is_ghost(node));
        auto begin = first_out_[node];
        auto end = first_out_[node] + first_out_offset_[node];
        for (EdgeId edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            on_edge(Edge(head, node));
        }
    }

    EdgeRange in_edges(NodeId node) const {
        auto begin = first_out_[node];
        auto end = first_out_[node] + first_out_offset_[node];
        return EdgeRange(node, head_.cbegin() + begin, head_.cbegin() + end);
    }

    template <typename NodeCmp>
    inline void orient(NodeCmp node_cmp) {
        auto is_outgoing = [&](NodeId tail, NodeId head) {
            return node_cmp(tail, head);
        };
        auto orient_neighborhoods = [&](NodeId node) {
            int64_t left = first_out_[node];
            int64_t right = first_out_[node] + degree_[node] - 1;
            while (left <= right) {
                while (left <= right && !is_outgoing(node, head_[left])) {
                    left++;
                }
                while (right >= (int64_t)first_out_[node] && is_outgoing(node, head_[right])) {
                    right--;
                }
                if (left <= right) {
                    std::iter_swap(head_.begin() + left, head_.begin() + right);
                    // left++;
                    // right--;
                }
            }
            first_out_offset_[node] = left - first_out_[node];
        };
        if (ghosts_expanded_) {
            for_each_local_node_and_ghost(orient_neighborhoods);
        } else {
            for_each_local_node(orient_neighborhoods);
        };
        oriented_ = true;
    }

    inline bool oriented() const {
        return oriented_;
    }

    inline void set_ghost_payload(NodeId local_node_id, GhostPayloadType payload) {
        assert(is_ghost(local_node_id));
        ghost_data_[local_node_id - local_node_count_].payload = std::move(payload);
    }

    inline void sort_neighborhoods() {
        for_each_local_node_and_ghost([&](NodeId node) {
            EdgeId begin = first_out_[node];
            EdgeId in_end = first_out_[node] + first_out_offset_[node];
            EdgeId out_end = first_out_[node] + degree_[node];
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
            return std::binary_search(head_.begin() + begin, head_.begin() + end, head_local,
                                      [&](NodeId a, NodeId b) { return to_global_id(a) < to_global_id(b); });
        } else {
            auto begin = first_out_[head_local];
            auto end = first_out_[head_local] + first_out_offset_[head_local];
            return std::binary_search(head_.begin() + begin, head_.begin() + end, tail_local,
                                      [&](NodeId a, NodeId b) { return to_global_id(a) < to_global_id(b); });
        }
    }

    //! returns the degree for the given local id
    //! if ghost_payload has degree this also works for ghosts
    inline Degree degree(NodeId node) const {
        if constexpr (payload_has_degree<payload_type>::value) {
            assert(is_local_from_local(node) || is_ghost(node));
            if (is_ghost(node)) {
                assert(graph_payload_.ghost_degree_available);
                if constexpr (std::is_convertible_v<payload_type, Degree>) {
                    return get_ghost_payload(node);
                } else {
                    return get_ghost_payload(node).degree;
                }
            } else {
                return degree_[node];
            }
        } else {
            assert(is_local_from_local(node));
            return degree_[node];
        }
    }

    inline Degree local_degree(NodeId node) const {
        assert(is_local_from_local(node) || is_ghost(node));
        return degree_[node];
    }

    inline Degree local_outdegree(NodeId node) const {
        assert(oriented());
        assert(is_local_from_local(node) || is_ghost(node));
        assert(ghosts_expanded_ || is_local_from_local(node));
        return degree_[node] - first_out_offset_[node];
    }

    inline Degree outdegree(NodeId node) const {
        assert(oriented());
        assert(is_local_from_local(node));
        return degree_[node] - first_out_offset_[node];
    }

    Degree initial_degree(NodeId node) const {
        assert(is_local_from_local(node));
        return first_out_[node + 1] - first_out_[node];
    }

    [[nodiscard]] inline NodeId to_global_id(NodeId local_node_id) const {
        assert((is_local_from_local(local_node_id) || is_ghost(local_node_id)));
        if (!is_ghost(local_node_id)) {
            if (consecutive_vertices_) {
                return node_range_.first + local_node_id;
            } else {
                return local_data_[local_node_id].global_id;
            }
        } else {
            return get_ghost_data(local_node_id).global_id;
        }
    }

    [[nodiscard]] inline Edge to_global_edge(const Edge& edge) const {
        return Edge{to_global_id(edge.tail), to_global_id(edge.head)};
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

    [[nodiscard]] inline const GraphPayload& get_graph_payload() const {
        return graph_payload_;
    }

    [[nodiscard]] inline GraphPayload& get_graph_payload() {
        return graph_payload_;
    }

    [[nodiscard]] inline const GhostPayloadType& get_ghost_payload(NodeId local_node_id) const {
        assert(is_ghost(local_node_id));
        NodeId ghost_index = local_node_id - local_node_count_;
        return ghost_data_[ghost_index].payload;
    }

    [[nodiscard]] inline GhostPayloadType& get_ghost_payload(NodeId local_node_id) {
        assert(is_ghost(local_node_id));
        NodeId ghost_index = local_node_id - local_node_count_;
        return ghost_data_[ghost_index].payload;
    }

    [[nodiscard]] inline const GhostData<GhostPayloadType>& get_ghost_data(NodeId local_node_id) const {
        assert(is_ghost(local_node_id));
        NodeId ghost_index = local_node_id - local_node_count_;
        return ghost_data_[ghost_index];
    }

    [[nodiscard]] inline const LocalData& get_local_data(NodeId local_node_id) const {
        assert(is_local_from_local(local_node_id));
        return local_data_[local_node_id];
    }

    template <typename = std::enable_if_t<payload_has_degree<payload_type>::value>>
    inline bool is_outgoing(const Edge& e) const {
        return std::forward_as_tuple(degree(e.tail), to_global_id(e.tail)) <
               std::forward_as_tuple(degree(e.head), to_global_id(e.head));
    }

    DistributedGraph() {
        first_out_.push_back(0);
    };

    inline NodeId total_node_count() const {
        return total_node_count_;
    }

    inline std::pair<NodeId, NodeId> node_range() const {
        return node_range_;
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
        if constexpr (payload_has_degree<GhostPayloadType>::value) {
            get_graph_payload().ghost_degree_available = false;
        }
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
        for (size_t i = 0; i < ghost_neighbors.size(); ++i) {
            first_out_[local_node_count_ + i + 1] = first_out_[local_node_count_ + i] + ghost_neighbors[i].size();
            degree_[local_node_count_ + i] = ghost_neighbors[i].size();
            for (NodeId neighbor : ghost_neighbors[i]) {
                head_[edge_index] = neighbor;
                edge_index++;
            }
        }
        ghosts_expanded_ = true;
    }

    DistributedGraph(cetric::graph::LocalGraphView&& G, PEID rank, PEID size)
        : first_out_(G.local_node_count() + 1),
          first_out_offset_(G.local_node_count()),
          degree_(G.local_node_count()),
          head_(),
          ghost_data_(),
          local_data_(G.local_node_count()),
          consecutive_vertices_(false),
          ghost_ranks_available_(false),
          oriented_(false),
          ghosts_expanded_(false),
          global_to_local_(),
          local_node_count_(G.local_node_count()),
          local_edge_count_{G.edge_heads.size()},
          total_node_count_{},
          node_range_(std::numeric_limits<NodeId>::max(), std::numeric_limits<NodeId>::min()),
          rank_(rank),
          size_(size) {
        global_to_local_.set_empty_key(-1);
        global_to_local_.set_deleted_key(-2);
        auto degree_sum = 0;
        for (size_t i = 0; i < local_node_count_; ++i) {
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
        consecutive_vertices_ = node_range_.second - node_range_.first + 1 == local_node_count_;

        first_out_[local_node_count_] = degree_sum;
        assert(first_out_.size() == local_node_count_ + 1);
        assert(first_out_[0] == 0);

        for (size_t i = 0; i < local_node_count_; ++i) {
            assert(degree_[i] == G.node_info[i].degree);
            assert(first_out_[i + 1] - first_out_[i] == degree_[i]);
        }
        assert(first_out_[first_out_.size() - 1] == G.edge_heads.size());
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
                        ghost_count++;
                    }
                }
                head_[edge_id] = to_local_id(head);
            }
        }
        ghost_data_.resize(ghost_count);
        for (const auto& kv : global_to_local_) {
            NodeId global_id = kv.first;
            NodeId local_id = kv.second;
            if (local_id < local_node_count_) {
                continue;
            }
            // assert(local_id  - local_node_count_ < ghost_data_.size());
            ghost_data_[local_id - local_node_count_].global_id = global_id;
            ghost_data_[local_id - local_node_count_].rank = -1;
        }
    }

    void find_ghost_ranks() {
        if (ghost_ranks_available()) {
            return;
        }
        // if (consecutive_vertices_) {
        // atomic_debug("consecutive vertices");
        std::vector<std::pair<NodeId, NodeId>> ranges(size_);
        gather_PE_ranges(node_range_.first, node_range_.second, ranges, MPI_COMM_WORLD, rank_, size_);
        for_each_ghost_node([&](NodeId node) {
            PEID rank = get_PE_from_node_ranges(to_global_id(node), ranges);
            ghost_data_[ghost_to_ghost_index(node)].rank = rank;
        });
        // } else {
        //     atomic_debug("No consecutive vertices");
        //     std::vector<NodeId> nodes(local_node_count_);
        //     for_each_local_node([&](NodeId node) {
        //         nodes[node] = to_global_id(node);
        //     });
        //     auto [all_nodes, displs] = CommunicationUtility::all_gather(nodes, MPI_NODE, MPI_COMM_WORLD, rank_,
        //     size_); for (PEID rank = 0; rank < size_; rank++) {
        //         for (int i = displs[rank]; i < displs[rank + 1]; ++i) {
        //             NodeId node = all_nodes[i];
        //             if (is_ghost_from_global(node)) {
        //                 NodeId local_id = to_local_id(node);
        //                 ghost_data_[ghost_to_ghost_index(local_id)].rank = rank;
        //             }
        //         }
        //     }
        // }
        ghost_ranks_available_ = true;
    }

    LocalGraphView to_local_graph_view(bool remove_isolated, bool keep_only_out_edges) {
        DistributedGraph G = std::move(*this);
        std::vector<LocalGraphView::NodeInfo> node_info;
        EdgeId edge_counter = 0;
        G.for_each_local_node([&](NodeId node) {
            auto degree = G.degree(node);
            // TODO we need to handle vertices with out degree 0
            // maybe prevent sending to them, as we should know their degree (?)
            if (remove_isolated && degree == 0) {
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

    bool ghost_ranks_available() const {
        return ghost_ranks_available_;
    }

    PEID rank() const {
        return rank_;
    }

    PEID size() const {
        return size_;
    }

    void check_consistency() {
#ifdef CETRIC_CHECK_GRAPH_CONSISTENCY
        for_each_local_node([&](NodeId node) {
            DEBUG_ASSERT(to_local_id(get_local_data(node).global_id) == node, debug_module{});
            Degree out_deg = 0;
            Degree in_deg = 0;
            Degree deg = 0;
            if (oriented()) {
                for_each_local_out_edge(node, [&](Edge) { out_deg++; });
                for_each_local_in_edge(node, [&](Edge) { in_deg++; });
            }
            for_each_edge(node, [&](Edge) { deg++; });
            if (oriented()) {
                auto message = fmt::format("[R{}] {} != {} for node {}, interface {}", rank_, outdegree(node), out_deg,
                                           to_global_id(node), get_local_data(node).is_interface);
                DEBUG_ASSERT(outdegree(node) == out_deg, debug_module{}, message.c_str());
                DEBUG_ASSERT(degree(node) == out_deg + in_deg, debug_module{}, message.c_str());
                DEBUG_ASSERT(degree(node) == deg, debug_module{}, message.c_str());
            } else {
                DEBUG_ASSERT(degree(node) == deg, debug_module{});
            }
        });
        for_each_ghost_node([&](NodeId node) {
            DEBUG_ASSERT(!is_local_from_local(node), debug_module{});
            DEBUG_ASSERT(is_ghost(node), debug_module{});
            DEBUG_ASSERT(to_local_id(get_ghost_data(node).global_id) == node, debug_module{});
        });
        for_each_local_node([&](NodeId node) {
            for_each_edge(
                node, [&](Edge e) { DEBUG_ASSERT(is_local_from_local(e.head) != is_ghost(e.head), debug_module{}); });
        });
#endif
    }

private:
    NodeId ghost_to_ghost_index(NodeId local_node_id) const {
        assert(is_ghost(local_node_id));
        return local_node_id - local_node_count_;
    }

    using node_map = google::dense_hash_map<NodeId, NodeId>;
    std::vector<EdgeId> first_out_;
    std::vector<EdgeId> first_out_offset_;
    std::vector<Degree> degree_;
    std::vector<NodeId> head_;
    std::vector<GhostData<GhostPayloadType>> ghost_data_;
    std::vector<LocalData> local_data_;
    bool consecutive_vertices_;
    bool ghost_ranks_available_;
    bool oriented_;
    bool ghosts_expanded_;
    GraphPayload graph_payload_;
    node_map global_to_local_;
    NodeId local_node_count_{};
    EdgeId local_edge_count_{};
    NodeId total_node_count_{};
    std::pair<NodeId, NodeId> node_range_;
    PEID rank_;
    PEID size_;
};

template <typename GhostPayloadType>
inline std::ostream& operator<<(std::ostream& out, DistributedGraph<GhostPayloadType>& G) {
    G.for_each_local_node([&](NodeId node) {
        G.for_each_local_out_edge(node, [&](Edge edge) {
            out << "(" << G.to_global_id(edge.tail) << ", " << G.to_global_id(edge.head) << ")" << std::endl;
        });
    });
    return out;
}

}  // namespace graph
}  // namespace cetric
#endif  // PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H
