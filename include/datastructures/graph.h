//
// Created by Tim Niklas Uhl on 19.11.20.
//

#pragma once

#include <graph-io/local_graph_view.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>
#include "../util.h"
#include "debug_assert.hpp"
#include "fmt/core.h"
#include "graph-io/graph.h"
#include "graph_definitions.h"

namespace cetric {
namespace graph {

class AdjacencyGraph {
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

        bool operator<(const NodeIterator& rhs) const {
            return this->node_ < rhs.node_;
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
        NodeIterator operator+(difference_type n) const {
            return NodeIterator(this->node_ + n);
        }

        difference_type operator-(const NodeIterator& rhs) const {
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
        difference_type operator-(const EdgeIterator& rhs) const {
            return this->iter_ - rhs.iter_;
        }

        EdgeIterator operator-(difference_type n) const {
            return EdgeIterator(this->tail_, this->iter_ - n);
        }

        bool operator<(const EdgeIterator& rhs) const {
            return this->iter_ < rhs.iter_;
        }

        EdgeIterator& operator+=(difference_type n) {
            this->iter_ += n;
            return *this;
        }

        EdgeIterator operator+(difference_type n) const {
            return EdgeIterator(this->tail_, this->iter_ + n);
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

    inline NodeId local_node_count() const {
        return first_out_.size() - 1;
    }

    inline EdgeId local_edge_count() const {
        return local_edge_count_;
    }

    template <typename NodeFunc>
    inline void for_each_local_node(NodeFunc on_node) const {
        for (size_t node = 0; node < local_node_count(); ++node) {
            on_node(node);
        }
    }

    template <typename NodeFunc, typename Partitioner = tbb::auto_partitioner>
    inline void parallel_for_each_local_node(NodeFunc on_node,
                                             size_t grainsize = 1,
                                             Partitioner&& partitioner = Partitioner{}) const {
        tbb::parallel_for(
            tbb::blocked_range<NodeId>(0, local_node_count(), grainsize),
            [&](auto const& r) {
                for (NodeId node = r.begin(); node != r.end(); node++) {
                    on_node(node);
                }
            },
            partitioner);
    }

    NodeRange local_nodes() const {
        return NodeRange(0, local_node_count());
    }

    template <typename NodeFunc>
    inline void intersect_neighborhoods(NodeId u, NodeId v, NodeFunc on_intersection) const {
        EdgeId u_current_edge = first_out_[u] + first_out_offset_[u];
        EdgeId u_end = first_out_[u] + degree_[u];
        EdgeId v_current_edge = first_out_[v] + first_out_offset_[v];
        EdgeId v_end = first_out_[v] + degree_[v];
        while (u_current_edge != u_end && v_current_edge != v_end) {
            NodeId u_node = head_[u_current_edge];
            NodeId v_node = head_[v_current_edge];
            if (u_node < v_node) {
                u_current_edge++;
            } else if (u_node > v_node) {
                v_current_edge++;
            } else {
                // u_node == v_node
                on_intersection(u_node);
                u_current_edge++;
                v_current_edge++;
            }
        }
    }

    template <typename NodeFunc, typename Iter>
    inline void intersect_neighborhoods(NodeId u, Iter begin, Iter end, NodeFunc on_intersection) const {
        EdgeId u_current_edge = first_out_[u] + first_out_offset_[u];
        EdgeId u_end = first_out_[u] + degree_[u];
        while (u_current_edge != u_end && begin != end) {
            NodeId u_node = head_[u_current_edge];
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

    template <typename EdgeFunc, typename Partitioner = tbb::auto_partitioner>
    inline void parallel_for_each_edge(NodeId node,
                                       EdgeFunc on_edge,
                                       size_t grainsize = 1,
                                       Partitioner&& partitioner = Partitioner{}) const {
        EdgeId begin = first_out_[node];
        // TODO change it!!!
        EdgeId end = first_out_[node] + degree_[node];
        // EdgeId end = first_out_[node] + first_out_[node + 1];
        tbb::parallel_for(
            tbb::blocked_range(begin, end, grainsize),
            [&](const auto& r) {
                for (size_t edge_id = r.begin(); edge_id < r.end(); ++edge_id) {
                    NodeId head = head_[edge_id];
                    Edge edge{node, head};
                    on_edge(edge);
                }
            },
            partitioner);
    }

    EdgeRange edges(NodeId node) const {
        EdgeId begin = first_out_[node];
        EdgeId end = first_out_[node] + degree_[node];
        return EdgeRange(node, head_.cbegin() + begin, head_.cbegin() + end);
    }

    template <typename EdgeFunc>
    inline void for_each_local_out_edge(NodeId node, EdgeFunc on_edge) const {
        auto begin = first_out_[node] + first_out_offset_[node];
        auto end = first_out_[node] + degree_[node];
        for (EdgeId edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            on_edge(Edge(node, head));
        }
    }

    template <typename EdgeFunc, typename Partitioner = tbb::auto_partitioner>
    inline void parallel_for_each_local_out_edge(NodeId node,
                                                 EdgeFunc on_edge,
                                                 size_t grainsize = 1,
                                                 Partitioner&& partitioner = Partitioner{}) const {
        auto begin = first_out_[node] + first_out_offset_[node];
        auto end = first_out_[node] + degree_[node];
        tbb::parallel_for(
            tbb::blocked_range(begin, end, grainsize),
            [&on_edge, this, node](const auto& r) {
                for (size_t edge_id = r.begin(); edge_id < r.end(); ++edge_id) {
                    NodeId head = head_[edge_id];
                    Edge edge{node, head};
                    on_edge(edge);
                }
            },
            partitioner);
    }

    EdgeRange out_edges(NodeId node) const {
        auto begin = first_out_[node] + first_out_offset_[node];
        auto end = first_out_[node] + degree_[node];
        return EdgeRange(node, head_.cbegin() + begin, head_.cbegin() + end);
    }

    template <typename EdgeFunc>
    inline void for_each_local_in_edge(NodeId node, EdgeFunc on_edge) const {
        auto begin = first_out_[node];
        auto end = first_out_[node] + first_out_offset_[node];
        for (EdgeId edge_id = begin; edge_id < end; ++edge_id) {
            NodeId head = head_[edge_id];
            on_edge(Edge(head, node));
        }
    }

    template <typename EdgeFunc, typename Partitioner = tbb::auto_partitioner>
    inline void parallel_for_each_local_in_edge(NodeId node,
                                                EdgeFunc on_edge,
                                                size_t grainsize = 1,
                                                Partitioner&& partitioner = Partitioner{}) const {
        auto begin = first_out_[node];
        auto end = first_out_[node] + first_out_offset_[node];
        tbb::parallel_for(
            tbb::blocked_range(begin, end, grainsize),
            [&](const auto& r) {
                for (size_t edge_id = r.begin(); edge_id < r.end(); ++edge_id) {
                    NodeId head = head_[edge_id];
                    Edge edge{node, head};
                    on_edge(edge);
                }
            },
            partitioner);
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
        for_each_local_node(orient_neighborhoods);
        oriented_ = true;
    }

    inline bool oriented() const {
        return oriented_;
    }

    template <class ExecutionPolicy = execution_policy::sequential>
    inline void sort_neighborhoods(ExecutionPolicy&& policy [[maybe_unused]] = ExecutionPolicy{}) {
        auto on_node = [&](NodeId node) {
            EdgeId begin = first_out_[node];
            EdgeId in_end = first_out_[node] + first_out_offset_[node];
            EdgeId out_end = first_out_[node] + degree_[node];
            auto node_cmp = [&](NodeId a, NodeId b) {
                return a < b;
            };
            std::sort(head_.begin() + begin, head_.begin() + in_end, node_cmp);
            std::sort(head_.begin() + in_end, head_.begin() + out_end, node_cmp);
        };
        for_each_local_node(on_node);
    }

    inline Degree local_degree(NodeId node) const {
        return degree_[node];
    }

    inline Degree local_outdegree(NodeId node) const {
        assert(oriented());
        return degree_[node] - first_out_offset_[node];
    }

    Degree initial_degree(NodeId node) const {
        return first_out_[node + 1] - first_out_[node];
    }

    AdjacencyGraph() {
        first_out_.push_back(0);
    };

    AdjacencyGraph(graphio::Graph&& G)
        : first_out_(std::move(G.first_out_)),
          first_out_offset_(first_out_.size() - 1),
          degree_(first_out_.size() - 1),
          head_(std::move(G.head_)),
          oriented_(false),
          local_edge_count_(head_.size()) {
        for (size_t i = 0; i < degree_.size(); i++) {
            degree_[i] = first_out_[i + 1] - first_out_[i];
        }
    }

    AdjacencyGraph(graphio::LocalGraphView&& G)
        : first_out_(G.local_node_count() + 1),
          first_out_offset_(G.local_node_count()),
          degree_(G.local_node_count()),
          head_(),
          oriented_(false),
          local_edge_count_{G.edge_heads.size()} {
        auto degree_sum = 0;
        for (size_t i = 0; i < local_node_count(); ++i) {
            first_out_[i] = degree_sum;
            degree_sum += G.node_info[i].degree;
            NodeId local_id = i;
            degree_[local_id] = G.node_info[i].degree;
        }

        first_out_[local_node_count()] = degree_sum;
        assert(first_out_[0] == 0);

        for (size_t i = 0; i < local_node_count(); ++i) {
            assert(degree_[i] == G.node_info[i].degree);
            assert(first_out_[i + 1] - first_out_[i] == degree_[i]);
        }
        assert(first_out_[first_out_.size() - 1] == G.edge_heads.size());
        head_ = std::move(G.edge_heads);
    }

    std::vector<EdgeId> first_out_;
    std::vector<EdgeId> first_out_offset_;
    std::vector<Degree> degree_;
    std::vector<NodeId> head_;
    bool oriented_;
    EdgeId local_edge_count_{};
};

}  // namespace graph
}  // namespace cetric
