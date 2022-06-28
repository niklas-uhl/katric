//
// Created by Tim Niklas Uhl on 19.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H
#define PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H

#include <communicator.h>
#include <datastructures/graph_definitions.h>
#include <graph-io/local_graph_view.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <util.h>
#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/iterator_categories.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <default_hash.hpp>
#include <google/dense_hash_set>
#include <google/sparse_hash_map>
#include <iterator>
#include <kassert/kassert.hpp>
#include <limits>
#include <optional>
#include <set>
#include <sparsehash/dense_hash_set>
#include <sstream>
#include <stdexcept>
#include <tlx/vector_free.hpp>
#include <tuple>
#include <type_traits>
#include <vector>
#include "datastructures/distributed/helpers.h"
#include "debug_assert.hpp"
#include "fmt/core.h"

namespace cetric {
namespace load_balancing {
class LoadBalancer;
}
namespace graph {

using LocalGraphView = graphio::LocalGraphView;

struct ContinuousNodeIndexer {
    ContinuousNodeIndexer()
        : from(std::numeric_limits<NodeId>::max()), to(std::numeric_limits<NodeId>::max()), rank(-1) {}
    template <typename NodeIterator>
    ContinuousNodeIndexer(std::pair<NodeId, NodeId> const& node_range,
                          NodeIterator begin [[maybe_unused]],
                          NodeIterator end [[maybe_unused]],
                          PEID rank)
        : from(node_range.first), to(node_range.second), rank(rank) {}

    bool is_indexed(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank);
        return node.id() >= from && node.id() < to;
    }

    auto nodes() const {
        using counting_iterator_type =
            boost::counting_iterator<RankEncodedNodeId, boost::random_access_traversal_tag, std::ptrdiff_t>;
        return boost::make_iterator_range(counting_iterator_type{RankEncodedNodeId(from, rank)},
                                          counting_iterator_type{RankEncodedNodeId(to, rank)});
    }

    inline size_t get_index(RankEncodedNodeId node) const {
        KASSERT(is_indexed(node));
        auto idx = node.id() - from;
        return idx;
    }

    inline RankEncodedNodeId get_node(size_t idx) const {
        KASSERT(idx < to - from);
        return RankEncodedNodeId{from + idx, static_cast<uint16_t>(rank)};
    }
    inline size_t size() const {
        return to - from;
    }

    NodeId from;
    NodeId to;
    PEID rank;
};

struct SparseNodeIndexer {
    SparseNodeIndexer() : id_map(), idx_map(), rank(-1) {}

    template <typename NodeIterator>
    SparseNodeIndexer(std::pair<NodeId, NodeId> const& node_range [[maybe_unused]],
                      NodeIterator begin,
                      NodeIterator end,
                      PEID rank)
        : id_map(begin, end), idx_map(id_map.size()), rank(rank) {
        for (size_t i = 0; i < id_map.size(); i++) {
            idx_map[id_map[i]] = i;
        }
    }

    bool is_indexed(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank);
        return idx_map.find(node) != idx_map.end();
    }
    auto nodes() const {
        return boost::make_iterator_range(id_map.begin(), id_map.end());
    }

    inline size_t get_index(RankEncodedNodeId node) const {
        KASSERT(is_indexed(node));
        auto it = idx_map.find(node);
        KASSERT(it != idx_map.end());
        return it->second;
    }

    inline RankEncodedNodeId get_node(size_t idx) const {
        KASSERT(idx < id_map.size());
        return id_map[idx];
    }

    inline size_t size() const {
        return id_map.size();
    }

    std::vector<RankEncodedNodeId> id_map;
    node_map<size_t> idx_map;
    PEID rank;
};

template <typename NodeIndexer = ContinuousNodeIndexer>
class DistributedGraph {
    friend class cetric::load_balancing::LoadBalancer;
    friend class GraphBuilder;
    friend class CompactGraph;

public:
    template <typename NodeIdType>
    class NodeRange {
    public:
        explicit NodeRange(NodeIdType from, NodeIdType to) : from_(std::move(from)), to_(std::move(to)) {}
        boost::counting_iterator<NodeIdType> begin() const {
            return boost::counting_iterator(from_);
        }
        boost::counting_iterator<NodeIdType> end() const {
            return boost::counting_iterator(to_);
        }

    private:
        NodeIdType from_;
        NodeIdType to_;
    };

    enum RangeModifiability { modifiable, non_modifiable };
    template <typename NodeIdType, RangeModifiability modifiability>
    class EdgeRange {
    public:
        using head_iterator_type = std::conditional_t<modifiability == RangeModifiability::modifiable,
                                                      typename std::vector<NodeIdType>::iterator,
                                                      typename std::vector<NodeIdType>::const_iterator>;
        explicit EdgeRange(NodeIdType tail, head_iterator_type&& begin, head_iterator_type&& end)
            : tail_(tail), begin_(std::move(begin)), end_(std::move(end)) {}

        auto neighbors() const {
            return boost::make_iterator_range(begin_, end_);
        }
        auto edges() const {
            return boost::adaptors::transform(neighbors(), [tail = tail_](NodeIdType neighbor) {
                return graphio::Edge<NodeIdType>{RankEncodedNodeId(tail), neighbor};
            });
        }

    private:
        NodeIdType tail_;
        head_iterator_type begin_;
        head_iterator_type end_;
    };

    inline NodeId local_node_count() const {
        return node_range_.second - node_range_.first;
    }

    inline EdgeId local_edge_count() const {
        return local_edge_count_;
    }

    auto local_nodes() const {
        return node_indexer_.nodes();
    }

    EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable> adj(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank_, "Node " << node << " is not local.");
        auto idx = to_local_idx(node);
        EdgeId begin = first_out_[idx];
        EdgeId end = first_out_[idx] + degree_[idx];
        return EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable>{node, head_.cbegin() + begin,
                                                                                head_.cbegin() + end};
    }

    EdgeRange<RankEncodedNodeId, RangeModifiability::modifiable> adj(RankEncodedNodeId node) {
        KASSERT(node.rank() == rank_, "Node " << node << " is not local.\n" << get_stacktrace());
        auto idx = to_local_idx(node);
        EdgeId begin = first_out_[idx];
        EdgeId end = first_out_[idx] + degree_[idx];
        return EdgeRange<RankEncodedNodeId, RangeModifiability::modifiable>{node, head_.begin() + begin,
                                                                            head_.begin() + end};
    }

    EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable> out_adj(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank_, "Node " << node << " is not local.");
        auto idx = to_local_idx(node);
        auto begin = first_out_[idx] + first_out_offset_[idx];
        auto end = first_out_[idx] + degree_[idx];
        return EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable>{node, head_.cbegin() + begin,
                                                                                head_.cbegin() + end};
    }

    EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable> in_adj(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank_, "Node " << node << " is not local.");
        auto idx = to_local_idx(node);
        auto begin = first_out_[idx];
        auto end = first_out_[idx] + first_out_offset_[idx];
        return EdgeRange<RankEncodedNodeId, RangeModifiability::non_modifiable>{node, head_.cbegin() + begin,
                                                                                head_.cbegin() + end};
    }

    template <typename NodeCmp>
    inline void orient(RankEncodedNodeId node, NodeCmp&& node_cmp) {
        auto is_outgoing = [&](auto tail, auto head) {
            auto out = node_cmp(tail, head);
            return out;
        };
        auto idx = to_local_idx(node);
        int64_t left = first_out_[idx];
        int64_t right = first_out_[idx] + degree_[idx] - 1;
        while (left <= right) {
            while (left <= right && !is_outgoing(node, head_[left])) {
                left++;
            }
            while (right >= (int64_t)first_out_[idx] && is_outgoing(node, head_[right])) {
                right--;
            }
            if (left <= right) {
                std::iter_swap(head_.begin() + left, head_.begin() + right);
                // left++;
                // right--;
            }
        }
        first_out_offset_[idx] = left - first_out_[idx];
        // atomic_debug(head_);
        // if (ghosts_expanded_) {
        //     for_each_local_node_and_ghost(orient_neighborhoods);
        // } else {
        //     for_each_local_node(orient_neighborhoods);
        // };
    }

    inline bool oriented() const {
        return oriented_;
    }

    // inline void set_ghost_payload(NodeId local_node_id, GhostPayloadType payload) {
    //     // assert(is_ghost(local_node_id));
    //     ghost_data_[local_node_id - local_node_count_].payload = std::move(payload);
    // }

    template <typename NodeCmp>
    inline void sort_neighborhoods(RankEncodedNodeId node, NodeCmp&& node_cmp) {
        auto idx = to_local_idx(node);
        EdgeId begin = first_out_[idx];
        EdgeId in_end = first_out_[idx] + first_out_offset_[idx];
        EdgeId out_end = first_out_[idx] + degree_[idx];
        std::sort(head_.begin() + begin, head_.begin() + in_end, node_cmp);
        std::sort(head_.begin() + in_end, head_.begin() + out_end, node_cmp);
        {
            auto neighbors = out_adj(node).neighbors();
            KASSERT(std::is_sorted(neighbors.begin(), neighbors.end(), node_cmp));
        }
        {
            auto neighbors = in_adj(node).neighbors();
            KASSERT(std::is_sorted(neighbors.begin(), neighbors.end(), node_cmp));
        }
    }

    inline Degree degree(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank_);
        return degree_[to_local_idx(node)];
    }

    inline Degree outdegree(RankEncodedNodeId node) const {
        KASSERT(node.rank() == rank_);
        auto idx = to_local_idx(node);
        return degree_[idx] - first_out_offset_[idx];
    }

    DistributedGraph() {
        first_out_.push_back(0);
    };

    inline std::pair<NodeId, NodeId> node_range() const {
        return node_range_;
    }

    inline size_t to_local_idx(RankEncodedNodeId node) const {
        return node_indexer_.get_index(node);
    }

    inline bool is_interface_node_if_sorted_by_rank(RankEncodedNodeId node) const {
        // KASSERT(oriented());
        KASSERT(edges_last_outward_sorted(node));
        KASSERT(node.rank() == rank_);

        return !degree(node) == 0 && (adj(node).neighbors().end() - 1)->rank() != rank_;
    }

    inline bool is_interface_node(RankEncodedNodeId node) const {
        auto neighbors = adj(node).neighbors();
        auto it = std::find_if(neighbors.begin(), neighbors.end(),
                               [rank = rank_](RankEncodedNodeId node) { return node.rank() != rank; });
        bool is_interface = it != neighbors.end();
        KASSERT(is_interface == std::any_of(neighbors.begin(), neighbors.end(),
                                            [rank = rank_](RankEncodedNodeId node) { return node.rank() != rank; }));
        return is_interface;
    }

    bool edges_last_outward_sorted(RankEncodedNodeId node) const {
        auto it = std::find_if(adj(node).neighbors().begin(), adj(node).neighbors().end(),
                               [rank = rank_](auto v) { return v.rank() != rank; });
        auto it2 = std::find_if(it, adj(node).neighbors().end(), [rank = rank_](auto v) { return v.rank() == rank; });
        return it2 == adj(node).neighbors().end();
    }

    void remove_internal_edges(RankEncodedNodeId node) {
        KASSERT(node.rank() == rank_);
        KASSERT(edges_last_outward_sorted(node));
        if (!is_interface_node_if_sorted_by_rank(node)) {
            degree_[to_local_idx(node)] = 0;
            first_out_offset_[to_local_idx(node)] = 0;
            return;
        }
        auto neighbors = adj(node).neighbors();
        auto it = std::find_if(neighbors.begin(), neighbors.end(),
                               [this](RankEncodedNodeId node) { return node.rank() != rank_; });
        size_t old_degree = std::distance(neighbors.begin(), neighbors.end());
        size_t dist = std::distance(neighbors.begin(), it);
        size_t degree = std::distance(it, neighbors.end());
        first_out_[to_local_idx(node)] = first_out_[to_local_idx(node)] + dist;
        first_out_offset_[to_local_idx(node)] = first_out_[to_local_idx(node)];
        degree_[to_local_idx(node)] = degree;
        local_edge_count_ -= old_degree - degree;
    }

    // void expand_ghosts() {
    //     if (ghosts_expanded_) {
    //         return;
    //     }
    //     std::vector<std::vector<NodeId>> ghost_neighbors(ghost_count());
    //     size_t ghost_edges = 0;
    //     for_each_local_node([&](NodeId node) {
    //         for_each_edge(node, [&](RankEncodedEdge edge) {
    //             if (is_ghost(edge.head.id())) {
    //                 ghost_neighbors[ghost_to_ghost_index(edge.head.id())].push_back(edge.tail.id());
    //                 ghost_edges++;
    //             }
    //         });
    //     });
    //     size_t edge_index = head_.size();
    //     head_.resize(head_.size() + ghost_edges);
    //     local_edge_count_ = head_.size() + ghost_edges;
    //     first_out_.resize(first_out_.size() + ghost_count());
    //     first_out_offset_.resize(first_out_offset_.size() + ghost_count(), 0);
    //     degree_.resize(degree_.size() + ghost_count());
    //     for (size_t i = 0; i < ghost_neighbors.size(); ++i) {
    //         first_out_[local_node_count_ + i + 1] = first_out_[local_node_count_ + i] + ghost_neighbors[i].size();
    //         degree_[local_node_count_ + i] = ghost_neighbors[i].size();
    //         for (NodeId neighbor : ghost_neighbors[i]) {
    //             head_[edge_index] = RankEncodedNodeId{neighbor};
    //             edge_index++;
    //         }
    //     }
    //     ghosts_expanded_ = true;
    // }

    DistributedGraph(cetric::graph::LocalGraphView&& G, std::pair<NodeId, NodeId> node_range, PEID rank, PEID size)
        : first_out_(node_range.second - node_range.first + 1),
          first_out_offset_(first_out_.size() - 1),
          degree_(first_out_.size() - 1),
          head_(),
          ghost_ranks_available_(false),
          oriented_(false),
          local_edge_count_{G.edge_heads.size()},
          node_range_(std::move(node_range)),
          rank_(rank),
          size_(size),
          node_indexer_() {
        // global_to_local_.set_empty_key(-1);
        auto degree_sum = 0;
        // node_range_.first = G.node_info[0].global_id;
        // node_range_.second = G.node_info.back().global_id;
        NodeId current_node = node_range_.first;
        KASSERT(std::is_sorted(G.node_info.begin(), G.node_info.end(),
                               [](auto const& lhs, auto const& rhs) { return lhs.global_id < rhs.global_id; }),
                "Nodes are not sorted");
        if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
            for (auto const& node_info : G.node_info) {
                KASSERT(node_info.global_id >= node_range_.first);
                KASSERT(node_info.global_id < node_range_.second);
            }
        }
        auto nodes = boost::adaptors::transform(
            boost::make_iterator_range(G.node_info.begin(), G.node_info.end()), [rank](auto node_info) {
                return RankEncodedNodeId{node_info.global_id, static_cast<std::uint16_t>(rank)};
            });
        node_indexer_ = NodeIndexer(node_range, nodes.begin(), nodes.end(), rank);
        for (size_t i = 0; i < G.node_info.size(); i++) {
            while (current_node != G.node_info[i].global_id) {
                // insert empty nodes for all nodes in range that have no edges
                auto idx = to_local_idx(RankEncodedNodeId{current_node, static_cast<std::uint16_t>(rank_)});
                first_out_[idx] = degree_sum;
                degree_[idx] = 0;
                current_node++;
            }
            auto idx = to_local_idx(RankEncodedNodeId{current_node, static_cast<std::uint16_t>(rank_)});
            first_out_[idx] = degree_sum;
            degree_sum += G.node_info[i].degree;
            degree_[idx] = G.node_info[i].degree;
            current_node++;
        }
        while (current_node != node_range_.second) {
            // insert empty nodes for all nodes in range that have no edges
            auto idx = to_local_idx(RankEncodedNodeId{current_node, static_cast<std::uint16_t>(rank_)});
            first_out_[idx] = degree_sum;
            degree_[idx] = 0;
            current_node++;
        }

        first_out_[local_node_count()] = degree_sum;
        KASSERT(first_out_.size() == local_node_count() + 1);
        KASSERT(first_out_[0] == 0ul);

        // for (size_t i = 0; i < local_node_count(); ++i) {
        //     KASSERT(degree_[i] == G.node_info[i].degree);
        //     KASSERT(first_out_[i + 1] - first_out_[i] == degree_[i]);
        // }
        KASSERT(first_out_[first_out_.size() - 1] == G.edge_heads.size());
        head_.resize(G.edge_heads.size());
        std::transform(G.edge_heads.begin(), G.edge_heads.end(), head_.begin(), [this](auto id) {
            auto head = RankEncodedNodeId{id};
            if (head.id() >= node_range_.first && head.id() < node_range_.second) {
                head.set_rank(rank_);
            }
            return head;
        });
        tlx::vector_free(G.edge_heads);
    }

    template <typename ExecutionPolicy = execution_policy::sequential>
    void find_ghost_ranks(ExecutionPolicy&& = {}) {
        if (ghost_ranks_available()) {
            return;
        }
        // if (consecutive_vertices_) {
        // atomic_debug("consecutive vertices");
        std::vector<std::pair<NodeId, NodeId>> ranges(size_);
        gather_PE_ranges(node_range_.first, node_range_.second, ranges, MPI_COMM_WORLD, rank_, size_);
        auto nodes = local_nodes();
        if constexpr (std::is_same_v<ExecutionPolicy, execution_policy::parallel>) {
            tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&ranges, this](auto const& r) {
                for (auto node : r) {
                    for (auto& neighbor : this->adj(node).neighbors()) {
                        PEID rank = get_PE_from_node_ranges(neighbor.id(), ranges);
                        neighbor.set_rank(rank);
                    }
                }
            });
        } else {
            for (auto node : nodes) {
                for (auto& neighbor : this->adj(node).neighbors()) {
                    PEID rank = get_PE_from_node_ranges(neighbor.id(), ranges);
                    neighbor.set_rank(rank);
                }
            }
        }
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
        for (RankEncodedNodeId node : G.local_nodes()) {
            auto degree = G.degree(node);
            // TODO we need to handle vertices with out degree 0
            // maybe prevent sending to them, as we should know their degree (?)
            if (remove_isolated && degree == 0) {
                continue;
            }
            // auto global_id = G.to_global_id(node);
            if (keep_only_out_edges) {
                degree = G.outdegree(node);
            }
            node_info.emplace_back(node.id(), degree);
            if (keep_only_out_edges) {
                for (RankEncodedNodeId head : G.out_adj(node).neighbors()) {
                    G.head_[edge_counter] = head;
                    edge_counter++;
                }
            } else {
                for (RankEncodedNodeId head : G.adj(node).neighbors()) {
                    G.head_[edge_counter] = head;
                    edge_counter++;
                }
            }
        }
        G.head_.resize(edge_counter);
        G.head_.shrink_to_fit();
        auto head = std::move(G.head_);
        LocalGraphView view;
        view.node_info = std::move(node_info);
        view.edge_heads.resize(head.size());
        std::transform(head.begin(), head.end(), view.edge_heads.begin(), [](auto id) { return id.id(); });
        tlx::vector_free(head);
        return view;
    }

    // DistributedGraph<SparseNodeIndexer> compact() {
    //     DistributedGraph<SparseNodeIndexer> G_compact;
    //     auto remaining_nodes =
    //         boost::adaptors::filter(this->local_nodes(), [this](auto node) { return this->degree(node) > 0; });
    //     SparseNodeIndexer sparse_indexer(this->node_range_, remaining_nodes.begin(), remaining_nodes.end(), rank_);
    //     for (size_t i = 0; i < sparse_indexer.size(); ++i) {
    //         first_out_[i] = this->degree(sparse_indexer.get_node(i));
    //         degree_[i] = this->degree(sparse_indexer.get_node(i));
    //         first_out_offset_[i] = this->degree(sparse_indexer.get_node(i));
    //     }
    // }

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
    void check_edge_consistency() {
        std::vector<std::pair<NodeId, NodeId>> ranges(size_);
        gather_PE_ranges(node_range_.first, node_range_.second, ranges, MPI_COMM_WORLD, rank_, size_);
        NodeId max_node_id = ranges.back().second - 1;
        for (auto node : this->local_nodes()) {
            for (auto neighbor : this->adj(node).neighbors()) {
                KASSERT(neighbor.id() <= max_node_id, "neighbor " << neighbor << " of node " << node << " is invalid!");
            }
        }
    }

private:
    std::vector<EdgeId> first_out_;
    std::vector<EdgeId> first_out_offset_;
    std::vector<Degree> degree_;
    std::vector<RankEncodedNodeId> head_;
    bool ghost_ranks_available_;
    bool oriented_;
    EdgeId local_edge_count_{};
    std::pair<NodeId, NodeId> node_range_;
    PEID rank_;
    PEID size_;
    NodeIndexer node_indexer_;
};

template <typename GhostPayloadType>
inline std::ostream& operator<<(std::ostream& out, DistributedGraph<GhostPayloadType>& G) {
    for (auto node : G.local_nodes()) {
        for (auto edge : G.out_adj(node).edges()) {
            out << edge << std::endl;
        }
    }
    return out;
}

template <typename GhostPayloadType, typename SetType = std::set<RankEncodedNodeId>>
void find_ghosts(DistributedGraph<GhostPayloadType> const& G, SetType& ghosts) {
    for (auto node : G.local_nodes()) {
        for (auto neighbor : boost::adaptors::reverse(G.adj(node).neighbors())) {
            if (neighbor.rank() == G.rank()) {
                continue;
            }
            KASSERT(neighbor.rank() != G.rank());
            ghosts.insert(neighbor);
        }
    }
}

}  // namespace graph
}  // namespace cetric
#endif  // PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_H
