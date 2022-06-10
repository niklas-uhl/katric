#pragma once

#include <datastructures/graph_definitions.h>
#include <iterator>
#include <kassert/kassert.hpp>
#include <sparsehash/dense_hash_map>

namespace cetric {
template <typename T>
class AuxiliaryNodeData {
public:
    explicit AuxiliaryNodeData()
        : AuxiliaryNodeData(graph::RankEncodedNodeId::sentinel(), graph::RankEncodedNodeId::sentinel()) {}
    explicit AuxiliaryNodeData(graph::RankEncodedNodeId from, graph::RankEncodedNodeId to)
        : index_map_(), data_(to - from + 1) {}

    template <typename GhostIterator>
    explicit AuxiliaryNodeData(graph::RankEncodedNodeId from,
                               graph::RankEncodedNodeId to,
                               GhostIterator ghost_begin,
                               GhostIterator ghost_end)
        : index_map_(std::distance(ghost_begin, ghost_end)),
          data_(to - from + 1 + std::distance(ghost_begin, ghost_end)) {
        index_map_.set_empty_key(graph::RankEncodedNodeId::sentinel());
        index_map_.set_deleted_key(graph::RankEncodedNodeId::sentinel() - 1);
        size_t index = to - from;
        for (auto current = ghost_begin; current != ghost_end; current++) {
            index_map_[*current] = index;
            index++;
        }
    }

    template <typename GhostIterator>
    explicit AuxiliaryNodeData(GhostIterator ghost_begin, GhostIterator ghost_end)
        : AuxiliaryNodeData(graph::RankEncodedNodeId::sentinel(),
                            graph::RankEncodedNodeId::sentinel(),
                            ghost_begin,
                            ghost_end) {}

    const T& operator[](graph::RankEncodedNodeId node) const {
        return data_[get_index(node)];
    }

    T& operator[](graph::RankEncodedNodeId node) {
        return data_[get_index(node)];
    }

private:
    size_t get_index(graph::RankEncodedNodeId node) const {
        if (node >= from_ && node <= to_) {
            return node - from_;
        }
        auto it = index_map_.find(node);
        KASSERT(it != index_map_.end());
        return it->second;
    }
    graph::RankEncodedNodeId from_;
    graph::RankEncodedNodeId to_;
    google::dense_hash_map<graph::RankEncodedNodeId, size_t> index_map_;
    std::vector<T> data_;
};
}  // namespace cetric
