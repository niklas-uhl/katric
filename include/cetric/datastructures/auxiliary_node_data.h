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

#pragma once

#include <iterator>
#include <sparsehash/dense_hash_map>
#include <utility>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/range/join.hpp>
#include <fmt/ranges.h>
#include <kassert/kassert.hpp>

#include "cetric/datastructures/graph_definitions.h"

namespace cetric {
template <typename T>
class AuxiliaryNodeData {
public:
    explicit AuxiliaryNodeData()
        : AuxiliaryNodeData(graph::RankEncodedNodeId::sentinel(), graph::RankEncodedNodeId::sentinel()) {}
    explicit AuxiliaryNodeData(graph::RankEncodedNodeId from, graph::RankEncodedNodeId to)
        : from_(from),
          to_(to),
          index_map_(),
          data_(to - from) {
        index_map_.set_empty_key(graph::RankEncodedNodeId::sentinel());
    }

    template <typename GhostIterator>
    explicit AuxiliaryNodeData(
        graph::RankEncodedNodeId from, graph::RankEncodedNodeId to, GhostIterator ghost_begin, GhostIterator ghost_end
    )
        : from_(from),
          to_(to),
          index_map_(std::distance(ghost_begin, ghost_end)),
          data_(to - from + std::distance(ghost_begin, ghost_end)) {
        index_map_.set_empty_key(graph::RankEncodedNodeId::sentinel());
        index_map_.set_deleted_key(graph::RankEncodedNodeId::sentinel() - 1);
        size_t index = to - from;
        for (auto current = ghost_begin; current != ghost_end; current++) {
            index_map_[*current] = index;
            index++;
        }
    }
    template <typename GhostIterator>
    void add_ghosts(GhostIterator ghost_begin, GhostIterator ghost_end) {
        size_t index = data_.size();
        for (auto current = ghost_begin; current != ghost_end; current++) {
            if (!has_data_for(*current)) {
                index_map_[*current] = index;
                index++;
            }
        }
        data_.resize(index);
    }

    template <typename GhostIterator>
    explicit AuxiliaryNodeData(GhostIterator ghost_begin, GhostIterator ghost_end)
        : AuxiliaryNodeData(
            graph::RankEncodedNodeId::sentinel(), graph::RankEncodedNodeId::sentinel(), ghost_begin, ghost_end
        ) {}

    const T& operator[](graph::RankEncodedNodeId node) const {
        return data_[get_index(node)];
    }

    T& operator[](graph::RankEncodedNodeId node) {
        return data_[get_index(node)];
    }

    bool has_data_for(graph::RankEncodedNodeId node) const {
        return (node >= from_ && node < to_) || index_map_.find(node) != index_map_.end();
    }

    size_t size() const {
        return data_.size();
    }

    auto range() const {
        using counting_iterator_type =
            boost::counting_iterator<graph::RankEncodedNodeId, boost::random_access_traversal_tag, std::ptrdiff_t>;
        auto local = boost::adaptors::transform(
            boost::make_iterator_range(counting_iterator_type{from_}, counting_iterator_type{to_}),
            [&](graph::RankEncodedNodeId node) { return std::make_pair(node, get_index(node)); }
        );
        auto ghosts     = boost::make_iterator_range(index_map_.begin(), index_map_.end());
        auto full_range = boost::join(local, ghosts);
        return boost::adaptors::transform(full_range, Transformer{data_});
    }

    auto begin() const {
        return range().begin();
    }
    auto end() const {
        return range().end();
    }

private:
    struct Transformer {
        std::vector<T> const&                  data;
        std::pair<graph::RankEncodedNodeId, T> operator()(std::pair<graph::RankEncodedNodeId, size_t> kv) const {
            return std::make_pair(kv.first, data[kv.second]);
        }
    };
    size_t get_index(graph::RankEncodedNodeId node) const {
        // atomic_debug(fmt::format("Lookup node {}", node));
        if (node >= from_ && node <= to_) {
            return node - from_;
        }
        auto it = index_map_.find(node);
        int  rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        KASSERT(it != index_map_.end(), "[R" << rank << "] Failed lookup of " << node);
        return it->second;
    }
    graph::RankEncodedNodeId                                 from_;
    graph::RankEncodedNodeId                                 to_;
    google::dense_hash_map<graph::RankEncodedNodeId, size_t> index_map_;
    std::vector<T>                                           data_;
};
} // namespace cetric
