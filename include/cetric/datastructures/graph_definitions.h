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

#ifndef GRAPH_DEFINITIONS_H_8XAL43DH
#define GRAPH_DEFINITIONS_H_8XAL43DH

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <ostream>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <mpi.h>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <kamping/types/mpi_type_traits.hpp>

#include <spdlog/spdlog.h>

namespace cetric {
namespace graph {

using NodeId                      = std::uint64_t;
static constexpr size_t RANK_BITS = 16;
class RankEncodedNodeId {
public:
    static constexpr std::uint64_t     NON_RANK_BITS = sizeof(std::uint64_t) * 8 - RANK_BITS;
    static constexpr std::uint64_t     rank_mask     = ((1ull << RANK_BITS) - 1) << NON_RANK_BITS;
    static constexpr RankEncodedNodeId sentinel() {
        return RankEncodedNodeId{
            (1l << (cetric::graph::RankEncodedNodeId::NON_RANK_BITS + 1)) - 1,
            std::numeric_limits<uint16_t>::max()};
    }

    explicit constexpr RankEncodedNodeId(std::uint64_t id) : value_(id | rank_mask) {}
    explicit constexpr RankEncodedNodeId(std::uint64_t id, std::uint16_t rank)
        : value_(id | (std::uint64_t(rank) << NON_RANK_BITS)) {}
    explicit constexpr RankEncodedNodeId() : RankEncodedNodeId(0) {}
    // void operator=(const std::uint64_t& value) {
    //     value_ = value | rank_mask;
    // }

    inline std::uint16_t rank() const {
        return (value_ & rank_mask) >> NON_RANK_BITS;
    }

    inline std::uint64_t id() const {
        return (value_ & ~rank_mask);
    }

    inline std::uint64_t data() const {
        return value_;
    }

    inline std::uint64_t mask_rank(std::uint16_t rank) const {
        return value_ ^ (std::uint64_t(rank) << NON_RANK_BITS);
    }

    inline void set_rank(std::uint16_t rank) {
        value_ = (value_ & ~rank_mask) | (std::uint64_t(rank) << NON_RANK_BITS);
    }

    inline bool operator<(const RankEncodedNodeId& rhs) const {
        return value_ < rhs.value_;
    }
    inline bool operator>(const RankEncodedNodeId& rhs) const {
        return rhs < *this;
    }

    inline bool operator>=(const RankEncodedNodeId& rhs) const {
        return !(*this < rhs);
    }

    inline bool operator<=(const RankEncodedNodeId& rhs) const {
        return !(*this > rhs);
    }

    inline bool operator==(RankEncodedNodeId const& rhs) const {
        return value_ == rhs.value_;
    }
    inline bool operator!=(RankEncodedNodeId const& rhs) const {
        return value_ != rhs.value_;
    }
    /// Makes the type hashable with absl::Hash (used by absl::flat_hash_map/set).
    template <typename H>
    friend H AbslHashValue(H state, RankEncodedNodeId const& node) {
        return H::combine(std::move(state), node.value_);
    }
    // operator std::uint64_t() const {
    //     return value_;
    // }
    RankEncodedNodeId& operator++() {
        value_++;
        return *this;
    }
    RankEncodedNodeId operator++(int) {
        auto pre = *this;
        value_--;
        return pre;
    }
    RankEncodedNodeId& operator--() {
        value_++;
        return *this;
    }
    RankEncodedNodeId operator--(int) {
        auto pre = *this;
        value_--;
        return pre;
    }

    RankEncodedNodeId& operator+=(const size_t n) {
        value_ += n;
        return *this;
    }

    RankEncodedNodeId operator+(const size_t n) const {
        auto pre = *this;
        pre.value_ += n;
        return pre;
    }
    RankEncodedNodeId& operator-=(const size_t n) {
        value_ -= n;
        return *this;
    }

    RankEncodedNodeId operator-(const size_t n) const {
        auto pre = *this;
        pre.value_ -= n;
        return pre;
    }

    auto operator-(RankEncodedNodeId const& rhs) const {
        return this->value_ - rhs.value_;
    }

    explicit operator std::uint64_t() const {
        return data();
    }

private:
    std::uint64_t value_;
};

inline std::ostream& operator<<(std::ostream& os, const RankEncodedNodeId& node_id) {
    os << node_id.id() << "@";
    if (node_id.rank() == RankEncodedNodeId::sentinel().rank()) {
        os << "nil";
    } else {
        os << node_id.rank();
    }
    return os;
}
using EdgeId = std::uint64_t;
using Degree = NodeId;
template <typename NodeIdType>
struct BasicEdge {
    NodeIdType tail{};
    NodeIdType head{};

    BasicEdge() = default;
    BasicEdge(NodeIdType tail_, NodeIdType head_) : tail(tail_), head(head_) {}
    [[nodiscard]] BasicEdge reverse() const {
        return BasicEdge{head, tail};
    }
    friend bool operator==(BasicEdge const&, BasicEdge const&) = default;
    friend std::ostream& operator<<(std::ostream& out, BasicEdge const& edge) {
        return out << "(" << edge.tail << ", " << edge.head << ")";
    }
};
using Edge            = BasicEdge<NodeId>;
using RankEncodedEdge = BasicEdge<RankEncodedNodeId>;

template <typename NodeIdType = NodeId>
struct Triangle {
    NodeIdType x;
    NodeIdType y;
    NodeIdType z;

    void normalize() {
        if (x > y) {
            std::swap(x, y);
        }
        if (y > z) {
            std::swap(y, z);
        }
        if (x > y) {
            std::swap(x, y);
        }
        assert(x < y && y < z);
    }
};

template <typename NodeIdType>
inline bool operator==(const Triangle<NodeIdType>& t1, const Triangle<NodeIdType>& t2) {
    return t1.x == t2.x && t1.y == t2.y && t1.z == t2.z;
}

template <typename NodeIdType>
inline std::ostream& operator<<(std::ostream& out, const Triangle<NodeIdType>& t) {
    out << "(" << t.x << ", " << t.y << ", " << t.z << ")";
    return out;
}
} // namespace graph
} // namespace cetric

template <>
struct std::hash<cetric::graph::RankEncodedNodeId> {
    std::size_t operator()(cetric::graph::RankEncodedNodeId const& id) const noexcept {
        return std::hash<std::uint64_t>{}(id.data());
    }
};

namespace cetric {
using node_set = absl::flat_hash_set<graph::RankEncodedNodeId>;

template <typename K, typename V>
using default_map = absl::flat_hash_map<K, V>;

template <typename T>
using node_map = absl::flat_hash_map<graph::RankEncodedNodeId, T>;
} // namespace cetric

// template <>
// struct std::numeric_limits<cetric::graph::RankEncodedNodeId> {
//     static constexpr bool is_specialized = std::numeric_limits<uint64_t>::is_specialized;
//     static constexpr bool is_signed = std::numeric_limits<uint64_t>::is_signed;
//     static constexpr bool is_bounded = std::numeric_limits<uint64_t>::is_bounded;
//     static constexpr int digits = std::numeric_limits<uint64_t>::digits;

//     static constexpr cetric::graph::RankEncodedNodeId min() {
//         return cetric::graph::RankEncodedNodeId{0, 0};
//     }
//     static constexpr cetric::graph::RankEncodedNodeId max() {
//         return cetric::graph::RankEncodedNodeId{(1l << (cetric::graph::RankEncodedNodeId::NON_RANK_BITS + 1)) - 1,
//                                                 std::numeric_limits<uint16_t>::max()};
//     }
// };

// fmt >= 9 requires an explicit opt-in to use operator<<
template <>
struct fmt::formatter<cetric::graph::RankEncodedNodeId> : fmt::ostream_formatter {};

template <typename NodeIdType>
struct fmt::formatter<cetric::graph::Triangle<NodeIdType>> : fmt::ostream_formatter {};

// RankEncodedNodeId is transferred as its underlying 64 bit integer
template <>
struct kamping::types::mpi_type_traits<cetric::graph::RankEncodedNodeId> : kamping::types::mpi_type_traits<std::uint64_t> {};

#endif /* end of include guard: GRAPH_DEFINITIONS_H_8XAL43DH */
