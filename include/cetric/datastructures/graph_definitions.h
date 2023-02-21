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
#include <ostream>
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>
#include <mpi.h>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <graph-io/graph_definitions.h>
#include <hash/murmur2_hash.hpp>
#include <boost/mpi/datatype.hpp>

#include "cetric/atomic_debug.h"

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
//#ifdef MPI_VERSION
#ifdef MPI_UNINT64_T
    #define MPI_NODE MPI_UINT64_T
#else
static_assert(sizeof(unsigned long long) == 8, "We expect an unsigned long long to have 64 bit");
    #define MPI_NODE MPI_UNSIGNED_LONG_LONG
#endif
//#endif

using Edge            = graphio::Edge<>;
using RankEncodedEdge = graphio::Edge<RankEncodedNodeId>;

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
struct hash {
    size_t operator()(int rank) const {
        int local = rank;
        return murmur.MurmurHash64A(&local, sizeof(int), murmur.seed);
    }
    size_t operator()(graph::RankEncodedNodeId node) const {
        std::uint64_t local = node.data();
        return murmur.MurmurHash64A(&local, sizeof(local), murmur.seed);
    }
    utils_tm::hash_tm::murmur2_hash murmur;
};
struct node_set : public google::dense_hash_set<graph::RankEncodedNodeId, hash> {
    explicit node_set() : google::dense_hash_set<graph::RankEncodedNodeId, hash>() {
        this->set_empty_key(graph::RankEncodedNodeId::sentinel());
    }
    explicit node_set(size_t size) : google::dense_hash_set<graph::RankEncodedNodeId, hash>(size) {
        this->set_empty_key(graph::RankEncodedNodeId::sentinel());
    }
};

template <typename K, typename V>
using default_map = google::dense_hash_map<K, V, hash>;

template <typename T>
struct node_map : public google::dense_hash_map<graph::RankEncodedNodeId, T, hash> {
    explicit node_map() : google::dense_hash_map<graph::RankEncodedNodeId, T, hash>() {
        this->set_empty_key(graph::RankEncodedNodeId::sentinel());
    }
    explicit node_map(size_t size) : google::dense_hash_map<graph::RankEncodedNodeId, T, hash>(size) {
        this->set_empty_key(graph::RankEncodedNodeId::sentinel());
    }
};
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

namespace boost::mpi {
template <>
MPI_Datatype get_mpi_datatype<cetric::graph::RankEncodedNodeId>(cetric::graph::RankEncodedNodeId const&) {
    return get_mpi_datatype<std::uint64_t>();
}
template<>
struct is_mpi_builtin_datatype<cetric::graph::RankEncodedNodeId> : is_mpi_builtin_datatype<std::uint64_t> {};
}

#endif /* end of include guard: GRAPH_DEFINITIONS_H_8XAL43DH */
