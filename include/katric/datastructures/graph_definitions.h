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

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace katric {
namespace graph {

using NodeId                      = std::uint64_t;
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
using Edge = BasicEdge<NodeId>;

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
} // namespace katric

// fmt >= 9 requires an explicit opt-in to use operator<<
template <typename NodeIdType>
struct fmt::formatter<katric::graph::Triangle<NodeIdType>> : fmt::ostream_formatter {};

#endif /* end of include guard: GRAPH_DEFINITIONS_H_8XAL43DH */
