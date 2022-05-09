#ifndef GRAPH_DEFINITIONS_H_8XAL43DH
#define GRAPH_DEFINITIONS_H_8XAL43DH

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <ostream>
#include <graph-io/graph_definitions.h>

namespace cetric {
namespace graph {

using NodeId = std::uint64_t;
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

using Edge = graphio::Edge;

inline std::ostream& operator<<(std::ostream& out, const Edge& edge) {
    out << "(" << edge.tail << ", " << edge.head << ")";
    return out;
}

inline bool operator==(const Edge& x, const Edge& y) {
    return x.tail == y.tail && x.head == y.head;
}

struct Triangle {
    NodeId x;
    NodeId y;
    NodeId z;

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

inline bool operator==(const Triangle& t1, const Triangle& t2) {
    return t1.x == t2.x && t1.y == t2.y && t1.z == t2.z;
}

inline std::ostream& operator<<(std::ostream& out, const Triangle& t) {
    out << "(" << t.x << ", " << t.y << ", " << t.z << ")";
    return out;
}
}  // namespace graph
}  // namespace cetric

#endif /* end of include guard: GRAPH_DEFINITIONS_H_8XAL43DH */
