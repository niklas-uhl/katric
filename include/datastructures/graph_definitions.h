#ifndef GRAPH_DEFINITIONS_H_8XAL43DH
#define GRAPH_DEFINITIONS_H_8XAL43DH

#include <cassert>
#include <cinttypes>
#include <ostream>
#include <utility>

namespace cetric {
    namespace graph {

using NodeId = std::uint64_t;
using EdgeId = std::uint64_t;
using Degree = NodeId;
//#ifdef MPI_VERSION
#define MPI_NODE MPI_UINT64_T
//#endif


struct Edge {
    Edge(): tail(0), head(0) { }
    Edge(NodeId tail, NodeId head): tail(tail), head(head) { }
    Edge reverse() const {
        return Edge {head, tail};
    }
    template<typename VertexMap>
    Edge map(VertexMap map) {
        return Edge {map(tail), map(head)};
    }

    NodeId tail;
    NodeId head;
};

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


}
}


#endif /* end of include guard: GRAPH_DEFINITIONS_H_8XAL43DH */
