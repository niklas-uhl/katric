#pragma once

#include <datastructures/graph_definitions.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace cetric {
template <class Graph>

class SharedMemoryEdgeIterator {
public:
    SharedMemoryEdgeIterator(const Graph& G) : G(G) {}

    template <typename TriangleFunc>
    void run(TriangleFunc emit) {
        using namespace cetric::graph;
        auto node_range = G.local_nodes();
        tbb::parallel_for(tbb::blocked_range(node_range.begin(), node_range.end()), [&](auto r) {
            for (NodeId v : r) {
                auto edge_range = G.out_edges(v);
                tbb::parallel_for(tbb::blocked_range(edge_range.begin(), edge_range.end()), [&](auto r2) {
                    for (Edge edge : r2) {
                        auto u = edge.head;
                        G.intersect_neighborhoods(v, u, [&](auto x) {
                            emit(Triangle {u, v, x});
                        });
                    }
                });
            }
        });
    }

private:
    const Graph& G;
};
}  // namespace cetric
