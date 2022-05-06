#pragma once

template <class Graph>
class SharedMemoryEdgeIterator {
public:
    SharedMemoryEdgeIterator(const Graph& G) {}

    template <typename TriangleFunc>
    void run(TriangleFunc emit) {
    }
private:
};
