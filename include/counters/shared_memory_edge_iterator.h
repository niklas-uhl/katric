#pragma once

#include <datastructures/graph_definitions.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <algorithm>
#include <iostream>
#include "datastructures/graph.h"

namespace cetric::shared_memory {
enum class Partition { one_dimensional, two_dimensional };
inline std::ostream& operator<<(std::ostream& os, Partition partition) {
    switch (partition) {
        case Partition::one_dimensional:
            os << "1D";
            break;
        case Partition::two_dimensional:
            os << "2D";
            break;
    }
    return os;
}
enum class IntersectionMethod { merge, binary_search, hybrid };
inline std::ostream& operator<<(std::ostream& os, IntersectionMethod intersection_method) {
    switch (intersection_method) {
        case IntersectionMethod::merge:
            os << "merge";
            break;
        case IntersectionMethod::binary_search:
            os << "binary_search";
            break;
        case IntersectionMethod::hybrid:
            os << "hybrid";
            break;
    }
    return os;
}
enum class Partitioner { auto_partitioner, simple_partitioner, static_partitioner, affinity_partitioner };
inline std::ostream& operator<<(std::ostream& os, Partitioner partitioner) {
    switch (partitioner) {
        case Partitioner::auto_partitioner:
            os << "auto";
            break;
        case Partitioner::simple_partitioner:
            os << "simple";
            break;
        case Partitioner::static_partitioner:
            os << "static";
            break;
        case Partitioner::affinity_partitioner:
            os << "affinity";
            break;
    }
    return os;
}

struct Config {
    Partition partition = Partition::two_dimensional;
    IntersectionMethod intersection_method = IntersectionMethod::merge;
    Partitioner partitioner = Partitioner::auto_partitioner;
    size_t grainsize = 1;
};

template <class Graph>
class SharedMemoryEdgeIterator {
public:
    SharedMemoryEdgeIterator(const Graph& G) : G(G) {}

    template <typename TriangleFunc>
    void run(TriangleFunc emit) {
        Config conf;
        run(emit, conf);
    }

    template <typename TriangleFunc>
    void run(TriangleFunc emit, const Config& conf) {
        run_dispatcher(emit, conf);
    }

private:
    template <typename TriangleFunc>
    void run_dispatcher(TriangleFunc emit, const Config& conf) {
        switch (conf.partition) {
            case Partition::one_dimensional:
                run_dispatcher<Partition::one_dimensional>(emit, conf);
                break;
            case Partition::two_dimensional:
                run_dispatcher<Partition::two_dimensional>(emit, conf);
                break;
        }
    }

    template <Partition partition, typename TriangleFunc>
    void run_dispatcher(TriangleFunc emit, const Config& conf) {
        switch (conf.intersection_method) {
            case IntersectionMethod::merge:
                run_dispatcher<partition, graph::intersection_policy::merge>(emit, conf);
                break;
            case IntersectionMethod::binary_search:
                run_dispatcher<partition, graph::intersection_policy::binary_search>(emit, conf);
                break;
            case IntersectionMethod::hybrid:
                run_dispatcher<partition, graph::intersection_policy::hybrid>(emit, conf);
                break;
        }
    }

    template <Partition partition, typename IntersectionPolicy, typename TriangleFunc>
    void run_dispatcher(TriangleFunc emit, const Config& conf) {
        switch (conf.partitioner) {
            case Partitioner::auto_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::auto_partitioner>(emit, conf);
                break;
            case Partitioner::simple_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::simple_partitioner>(emit, conf);
                break;
            case Partitioner::static_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::static_partitioner>(emit, conf);
                break;
            case Partitioner::affinity_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::affinity_partitioner>(emit, conf);
                break;
        }
    }

    template <Partition partition, typename IntersectionPolicy, typename Partitioner, typename TriangleFunc>
    void run_dispatcher(TriangleFunc emit, const Config& conf) {
        using namespace cetric::graph;
        auto node_range = G.local_nodes();
        auto on_edge = [this, &emit](Edge edge) {
            auto v = edge.tail;
            auto u = edge.head;
            G.intersect_neighborhoods(
                v, u,
                [&](auto x) {
                    emit(Triangle{u, v, x});
                },
                IntersectionPolicy{});
        };
        Partitioner p;
        tbb::parallel_for(
            tbb::blocked_range(node_range.begin(), node_range.end(), conf.grainsize),
            [&](auto r) {
                for (NodeId v : r) {
                    auto edge_range = G.out_edges(v);
                    if constexpr (partition == Partition::one_dimensional) {
                        std::for_each(edge_range.begin(), edge_range.end(), on_edge);
                    } else {
                        tbb::parallel_for(
                            tbb::blocked_range(edge_range.begin(), edge_range.end(), conf.grainsize),
                            [&](auto r2) { std::for_each(r2.begin(), r2.end(), on_edge); }, p);
                    }
                }
            },
            p);
    }
    const Graph& G;
};
}  // namespace cetric::shared_memory
