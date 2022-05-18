#pragma once

#include <datastructures/graph_definitions.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <algorithm>
#include <cyclic_range_adaptor.hpp>
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
    size_t num_threads = 0;
    bool skip_previous_edges = false;
};

template <class Graph>
class SharedMemoryEdgeIterator {
public:
    SharedMemoryEdgeIterator(const Graph& G) : G(G) {}

    template <typename TriangleFunc, typename NodeOrdering = std::less<>>
    void run(TriangleFunc emit, const Config& conf, NodeOrdering&& node_ordering = {}) {
        run_dispatcher(emit, conf, std::move(node_ordering));
    }

private:
    template <typename TriangleFunc, typename NodeOrdering>
    void run_dispatcher(TriangleFunc emit, const Config& conf, NodeOrdering&& node_ordering) {
        switch (conf.partition) {
            case Partition::one_dimensional:
                run_dispatcher<Partition::one_dimensional>(emit, conf, std::move(node_ordering));
                break;
            case Partition::two_dimensional:
                run_dispatcher<Partition::two_dimensional>(emit, conf, std::move(node_ordering));
                break;
        }
    }

    template <Partition partition, typename TriangleFunc, typename NodeOrdering>
    void run_dispatcher(TriangleFunc emit, const Config& conf, NodeOrdering&& node_ordering) {
        switch (conf.intersection_method) {
            case IntersectionMethod::merge:
                run_dispatcher<partition, graph::intersection_policy::merge>(emit, conf, std::move(node_ordering));
                break;
            case IntersectionMethod::binary_search:
                run_dispatcher<partition, graph::intersection_policy::binary_search>(emit, conf,
                                                                                     std::move(node_ordering));
                break;
            case IntersectionMethod::hybrid:
                run_dispatcher<partition, graph::intersection_policy::hybrid>(emit, conf, std::move(node_ordering));
                break;
        }
    }

    template <Partition partition, typename IntersectionPolicy, typename TriangleFunc, typename NodeOrdering>
    void run_dispatcher(TriangleFunc emit, const Config& conf, NodeOrdering&& node_ordering) {
        switch (conf.partitioner) {
            case Partitioner::auto_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::auto_partitioner>(emit, conf,
                                                                                     std::move(node_ordering));
                break;
            case Partitioner::static_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::static_partitioner>(emit, conf,
                                                                                       std::move(node_ordering));
                break;
            case Partitioner::affinity_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::affinity_partitioner>(emit, conf,
                                                                                         std::move(node_ordering));
                break;
            case Partitioner::simple_partitioner:
                run_dispatcher<partition, IntersectionPolicy, tbb::simple_partitioner>(emit, conf,
                                                                                       std::move(node_ordering));
                break;
        }
    }

    template <Partition partition,
              typename IntersectionPolicy,
              typename Partitioner,
              typename TriangleFunc,
              typename NodeOrdering>
    void run_dispatcher(TriangleFunc emit, const Config& conf, NodeOrdering&& node_ordering) {
        if (conf.skip_previous_edges) {
            run_dispatcher<partition, IntersectionPolicy, Partitioner, true>(emit, conf, std::move(node_ordering));
        } else {
            run_dispatcher<partition, IntersectionPolicy, Partitioner, false>(emit, conf, std::move(node_ordering));
        }
    }

    template <Partition partition,
              typename IntersectionPolicy,
              typename Partitioner,
              bool skip_previous_edges,
              typename TriangleFunc,
              typename NodeOrdering>
    void run_dispatcher(TriangleFunc emit, const Config& conf, NodeOrdering&& node_ordering) {
        using namespace cetric::graph;
        auto node_range = G.local_nodes();
        Partitioner p;

        // nw::graph::cyclic_range_adaptor tbb_range(node_range.begin(), node_range.end(), conf.num_threads);
        tbb::blocked_range tbb_range(node_range.begin(), node_range.end(), conf.grainsize);
        tbb::parallel_for(
            tbb_range,
            [emit, this, &conf, &p, node_ordering](auto r) {
                for (NodeId v : r) {
                    auto neighbors = G.out_neighbors(v);
                    if constexpr (partition == Partition::one_dimensional) {
                        for (auto current = neighbors.begin(); current != neighbors.end(); current++) {
                            NodeId u = *current;
                            decltype(neighbors.begin()) begin;
                            if constexpr (skip_previous_edges) {
                                begin = current;
                            } else {
                                begin = G.out_neighbors(v).begin();
                            }
                            G.intersect_neighborhoods(
                                begin, neighbors.end(), u,
                                [&](auto x) {
                                    emit(Triangle{u, v, x});
                                },
                                std::move(node_ordering), IntersectionPolicy{});
                        }
                    } else {
                        tbb::parallel_for(
                            tbb::blocked_range(neighbors.begin(), neighbors.end(), conf.grainsize),
                            [this, emit, v, node_ordering](auto r2) {
                                for (auto current = r2.begin(); current != r2.end(); current++) {
                                    NodeId u = *current;
                                    decltype(neighbors.begin()) begin;
                                    if constexpr (skip_previous_edges) {
                                        begin = current;
                                    } else {
                                        begin = G.out_neighbors(v).begin();
                                    }
                                    G.intersect_neighborhoods(
                                        begin, G.out_neighbors(v).end(), u,
                                        [&](auto x) {
                                            emit(Triangle{u, v, x});
                                        },
                                        std::move(node_ordering), IntersectionPolicy{});
                                }
                            },
                            p);
                    }
                }
            },
            p);
    }
    const Graph& G;
};
}  // namespace cetric::shared_memory
