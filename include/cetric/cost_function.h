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

//
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
#define PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <sparsehash/dense_hash_map>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fmt/core.h>
#include <graph-io/local_graph_view.h>
#include <tbb/task_arena.h>

#include "cetric/atomic_debug.h"
#include "cetric/communicator.h"
#include "cetric/datastructures/auxiliary_node_data.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/distributed/graph_communicator.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/statistics.h"
#include "cetric/timer.h"
#include "cetric/util.h"

// using namespace cetric::graph;

// class OutDegreeCache {
// public:
//     OutDegreeCache() : data_() {}
//     OutDegreeCache(const DistributedGraph<>& G) : data_() {
//         auto ghosts = find_ghosts(G);
//         assert(G.get_graph_payload().ghost_degree_available);
//         for (RankEncodedNodeId node : G.local_nodes()) {
//             Degree out_deg = 0;
//             for (auto edge : G.edges(node)) {
//                 if (edge.head.rank() != G.rank()) {
//                 }
//             }
//         }

//         G.for_each_local_node([&](NodeId local_node_id) {
//             Degree out_deg = 0;
//             G.for_each_edge(local_node_id, [&](auto e) {
//                 if (G.is_outgoing(e)) {
//                     out_deg++;
//                 }
//             });
//             data_[local_node_id] = out_deg;
//         });
//     }

//     inline void set(NodeId local_node_id, Degree degree) {
//         data_[local_node_id] = degree;
//     }

//     inline Degree get(NodeId local_node_id) {
//         return data_[local_node_id];
//     }

// private:
//     std::vector<Degree> data_;
// };
namespace cetric {

template <typename Graph>
class CostFunction {
public:
    static inline void
    init_nop(Graph&, node_set&, AuxiliaryNodeData<Degree>&, AuxiliaryNodeData<Degree>&, bool, bool, cetric::profiling::MessageStatistics&) {
    }

    static inline void init_local_outdegree(
        Graph&                                G,
        node_set&                             ghosts,
        AuxiliaryNodeData<Degree>&            degree,
        AuxiliaryNodeData<Degree>&            out_degree,
        bool                                  dense_degree_exchange,
        bool                                  compact_degree_exchange,
        cetric::profiling::MessageStatistics& stats
    ) {
        init_degree(G, ghosts, degree, out_degree, dense_degree_exchange, compact_degree_exchange, stats);
        if (out_degree.size() == 0) {
            out_degree = AuxiliaryNodeData<Degree>(
                RankEncodedNodeId{G.node_range().first, static_cast<uint16_t>(G.rank())},
                RankEncodedNodeId{G.node_range().second, static_cast<uint16_t>(G.rank())}
            );
            auto deg = [&G, &degree](RankEncodedNodeId node) {
                if (node.rank() == G.rank()) {
                    return G.degree(node);
                }
                return degree[node];
            };
            for (RankEncodedNodeId v: G.local_nodes()) {
                auto   neighbors = G.adj(v).neighbors();
                Degree v_out_deg = std::count_if(neighbors.begin(), neighbors.end(), [&deg, &v](auto u) {
                    return std::tuple(deg(u), u.id()) > std::tuple(deg(v), v.id());
                });
                out_degree[v]    = v_out_deg;
            }
        }
    }

    static inline void init_degree(
        Graph&                     G,
        node_set&                  ghosts,
        AuxiliaryNodeData<Degree>& degree,
        AuxiliaryNodeData<Degree>&,
        bool                                  dense_degree_exchange,
        bool                                  compact_degree_exchange,
        cetric::profiling::MessageStatistics& stats
    ) {
        // (void)cache;
        DegreeCommunicator comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
        if (ghosts.size() == 0) {
            find_ghosts(G, ghosts);
        }
        if (degree.size() == 0) {
            degree = AuxiliaryNodeData<Degree>(ghosts.begin(), ghosts.end());
            comm.get_ghost_degree(
                [&](RankEncodedNodeId node, Degree deg) {
                    KASSERT(ghosts.find(node) != ghosts.end());
                    degree[node] = deg;
                },
                stats,
                !dense_degree_exchange,
                compact_degree_exchange
            );
        }
    }

    static inline void init_all(
        Graph&                                G,
        node_set&                             ghosts,
        AuxiliaryNodeData<Degree>&            degree,
        AuxiliaryNodeData<Degree>&            out_degree,
        bool                                  dense_degree_exchange,
        bool                                  compact_degree_exchange,
        cetric::profiling::MessageStatistics& stats
    ) {
        init_degree(G, ghosts, degree, out_degree, dense_degree_exchange, compact_degree_exchange, stats);
        init_local_outdegree(G, ghosts, degree, out_degree, dense_degree_exchange, compact_degree_exchange, stats);
        if (ghosts.size() == 0) {
            find_ghosts(G, ghosts);
        }
        if (out_degree.size() <= G.local_node_count()) {
            out_degree.add_ghosts(ghosts.begin(), ghosts.end());
            DegreeCommunicator comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
            comm.get_ghost_data(
                [&](auto node) { return out_degree[node]; },
                [&](auto node, auto data) {
                    KASSERT(ghosts.find(node) != ghosts.end());
                    out_degree[node] = data.id();
                },
                stats,
                !dense_degree_exchange,
                compact_degree_exchange
            );
            // comm.get_ghost_degree([&](RankEncodedNodeId node, Degree deg) { degree[node] = deg; }, stats);
        }
        //     DegreeCommunicator<Graph> comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
        //     if (!G.get_graph_payload().ghost_degree_available) {
        //         comm.get_ghost_degree(
        //             [&](NodeId global_id, Degree degree) {
        //                 G.get_ghost_payload(G.to_local_id(global_id)).degree = degree;
        //             },
        //             stats);
        //         G.get_graph_payload().ghost_degree_available = true;
        //     }
        //     cache = OutDegreeCache(G);
        // comm.get_ghost_outdegree(
        //     [&](auto e) { return G.is_outgoing(e); },
        //     [&](NodeId global_id, Degree outdegree) { cache.set(G.to_local_id(global_id), outdegree); }, stats);
    }

    template <typename CostFunctionType, typename ExecutionPolicy = execution_policy::sequential>
    explicit CostFunction(
        Graph&                                      G,
        const std::string&                          name,
        AuxiliaryNodeData<Degree> const&            degree,
        AuxiliaryNodeData<Degree> const&            outdegree,
        CostFunctionType&                           cost_function,
        cetric::profiling::LoadBalancingStatistics& stats,
        ExecutionPolicy&&                           policy = {}
    )
        : G(G),
          degree_(degree),
          out_degree_(outdegree),
          node_to_idx_(G.local_node_count()),
          cost(G.local_node_count()),
          name(name) {
        node_to_idx_.set_empty_key(-1);
        cetric::profiling::Timer t;
        size_t                   node_index = 0;
        auto                     nodes      = G.local_nodes();
        if constexpr (std::is_same_v<ExecutionPolicy, execution_policy::parallel>) {
            tbb::task_arena arena(policy.num_threads);
            arena.execute([&] {
                tbb::parallel_for(
                    tbb::blocked_range(nodes.begin(), nodes.end()),
                    [this, &node_index, &cost_function, &G](auto const& r) {
                        for (RankEncodedNodeId node: r) {
                            node_to_idx_[node.id()] = node_index;
                            cost[node_index]        = cost_function(*this, G, node);
                            node_index++;
                        }
                    }
                );
            });
        } else {
            for (RankEncodedNodeId node: nodes) {
                node_to_idx_[node.id()] = node_index;
                cost[node_index]        = cost_function(*this, G, node);
                node_index++;
            }
        }
        stats.cost_function_evaluation_time = t.elapsed_time();
    }

    inline size_t operator()(const LocalGraphView& G_local, NodeId local_node_id) {
        auto idx = node_to_idx_[G_local.node_info[local_node_id].global_id];
        return cost[idx];
    }

    inline Degree degree(RankEncodedNodeId node) {
        // DEBUG_ASSERT(G.is_local_from_local(local_node_id) || G.get_graph_payload().ghost_degree_available,
        //              debug_module{}, name.c_str());
        // return G.degree(local_node_id);
        if (node.rank() == G.rank()) {
            return G.degree(node);
        } else {
            return degree_[node];
        }
    }

    inline Degree out_degree(RankEncodedNodeId node) {
        // return degree_cache.outdegree(local_node_id);
        return out_degree_[node];
    }
    inline PEID rank(NodeId local_node_id) {
        assert(G.ghost_ranks_available());
        G.find_ghost_ranks();
        if (G.is_local(local_node_id)) {
            return G.rank();
        } else {
            return G.get_ghost_data(local_node_id).rank;
        }
    }

private:
    Graph&                           G;
    AuxiliaryNodeData<Degree> const& degree_;
    AuxiliaryNodeData<Degree> const& out_degree_;
    default_map<NodeId, size_t>      node_to_idx_;
    std::vector<size_t>              cost;
    std::string                      name;
};

template <typename Graph>
struct CostFunctionRegistry {
    template <typename ExecutionPolicy = execution_policy::sequential>
    static CostFunction<Graph>
    get(const std::string&                          name,
        Graph&                                      G,
        const Config&                               conf,
        cetric::profiling::LoadBalancingStatistics& stats,
        ExecutionPolicy&&                           policy = {}) {
        (void)conf;
        using RefType   = CostFunction<Graph>&;
        using GraphType = Graph&;
        std::unordered_map<
            std::string,
            std::pair<
                std::function<
                    void(GraphType, node_set&, AuxiliaryNodeData<Degree>&, AuxiliaryNodeData<Degree>&, bool, bool, cetric::profiling::MessageStatistics&)>,
                std::function<size_t(RefType, GraphType, RankEncodedNodeId)>>>
            cost_functions = {
                {"N",
                 {CostFunction<Graph>::init_nop,
                  [](RefType, GraphType, RankEncodedNodeId) {
                      return 1;
                  }}},
                {"D",
                 {CostFunction<Graph>::init_nop,
                  [](RefType ctx, GraphType, RankEncodedNodeId node) {
                      return ctx.degree(node);
                  }}},
                {"DH",
                 {CostFunction<Graph>::init_local_outdegree,
                  [](RefType ctx, GraphType, RankEncodedNodeId node) {
                      return ctx.out_degree(node);
                  }}},
                {"DDH",
                 {CostFunction<Graph>::init_local_outdegree,
                  [](RefType ctx, GraphType, RankEncodedNodeId node) {
                      return ctx.degree(node) * ctx.out_degree(node);
                  }}},
                {"DH2",
                 {CostFunction<Graph>::init_local_outdegree,
                  [](RefType ctx, GraphType, RankEncodedNodeId node) {
                      Degree out_deg = ctx.out_degree(node);
                      return out_deg * out_deg;
                  }}},
                {"DPD",
                 {CostFunction<Graph>::init_all,
                  [](RefType ctx, GraphType G, RankEncodedNodeId v) {
                      size_t cost = 0;
                      for (RankEncodedNodeId u: G.adj(v).neighbors()) {
                          if (std::tuple(ctx.degree(u), u.id()) > std::tuple(ctx.degree(v), v.id())) {
                              cost += ctx.out_degree(v) + ctx.out_degree(u);
                          }
                      }
                      return cost;
                  }}},
                {"IDPD", {CostFunction<Graph>::init_all, [](RefType ctx, GraphType G, RankEncodedNodeId v) {
                              size_t cost = 0;
                              for (RankEncodedNodeId u: G.adj(v).neighbors()) {
                                  if (std::tuple(ctx.degree(u), u.id()) <= std::tuple(ctx.degree(v), v.id())) {
                                      cost += ctx.out_degree(v) + ctx.out_degree(u);
                                  }
                              }
                              return cost;
                          }}}};
        auto it = cost_functions.find(name);
        if (it == cost_functions.end()) {
            throw std::runtime_error("Unsupported cost function");
        }
        auto [init, cost] = it->second;
        AuxiliaryNodeData<Degree> degree;
        AuxiliaryNodeData<Degree> outdegree;
        node_set                  ghosts;
        cetric::profiling::Timer  t;
        init(
            G,
            ghosts,
            degree,
            outdegree,
            conf.dense_degree_exchange,
            conf.compact_degree_exchange,
            stats.message_statistics
        );
        stats.cost_function_communication_time += t.elapsed_time();
        return CostFunction<Graph>(G, name, degree, outdegree, cost, stats, policy);
    }
};
} // namespace cetric

#endif // PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
