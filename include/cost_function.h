//
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
#define PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H

#include <datastructures/distributed/distributed_graph.h>
#include <datastructures/distributed/graph_communicator.h>
#include <datastructures/distributed/local_graph_view.h>
#include <datastructures/graph_definitions.h>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <sparsehash/dense_hash_map>
#include <stdexcept>
#include <unordered_map>
#include "communicator.h"
#include "fmt/core.h"
#include "util.h"

#include <statistics.h>
#include <type_traits>
#include <utility>

using namespace cetric::graph;

class OutDegreeCache {
public:
    OutDegreeCache() : G(nullptr), data_() {}
    OutDegreeCache(const DistributedGraph<>& G)
        : G(&G), data_(G.local_node_count() + G.ghost_count(), std::numeric_limits<Degree>::max()) {}

    void set(NodeId local_node_id, Degree degree) {
        assert(G->is_local_from_local(local_node_id) || G->is_ghost(local_node_id));
        data_[local_node_id] = degree;
    }

    Degree get(NodeId local_node_id) {
        assert(G->is_local_from_local(local_node_id) || G->is_ghost(local_node_id));
        if (data_[local_node_id] == std::numeric_limits<Degree>::max()) {
            assert(G->is_local_from_local(local_node_id));
            Degree out_deg = 0;
            G->for_each_edge(local_node_id, [&](Edge e) {
                if (G->is_outgoing(e)) {
                    out_deg++;
                }
            });
            data_[local_node_id] = out_deg;
        }
        return data_[local_node_id];
    }

private:
    const DistributedGraph<>* G;
    std::vector<Degree> data_;
};

template <typename Graph>
class CostFunction {
public:
    static void init_nop(Graph& G, OutDegreeCache& cache, cetric::profiling::MessageStatistics& stats) {
        (void) G;
        (void)cache;
        (void) stats;
    }
    static void init_degree(Graph& G, OutDegreeCache& cache, cetric::profiling::MessageStatistics& stats) {
        (void) cache;
        GraphCommunicator comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
        if (!G.get_graph_payload().ghost_degree_available) {
            comm.get_ghost_degree(
                [&](NodeId global_id, Degree degree) { G.get_ghost_payload(G.to_local_id(global_id)).degree = degree; },
                stats);
            G.get_graph_payload().ghost_degree_available = true;
        }
    }

    static void init_all(Graph& G, OutDegreeCache& cache, cetric::profiling::MessageStatistics& stats) {
        cache = OutDegreeCache(G);
        GraphCommunicator<Graph> comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
        if (!G.get_graph_payload().ghost_degree_available) {
            comm.get_ghost_degree(
                [&](NodeId global_id, Degree degree) { G.get_ghost_payload(G.to_local_id(global_id)).degree = degree; },
                stats);
            G.get_graph_payload().ghost_degree_available = true;
        }
        comm.get_ghost_outdegree(
            [&](Edge e) { return G.is_outgoing(e); },
            [&](NodeId global_id, Degree outdegree) { cache.set(G.to_local_id(global_id), outdegree); }, stats);
    }

    template <typename CostFunctionType>
    explicit CostFunction(Graph& G, const std::string& name, OutDegreeCache& cache, CostFunctionType&& cost_function)
        : G(G), outdegree_cache(cache), global_to_index(G.local_node_count()), cost(G.local_node_count()), name(name) {
        global_to_index.set_empty_key(-1);
        G.for_each_local_node([&](NodeId node) {
            global_to_index[G.to_global_id(node)] = node;
            cost[node] = cost_function(*this, G, node);
        });
    }

    size_t operator()(const LocalGraphView& G_local, NodeId local_node_id) {
        auto idx = global_to_index[G_local.node_info[local_node_id].global_id];
        return cost[idx];
    }

    typename std::enable_if<payload_has_degree<typename Graph::payload_type>::value, Degree>::type degree(
        NodeId local_node_id) {
        DEBUG_ASSERT(G.is_local_from_local(local_node_id) || G.get_graph_payload().ghost_degree_available,
                     debug_module{}, name.c_str());
        return G.degree(local_node_id);
    }

    typename std::enable_if<payload_has_degree<typename Graph::payload_type>::value, Degree>::type out_degree(
        NodeId local_node_id) {
        // return G.outdegree(local_node_id);
        Degree out_deg = 0;
        if (G.is_ghost(local_node_id)) {
            // TODO restructure
            return outdegree_cache.get(local_node_id);
            // return G.get_ghost_payload(local_node_id).outdegree;
            // return G.outdegree(local_node_id);
        }
        G.for_each_edge(local_node_id, [&](Edge e) {
            if (G.is_outgoing(e)) {
                out_deg++;
            }
        });
        return out_deg;
    }
    PEID rank(NodeId local_node_id) {
        assert(G.ghost_ranks_available());
        G.find_ghost_ranks();
        if (G.is_local(local_node_id)) {
            return G.rank();
        } else {
            return G.get_ghost_data(local_node_id).rank;
        }
    }

private:
    Graph& G;
    OutDegreeCache& outdegree_cache;
    google::dense_hash_map<NodeId, NodeId> global_to_index;
    std::vector<size_t> cost;
    std::string name;
};

template <typename Graph>
struct CostFunctionRegistry {
    static CostFunction<Graph> get(const std::string& name,
                                   Graph& G,
                                   const Config& conf,
                                   cetric::profiling::MessageStatistics& stats) {
        (void)conf;
        using RefType = CostFunction<Graph>&;
        using GraphType = Graph&;
        std::unordered_map<
            std::string,
            std::pair<std::function<void(GraphType, OutDegreeCache&, cetric::profiling::MessageStatistics&)>,
                      std::function<size_t(RefType, GraphType, NodeId)>>>
            cost_functions = {
                {"N",
                 {CostFunction<Graph>::init_nop,
                  [](RefType, GraphType, NodeId) {
                      return 1;
                  }}},
                {"D",
                 {CostFunction<Graph>::init_nop,
                  [](RefType ctx, GraphType, NodeId node) {
                      return ctx.degree(node);
                  }}},
                {"DH",
                 {CostFunction<Graph>::init_nop,
                  [](RefType ctx, GraphType, NodeId node) {
                      return ctx.out_degree(node);
                  }}},
                {"DDH",
                 {CostFunction<Graph>::init_nop,
                  [](RefType ctx, GraphType, NodeId node) {
                      return ctx.degree(node) * ctx.out_degree(node);
                  }}},
                {"DH2",
                 {CostFunction<Graph>::init_nop,
                   [](RefType ctx, GraphType, NodeId node) {
                       Degree out_deg = ctx.out_degree(node);
                       return out_deg * out_deg;
                   }}},
                 {"DPD",
                  {CostFunction<Graph>::init_all,
                   [](RefType ctx, GraphType G, NodeId v) {
                       size_t cost = 0;
                       G.for_each_edge(v, [&](Edge e) {
                           if (G.is_outgoing(e)) {
                               NodeId u = e.head;
                               cost += ctx.out_degree(v) + ctx.out_degree(u);
                           }
                       });
                       return cost;
                   }}},
                 {"IDPD", {CostFunction<Graph>::init_all, [](RefType ctx, GraphType G, NodeId v) {
                               size_t cost = 0;
                               G.for_each_edge(v, [&](Edge e) {
                                   if (!G.is_outgoing(e)) {
                                       NodeId u = e.head;
                                       cost += ctx.out_degree(v) + ctx.out_degree(u);
                                   }
                               });
                               return cost;
                           }}}};
        auto it = cost_functions.find(name);
        if (it == cost_functions.end()) {
            throw std::runtime_error("Unsupported cost function");
        }
        auto [init, cost] = it->second;
        G.check_consistency();
        OutDegreeCache cache;
        init(G, cache, stats);
        return CostFunction<Graph>(G, name, cache, cost);
    }
};

#endif  // PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
