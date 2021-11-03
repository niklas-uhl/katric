//
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
#define PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H

#include "communicator.h"
#include <cstddef>
#include <datastructures/distributed/distributed_graph.h>
#include <datastructures/distributed/graph_communicator.h>
#include <datastructures/distributed/local_graph_view.h>
#include <datastructures/graph_definitions.h>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <statistics.h>
#include <type_traits>
#include <utility>

using namespace cetric::graph;

template <typename Graph>
class CostFunction {
public:

    static void init_degree(Graph& G) {
        GraphCommunicator comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
        //TODO replace this dummy
        cetric::profiling::MessageStatistics dummy;
        if (!G.get_graph_payload().ghost_degree_available) {
          comm.get_ghost_degree(
              [&](NodeId global_id, Degree degree) {
                G.get_ghost_payload(G.to_local_id(global_id)).degree = degree;
              },
              dummy);
          G.get_graph_payload().ghost_degree_available = true;
        }
    }

    static void init_all(Graph &G) {
        GraphCommunicator<Graph> comm(G, G.rank(), G.size(), as_int(MessageTag::CostFunction));
        cetric::profiling::MessageStatistics dummy;
        if (!G.get_graph_payload().ghost_degree_available) {
          comm.get_ghost_degree(
              [&](NodeId global_id, Degree degree) {
                G.get_ghost_payload(G.to_local_id(global_id)).degree = degree;
              },
              dummy);
          G.get_graph_payload().ghost_degree_available = true;
        }
        if (!G.get_graph_payload().ghost_outdegree_available) {
            comm.get_ghost_outdegree(
                [&](Edge e) { return G.is_outgoing(e); },
                [&](NodeId global_id, Degree outdegree) {
                    G.get_ghost_payload(G.to_local_id(global_id)).outdegree =
                        outdegree;
                },
                dummy);
            G.get_graph_payload().ghost_outdegree_available = true;
        }
    }

    template<typename CostFunctionType>
    explicit CostFunction(Graph &G,
                         CostFunctionType &&cost_function): G(G), cost(G.local_node_count()) {
        G.for_each_local_node(
            [&](NodeId node) { cost[node] = cost_function(*this, G, node); });
    }

    size_t operator()(const LocalGraphView& G_local, NodeId local_node_id) { return cost[local_node_id]; }

    typename std::enable_if<payload_has_degree<typename Graph::payload_type>::value, Degree>::type
    degree(NodeId local_node_id) {
        return G.degree(local_node_id);
    }

    typename std::enable_if<
        payload_has_outdegree<typename Graph::payload_type>::value,
        Degree>::type
    out_degree(NodeId local_node_id) {
      return G.outdegree(local_node_id);
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
    Graph &G;
    std::vector<size_t> cost;
};

template <typename Graph> struct CostFunctionRegistry {
    static CostFunction<Graph> get(const std::string &name, Graph &G,
                                   const Config &conf) {
        (void) conf;
        using RefType = CostFunction<Graph>&;
        using GraphType = Graph&;
        std::unordered_map<
            std::string,
            std::pair<std::function<void(GraphType)>,
                      std::function<size_t(RefType, GraphType, NodeId)>>>
            cost_functions = {
            {"N",
             {[](auto) {},
              [](RefType, GraphType, NodeId) {
                  return 1;
              }}},
            {"D",
             {[](auto) {},
              [](RefType ctx, GraphType, NodeId node) {
                  return ctx.degree(node);
              }}},
            {"DH",
             {[](auto) {},
              [](RefType ctx, GraphType, NodeId node) {
                  return ctx.out_degree(node);
              }}},
            {"DDH",
             {[](auto) {},
              [](RefType ctx, GraphType, NodeId node){
                  return ctx.degree(node) * ctx.out_degree(node);
              }}},
            {"DH2",
             {[](auto) {},
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
            {"IDPD",
             {CostFunction<Graph>::init_all,
              [](RefType ctx, GraphType G, NodeId v) {
                  size_t cost = 0;
                  G.for_each_edge(v, [&](Edge e) {
                      if (!G.is_outgoing(e)) {
                          NodeId u = e.head;
                          cost += ctx.out_degree(v) + ctx.out_degree(u);
                      }
                  });
                  return cost;
              }
             }}};
        auto it = cost_functions.find(name);
        if (it == cost_functions.end()) {
            throw std::runtime_error("Unsupported cost function");
        }
        auto [init, cost] = *it.second;
        init(G);
        return CostFunction<Graph>(G, cost);
    }
};

#endif //PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
