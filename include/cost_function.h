//
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
#define PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H

#include <cstddef>
#include <datastructures/distributed/distributed_graph.h>
#include <datastructures/distributed/graph_communicator.h>
#include <datastructures/distributed/local_graph_view.h>
#include <datastructures/graph_definitions.h>
#include <numeric>
#include <unordered_map>

#include <statistics.h>
#include <type_traits>
#include <utility>

using namespace cetric::graph;

template<class GraphType>
std::vector<Degree> out_degree_from_comm(const GraphType& G, const GraphCommunicator<GraphType>& comm) {
    std::vector<Degree> out_degree(G.local_node_count());
    auto degree = [&](NodeId node) {
        if (G.is_ghost(node)) {
            return comm.get_ghost_degree(node);
        } else {
            return G.degree(node);
        }
    };
    auto is_outgoing = [&](const Edge& e) {
        return std::make_pair(degree(e.tail), G.to_global_id(e.tail)) < std::make_pair(degree(e.head), G.to_global_id(e.head));
    };
    G.for_each_local_node([&](NodeId node) {
        Degree out_deg = 0;
        G.for_each_edge(node, [&](Edge edge) {
            if (is_outgoing(edge)) {
                out_deg++;
            }
        });
        out_degree[node] = out_deg;
    });

    return out_degree;
}


template <typename PayloadType>
class CostFunction {
public:
    template<typename CostFunctionType>
    explicit CostFunction(DistributedGraph<PayloadType> &G,
                          CostFunctionType &&cost_function): G(G), cost(G.local_node_count()) {
        G.for_each_local_node(
            [&](NodeId node) { cost[node] = cost_function(*this, G, node); });
    }

    size_t operator()(NodeId local_node_id) { return cost[local_node_id]; }

    typename std::enable_if<payload_has_degree<PayloadType>::value, Degree>::type
    degree(NodeId local_node_id) {
        return G.degree(local_node_id);
    }

    typename std::enable_if<payload_has_outdegree<PayloadType>::value, Degree>::type
    out_degree(NodeId local_node_id) {
        return G.outdegree(local_node_id);
    }
    PEID rank(NodeId local_node_id) {
        G.find_ghost_ranks();
        if (G.is_local(local_node_id)) {
            return G.rank();
        } else {
            return G.get_ghost_data(local_node_id).rank;
        }
    }
    DistributedGraph<PayloadType> &G;
    std::vector<size_t> cost;
};

template <typename PayloadType> struct CostFunctionRegistry {
    static CostFunction<PayloadType> get(const std::string &name, DistributedGraph<PayloadType> &G,
                    const Config &conf) {
        (void) conf;
        using RefType = CostFunction<PayloadType>&;
        using GraphType = DistributedGraph<PayloadType>&;
        std::unordered_map<
            std::string,
            std::function<size_t(CostFunction<PayloadType> &,
                                 DistributedGraph<PayloadType> &, NodeId)>>
            cost_functions = {
            std::make_pair("N",
                           [](RefType, GraphType, NodeId) { return 1; }),
            std::make_pair("D",
                           [](RefType ctx, GraphType, NodeId node) {
                               return ctx.degree(node);
                           }),
            std::make_pair("DH",
                           [](RefType ctx, GraphType, NodeId node) {
                               return ctx.out_degree(node);
                           })

        };
        return CostFunction<PayloadType>(G, cost_functions[name]);
    }
};

struct AbstractCostFunction {
    virtual size_t operator()(const LocalGraphView& G, NodeId node) const = 0;
    virtual ~AbstractCostFunction() = default;
    const cetric::profiling::MessageStatistics& get_comm_stats() {
       return stats;
    }
protected:
    cetric::profiling::MessageStatistics stats;
};

template<class GraphType>
struct UniformCostFunction : AbstractCostFunction {
    explicit UniformCostFunction(const GraphType&, PEID, PEID) { }
    size_t operator()(const LocalGraphView& G, NodeId) const override {
        (void) G;
        return 1;
    }
};

template<class GraphType>
struct DegreeCostFunction : AbstractCostFunction {
    explicit DegreeCostFunction(const GraphType& G, PEID, PEID): G(G) { }
    size_t operator()(const LocalGraphView& G, NodeId node) const override {
        return G.node_info[node].degree;
    }

private:
    const GraphType& G;
};

template<class GraphType>
struct OutDegreeCostFunction : AbstractCostFunction {
    explicit OutDegreeCostFunction(GraphType& G, PEID rank, PEID size): comm(G, rank, size, as_int(MessageTag::CostFunction)), out_degree_(G.local_node_count()) {
        comm.get_ghost_degree([&](NodeId global_node_id, Degree degree) {
            G.get_ghost_payload(G.to_local_id(global_node_id)).degree = degree;
        }, stats);
        G.for_each_local_node([&](NodeId node) {
            G.for_each_edge(node, [&](Edge e) {
                if (G.is_outgoing(e)) {
                    out_degree_[node]++;
                }
            });
        });
    }
    size_t operator()(const LocalGraphView& G, NodeId node) const override {
        (void) G;
        return out_degree_[node];
    }

private:
    GraphCommunicator<GraphType> comm;
    std::vector<Degree> out_degree_;
};

template<class GraphType>
struct DegreeAndOutDegreeCostFunction : AbstractCostFunction {
    explicit DegreeAndOutDegreeCostFunction(GraphType& G, PEID rank, PEID size): G(G), comm(G, rank, size, as_int(MessageTag::CostFunction)), out_degree_(G.local_node_count()) {
        comm.get_ghost_degree([&](NodeId global_node_id, Degree degree) {
            G.get_ghost_payload(G.to_local_id(global_node_id)).degree = degree;
        }, stats);
        G.for_each_local_node([&](NodeId node) {
            G.for_each_edge(node, [&](Edge e) {
                if (G.is_outgoing(e)) {
                    out_degree_[node]++;
                }
            });
        });
    }
    size_t operator()(const LocalGraphView& G, NodeId node) const override {
      return G.node_info[node].degree * out_degree_[node];
    }

private:
    const GraphType& G;
    GraphCommunicator<GraphType> comm;
    std::vector<Degree> out_degree_;
};

template<class GraphType>
struct OutDegreeSquaredCostFunction : AbstractCostFunction {
    explicit OutDegreeSquaredCostFunction(GraphType& G, PEID rank, PEID size): G(G), comm(G, rank, size, as_int(MessageTag::CostFunction)), out_degree_(G.local_node_count()) {
        comm.get_ghost_degree([&](NodeId global_node_id, Degree degree) {
            G.get_ghost_payload(G.to_local_id(global_node_id)).degree = degree;
        }, stats);
        G.for_each_local_node([&](NodeId node) {
            G.for_each_edge(node, [&](Edge e) {
                if (G.is_outgoing(e)) {
                    out_degree_[node]++;
                }
            });
        });
    }
    size_t operator()(const LocalGraphView& G, NodeId node) const override {
        (void) G;
        return out_degree_[node] * out_degree_[node];
    }

private:
    const GraphType& G;
    GraphCommunicator<GraphType> comm;
    std::vector<Degree> out_degree_;
};

template<class GraphType>
struct OutNeighborOutDegreeCostFunction : AbstractCostFunction {
    explicit OutNeighborOutDegreeCostFunction(GraphType& G, PEID rank, PEID size): G(G), comm(G, rank, size, as_int(MessageTag::CostFunction)), out_degree_(G.local_node_count()), cost_(G.local_node_count()) {
        comm.get_ghost_degree([&](NodeId global_node_id, Degree degree) {
            G.get_ghost_payload(G.to_local_id(global_node_id)).degree = degree;
        }, stats);
        comm.get_ghost_outdegree([&](Edge e) {
            return G.is_outgoing(e);
        }, [&](NodeId global_id, Degree outdegree) {
            G.get_ghost_payload(G.to_local_id(global_id)).outdegree = outdegree;
        }, stats);
        G.for_each_local_node([&](NodeId v) {
            cost_[v] = 0;
            G.for_each_edge(v, [&](Edge edge) {
                if (G.is_outgoing(edge)) {
                    NodeId u = edge.head;
                    cost_[v] += G.outdegree(v) + G.outdegree(u);
                }
           });
        });
    }
    size_t operator()(const LocalGraphView& G, NodeId node) const override {
        (void) G;
        return cost_[node];
    }

private:
    const GraphType& G;
    GraphCommunicator<GraphType> comm;
    std::vector<Degree> out_degree_;
    std::vector<size_t> cost_;
};


template<class GraphType>
struct InNeighborOutDegreeCostFunction : AbstractCostFunction {
    explicit InNeighborOutDegreeCostFunction(GraphType& G, PEID rank, PEID size): G(G), comm(G, rank, size, as_int(MessageTag::CostFunction)), out_degree_(G.local_node_count()), cost_(G.local_node_count()) {
        comm.get_ghost_degree([&](NodeId global_node_id, Degree degree) {
            G.get_ghost_payload(G.to_local_id(global_node_id)).degree = degree;
        }, stats);
        comm.get_ghost_outdegree([&](Edge e) {
            return G.is_outgoing(e);
        }, [&](NodeId global_id, Degree outdegree) {
            G.get_ghost_payload(G.to_local_id(global_id)).outdegree = outdegree;
        }, stats);
        G.for_each_local_node([&](NodeId v) {
            cost_[v] = 0;
            G.for_each_edge(v, [&](Edge edge) {
                if (!G.is_outgoing(edge)) {
                    NodeId u = edge.head;
                    cost_[v] += G.outdegree(v) + G.outdegree(u);
                }
           });
        });
    }
    size_t operator()(const LocalGraphView& G, NodeId node) const override {
        (void) G;
        return cost_[node];
    }

private:
    const GraphType& G;
    GraphCommunicator<GraphType> comm;
    std::vector<Degree> out_degree_;
    std::vector<size_t> cost_;
};


template<class GraphType>
std::unique_ptr<AbstractCostFunction> get_cost_function_by_name(const std::string& name, GraphType& G, PEID rank, PEID size) {
    if (name == "N") {
        return std::make_unique<UniformCostFunction<GraphType>>(G, rank, size);
    } else if (name == "D") {
        return std::make_unique<DegreeCostFunction<GraphType>>(G, rank, size);
    } else if (name == "DH") {
        return std::make_unique<OutDegreeCostFunction<GraphType>>(G, rank, size);
    } else if (name == "DDH") {
        return std::make_unique<DegreeAndOutDegreeCostFunction<GraphType>>(G, rank, size);
    } else if (name == "DH2") {
        return std::make_unique<OutDegreeSquaredCostFunction<GraphType>>(G, rank, size);
    } else if (name == "DPD") {
        return std::make_unique<OutNeighborOutDegreeCostFunction<GraphType>>(G, rank, size);
    } else if (name == "IDPD") {
        return std::make_unique<InNeighborOutDegreeCostFunction<GraphType>>(G, rank, size);
    } else {
        throw std::runtime_error("Unsupported cost function");
    }
}


#endif //PARALLEL_TRIANGLE_COUNTER_COST_FUNCTION_H
