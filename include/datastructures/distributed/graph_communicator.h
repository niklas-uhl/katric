//
// Created by Tim Niklas Uhl on 30.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
#define PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H

#include <datastructures/graph_definitions.h>
#include <comm_utils.h>
#include <util.h>
#include <optional>
#include <vector>
#include <statistics.h>

using namespace cetric::graph;

template<class Graph>
class GraphCommunicator {
public:
    GraphCommunicator(const Graph &G, PEID rank, PEID size, int message_tag) : G(G), rank_(rank), size_(size),
                                                                         send_buffers(size), receive_buffers(size),
                                                                         neighboring_PEs(size),
                                                                         ghost_degree_(G.ghost_count()),
                                                                         ghost_out_degree_(G.ghost_count()),
                                                                         message_tag_(message_tag) {
        send_buffers.set_empty_key(-1);
        receive_buffers.set_empty_key(-1);
        neighboring_PEs.set_empty_key(-1);
    }

    void distribute_degree(cetric::profiling::MessageStatistics& stats) {
        send_buffers.clear();
        receive_buffers.clear();
        for (NodeId node = 0; node < G.local_node_count(); ++node) {
            local_max_degree = std::max(local_max_degree.value_or(0), G.degree(node));
            neighboring_PEs.clear();
            if (G.get_local_data(node).is_interface) {
                Degree deg = G.degree(node);
                G.for_each_edge(node, [&](Edge edge) {
                    if (G.is_ghost(edge.head)) {
                        PEID rank = G.get_ghost_data(edge.head).rank;
                        if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
                            send_buffers[rank].emplace_back(G.to_global_id(node));
                            send_buffers[rank].emplace_back(deg);
                            neighboring_PEs.insert(rank);
                        }
                    }
                });
            }
        }
        CommunicationUtility::sparse_all_to_all(send_buffers, receive_buffers, MPI_NODE, rank_, size_, stats, message_tag_);
        for (const auto& elem : receive_buffers) {
            const std::vector<NodeId>& buffer = elem.second;
            assert(buffer.size() % 2 == 0);
            for (size_t i = 0; i < buffer.size(); i+=2) {
                NodeId node = buffer[i];
                Degree degree = buffer[i + 1];
                assert(G.is_ghost_from_global(node));
                NodeId local_node = G.to_local_id(node);
                ghost_degree_[local_node - G.local_node_count()] = degree;
            }
        }
        degree_broadcast_ = true;
    }

    void distribute_out_degree(const std::vector<Degree>& out_degree, cetric::profiling::MessageStatistics& stats) {
        assert(degree_broadcast_);
        send_buffers.clear();
        receive_buffers.clear();
        for (NodeId node = 0; node < G.local_node_count(); ++node) {
            local_max_out_degree = std::max(local_max_out_degree.value_or(0), G.degree(node));
            neighboring_PEs.clear();
            if (G.get_local_data(node).is_interface) {
                G.for_each_edge(node, [&](Edge edge) {
                    if (G.is_ghost(edge.head)) {
                        PEID rank = G.get_ghost_data(edge.head).rank;
                        if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
                            send_buffers[rank].emplace_back(G.to_global_id(node));
                            send_buffers[rank].emplace_back(out_degree[node]);
                            neighboring_PEs.insert(rank);
                        }
                    }
                });
            }
        }
        CommunicationUtility::sparse_all_to_all(send_buffers, receive_buffers, MPI_NODE, rank_, size_, stats, message_tag_ + 1);
        for (const auto& elem : receive_buffers) {
            const std::vector<NodeId>& buffer = elem.second;
            assert(buffer.size() % 2 == 0);
            for (size_t i = 0; i < buffer.size(); i+=2) {
                NodeId node = buffer[i];
                Degree degree = buffer[i + 1];
                NodeId local_node = G.to_local_id(node);
                ghost_out_degree_[local_node - G.local_node_count()] = degree;
            }
        }
        out_degree_broadcast_ = true;
    }


    void distribute_out_degree(cetric::profiling::MessageStatistics& stats) {
        auto out_degree = get_out_degree(stats);
        distribute_out_degree(out_degree, stats);
    }

    Degree get_ghost_degree(NodeId node) const {
        return ghost_degree_[node - G.local_node_count()];
    }

    Degree get_ghost_out_degree(NodeId node) const {
        return ghost_out_degree_[node - G.local_node_count()];
    }

    Degree get_max_degree() {
        if (!global_max_degree.has_value()) {
            if (!local_max_degree.has_value()) {
                G.for_each_local_node([&](NodeId node) {
                        local_max_degree = std::max(local_max_degree.value_or(0), G.degree(node));
                });
            }
            Degree max;
            MPI_Allreduce(&local_max_degree.value(), &max, 1, MPI_NODE, MPI_MAX, MPI_COMM_WORLD);
            global_max_degree = max;
        }
        return global_max_degree.value();
    }

    Degree get_max_out_degree() {
        if (!global_max_out_degree.has_value()) {
            if (!local_max_out_degree.has_value()) {
                auto out_degree = get_out_degree();
                local_max_out_degree = max_element(out_degree.begin(), out_degree.end());
            }
            Degree max;
            MPI_Allreduce(&local_max_out_degree.value(), &max, 1, MPI_NODE, MPI_MAX, MPI_COMM_WORLD);
            global_max_out_degree = max;
        }
        return global_max_out_degree.value();
    }

private:
    std::vector<Degree> get_out_degree(cetric::profiling::MessageStatistics& stats) {
        if (!degree_broadcast_) {
            distribute_degree(stats);
        }
        std::vector<Degree> out_degree(G.local_node_count());
        auto degree = [&](NodeId node) {
            if (G.is_ghost(node)) {
                return get_ghost_degree(node);
            } else {
                return G.degree(node);
            }
        };
        auto is_outgoing = [&](const Edge& e) {
            return std::make_pair(degree(e.tail), G.to_global_id(e.tail)) <
                std::make_pair(degree(e.head), G.to_global_id(e.head));
        };
        for (NodeId node = 0; node < G.local_node_count(); ++node) {
            bool is_interface = G.get_local_data(node).is_interface;
            if (is_interface) {
                Degree out_deg = 0;
                G.for_each_edge(node, [&](Edge edge) {
                    if (is_outgoing(edge)) {
                        out_deg++;
                    }
                });
                out_degree[node] = out_deg;
            }
        }
        return out_degree;
    }

    const Graph& G;
    PEID rank_;
    PEID size_;
    google::dense_hash_map<PEID, std::vector<NodeId>> send_buffers;
    google::dense_hash_map<PEID, std::vector<NodeId>> receive_buffers;
    google::dense_hash_set<PEID> neighboring_PEs;
    std::optional<NodeId> local_max_degree;
    std::optional<NodeId> local_max_out_degree;
    std::optional<NodeId> global_max_degree;
    std::optional<NodeId> global_max_out_degree;
    std::vector<Degree> ghost_degree_;
    std::vector<Degree> ghost_out_degree_;
    bool degree_broadcast_ = false;
    bool out_degree_broadcast_ = false;
    int message_tag_;
};


#endif //PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
