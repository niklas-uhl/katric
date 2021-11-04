//
// Created by Tim Niklas Uhl on 30.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
#define PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H

#include <datastructures/graph_definitions.h>
#include <google/dense_hash_map>
#include <communicator.h>
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
                                                                         message_tag_(message_tag) {
        send_buffers.set_empty_key(-1);
        receive_buffers.set_empty_key(-1);
        neighboring_PEs.set_empty_key(-1);
    }

    template<typename DegreeFunc>
    void get_ghost_degree(DegreeFunc&& on_degree_receive, cetric::profiling::MessageStatistics& stats) {
        assert(G.ghost_ranks_available());
        send_buffers.clear();
        receive_buffers.clear();
        for (NodeId node = 0; node < G.local_node_count(); ++node) {
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
                on_degree_receive(node, degree);
            }
        }
    }

    template <typename OnDegreeFunc>
    void get_ghost_outdegree(OnDegreeFunc &&on_degree_receive,
                             cetric::profiling::MessageStatistics &stats) {
        assert(G.oriented());
        get_ghost_outdegree([](auto) {}, on_degree_receive, stats);
    }

    template<typename EdgePred, typename OnDegreeFunc>
    void get_ghost_outdegree(EdgePred&& is_outgoing, OnDegreeFunc&& on_degree_receive, cetric::profiling::MessageStatistics& stats) {
        assert(G.ghost_ranks_available());
        auto get_out_degree = [&](NodeId local_node_id) {
            if (G.oriented()) {
                return G.outdegree(local_node_id);
            } else {
                Degree outdegree = 0;
                G.for_each_edge(local_node_id, [&](Edge e) {
                    if (is_outgoing(e)) {
                        outdegree++;
                    }
                });
                return outdegree;
            }
        };
        // assert(G.oriented());
        send_buffers.clear();
        receive_buffers.clear();
        for (NodeId node = 0; node < G.local_node_count(); ++node) {
            neighboring_PEs.clear();
            if (G.get_local_data(node).is_interface) {
                G.for_each_edge(node, [&](Edge edge) {
                  if (G.is_ghost(edge.head)) {
                    PEID rank = G.get_ghost_data(edge.head).rank;
                    if (neighboring_PEs.find(rank) == neighboring_PEs.end()) {
                      send_buffers[rank].emplace_back(G.to_global_id(node));
                      send_buffers[rank].emplace_back(get_out_degree(node));
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
                on_degree_receive(node, degree);
            }
        }
    }

private:

    const Graph& G;
    PEID rank_;
    PEID size_;
    google::dense_hash_map<PEID, std::vector<NodeId>> send_buffers;
    google::dense_hash_map<PEID, std::vector<NodeId>> receive_buffers;
    google::dense_hash_set<PEID> neighboring_PEs;
    int message_tag_;
};


#endif //PARALLEL_TRIANGLE_COUNTER_GRAPH_COMMUNICATOR_H
