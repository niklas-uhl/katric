//
// Created by Tim Niklas Uhl on 20.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_IO_H
#define PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_IO_H

#include <filesystem>

#include <datastructures/graph_definitions.h>
#include <datastructures/distributed/local_graph_view.h>
#include <limits>
#include <locale>
#include <communicator.h>
#include <sparsehash/dense_hash_set>
#include <string>
#include <type_traits>
#include <io/definitions.h>
#include <statistics.h>
#include "util.h"
#include "graph_io.h"
#include <google/dense_hash_set>

namespace cetric {
    using namespace cetric::graph;

    using node_set = google::dense_hash_set<NodeId>;

struct GraphInfo {

    NodeId total_node_count;
    NodeId local_from;
    NodeId local_to;

    static GraphInfo even_distribution(NodeId total_node_count, PEID rank, PEID size) {
        GraphInfo graph_info;
        NodeId remaining_nodes = total_node_count % size;
        NodeId local_node_count = (total_node_count / size) + static_cast<NodeId>(static_cast<size_t>(rank) < remaining_nodes);
        NodeId local_from = (rank * local_node_count) + static_cast<NodeId>(static_cast<size_t>(rank) >= remaining_nodes ? remaining_nodes : 0);
        NodeId local_to = local_from + local_node_count;
        graph_info.total_node_count = total_node_count;
        graph_info.local_from = local_from;
        graph_info.local_to = local_to;
        return graph_info;
    }

    NodeId local_node_count() const {
        return local_to - local_from;
    }
};

void read_metis_distributed(const std::string& input, const GraphInfo& graph_info, std::vector<Edge>& edge_list, node_set& ghosts, PEID rank, PEID size);

void read_metis_distributed(const std::string& input, const GraphInfo& graph_info, std::vector<EdgeId>& first_out, std::vector<NodeId>& head, PEID rank, PEID size);

void gather_PE_ranges(NodeId local_from, NodeId local_to, std::vector<std::pair<NodeId, NodeId>>& ranges, const MPI_Comm& comm, PEID rank, PEID size);

PEID get_PE_from_node_ranges(NodeId node, const std::vector<std::pair<NodeId, NodeId>>& ranges);

template<class EdgeList>
void fix_broken_edge_list(EdgeList& edge_list, const std::vector<std::pair<NodeId, NodeId>>& ranges, node_set& ghosts, PEID rank, PEID size) {
    NodeId local_from = ranges[rank].first;
    NodeId local_to = ranges[rank].second;
    BufferedCommunicator<NodeId> communicator(std::numeric_limits<size_t>::max(), MPI_NODE, rank, size, as_int(MessageTag::RHGFix));
    cetric::profiling::MessageStatistics dummy_stats;
    auto handle_message = [&](PEID, const std::vector<NodeId> &message) {
        for (size_t i = 0; i < message.size(); i += 2) {
            edge_list.emplace_back(message[i + 1], message[i]);
            ghosts.insert(message[i]);
        }
    };
    for (auto &edge : edge_list) {
        NodeId tail;
        NodeId head;
        if constexpr (!std::is_same<EdgeList, std::vector<Edge>>::value) {
            tail = edge.first;
            head = edge.second;
        } else {
            tail = edge.tail;
            head = edge.head;
        }
        if (tail >= local_from && tail < local_to) {
            if (head < local_from || head >= local_to) {
                communicator.add_message({tail, head}, get_PE_from_node_ranges(head, ranges), handle_message,
                                         dummy_stats);
            }
        }
    }
    communicator.all_to_all(handle_message, dummy_stats);

    if constexpr (!std::is_same<EdgeList, std::vector<Edge>>::value) {
        std::sort(edge_list.begin(), edge_list.end());
    } else {
        std::sort(edge_list.begin(), edge_list.end(), [&](const Edge& e1, const Edge& e2) {
            return std::tie(e1.tail, e1.head) < std::tie(e2.tail, e2.head);
        });
    }

    //kagen sometimes produces duplicate edges
    auto it = std::unique(edge_list.begin(), edge_list.end());
    edge_list.erase(it, edge_list.end());
}

LocalGraphView gen_local_graph(const Config& conf, PEID rank, PEID size);

LocalGraphView read_local_metis_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size);

LocalGraphView read_local_partitioned_edgelist(const std::string& input, const Config& conf, PEID rank, PEID size);

LocalGraphView read_local_binary_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size);

void read_graph_info_from_binary(const std::string& input, NodeId& node_count, EdgeId& edge_count);

LocalGraphView read_local_graph(const std::string& input, InputFormat format, PEID rank, PEID size);

std::pair<NodeId, NodeId> get_node_range(const std::string& input, PEID rank, PEID size);

}
#endif //PARALLEL_TRIANGLE_COUNTER_DISTRIBUTED_GRAPH_IO_H
