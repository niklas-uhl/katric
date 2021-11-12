#ifndef NEW_LOAD_BALANCING_H_CYE8RYEL
#define NEW_LOAD_BALANCING_H_CYE8RYEL

#include <config.h>
#include <cost_function.h>
#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "datastructures/distributed/local_graph_view.h"
#include "datastructures/graph_definitions.h"
#include "debug_assert.hpp"
#include "util.h"
#include <fmt/core.h>

namespace cetric {
namespace load_balancing {

class LoadBalancer {
public:
    template <typename CostFunction>
    static graph::LocalGraphView run(graph::LocalGraphView&& G, CostFunction& cost_function, const Config& conf) {
        auto to_send = reassign_nodes(G, cost_function, conf);

        // TODO we should also try sparse all to all
        std::vector<std::pair<NodeId, NodeId>> to_receive(conf.PEs);
        MPI_Alltoall(to_send.data(), 2, MPI_NODE, to_receive.data(), 2, MPI_NODE, MPI_COMM_WORLD);

        std::vector<int> send_counts(conf.PEs);
        std::vector<int> send_displs(conf.PEs);
        std::vector<int> recv_counts(conf.PEs);
        std::vector<int> recv_displs(conf.PEs);

        int send_running_sum = 0;
        int recv_running_sum = 0;
        for (size_t i = 0; i < send_counts.size(); ++i) {
            send_counts[i] = to_send[i].first;
            send_displs[i] = send_running_sum;
            send_running_sum += send_counts[i];

            recv_counts[i] = to_receive[i].first;
            recv_displs[i] = recv_running_sum;
            recv_running_sum += recv_counts[i];
        }

        MPI_Datatype mpi_node_info;
        MPI_Type_contiguous(2, MPI_NODE, &mpi_node_info);
        MPI_Type_commit(&mpi_node_info);

        auto [nodes_before, _displs] =
            CommunicationUtility::gather(G.node_info, mpi_node_info, MPI_COMM_WORLD, 0, conf.rank, conf.PEs);

        std::vector<LocalGraphView::NodeInfo> node_info_recv(recv_displs[conf.PEs - 1] + recv_counts[conf.PEs - 1]);

        MPI_Alltoallv(G.node_info.data(), send_counts.data(), send_displs.data(), mpi_node_info, node_info_recv.data(),
                      recv_counts.data(), recv_displs.data(), mpi_node_info, MPI_COMM_WORLD);
        G.node_info.resize(0);
        G.node_info.shrink_to_fit();

        send_running_sum = 0;
        recv_running_sum = 0;
        for (size_t i = 0; i < send_counts.size(); ++i) {
            send_counts[i] = to_send[i].second;
            send_displs[i] = send_running_sum;
            send_running_sum += send_counts[i];

            recv_counts[i] = to_receive[i].second;
            recv_displs[i] = recv_running_sum;
            recv_running_sum += recv_counts[i];
        }
        std::vector<NodeId> head(recv_displs[conf.PEs - 1] + recv_counts[conf.PEs - 1]);
        MPI_Alltoallv(G.edge_heads.data(), send_counts.data(), send_displs.data(), MPI_NODE, head.data(),
                      recv_counts.data(), recv_displs.data(), MPI_NODE, MPI_COMM_WORLD);
        G.edge_heads.resize(0);
        G.edge_heads.shrink_to_fit();

        LocalGraphView G_balanced;
        G_balanced.node_info = std::move(node_info_recv);
        G_balanced.edge_heads = std::move(head);

        return G_balanced;
    }

private:
    template <typename CostFunction>
    static std::vector<std::pair<NodeId, EdgeId>> reassign_nodes(const graph::LocalGraphView& G,
                                                                 CostFunction& cost_function,
                                                                 const Config& conf) {
        using namespace cetric::graph;
        std::vector<size_t> cost(G.node_info.size());
        size_t prefix_sum = 0;
        for (size_t node = 0; node < G.node_info.size(); ++node) {
            cost[node] = prefix_sum;
            size_t node_cost = cost_function(G, node);
            prefix_sum += node_cost;
        }

        size_t global_prefix;
        MPI_Exscan(&prefix_sum, &global_prefix, 1, MPI_NODE, MPI_SUM, MPI_COMM_WORLD);
        if (conf.rank == 0) {
            global_prefix = 0;
        }
        size_t total_cost;
        if (conf.rank == conf.PEs - 1) {
            total_cost = global_prefix + prefix_sum;
        }
        MPI_Bcast(&total_cost, 1, MPI_NODE, conf.PEs - 1, MPI_COMM_WORLD);
        size_t per_pe_cost = (total_cost + conf.PEs - 1) / conf.PEs;
        std::vector<std::pair<NodeId, EdgeId>> to_send(conf.PEs);
        for (size_t node = 0; node < cost.size(); ++node) {
            cost[node] += global_prefix;
            PEID new_pe = std::min(static_cast<int>(cost[node] / per_pe_cost), conf.PEs - 1);
            to_send[new_pe].first++;
            to_send[new_pe].second += G.node_info[node].degree;
        }

        cost.resize(0);
        cost.shrink_to_fit();
        return to_send;
    }
};

}  // namespace load_balancing
}  // namespace cetric

#endif /* cargend of include guard: NEW_LOAD_BALANCING_H_CYE8RYEL */
