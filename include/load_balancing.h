#ifndef NEW_LOAD_BALANCING_H_CYE8RYEL
#define NEW_LOAD_BALANCING_H_CYE8RYEL

#include <algorithm>
#include <config.h>
#include <datastructures/distributed/compact_graph.h>
#include <numeric>
#include <string>
#include <vector>
#include <mpi.h>

namespace cetric {
namespace load_balancing {

class LoadBalancer {
    public:
    LoadBalancer(CompactGraph& G, const Config& conf) {
        std::vector<size_t> cost(G.local_node_count());
        size_t prefix_sum = 0;
        G.for_each_local_node([&](NodeId node) {
            cost[node] = prefix_sum;
            prefix_sum += G.degree(node);
        });
        //atomic_debug(cost);
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
        for (size_t i = 0; i < cost.size(); ++i) {
            cost[i] += global_prefix;
        }
        std::vector<std::pair<NodeId, NodeId>> to_send(conf.PEs);
        G.for_each_local_node([&](NodeId node) {
            PEID new_pe = cost[node] / per_pe_cost;
            //atomic_debug("cost: "+ std::to_string(cost[node]));
            atomic_debug("Node " + std::to_string( G.to_global_id(node) ) + " assigned to PE " + std::to_string(new_pe));
            to_send[new_pe].first++;
            to_send[new_pe].second += G.degree(node);
        });
        std::vector<std::pair<NodeId, NodeId>> to_receive(conf.PEs);
        MPI_Alltoall(to_send.data(), 2, MPI_NODE, to_receive.data(), 2, MPI_NODE, MPI_COMM_WORLD);

        std::vector<int> send_counts(conf.PEs);
        std::vector<int> send_displs(conf.PEs);
        std::vector<int> recv_counts(conf.PEs);
        std::vector<int> recv_displs(conf.PEs);

        std::transform(to_send.begin(), to_send.end(), send_counts.begin(), [](auto pair) { return pair.first * 2; });
        std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);

        std::transform(to_receive.begin(), to_receive.end(), recv_counts.begin(), [](auto pair) { return pair.first * 2; });
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

        for(size_t i = 0; i < G.edge_pointer.size() - 1; i += 2) {
            G.edge_pointer[i] = G.degree(i / 2);
        }

        std::vector<NodeId> edge_pointer(recv_displs[conf.PEs - 1] + recv_counts[conf.PEs - 1] + 1);
        MPI_Alltoallv(G.edge_pointer.data(), send_counts.data(), send_displs.data(), MPI_NODE, edge_pointer.data(), recv_counts.data(), recv_displs.data(), MPI_NODE, MPI_COMM_WORLD);
        /* atomic_debug(G.edge_pointer); */
        /* atomic_debug(edge_pointer); */
        
        int send_running_sum = 0;
        int recv_running_sum = 0;
        for(size_t i = 0; i < send_counts.size(); ++i) {
            send_counts[i] = to_send[i].second;
            send_displs[i] = send_running_sum;
            send_running_sum += send_counts[i];

            recv_counts[i] = to_receive[i].second;
            recv_displs[i] = recv_running_sum;
            recv_running_sum += recv_counts[i];
        }
        std::vector<NodeId> head(recv_displs[conf.PEs - 1] + recv_counts[conf.PEs - 1]);
        MPI_Alltoallv(G.head.data(), send_counts.data(), send_displs.data(), MPI_NODE, head.data(), recv_counts.data(), recv_displs.data(), MPI_NODE, MPI_COMM_WORLD);
        
        NodeId running_sum = 0;
        //atomic_debug(edge_pointer);
        for(size_t i = 0; i < edge_pointer.size(); i+=2) {
            auto val = edge_pointer[i];
            edge_pointer[i] = running_sum;
            running_sum += val;
        }
        edge_pointer[edge_pointer.size() - 1] = running_sum;
        //atomic_debug(edge_pointer);
        //atomic_debug(head);
        G.edge_pointer = std::move(edge_pointer);
        G.head = std::move(head);
    }
};

}
namespace communication {

class GraphRelocator {
};

}
}

#endif /* cargend of include guard: NEW_LOAD_BALANCING_H_CYE8RYEL */
