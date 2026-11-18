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

#ifndef NEW_LOAD_BALANCING_H_CYE8RYEL
#define NEW_LOAD_BALANCING_H_CYE8RYEL

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <caliper/cali.h>
#include <dstl/redistribute.hpp>
#include <kamping/types/contiguous_type.hpp>
#include <kamping/types/mpi_type_traits.hpp>
#include <kamping/types/scoped_datatype.hpp>
#include <kamping/v2/views/with_type_view.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "cetric/communicator.h"
#include "cetric/config.h"
#include "cetric/datastructures/local_graph_view.h"
#include "cetric/message_statistics.h"

namespace cetric {
namespace load_balancing {

/// Primary load balancing: splits the global prefix sum of a per-vertex cost into p contiguous vertex ranges of
/// about equal cost and moves the vertices (with their adjacency lists) there. The vertices stay in global-ID order,
/// so the move is a dstl::redistribute of the node and edge arrays.
class LoadBalancer {
public:
    template <typename CostFunction>
    static graph::LocalGraphView
    run(graph::LocalGraphView&& G, CostFunction& cost_function, const Config& conf, MPI_Comm comm) {
        CALI_CXX_MARK_SCOPE("redistribution");
        auto const counts = counts_per_destination(G, cost_function, conf, comm);

        // what this rank receives: nodes and edges, summed over all sources
        std::array<std::uint64_t, 2> recv{};
        MPI_Reduce_scatter_block(counts.data(), recv.data(), 2, MPI_UINT64_T, MPI_SUM, comm);
        KASSERT(recv[1] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()));

        profiling::MessageStatistics stats;
        for (int pe = 0; pe < conf.PEs; ++pe) {
            auto const nodes = counts[2 * static_cast<std::size_t>(pe)];
            auto const edges = counts[2 * static_cast<std::size_t>(pe) + 1];
            if (pe == conf.rank) {
                stats.receive_volume = recv[0] + recv[1] - nodes - edges;
                continue;
            }
            stats.sent_messages += (nodes > 0) + (edges > 0);
            stats.send_volume += nodes + edges;
        }

        // The destinations are monotone in global-ID order, so the concatenation of the senders' arrays is the
        // concatenation of the receivers' arrays.
        graph::LocalGraphView          G_balanced;
        kamping::types::ScopedDatatype node_info_type{node_info_datatype()};
        G_balanced.node_info.resize(recv[0]);
        G_balanced.edge_heads.resize(recv[1]);
        int const tag = as_int(MessageTag::LoadBalancing);
        namespace views = kamping::v2::views;
        dstl::redistribute(
            G.node_info | views::with_type(node_info_type.data_type()),
            G_balanced.node_info | views::with_type(node_info_type.data_type()),
            comm,
            tag
        );
        dstl::redistribute(G.edge_heads, G_balanced.edge_heads, comm, tag);
        stats.report();
        return G_balanced;
    }

private:
    static MPI_Datatype node_info_datatype() {
        static_assert(sizeof(graph::LocalGraphView::NodeInfo) == 2 * sizeof(graph::NodeId));
        return kamping::types::mpi_type_traits<std::array<graph::NodeId, 2>>::data_type();
    }

    /// Number of nodes and edges (interleaved, 2 per rank) this rank sends to each rank: node v goes to the rank whose
    /// share [r * C/p, (r+1) * C/p) of the total cost C contains the cost prefix sum before v.
    template <typename CostFunction>
    static std::vector<std::uint64_t> counts_per_destination(
        const graph::LocalGraphView& G, CostFunction& cost_function, const Config& conf, MPI_Comm comm
    ) {
        std::vector<std::size_t> cost(G.node_info.size());
        std::size_t              prefix_sum = 0;
        for (std::size_t node = 0; node < G.node_info.size(); ++node) {
            cost[node] = prefix_sum;
            prefix_sum += cost_function(G, node);
        }

        std::size_t global_prefix = 0;
        MPI_Exscan(&prefix_sum, &global_prefix, 1, kamping::types::mpi_type_traits<std::size_t>::data_type(), MPI_SUM, comm);
        if (conf.rank == 0) {
            global_prefix = 0;
        }
        std::size_t total_cost = global_prefix + prefix_sum;
        MPI_Bcast(&total_cost, 1, kamping::types::mpi_type_traits<std::size_t>::data_type(), conf.PEs - 1, comm);
        std::size_t const per_pe_cost = std::max<std::size_t>((total_cost + conf.PEs - 1) / conf.PEs, 1);

        std::vector<std::uint64_t> counts(2 * static_cast<std::size_t>(conf.PEs), 0);
        for (std::size_t node = 0; node < cost.size(); ++node) {
            auto const new_pe = std::min<std::size_t>((cost[node] + global_prefix) / per_pe_cost, conf.PEs - 1);
            counts[2 * new_pe]++;
            counts[2 * new_pe + 1] += G.node_info[node].degree;
        }
        return counts;
    }
};

} // namespace load_balancing
} // namespace cetric

#endif /* cargend of include guard: NEW_LOAD_BALANCING_H_CYE8RYEL */
