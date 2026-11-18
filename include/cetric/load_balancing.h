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

// Primary load balancing: repartitions the graph into contiguous vertex ranges of about equal cost.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <dstl/redistribute.hpp>
#include <kamping/v2/collectives/bcast.hpp>
#include <kamping/v2/collectives/scan.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/sentinels.hpp>
#include <kamping/v2/views/ref_single_view.hpp>
#include <kagen.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "cetric/message_tags.h"
#include "cetric/message_statistics.h"
#include "cetric/util.h"

namespace cetric::load_balancing {

/// Moves the vertices of `G` (KaGen's CSR, with their adjacency lists) so that each rank gets a contiguous range of
/// about C/p of the total cost C: vertex v goes to the rank whose share [r C/p, (r+1) C/p) contains the cost before v
/// (`cost[i]` is the cost of vertex first + i). The vertices stay in global-id order, so the move is a
/// dstl::redistribute of the degrees and of the heads. Region `redistribution`.
inline kagen::Graph run(kagen::Graph&& G, std::span<std::size_t const> cost, MPI_Comm comm) {
    CALI_CXX_MARK_SCOPE("redistribution");
    kamping::v2::comm_view const cv{comm};
    auto const                   size = static_cast<std::size_t>(cv.size());
    auto const                   rank = static_cast<std::size_t>(cv.rank());
    std::size_t const            n    = G.vertex_range.second - G.vertex_range.first;
    KASSERT(cost.size() == n);

    // the cost before each vertex, globally: the ranks before (one scan), then the vertices before on this rank; the
    // last rank's sum is the total
    namespace views                = kamping::v2::views;
    std::uint64_t const local_cost = std::reduce(cost.begin(), cost.end(), std::uint64_t{0});
    std::uint64_t       total      = local_cost; // inclusive prefix here, the total after the broadcast
    kamping::v2::scan(kamping::v2::inplace, views::ref_single(total), comm);
    std::vector<std::size_t> cost_before(n);
    std::exclusive_scan(cost.begin(), cost.end(), cost_before.begin(), total - local_cost);
    kamping::v2::bcast(views::ref_single(total), static_cast<int>(size) - 1, comm);
    std::size_t const per_pe_cost = std::max<std::size_t>((total + size - 1) / size, 1);

    // vertices and edges this rank sends to each rank, interleaved; and what it receives
    std::vector<std::uint64_t> send(2 * size, 0);
    auto const degree = [&G](std::size_t i) {
        return G.xadj[i + 1] - G.xadj[i];
    };
    for (std::size_t i = 0; i < n; ++i) {
        auto const dest = std::min(cost_before[i] / per_pe_cost, size - 1);
        send[2 * dest]++;
        send[2 * dest + 1] += degree(i);
    }
    // kamping-v2 has no reduce-scatter yet
    std::array<std::uint64_t, 2> recv{};
    MPI_Reduce_scatter_block(send.data(), recv.data(), 2, MPI_UINT64_T, MPI_SUM, comm);
    std::uint64_t const recv_nodes = recv[0];
    std::uint64_t const recv_edges = recv[1];
    KASSERT(recv_edges <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()));

    profiling::MessageStatistics stats;
    for (std::size_t r = 0; r < size; ++r) {
        if (r == rank) {
            stats.receive_volume = recv_nodes + recv_edges - send[2 * r] - send[2 * r + 1];
            continue;
        }
        stats.sent_messages += (send[2 * r] > 0) + (send[2 * r + 1] > 0);
        stats.send_volume += send[2 * r] + send[2 * r + 1];
    }

    // the new range starts after the vertices of the ranks before
    std::uint64_t last = recv_nodes;
    kamping::v2::scan(kamping::v2::inplace, views::ref_single(last), comm);

    std::vector<kagen::SInt> degrees(n);
    for (std::size_t i = 0; i < n; ++i) {
        degrees[i] = degree(i);
    }
    std::vector<kagen::SInt> balanced_degrees(recv_nodes);
    kagen::Graph             balanced;
    balanced.representation = kagen::GraphRepresentation::CSR;
    balanced.vertex_range   = {last - recv_nodes, last};
    balanced.adjncy.resize(recv_edges);
    auto const tag = MessageTag::LoadBalancing;
    dstl::redistribute(degrees, balanced_degrees, comm, tag);
    dstl::redistribute(G.adjncy, balanced.adjncy, comm, tag);
    balanced.xadj.resize(recv_nodes + 1, 0);
    std::inclusive_scan(balanced_degrees.begin(), balanced_degrees.end(), balanced.xadj.begin() + 1);
    stats.report();
    return balanced;
}

} // namespace cetric::load_balancing
