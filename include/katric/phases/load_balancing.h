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

// Primary load balancing: repartitions the graph into contiguous vertex ranges of about equal cost
// (`--primary-cost-function`). Region `primary_load_balancing`.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <dstl/redistribute.hpp>
#include <kamping/v2/collectives/allreduce.hpp>
#include <kamping/v2/collectives/bcast.hpp>
#include <kamping/v2/collectives/scan.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/ops.hpp>
#include <kamping/v2/sentinels.hpp>
#include <kamping/v2/views/ref_single_view.hpp>
#include <kagen.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/ghost_exchange.h"
#include "katric/graph/graph.h"
#include "katric/message_tags.h"
#include "katric/phases/preprocessing.h"
#include "katric/message_statistics.h"

namespace katric {

/// Moves the vertices of `G` (KaGen's CSR, with their adjacency lists) so that each rank gets a contiguous range of
/// about C/p of the total cost C: vertex v goes to the rank whose share [r C/p, (r+1) C/p) contains the cost before v
/// (`cost[i]` is the cost of vertex first + i). The vertices stay in global-id order, so the move is a
/// dstl::redistribute of the degrees and of the heads. Region `redistribution`.
inline auto redistribute(kagen::Graph&& G, std::span<std::size_t const> cost, kamping::v2::comm_view comm) -> kagen::Graph {
    CALI_CXX_MARK_SCOPE("redistribution");
    auto const        size = static_cast<std::size_t>(comm.size());
    auto const        rank = static_cast<std::size_t>(comm.rank());
    std::size_t const n    = G.vertex_range.second - G.vertex_range.first;
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
    MPI_Reduce_scatter_block(send.data(), recv.data(), 2, MPI_UINT64_T, MPI_SUM, kamping::v2::handle(comm));
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

/// The orientation by (degree, global id) of the undirected graph `G`, decided
/// once for the cost functions: whether each edge (in adjacency order) is an
/// out-edge, and each local vertex's out-degree. Needs the ghosts' degrees.
struct CostOrientation {
  std::vector<bool> is_out_edge;
  std::vector<std::uint64_t> local_out_degree;
};

inline auto cost_orientation(graph::Graph const& G) -> CostOrientation {
  using graph::VertexHandle;
  CALI_CXX_MARK_SCOPE("cost_function_evaluation");
  auto const& numbering = G.numbering();
  CostOrientation o{std::vector<bool>(G.num_local_edges()),
                    std::vector<std::uint64_t>(G.num_local_vertices(), 0)};
  auto const key = degree_key(G);
  std::size_t edge = 0;
  for (VertexHandle const v : G.local_vertices()) {
    auto const key_v = key(v);
    std::uint64_t out_degree_v = 0;
    for (VertexHandle const u : G.local_neighbors(v)) {
      bool const out = key_v < key(u);
      o.is_out_edge[edge++] = out;
      if (out) {
        ++out_degree_v;
      }
    }
    o.local_out_degree[numbering.to_local_index(v)] = out_degree_v;
  }
  return o;
}

/// The primary load balancing's cost of every local vertex
/// (`--primary-cost-function` DPD or IDPD, PATRIC's cost functions): DPD = Σ
/// over out-neighbors u of d+(v) + d+(u) (the tail-compute work), IDPD = the
/// same over in-neighbors (push), both with the orientation `o`.
/// `ghost_out_degree(ghost)` gives a ghost's out-degree: exchanged, or bounded
/// by its degree (for IDPD that is IDD, which needs only the degree exchange
/// and is an upper bound of IDPD per vertex). Region
/// `cost_function_evaluation`.
template <typename GhostOutDegree>
auto vertex_costs(graph::Graph const& G,
                  Config const& conf,
                  CostOrientation const& o,
                  GhostOutDegree const& ghost_out_degree)
    -> std::vector<std::size_t> {
  using graph::VertexHandle;
  CALI_CXX_MARK_SCOPE("cost_function_evaluation");
  auto const& numbering = G.numbering();
  auto const out_degree = [&](VertexHandle vertex) -> std::uint64_t {
    return numbering.is_local(vertex)
               ? o.local_out_degree[numbering.to_local_index(vertex)]
               : ghost_out_degree(vertex);
  };
  bool const over_out_edges = conf.primary_cost_function == CostFunction::DPD;
  std::vector<std::size_t> cost(G.num_local_vertices(), 0);
  std::size_t edge = 0;
  for (VertexHandle const v : G.local_vertices()) {
    std::size_t& cost_v = cost[numbering.to_local_index(v)];
    for (VertexHandle const u : G.local_neighbors(v)) {
      if (o.is_out_edge[edge++] == over_out_edges) {
        cost_v += out_degree(v) + out_degree(u);
      }
    }
  }
  return cost;
}

/// Whether the cost's max/avg over the ranks (of each rank's sum) exceeds
/// `threshold`; collective (two allreduces). Reports each rank's sum as the
/// counter `counter` (so its max/avg is in the counters).
inline auto exceeds_threshold(std::span<std::size_t const> cost,
                              double threshold,
                              kamping::v2::comm_view comm,
                              char const* counter) -> bool {
  namespace views = kamping::v2::views;
  std::uint64_t const local =
      std::reduce(cost.begin(), cost.end(), std::uint64_t{0});
  std::uint64_t max = 0;
  std::uint64_t total = 0;
  kamping::v2::allreduce(std::views::single(local), views::ref_single(max),
                         kamping::v2::ops::max<>{}, comm);
  kamping::v2::allreduce(std::views::single(local), views::ref_single(total),
                         comm);
  double const avg =
      static_cast<double>(total) / static_cast<double>(comm.size());
  profiling::counter_add(counter, static_cast<std::int64_t>(local));
  return total > 0 && static_cast<double>(max) > threshold * avg;
}

/// Whether the primary load balancing should redistribute `G`: the cost's
/// max/avg over the ranks exceeds `--rebalance-threshold`. N is the vertex
/// count. DPD and IDPD need the ghosts' degrees (exchanged here; kept in `G`,
/// so preprocessing does not exchange them again) and their out-degrees; with
/// `--rebalance-precheck degree-bound`, the cost is first computed with the
/// out-degrees bounded by the degrees (counter `rank_cost_bound`), and if that
/// does not exceed the threshold, `G` is kept without the out-degree exchange.
/// Returns the cost if it does, nothing otherwise. Counters `rank_cost` (if
/// computed) and `rebalanced` (1 on every rank if so). Regions
/// `cost_function_communication` (the exchanges of the ghosts' degrees and
/// out-degrees) and `cost_function_evaluation` (the orientation and the costs,
/// entered several times).
inline auto costs_if_imbalanced(graph::Graph& G, Config const& conf)
    -> std::optional<std::vector<std::size_t>> {
  using graph::VertexHandle;
  auto const comm = G.comm();
  auto const threshold = conf.rebalance_threshold;
  auto const decide = [&](std::vector<std::size_t> cost)
      -> std::optional<std::vector<std::size_t>> {
    bool const exceeds = exceeds_threshold(cost, threshold, comm, "rank_cost");
    profiling::counter_add("rebalanced", exceeds ? 1 : 0);
    return exceeds ? std::optional(std::move(cost)) : std::nullopt;
  };
  if (conf.primary_cost_function == CostFunction::N) {
    return decide(std::vector<std::size_t>(G.num_local_vertices(), 1));
  }
  KASSERT(conf.primary_cost_function == CostFunction::DPD ||
          conf.primary_cost_function == CostFunction::IDPD);
  auto const& numbering = G.numbering();
  {
    CALI_CXX_MARK_SCOPE("cost_function_communication");
    G.exchange_ghost_degrees(conf.degree_exchange, MessageTag::CostFunction);
  }
  auto const orientation = cost_orientation(G);
  if (conf.rebalance_precheck == RebalancePrecheck::degree_bound) {
    auto const bound = vertex_costs(G, conf, orientation,
                                    [&](VertexHandle ghost) -> std::uint64_t {
                                      return G.undirected_degree(ghost);
                                    });
    if (!exceeds_threshold(bound, threshold, comm, "rank_cost_bound")) {
      profiling::counter_add("rebalanced", 0);
      return std::nullopt;
    }
  }
  std::vector<std::uint64_t> ghost_out_degree;
  {
    CALI_CXX_MARK_SCOPE("cost_function_communication");
    ghost_out_degree = push_to_ghosts(
        G,
        [&](VertexHandle v) {
          return orientation.local_out_degree[numbering.to_local_index(v)];
        },
        conf.degree_exchange, MessageTag::CostFunctionOutDegree);
  }
  return decide(vertex_costs(
      G, conf, orientation, [&](VertexHandle ghost) -> std::uint64_t {
        return ghost_out_degree[numbering.to_ghost_index(ghost)];
      }));
}

/// Repartitions `G` into contiguous vertex ranges of equal cost
/// (`--primary-cost-function`; with `none`, returns `G`), if the cost's max/avg
/// over the ranks exceeds `--rebalance-threshold` (costs_if_imbalanced());
/// otherwise returns `G`, with the degrees and the ghost push pattern the cost
/// function computed, so that preprocessing does not exchange the degrees
/// again. Consumes `G`: it is freed before the vertices move.
inline auto primary_load_balancing(graph::Graph&& G, Config const& conf)
    -> graph::Graph {
  CALI_CXX_MARK_SCOPE("primary_load_balancing");
  if (conf.primary_cost_function == CostFunction::none) {
    return std::move(G);
  }
  auto const comm = G.comm();
  auto const cost = costs_if_imbalanced(G, conf);
  if (!cost) {
    return std::move(G);
  }
  return graph::Graph::from_kagen(
      redistribute(graph::to_kagen_graph(std::move(G)), *cost, comm), comm);
}

}  // namespace katric
