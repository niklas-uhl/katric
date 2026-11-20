// Pushing a per-vertex value to the ranks that hold the vertex as a ghost.
// Adapted from KaCCv2's kacc::push_to_ghosts
// (src/kacc/graph/ghost_exchange.hpp), a candidate for dSTL. Differences:
// katric's graph instead of KaCCv2's graph concepts, one value per vertex and
// no ids on the wire (each sender's values land in a block of the receiver's
// ghosts known in advance), the messages are bucketed by destination with one
// counting sort instead of a hash map of per-rank vectors, the transport is a
// runtime choice that includes dSTL's sparse all-to-all, and the counters go to
// Caliper.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>
#include <dstl/grid_alltoallv.hpp>
#include <dstl/grid_comm.hpp>
#include <dstl/sparse_alltoall.hpp>
#include <kamping/v2/collectives/alltoallv.hpp>
#include <kamping/v2/p2p/mrecv.hpp>
#include <kamping/v2/views/auto_recv_v.hpp>
#include <kamping/v2/views/with_counts_view.hpp>
#include <kamping/v2/views/with_displs_view.hpp>
#include <kassert/kassert.hpp>

#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/message_statistics.h"
#include "katric/message_tags.h"

namespace katric {

/// Sends `value(u)` of every local vertex u once to each rank that holds u as a
/// ghost, and returns the values of this rank's ghosts, indexed by ghost index.
/// Collective; `G` must be undirected (both directions of every edge stored).
///
/// No ids travel: since `G` is undirected, rank s sends rank r exactly r's
/// ghosts owned by s (s sends u to r iff u has a neighbor on r iff u is a
/// neighbor of one of r's vertices), in ascending order (a vertex's
/// neighborhood is scanned at once, local vertices in order, and the counting
/// sort below is stable). The vertex ranges are contiguous and in rank order,
/// so r's ghosts owned by s are one block of its ghosts, ascending. So r knows
/// the size and offset of every sender's part and receives it directly into
/// place. Deduplication: remembering per rank the last vertex sent there
/// suffices.
///
/// `tag` must not be in use on `G.comm()` during the call (sparse mode).
/// Reports the message counters of the exchange (sent/received messages and
/// volume, one word per value) to the open Caliper region. Caliper regions
/// `prepare` (bucketing) and `exchange`.
template <typename Graph, typename ValueFunc>
[[nodiscard]] auto push_to_ghosts(Graph const& G,
                                  ValueFunc const& value,
                                  GhostExchangeMode mode,
                                  MessageTag tag)
    -> std::vector<std::uint64_t> {
  auto const comm = G.comm();
  auto const size = static_cast<std::size_t>(comm.size());
  int const rank = comm.rank();
  constexpr auto never = std::numeric_limits<std::size_t>::max();
  std::vector<int> send_counts(size, 0);
  std::vector<int> send_displs(size, 0);
  std::vector<std::uint64_t> send_buf;
  // what this rank receives: from each rank, the values of its ghosts owned
  // there (one block per owner)
  std::vector<int> recv_counts(size, 0);
  std::vector<int> recv_displs(size, 0);
  {
    CALI_CXX_MARK_SCOPE("prepare");
    // Append (value, destination) in one pass over the adjacency, then place
    // the values by destination with a counting sort: no per-rank containers,
    // and no second pass over the adjacency.
    std::vector<std::pair<std::uint64_t, int>> outgoing;
    std::vector<std::size_t> last_source_sent_to(size, never);
    for (auto const u : G.local_vertices()) {
      auto const u_local = G.to_local_index(u);
      for (auto const v : G.neighbors(u)) {
        int const v_rank = G.owner(v);
        if (v_rank == rank ||
            last_source_sent_to[static_cast<std::size_t>(v_rank)] == u_local) {
          continue;
        }
        outgoing.emplace_back(static_cast<std::uint64_t>(value(u)), v_rank);
        last_source_sent_to[static_cast<std::size_t>(v_rank)] = u_local;
      }
    }
    for (auto const& [message, dest] : outgoing) {
      send_counts[static_cast<std::size_t>(dest)]++;
    }
    for (std::size_t rk = 1; rk < size; ++rk) {
      send_displs[rk] = send_displs[rk - 1] + send_counts[rk - 1];
    }
    send_buf.resize(outgoing.size());
    std::vector<int> next = send_displs;
    for (auto const& [message, dest] : outgoing) {
      send_buf[static_cast<std::size_t>(
          next[static_cast<std::size_t>(dest)]++)] = message;
    }
    for (auto const ghost : G.ghosts()) {
      recv_counts[static_cast<std::size_t>(G.owner(ghost))]++;
    }
    for (std::size_t rk = 1; rk < size; ++rk) {
      recv_displs[rk] = recv_displs[rk - 1] + recv_counts[rk - 1];
    }
  }
  profiling::MessageStatistics stats;
  for (std::size_t rk = 0; rk < size; ++rk) {
    if (send_counts[rk] > 0) {
      stats.sent_messages++;
      stats.send_volume += static_cast<std::size_t>(send_counts[rk]);
    }
    if (recv_counts[rk] > 0) {
      stats.received_messages++;
      stats.receive_volume += static_cast<std::size_t>(recv_counts[rk]);
    }
  }
  std::vector<std::uint64_t> values(G.num_ghosts());
  {
    CALI_CXX_MARK_SCOPE("exchange");
    namespace views = kamping::v2::views;
    switch (mode) {
      case GhostExchangeMode::sparse: {
        std::vector<std::pair<int, std::span<std::uint64_t const>>> messages;
        for (std::size_t rk = 0; rk < size; ++rk) {
          if (send_counts[rk] > 0) {
            messages.emplace_back(
                static_cast<int>(rk),
                std::span<std::uint64_t const>(send_buf).subspan(
                    static_cast<std::size_t>(send_displs[rk]),
                    static_cast<std::size_t>(send_counts[rk])));
          }
        }
        dstl::sparse_alltoall(
            dstl::sparse_alltoall_algorithm::nbx, messages,
            [&](kamping::v2::probe_result probe) {
              auto const source = static_cast<std::size_t>(probe.source());
              KASSERT(
                  probe.count<std::uint64_t>() == recv_counts[source],
                  "rank "
                      << source
                      << " sent a part that does not match the ghosts it owns");
              kamping::v2::mrecv(
                  std::span<std::uint64_t>(values).subspan(
                      static_cast<std::size_t>(recv_displs[source]),
                      static_cast<std::size_t>(recv_counts[source])),
                  std::move(probe));
            },
            tag, comm);
        break;
      }
      case GhostExchangeMode::dense: {
        kamping::v2::alltoallv(send_buf | views::with_counts(send_counts) |
                                   views::with_displs(send_displs),
                               values | views::with_counts(recv_counts) |
                                   views::with_displs(recv_displs),
                               comm);
        break;
      }
      case GhostExchangeMode::grid: {
        // ordered by source: the parts arrive as from a flat alltoallv, i.e. in
        // the order of the blocks
        std::vector<std::uint64_t> received;
        dstl::grid_alltoallv(send_buf | views::with_counts(send_counts) |
                                 views::with_displs(send_displs),
                             received | views::auto_recv_v,
                             G.numbering().grid(),
                             dstl::layout::ordered_by_source{});
        KASSERT(received.size() == values.size(),
                "the received parts do not match the ghosts");
        values = std::move(received);
        break;
      }
    }
  }
  stats.report();
  return values;
}

}  // namespace katric
