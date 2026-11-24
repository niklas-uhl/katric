// Pushing a per-vertex value to the ranks that hold the vertex as a ghost.
// Adapted from KaCCv2's kacc::push_to_ghosts
// (src/kacc/graph/ghost_exchange.hpp), a candidate for dSTL. Differences:
// katric's graph instead of KaCCv2's graph concepts, one value per vertex and
// no ids on the wire (each sender's values land in a block of the receiver's
// ghosts known in advance), the messages are bucketed by destination with a
// counting sort instead of a hash map of per-rank vectors, and only once per
// graph (GhostPushPattern; later exchanges gather into it), the transport is a
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

/// What push_to_ghosts() sends to and receives from every rank on a graph. It
/// depends only on the undirected graph (its cut edges and ghosts), so the
/// graph builds it once (Graph::ghost_push_pattern()) and every exchange on
/// that graph reuses it: an exchange is then a gather of the values plus the
/// transfer.
struct GhostPushPattern {
  /// The local index of the vertex whose value goes into each slot of the send
  /// buffer: one block per destination rank, ascending within each block.
  std::vector<std::uint32_t> send_vertex;
  std::vector<int> send_counts;
  std::vector<int> send_displs;
  /// From each rank, the number of this rank's ghosts it owns: one block of
  /// the ghosts per owner, in rank order.
  std::vector<int> recv_counts;
  std::vector<int> recv_displs;
};

/// The pattern of push_to_ghosts() on `G` (undirected, both directions of every
/// edge stored): local vertex u goes once to each rank that holds u as a ghost.
///
/// No ids travel: since `G` is undirected, rank s sends rank r exactly r's
/// ghosts owned by s (s sends u to r iff u has a neighbor on r iff u is a
/// neighbor of one of r's vertices), in ascending order (local vertices are
/// scanned in order, and the counting sort below is stable).
/// The vertex ranges are contiguous and in rank order, so r's ghosts owned by s
/// are one block of its ghosts, ascending. So r knows the size and offset of
/// every sender's part and receives it directly into place. Deduplication:
/// remembering per rank the last vertex sent there suffices.
///
/// One pass over the adjacency appends (vertex, destination) pairs to a buffer
/// reserved for every adjacency entry (no reallocation) and counts per
/// destination; a counting sort then places the vertex indices into their
/// blocks. A second pass over the adjacency instead of the buffer takes ~1.6x
/// as long (bucketing benchmark, 2026-11-21).
template <typename Graph>
[[nodiscard]] auto build_ghost_push_pattern(Graph const& G)
    -> GhostPushPattern {
  auto const size = static_cast<std::size_t>(G.comm().size());
  [[maybe_unused]] int const rank = G.comm().rank();
  KASSERT(G.num_local_vertices() <= std::numeric_limits<std::uint32_t>::max(),
          "local indices do not fit the pattern's 32 bits");
  GhostPushPattern pattern{{},
                           std::vector<int>(size, 0),
                           std::vector<int>(size, 0),
                           std::vector<int>(size, 0),
                           std::vector<int>(size, 0)};
  // (local index of u, destination), local vertices ascending
  std::vector<std::pair<std::uint32_t, std::uint32_t>> outgoing;
  outgoing.reserve(G.num_local_edges());
  constexpr auto never = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> last_source_sent_to(size, never);
  for (auto const u : G.local_vertices()) {
    auto const u_local = G.to_local_index(u);
    for (auto const v : G.neighbors(u)) {
      if (G.is_local(v)) {
        continue;
      }
      auto const v_rank = static_cast<std::size_t>(G.owner(v));
      if (last_source_sent_to[v_rank] == u_local) {
        continue;
      }
      last_source_sent_to[v_rank] = u_local;
      outgoing.emplace_back(static_cast<std::uint32_t>(u_local),
                            static_cast<std::uint32_t>(v_rank));
      pattern.send_counts[v_rank]++;
    }
  }
  for (std::size_t rk = 1; rk < size; ++rk) {
    pattern.send_displs[rk] =
        pattern.send_displs[rk - 1] + pattern.send_counts[rk - 1];
  }
  pattern.send_vertex.resize(outgoing.size());
  std::vector<int> next = pattern.send_displs;
  for (auto const& [u_local, dest] : outgoing) {
    pattern.send_vertex[static_cast<std::size_t>(next[dest]++)] = u_local;
  }
  KASSERT(pattern.send_counts[static_cast<std::size_t>(rank)] == 0);
  auto const ghost_offsets = G.numbering().ghost_offsets();
  for (std::size_t rk = 0; rk < size; ++rk) {
    pattern.recv_counts[rk] =
        static_cast<int>(ghost_offsets[rk + 1] - ghost_offsets[rk]);
    pattern.recv_displs[rk] = static_cast<int>(ghost_offsets[rk]);
  }
  return pattern;
}

/// Sends `value(u)` of every local vertex u once to each rank that holds u as a
/// ghost, and returns the values of this rank's ghosts, indexed by ghost index.
/// Collective; `G` must be undirected (both directions of every edge stored).
/// Uses `G`'s GhostPushPattern (built on the first exchange on `G`).
///
/// `tag` must not be in use on `G.comm()` during the call (sparse mode).
/// Reports the message counters of the exchange (sent/received messages and
/// volume, one word per value) to the open Caliper region. Caliper regions
/// `prepare` (the gather of the values; on the first exchange on `G` also the
/// sub-region `pattern`, building the pattern) and `exchange`.
template <typename Graph, typename ValueFunc>
[[nodiscard]] auto push_to_ghosts(Graph& G,
                                  ValueFunc const& value,
                                  GhostExchangeMode mode,
                                  MessageTag tag)
    -> std::vector<std::uint64_t> {
  auto const comm = G.comm();
  auto const size = static_cast<std::size_t>(comm.size());
  std::vector<std::uint64_t> send_buf;
  GhostPushPattern const& pattern = [&]() -> GhostPushPattern const& {
    CALI_CXX_MARK_SCOPE("prepare");
    GhostPushPattern const& built = G.ghost_push_pattern();
    send_buf.resize(built.send_vertex.size());
    auto const& numbering = G.numbering();
    for (std::size_t slot = 0; slot < send_buf.size(); ++slot) {
      send_buf[slot] = static_cast<std::uint64_t>(
          value(numbering.local_handle(built.send_vertex[slot])));
    }
    return built;
  }();
  auto const& send_counts = pattern.send_counts;
  auto const& send_displs = pattern.send_displs;
  auto const& recv_counts = pattern.recv_counts;
  auto const& recv_displs = pattern.recv_displs;
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
