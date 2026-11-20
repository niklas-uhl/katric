// The global phase on compact ids: each rank sends N+(v) of its senders v to
// the ranks of v's cut out-edges and intersects the received neighborhoods with
// its own. Messages carry global ids in global order: [v, N+(v)...].
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <message-queue/buffered_queue.h>
#include <mpi.h>

#include "kassert/kassert.hpp"
#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/graph/graph.h"
#include "katric/indirect_message_queue.h"
#include "katric/intersection.h"
#include "katric/message_statistics.h"
#include "katric/triangle_counter.h"

namespace katric {

/// The message queue's buffer threshold (`--threshold`, `--threshold-scale`).
/// Taken from the load-balanced graph before the orientation (`local-edges`
/// counts the undirected edges).
inline auto get_threshold(graph::Graph const& G, Config const& conf)
    -> std::size_t {
  switch (conf.threshold) {
    case Threshold::local_nodes:
      return conf.threshold_scale * G.num_local_vertices();
    case Threshold::local_edges:
      return conf.threshold_scale * G.num_local_edges();
    case Threshold::none:
      return std::numeric_limits<std::size_t>::max();
  }
  return 0;
}

/// Work counters of the global phase.
struct GlobalPhaseCounters {
  std::int64_t skipped_nodes = 0;
  std::int64_t interface_nodes = 0;
  std::int64_t wedge_checks = 0;
  std::int64_t intersection_size_global = 0;

  /// Adds the counts to the open Caliper region.
  void report() const {
    profiling::counter_add("skipped_nodes", skipped_nodes);
    profiling::counter_add("interface_nodes", interface_nodes);
    profiling::counter_add("wedge_checks", wedge_checks);
    profiling::counter_add("intersection_size_global",
                           intersection_size_global);
  }
};


/// Separates the messages in the queue's buffers.
constexpr graph::VertexId message_end =
    std::numeric_limits<graph::VertexId>::max();

/// Appends a message (any range of global ids, e.g. a view of a message sent
/// to several ranks) to the queue's buffer for one rank, terminated by
/// `message_end`.
struct Merger {
  template <std::ranges::sized_range Message>
  auto operator()(std::vector<graph::VertexId>& buffer,
                  Message const& message,
                  int /*tag*/) const -> std::size_t {
    buffer.insert(buffer.end(), std::ranges::begin(message),
                  std::ranges::end(message));
    buffer.push_back(message_end);
    return std::ranges::size(message) + 1;
  }
};

/// Splits a received buffer into its messages at `message_end` and hands each
/// to `on_message` as a view into the buffer, valid during the call (the
/// handler runs synchronously).
struct Splitter {
  template <typename MessageFunc>
  void operator()(std::vector<graph::VertexId> const& buffer,
                  MessageFunc&& on_message,
                  int sender) const {
    auto slice_begin = buffer.begin();
    while (slice_begin != buffer.end()) {
      auto const slice_end = std::find(slice_begin, buffer.end(), message_end);
      on_message(std::span<graph::VertexId const>(slice_begin, slice_end),
                 sender);
      slice_begin = slice_end + 1;
    }
  }
};

/// Handles a received message [v, N+(v)...] (global ids, ascending; with
/// contraction only v's cut out-edges): for every local vertex u in it, records
/// the triangles (v, u, w) for w in N+(v) ∩ N+(u).
inline void process_message(graph::Graph const& G,
                            Config const& conf,
                            std::span<graph::VertexId const> message,
                            GlobalPhaseCounters& counters,
                            TriangleCounter& triangles) {
  graph::VertexId const v = message.front();
  auto const entries = message.subspan(1);
  KASSERT(!G.is_local_id(v));
  KASSERT(std::ranges::is_sorted(entries));
  for (graph::VertexId const u_global : entries) {
    if (!G.is_local_id(u_global)) {
      continue;
    }
    auto const u_neighbors =
        G.global_neighbors(G.handle(u_global));
    // nothing to intersect (CETRIC: u has no cut edges)
    if (u_neighbors.empty()) {
      continue;
    }
    // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
    counters.wedge_checks += u_neighbors.size();
    counters.intersection_size_global += entries.size() + u_neighbors.size();
    intersection(
        u_neighbors, entries,
        [&](graph::VertexId w) { triangles.record(Triangle{v, u_global, w}); },
        conf);
  }
}

/// The message of `v`: [v, N+(v)...] in global ids, ascending (with
/// contraction only v's cut out-edges).
inline auto make_message(graph::Graph const& G, graph::VertexHandle v)
    -> std::vector<graph::VertexId> {
  auto const neighbors = G.global_neighbors(v);
  std::vector<graph::VertexId> message;
  message.reserve(1 + neighbors.size());
  message.push_back(G.global_id(v));
  message.insert(message.end(), neighbors.begin(), neighbors.end());
  return message;
}

/// Message trimming (`--message-trimming`): a triangle (v, u, w) found at rank
/// r has u on r and w after u in orientation order (w in N+(u)), so r needs
/// only the entries of N+(v) from its first receiver u_r (in orientation order,
/// key (undirected degree, global id)) on; with no entry after u_r, it needs no
/// message. Posts the message for each destination of `v` (built in `message`,
/// reused) and counts the entries (incl. the sender id) not sent against
/// sending N+(v) whole, and the messages not sent at all.
template <typename Queue>
void post_trimmed_messages(graph::Graph const& G,
                           graph::VertexHandle v,
                           Queue& queue,
                           std::vector<graph::VertexId>& message,
                           std::size_t& trimmed_entries,
                           std::size_t& dropped_messages) {
  auto const neighbors = G.global_neighbors(v);
  auto const degrees = G.global_neighbor_degrees(v);
  auto const key = [&](std::size_t i) {
    return std::pair(degrees[i], neighbors[i]);
  };
  std::size_t run_begin = 0;
  while (run_begin < neighbors.size()) {
    if (G.is_local_id(neighbors[run_begin])) {
      ++run_begin;
      continue;
    }
    // the heads of one destination are consecutive (see global_phase)
    int const destination = G.owner_of_id(neighbors[run_begin]);
    graph::VertexId const destination_end = G.range_of(destination).second;
    auto first = key(run_begin);
    std::size_t run_end = run_begin + 1;
    for (; run_end < neighbors.size() && neighbors[run_end] < destination_end;
         ++run_end) {
      first = std::min(first, key(run_end));
    }
    message.clear();
    message.push_back(G.global_id(v));
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
      if (key(i) >= first) {
        message.push_back(neighbors[i]);
      }
    }
    // u_r itself is in the message; anything more comes after it
    if (message.size() > 2) {
      trimmed_entries += neighbors.size() + 1 - message.size();
      queue.post_message(std::span<graph::VertexId const>(message),
                         destination);
    } else {
      trimmed_entries += neighbors.size() + 1;
      dropped_messages++;
    }
    run_begin = run_end;
  }
}

/// The global phase: sends the message of every local vertex v with cut edges
/// once to each rank holding the head of a cut out-edge of v, and handles the
/// received messages. `threshold` is the message queue's buffer size.
///
/// A neighborhood is ascending and the vertex ranges are contiguous in rank
/// order, so the heads on one rank form a run: an entry beyond the current
/// owner's range starts the next run, whose owner is found by a binary search
/// over the partition table from the current owner on. Per neighborhood: a
/// comparison per entry and a search per destination.
inline void global_phase(graph::Graph const& G,
                         Config const& conf,
                         std::size_t threshold,
                         TriangleCounter& triangles) {
  CALI_CXX_MARK_SCOPE("global_phase");
  auto const triangles_before = triangles.count();
  // the legacy message queue communicates on MPI_COMM_WORLD, whatever the
  // graph's communicator is (until BriefKAsten replaces it)
  int comparison = MPI_UNEQUAL;
  MPI_Comm_compare(G.comm().mpi_handle(), MPI_COMM_WORLD, &comparison);
  if (comparison != MPI_IDENT && comparison != MPI_CONGRUENT) {
    throw std::invalid_argument(
        "the global phase needs MPI_COMM_WORLD (legacy message queue)");
  }
  auto base_queue =
      message_queue::make_buffered_queue<graph::VertexId>(Merger{}, Splitter{});
  // `--communication-policy`: calls `f` with the queue, direct or through the
  // 2D grid adaptor
  auto const with_queue = [&](auto&& f) -> void {
    if (conf.communication_policy == CommunicationPolicy::grid) {
      IndirectMessageQueueAdaptor queue(base_queue);
      f(queue);
    } else {
      f(base_queue);
    }
  };
  GlobalPhaseCounters counters;
  // Time spent building and posting messages and handling received ones (the
  // intersections), accumulated per call because a Caliper region per message
  // would cost more than the work it measures. `send_time` is the loop outside
  // `poll` (building and posting, skipping vertices without cut edges);
  // `poll_time` is the time in `poll` outside the handler (the queue's own
  // work); one clock reading per boundary, two per sender. `handle_time`
  // includes messages handled during termination, which is also counted
  // separately, so the region `termination` minus that part is the wait for
  // the other ranks.
  using clock = std::chrono::steady_clock;
  clock::duration send_time{};
  clock::duration handle_time{};
  clock::duration poll_time{};
  clock::duration termination_handle_time{};
  auto on_message = [&](std::span<graph::VertexId const> message,
                        int /*sender*/) {
    auto const start = clock::now();
    process_message(G, conf, message, counters, triangles);
    handle_time += clock::now() - start;
  };
  // message trimming: the message being built, and what was not sent
  std::vector<graph::VertexId> trimmed_message;
  std::size_t trimmed_entries = 0;
  std::size_t dropped_messages = 0;
  with_queue([&](auto& queue) -> void {
    queue.set_threshold(threshold);
    auto send_start = clock::now();
    for (graph::VertexHandle const v : G.local_vertices()) {
      if (!G.has_cut_edges(v)) {
        continue;
      }
      counters.interface_nodes++;
      auto const neighbors = G.global_neighbors(v);
      if (conf.pseudo2core && neighbors.size() < 2) {
        counters.skipped_nodes++;
      } else if (conf.message_trimming) {
        post_trimmed_messages(G, v, queue, trimmed_message, trimmed_entries,
                              dropped_messages);
      } else {
        auto const message = make_message(G, v);
        // Once to each rank owning a head. The heads are ascending and the
        // ranks' ranges contiguous, so the heads of one rank are consecutive:
        // a head before the end of the last destination's range belongs to it
        // (legacy's last_proc, without looking up every head's owner).
        graph::VertexId destination_end = 0;
        for (graph::VertexId const w : neighbors) {
          if (G.is_local_id(w) || w < destination_end) {
            continue;
          }
          int const destination = G.owner_of_id(w);
          destination_end = G.range_of(destination).second;
          queue.post_message(std::span<graph::VertexId const>(message),
                             destination);
        }
      }
      auto const poll_start = clock::now();
      send_time += poll_start - send_start;
      auto const handle_before = handle_time;
      queue.poll(on_message);
      send_start = clock::now();
      poll_time += (send_start - poll_start) - (handle_time - handle_before);
    }
    send_time += clock::now() - send_start;
    {
      CALI_CXX_MARK_SCOPE("termination");
      auto const handle_before = handle_time;
      queue.terminate(on_message);
      termination_handle_time = handle_time - handle_before;
    }
    auto const& stats = queue.stats();
    profiling::MessageStatistics{
        .sent_messages = stats.sent_messages,
        .received_messages = stats.received_messages,
        .send_volume = stats.send_volume,
        .receive_volume = stats.receive_volume}
        .report();
  });
  counters.report();
  if (conf.message_trimming) {
    profiling::counter_add("trimmed_entries",
                           static_cast<std::int64_t>(trimmed_entries));
    profiling::counter_add("dropped_messages",
                           static_cast<std::int64_t>(dropped_messages));
  }
  auto const ns = [](clock::duration d) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
  };
  profiling::counter_add("send_ns", ns(send_time));
  profiling::counter_add("handle_ns", ns(handle_time));
  profiling::counter_add("poll_ns", ns(poll_time));
  profiling::counter_add("termination_handle_ns", ns(termination_handle_time));
  profiling::counter_add(
      "type3_triangles",
      static_cast<std::int64_t>(triangles.count() - triangles_before));
}

/// After the local phase: the neighborhoods as global ids for the global phase
/// (region `global_ids`); with contraction only the cut edges (region
/// `contraction`, counters: the vertices with cut edges (compact_nodes) and
/// their cut out-edges (compact_edges)).
inline void global_ids(graph::Graph& G, Config const& conf) {
  if (!conf.contraction) {
    CALI_CXX_MARK_SCOPE("global_ids");
    G.to_global_ids(false, conf.message_trimming);
    return;
  }
  CALI_CXX_MARK_SCOPE("contraction");
  G.to_global_ids(true, conf.message_trimming);
  profiling::counter_add(
      "compact_nodes",
      static_cast<std::int64_t>(std::ranges::count_if(
          G.local_vertices(),
          [&](graph::VertexHandle v) -> bool { return G.has_cut_edges(v); })));
  profiling::counter_add("compact_edges",
                         static_cast<std::int64_t>(G.num_global_neighbors()));
}


}  // namespace katric
