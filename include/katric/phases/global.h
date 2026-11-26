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
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <briefkasten/aggregators.hpp>
#include <briefkasten/buffered_queue.hpp>
#include <briefkasten/grid_indirection.hpp>
#include <briefkasten/indirection.hpp>
#include <briefkasten/queue_builder.hpp>
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

/// The legacy message queue's buffer threshold (`--threshold`,
/// `--threshold-scale`).
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
/// reused) with `post(message, destination)` and counts the entries (incl. the
/// sender id) not sent against sending N+(v) whole, and the messages not sent
/// at all.
template <typename Post>
void post_trimmed_messages(graph::Graph const& G,
                           graph::VertexHandle v,
                           Post&& post,
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
      post(std::span<graph::VertexId const>(message), destination);
    } else {
      trimmed_entries += neighbors.size() + 1;
      dropped_messages++;
    }
    run_begin = run_end;
  }
}

/// BriefKAsten's configuration from `--buffer-size`, `--request-slots`,
/// `--flow-control`. Everything else is BriefKAsten's default (a capacity per
/// destination buffer, no global threshold, idle flush of every buffer on
/// entering `terminate`).
inline auto briefkasten_config(Config const& conf) -> briefkasten::Config {
  briefkasten::Config config;
  config.local_threshold_bytes = conf.buffer_size;
  config.num_request_slots = conf.request_slots;
  config.bounded_pool = !conf.legacy_pool;
  // messages larger than a buffer occur (a vertex's out-neighborhood: RMAT ~25 KB at 768 cores); BriefKAsten
  // makes them opt-in, since the large-message receive costs every poll an extra probe
  config.allow_large_messages = true;
  config.drop_shells = conf.drop_shells;
  config.num_credit_buffers =
      conf.flow_control ? std::nullopt : std::optional<std::size_t>(0);
  return config;
}

/// A BriefKAsten queue of global ids with `Receiver`. A buffer holds records
/// [receiver, length, message...] (the receiver for the grid's proxies), and a
/// received message is a span into the buffer.
template <template <typename> typename Receiver>
auto make_briefkasten_queue(MPI_Comm comm, briefkasten::Config const& config) {
  return briefkasten::BufferedMessageQueueBuilder<graph::VertexId>(comm, config)
      .with_merger(briefkasten::aggregation::EnvelopeSerializationMerger{})
      .with_splitter(
          briefkasten::aggregation::EnvelopeSerializationSplitter<
              graph::VertexId>{})
      .template with_receiver<Receiver>()
      .build();
}

/// Calls `f` with the global phase's BriefKAsten queue on `G`'s communicator:
/// `--receiver`, direct or through BriefKAsten's grid
/// (`--communication-policy`). Synchronous mode: handling a message creates no
/// new work (a proxy's forwarding is counted as pending by the queue), so an
/// arrival need not abort termination; `terminate` returns once all ranks are
/// done, draining every buffer (relays included) before each counting round.
template <typename F>
void with_briefkasten_queue(graph::Graph const& G, Config const& conf, F&& f) {
  MPI_Comm const comm = G.comm().mpi_handle();
  auto const config = briefkasten_config(conf);
  auto const with = [&](auto queue) -> void {
    if (conf.communication_policy == CommunicationPolicy::grid) {
      briefkasten::IndirectionAdapter grid_queue{
          std::move(queue), briefkasten::GridIndirectionScheme{comm}};
      grid_queue.synchronous_mode();
      f(grid_queue, std::as_const(grid_queue.queue()));
    } else {
      queue.synchronous_mode();
      f(queue, std::as_const(queue));
    }
  };
  switch (conf.receiver) {
    case Receiver::persistent:
      with(make_briefkasten_queue<briefkasten::PersistentReceiver>(comm,
                                                                    config));
      break;
    case Receiver::probe:
      with(make_briefkasten_queue<briefkasten::ProbeReceiver>(comm, config));
      break;
    case Receiver::posted:
      with(make_briefkasten_queue<briefkasten::PrepostingReceiver>(comm,
                                                                    config));
      break;
  }
}

/// Reports the counters of the BriefKAsten queue (the one under the grid
/// adapter) to the open Caliper region. Messages are buffers (each hop),
/// send volume the elements sent (each hop, with the 2-word header of every
/// message), receive volume the elements of the messages delivered here (with
/// their headers; what a proxy relays is not included).
template <typename Queue>
void report_briefkasten_counters(Queue const& queue,
                                 std::size_t receive_volume) {
  auto const counts = queue.message_counts();
  profiling::MessageStatistics{.sent_messages = counts.send,
                               .received_messages = counts.receive,
                               .send_volume = queue.num_elements_flushed(),
                               .receive_volume = receive_volume}
      .report();
  auto const add = [](char const* name, std::size_t value) {
    profiling::counter_add(name, static_cast<std::int64_t>(value));
  };
  add("overflows", queue.num_overflows());
  add("buffer_stalls", queue.num_buffer_stalls());
  add("send_capacity_waits", queue.num_send_capacity_waits());
  add("parked_for_credit", queue.num_parked_for_credit());
  add("parked_for_capacity", queue.num_parked_for_capacity());
  add("oversize_passes", queue.num_oversize_passes());
  add("large_messages", queue.num_large_messages());
  add("large_message_elements", queue.num_large_message_elements());
  add("forced_flushes", queue.num_forced_flushes());
  add("idle_flushes", queue.num_idle_flushes());
  add("idle_flush_elements", queue.num_idle_flush_elements());
  add("polls", queue.num_polls());
  add("unproductive_polls", queue.num_unproductive_polls());
  add("termination_rounds", queue.num_termination_rounds());
  // the most buffers waiting for a send slot at once (use the max over ranks)
  add("peak_send_backlog", queue.peak_send_backlog());
  // the proxy's reserve (buffers beyond the pool cap): most at once, the overdraft it allocated, and own posts
  // that waited for it to drain (use the max over ranks)
  add("peak_reserve_buffers", queue.peak_reserve_buffers());
  add("redirect_overdraft", queue.redirect_overdraft());
  add("reserve_waits", queue.num_reserve_waits());
  add("dropped_shells", queue.num_dropped_shells());
}

/// The global phase: sends the message of every local vertex v with cut edges
/// once to each rank holding the head of a cut out-edge of v, and handles the
/// received messages, through the legacy message queue (`threshold` is its
/// buffer size) or BriefKAsten (`--message-queue`).
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
  GlobalPhaseCounters counters;
  // Time spent building and posting messages and handling received ones (the
  // intersections), accumulated per call because a Caliper region per message
  // would cost more than the work it measures. `send_time` is the loop outside
  // `poll` (building and posting, skipping vertices without cut edges; without
  // messages handled while posting, which BriefKAsten does when it waits for a
  // buffer or a send slot); `poll_time` is the time in `poll` outside the
  // handler (the queue's own work); one clock reading per boundary, two per
  // sender. `handle_time` includes messages handled during termination, which
  // is also counted separately, so the region `termination` minus that part is
  // the wait for the other ranks.
  using clock = std::chrono::steady_clock;
  clock::duration send_time{};
  clock::duration handle_time{};
  clock::duration poll_time{};
  clock::duration termination_handle_time{};
  auto on_message = [&](std::span<graph::VertexId const> message) {
    auto const start = clock::now();
    process_message(G, conf, message, counters, triangles);
    handle_time += clock::now() - start;
  };
  // message trimming: the message being built, and what was not sent
  std::vector<graph::VertexId> trimmed_message;
  std::size_t trimmed_entries = 0;
  std::size_t dropped_messages = 0;
  // The send loop and the termination, with `post(message, destination)`,
  // `poll()` and `terminate()` (returns once all ranks are done) of the queue.
  auto const exchange = [&](auto&& post, auto&& poll, auto&& terminate) {
    auto send_start = clock::now();
    auto handle_before_send = handle_time;
    for (graph::VertexHandle const v : G.local_vertices()) {
      if (!G.has_cut_edges(v)) {
        continue;
      }
      counters.interface_nodes++;
      auto const neighbors = G.global_neighbors(v);
      if (conf.pseudo2core && neighbors.size() < 2) {
        counters.skipped_nodes++;
      } else if (conf.message_trimming) {
        post_trimmed_messages(G, v, post, trimmed_message, trimmed_entries,
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
          post(std::span<graph::VertexId const>(message), destination);
        }
      }
      auto const poll_start = clock::now();
      auto const handle_before = handle_time;
      send_time += (poll_start - send_start) - (handle_time - handle_before_send);
      poll();
      send_start = clock::now();
      handle_before_send = handle_time;
      poll_time += (send_start - poll_start) - (handle_time - handle_before);
    }
    send_time += (clock::now() - send_start) - (handle_time - handle_before_send);
    CALI_CXX_MARK_SCOPE("termination");
    auto const handle_before = handle_time;
    terminate();
    termination_handle_time = handle_time - handle_before;
  };
  if (conf.message_queue == MessageQueue::legacy) {
    // the legacy message queue communicates on MPI_COMM_WORLD, whatever the
    // graph's communicator is
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(G.comm().mpi_handle(), MPI_COMM_WORLD, &comparison);
    if (comparison != MPI_IDENT && comparison != MPI_CONGRUENT) {
      throw std::invalid_argument(
          "the legacy message queue needs MPI_COMM_WORLD");
    }
    auto base_queue = message_queue::make_buffered_queue<graph::VertexId>(
        Merger{}, Splitter{});
    auto const on_legacy_message =
        [&](std::span<graph::VertexId const> message, int /*sender*/) {
          on_message(message);
        };
    // `--communication-policy`: direct or through katric's 2D grid adaptor
    auto const run = [&](auto& queue) -> void {
      queue.set_threshold(threshold);
      exchange(
          [&](std::span<graph::VertexId const> message, int destination) {
            queue.post_message(message, destination);
          },
          [&] { queue.poll(on_legacy_message); },
          [&] { queue.terminate(on_legacy_message); });
      auto const& stats = queue.stats();
      profiling::MessageStatistics{
          .sent_messages = stats.sent_messages,
          .received_messages = stats.received_messages,
          .send_volume = stats.send_volume,
          .receive_volume = stats.receive_volume}
          .report();
    };
    if (conf.communication_policy == CommunicationPolicy::grid) {
      IndirectMessageQueueAdaptor queue(base_queue);
      run(queue);
    } else {
      run(base_queue);
    }
  } else {
    std::size_t receive_volume = 0;
    auto const on_briefkasten_message = [&](auto envelope) {
      receive_volume += envelope.message.size() + 2;
      on_message(envelope.message);
    };
    with_briefkasten_queue(G, conf, [&](auto& queue, auto const& base_queue) {
      exchange(
          [&](std::span<graph::VertexId const> message, int destination) {
            // a message is taken as an rvalue range (a copy of the span)
            queue.post_message_blocking(
                std::span<graph::VertexId const>(message), destination,
                on_briefkasten_message);
          },
          // `--poll-skip`: polls on every n-th call (every call for 1)
          [&] { queue.poll_throttled(on_briefkasten_message, conf.poll_skip); },
          [&] {
            // synchronous mode: never aborted by an arrival
            [[maybe_unused]] bool const terminated =
                queue.terminate(on_briefkasten_message);
            KASSERT(terminated);
          });
      report_briefkasten_counters(base_queue, receive_volume);
    });
  }
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
