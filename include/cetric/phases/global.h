// The global phase on compact ids: each rank sends N+(v) of its senders v to the ranks of v's cut out-edges and
// intersects the received neighborhoods with its own. Messages carry global ids in global order: [v, N+(v)...].
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

#include <message-queue/buffered_queue.h>

#include "cetric/caliper.h"
#include "cetric/communication_policy.h"
#include "cetric/config.h"
#include "cetric/intersection.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/datastructures/span.h"
#include "cetric/graph/graph.h"
#include "cetric/indirect_message_queue.h"
#include "cetric/phase_counters.h"
#include "cetric/util.h"
#include "kassert/kassert.hpp"

namespace cetric::global {

/// Separates the messages in the queue's buffers.
constexpr graph::NodeId sentinel = std::numeric_limits<graph::NodeId>::max();

/// Appends a message to the queue's buffer for one rank, terminated by a sentinel.
struct Merger {
    std::size_t operator()(std::vector<graph::NodeId>& buffer, std::vector<graph::NodeId> msg, int /*tag*/) const {
        buffer.insert(buffer.end(), msg.begin(), msg.end());
        buffer.push_back(sentinel);
        return msg.size() + 1;
    }
};

/// Splits a received buffer into its messages at the sentinels.
struct Splitter {
    template <typename MessageFunc>
    void operator()(std::vector<graph::NodeId> buffer, MessageFunc&& on_message, PEID sender) const {
        auto const buffer_ptr  = std::make_shared<std::vector<graph::NodeId>>(std::move(buffer));
        auto       slice_begin = buffer_ptr->begin();
        while (slice_begin != buffer_ptr->end()) {
            auto const slice_end = std::find(slice_begin, buffer_ptr->end(), sentinel);
            on_message(
                SharedVectorSpan(
                    buffer_ptr,
                    std::distance(buffer_ptr->begin(), slice_begin),
                    std::distance(buffer_ptr->begin(), slice_end)
                ),
                sender
            );
            slice_begin = slice_end + 1;
        }
    }
};

/// Handles a received message [v, N+(v)...] (global ids, ascending): for every local vertex u in it, emits the
/// triangles (v, u, w) for w in N+(v) ∩ N+(u), merging against u's neighbors in global ids.
template <typename Graph, typename It, typename TriangleFunc>
void process_message(
    Graph const& G, Config const& conf, It begin, It end, GlobalPhaseCounters& counters, TriangleFunc& emit
) {
    auto const&         layer   = *G.layer();
    graph::NodeId const v       = *begin;
    auto const          entries = std::span<graph::NodeId const>(std::to_address(std::next(begin)), std::to_address(end));
    KASSERT(!layer.owns_global(v));
    KASSERT(std::ranges::is_sorted(entries));
    for (graph::NodeId const u_global: entries) {
        if (!layer.owns_global(u_global)) {
            continue;
        }
        graph::Handle const u = u_global - layer.first();
        // CETRIC: u may have had no out-edges and was removed by the contraction
        if (!G.is_local(u)) {
            continue;
        }
        auto const u_neighbors = G.neighbors_as_global_ids(u);
        // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
        counters.wedge_checks += u_neighbors.size();
        counters.intersection_size_global += entries.size() + u_neighbors.size();
        cetric::intersection(
            u_neighbors.begin(),
            u_neighbors.end(),
            entries.begin(),
            entries.end(),
            [&](graph::NodeId w) {
                counters.type3_triangles++;
                emit(graph::Triangle<graph::NodeId>{v, u_global, w});
            },
            std::less<>{},
            conf
        );
    }
}

/// The message of `v`: [v, N+(v)...] in global ids, ascending.
template <typename Graph>
std::vector<graph::NodeId> make_message(Graph const& G, graph::Handle v) {
    auto const                 neighbors = G.neighbors_as_global_ids(v);
    std::vector<graph::NodeId> message;
    message.reserve(1 + neighbors.size());
    message.push_back(G.global_id(v));
    message.insert(message.end(), neighbors.begin(), neighbors.end());
    return message;
}

/// The global phase: sends the message of every vertex v of `senders` once to each rank holding the head of a cut
/// out-edge of v (the ghosts of a neighborhood are in global order, so the heads on one rank are consecutive), and handles the
/// received messages. `threshold` is the message queue's buffer size.
template <class CommunicationPolicy, typename Graph, typename Senders, typename TriangleFunc>
void run(Graph const& G, Config const& conf, std::size_t threshold, Senders const& senders, TriangleFunc emit) {
    auto base_queue = message_queue::make_buffered_queue<graph::NodeId>(Merger{}, Splitter{});
    auto queue      = [&]() {
        if constexpr (std::is_same_v<CommunicationPolicy, GridPolicy>) {
            return IndirectMessageQueueAdaptor(base_queue);
        } else {
            return DirectMessageQueueAdaptor(base_queue);
        }
    }();
    queue.set_threshold(threshold);
    GlobalPhaseCounters counters;
    // the rank the message of a sender went to last, by its local index
    std::vector<PEID> last_proc(G.local_node_count(), -1);
    // Time spent building and posting messages and handling received ones (the intersections), accumulated per call
    // because a Caliper region per message would cost more than the work it measures. The rest of the phase's
    // exclusive time is polling; the region `termination` is the wait for the other ranks.
    using clock = std::chrono::steady_clock;
    clock::duration send_time{};
    clock::duration handle_time{};
    auto            on_message = [&](SharedVectorSpan<graph::NodeId> span, PEID /*sender*/) {
        auto const start = clock::now();
        process_message(G, conf, span.begin(), span.end(), counters, emit);
        handle_time += clock::now() - start;
    };
    for (graph::Handle const v: senders) {
        counters.interface_nodes++;
        if (conf.pseudo2core && G.neighbors(v).size() < 2) {
            counters.skipped_nodes++;
        } else {
            auto const start = clock::now();
            auto const v_idx = G.to_local_idx(v);
            for (graph::Handle const u: G.neighbors(v)) {
                if (!G.is_owned(u) && last_proc[v_idx] != G.owner(u)) {
                    queue.post_message(make_message(G, v), G.owner(u));
                    last_proc[v_idx] = G.owner(u);
                }
            }
            send_time += clock::now() - start;
        }
        queue.poll(on_message);
    }
    {
        CALI_CXX_MARK_SCOPE("termination");
        queue.terminate(on_message);
    }
    counters.report(queue.stats());
    profiling::counter_add("send_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(send_time).count());
    profiling::counter_add("handle_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(handle_time).count());
}

} // namespace cetric::global
