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
/// triangles (v, u, w) for w in N+(v) ∩ N+(u).
template <typename It, typename TriangleFunc>
void process_message(
    graph::GlobalIdGraph const& G, Config const& conf, It begin, It end, GlobalPhaseCounters& counters, TriangleFunc& emit
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
        // CETRIC: u may have had no cut edges and was removed by the contraction
        if (!G.is_kept(u)) {
            continue;
        }
        auto const u_neighbors = G.neighbors(u);
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
inline std::vector<graph::NodeId> make_message(graph::GlobalIdGraph const& G, graph::Handle v) {
    auto const                 neighbors = G.neighbors(v);
    std::vector<graph::NodeId> message;
    message.reserve(1 + neighbors.size());
    message.push_back(G.layer()->global_id(v));
    message.insert(message.end(), neighbors.begin(), neighbors.end());
    return message;
}

/// The global phase: sends the message of every vertex v of `senders` once to each rank holding the head of a cut
/// out-edge of v, and handles the received messages. `threshold` is the message queue's buffer size.
///
/// A neighborhood is ascending and the vertex ranges are contiguous in rank order, so the heads on one rank form a run:
/// an entry beyond the current owner's range starts the next run, whose owner is found by a binary search over the
/// partition table from the current owner on. Per neighborhood: a comparison per entry and a search per destination.
template <class CommunicationPolicy, typename Senders, typename TriangleFunc>
void run(
    graph::GlobalIdGraph const& G, Config const& conf, std::size_t threshold, Senders const& senders, TriangleFunc emit
) {
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
    auto const&         layer     = *G.layer();
    auto const&         partition = layer.partition();
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
            auto const    start     = clock::now();
            PEID          owner     = 0;
            graph::NodeId owner_end = 0; // end of `owner`'s range; entries below it go to the same rank
            for (graph::NodeId const w: G.neighbors(v)) {
                if (layer.owns_global(w) || w < owner_end) {
                    continue;
                }
                owner     = partition.get_rank_from(w, owner);
                owner_end = partition.partition_range(owner).second;
                queue.post_message(make_message(G, v), owner);
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
