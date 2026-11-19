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

#include <dstl/distribution.hpp>
#include <message-queue/buffered_queue.h>

#include "katric/caliper.h"
#include "katric/communication_policy.h"
#include "katric/config.h"
#include "katric/intersection.h"
#include "katric/datastructures/span.h"
#include "katric/graph/graph.h"
#include "katric/indirect_message_queue.h"
#include "katric/phase_counters.h"
#include "katric/triangle_counter.h"
#include "katric/util.h"
#include "kassert/kassert.hpp"

namespace katric::global {

/// Separates the messages in the queue's buffers.
constexpr graph::VertexId sentinel = std::numeric_limits<graph::VertexId>::max();

/// Appends a message to the queue's buffer for one rank, terminated by a sentinel.
struct Merger {
    std::size_t operator()(std::vector<graph::VertexId>& buffer, std::vector<graph::VertexId> msg, int /*tag*/) const {
        buffer.insert(buffer.end(), msg.begin(), msg.end());
        buffer.push_back(sentinel);
        return msg.size() + 1;
    }
};

/// Splits a received buffer into its messages at the sentinels.
struct Splitter {
    template <typename MessageFunc>
    void operator()(std::vector<graph::VertexId> buffer, MessageFunc&& on_message, PEID sender) const {
        auto const buffer_ptr  = std::make_shared<std::vector<graph::VertexId>>(std::move(buffer));
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
    auto const&         numbering = G.numbering();
    graph::VertexId const v       = *begin;
    auto const          entries = std::span<graph::VertexId const>(std::to_address(std::next(begin)), std::to_address(end));
    KASSERT(!numbering.is_local_id(v));
    KASSERT(std::ranges::is_sorted(entries));
    for (graph::VertexId const u_global: entries) {
        if (!numbering.is_local_id(u_global)) {
            continue;
        }
        graph::VertexHandle const u = numbering.handle(u_global);
        // CETRIC: u may have had no cut edges and was removed by the contraction
        if (!G.is_kept(u)) {
            continue;
        }
        auto const u_neighbors = G.neighbors(u);
        // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
        counters.wedge_checks += u_neighbors.size();
        counters.intersection_size_global += entries.size() + u_neighbors.size();
        katric::intersection(
            u_neighbors.begin(),
            u_neighbors.end(),
            entries.begin(),
            entries.end(),
            [&](graph::VertexId w) {
                counters.type3_triangles++;
                emit(Triangle{v, u_global, w});
            },
            std::less<>{},
            conf
        );
    }
}

/// The message of `v`: [v, N+(v)...] in global ids, ascending.
inline std::vector<graph::VertexId> make_message(graph::GlobalIdGraph const& G, graph::VertexHandle v) {
    auto const                 neighbors = G.neighbors(v);
    std::vector<graph::VertexId> message;
    message.reserve(1 + neighbors.size());
    message.push_back(G.numbering().global_id(v));
    message.insert(message.end(), neighbors.begin(), neighbors.end());
    return message;
}

/// The rank owning global id `w`, searching the partition table only from rank `from` on (`w` must not lie before
/// rank `from`'s range): for ascending lookups along a sorted neighborhood.
inline auto owner_from(dstl::distribution const& distribution, graph::VertexId w, PEID from) -> PEID {
    auto const begins = distribution.displs();
    KASSERT(from >= 0 && from < distribution.size() && begins[from] <= w);
    // the last rank whose range starts at or before w (an empty rank starts where its successor does)
    auto const it = std::ranges::upper_bound(begins.begin() + from, begins.end(), w);
    return static_cast<PEID>(it - begins.begin()) - 1;
}

/// The global phase: sends the message of every vertex v of `senders` once to each rank holding the head of a cut
/// out-edge of v, and handles the received messages. `threshold` is the message queue's buffer size. Returns the
/// number of triangles found on this rank.
///
/// A neighborhood is ascending and the vertex ranges are contiguous in rank order, so the heads on one rank form a run:
/// an entry beyond the current owner's range starts the next run, whose owner is found by a binary search over the
/// partition table from the current owner on. Per neighborhood: a comparison per entry and a search per destination.
template <class CommunicationPolicy, typename Senders, typename TriangleFunc>
auto run(
    graph::GlobalIdGraph const& G, Config const& conf, std::size_t threshold, Senders const& senders, TriangleFunc emit
) -> std::size_t {
    auto base_queue = message_queue::make_buffered_queue<graph::VertexId>(Merger{}, Splitter{});
    auto queue      = [&]() {
        if constexpr (std::is_same_v<CommunicationPolicy, GridPolicy>) {
            return IndirectMessageQueueAdaptor(base_queue);
        } else {
            return DirectMessageQueueAdaptor(base_queue);
        }
    }();
    queue.set_threshold(threshold);
    GlobalPhaseCounters counters;
    auto const&         numbering    = G.numbering();
    auto const&         distribution = numbering.distribution();
    // Time spent building and posting messages and handling received ones (the intersections), accumulated per call
    // because a Caliper region per message would cost more than the work it measures. `poll_time` is the time in
    // `poll` outside the handler (the queue's own work); `handle_time` includes messages handled during termination,
    // which is also counted separately, so the region `termination` minus that part is the wait for the other ranks.
    using clock = std::chrono::steady_clock;
    clock::duration send_time{};
    clock::duration handle_time{};
    clock::duration poll_time{};
    clock::duration termination_handle_time{};
    auto            on_message = [&](SharedVectorSpan<graph::VertexId> span, PEID /*sender*/) {
        auto const start = clock::now();
        process_message(G, conf, span.begin(), span.end(), counters, emit);
        handle_time += clock::now() - start;
    };
    for (graph::VertexHandle const v: senders) {
        counters.interface_nodes++;
        if (conf.pseudo2core && G.neighbors(v).size() < 2) {
            counters.skipped_nodes++;
        } else {
            auto const    start     = clock::now();
            PEID          owner     = 0;
            graph::VertexId owner_end = 0; // end of `owner`'s range; entries below it go to the same rank
            for (graph::VertexId const w: G.neighbors(v)) {
                if (numbering.is_local_id(w) || w < owner_end) {
                    continue;
                }
                owner     = owner_from(distribution, w, owner);
                owner_end = distribution.index_range_end(owner);
                queue.post_message(make_message(G, v), owner);
            }
            send_time += clock::now() - start;
        }
        auto const poll_start    = clock::now();
        auto const handle_before = handle_time;
        queue.poll(on_message);
        poll_time += (clock::now() - poll_start) - (handle_time - handle_before);
    }
    {
        CALI_CXX_MARK_SCOPE("termination");
        auto const handle_before = handle_time;
        queue.terminate(on_message);
        termination_handle_time = handle_time - handle_before;
    }
    counters.report(queue.stats());
    auto const ns = [](clock::duration d) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
    };
    profiling::counter_add("send_ns", ns(send_time));
    profiling::counter_add("handle_ns", ns(handle_time));
    profiling::counter_add("poll_ns", ns(poll_time));
    profiling::counter_add("termination_handle_ns", ns(termination_handle_time));
    return static_cast<std::size_t>(counters.type3_triangles);
}

} // namespace katric::global
