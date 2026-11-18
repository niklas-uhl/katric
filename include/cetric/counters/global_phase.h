// The global phase: each rank sends N+(v) of its interface vertices v to the ranks of v's cut out-edges and intersects
// the received neighborhoods with its own.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <message-queue/buffered_queue.h>

#include "cetric/caliper.h"
#include "cetric/communication_policy.h"
#include "cetric/config.h"
#include "cetric/counters/intersection.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/datastructures/span.h"
#include "cetric/indirect_message_queue.h"
#include "cetric/util.h"
#include "kassert/kassert.hpp"

namespace cetric {

/// The message queue's buffer threshold (`--threshold`, `--threshold-scale`).
template <typename Graph>
std::size_t get_threshold(Graph const& G, Config const& conf) {
    switch (conf.threshold) {
        case Threshold::local_nodes:
            return conf.threshold_scale * G.local_node_count();
        case Threshold::local_edges:
            return conf.threshold_scale * G.local_edge_count();
        case Threshold::none:
            return std::numeric_limits<std::size_t>::max();
    }
    return 0;
}

/// Work counters of the global phase.
struct GlobalPhaseCounters {
    std::size_t skipped_nodes            = 0;
    std::size_t interface_nodes          = 0;
    std::size_t wedge_checks             = 0;
    std::size_t intersection_size_global = 0;
    std::size_t type3_triangles          = 0;

    /// Adds the counts and the message statistics of the phase's queue to the open Caliper region.
    template <typename MessageStatistics>
    void report(MessageStatistics const& messages) const {
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("interface_nodes", interface_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_global", intersection_size_global);
        profiling::counter_add("type3_triangles", type3_triangles);
        profiling::counter_add("sent_messages", messages.sent_messages);
        profiling::counter_add("send_volume", messages.send_volume);
        profiling::counter_add("received_messages", messages.received_messages);
        profiling::counter_add("receive_volume", messages.receive_volume);
    }
};

namespace global_phase {

/// Appends a message to the queue's buffer for one rank, terminated by a sentinel.
struct Merger {
    std::size_t operator()(
        std::vector<graph::RankEncodedNodeId>& buffer, std::vector<graph::RankEncodedNodeId> msg, int /*tag*/
    ) const {
        buffer.insert(buffer.end(), msg.begin(), msg.end());
        buffer.push_back(graph::RankEncodedNodeId::sentinel());
        return msg.size() + 1;
    }
};

/// Splits a received buffer into its messages at the sentinels.
struct Splitter {
    template <typename MessageFunc>
    void operator()(std::vector<graph::RankEncodedNodeId> buffer, MessageFunc&& on_message, PEID sender) const {
        auto const buffer_ptr  = std::make_shared<std::vector<graph::RankEncodedNodeId>>(std::move(buffer));
        auto       slice_begin = buffer_ptr->begin();
        while (slice_begin != buffer_ptr->end()) {
            auto const slice_end = std::find(slice_begin, buffer_ptr->end(), graph::RankEncodedNodeId::sentinel());
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

/// The work for one local receiver u of N+(v): emits the triangles (v, u, w) for w in N+(v) ∩ N+(u).
template <typename Graph, typename It, typename TriangleFunc, typename NodeOrdering>
void intersect_with_message(
    Graph const&             G,
    Config const&            conf,
    graph::RankEncodedNodeId v,
    graph::RankEncodedNodeId u,
    It                       v_first,
    It                       v_last,
    GlobalPhaseCounters&     counters,
    TriangleFunc&            emit,
    NodeOrdering const&      node_ordering
) {
    // CETRIC: u may have had no out-edges and was removed by the contraction
    if (conf.algorithm == Algorithm::cetric_x && !G.is_local(u)) {
        return;
    }
    auto const u_neighbors = G.out_adj(u).neighbors();
    // for the edge (v, u) we check the open wedges (v, u, w) for w in N+(u)
    counters.wedge_checks += u_neighbors.end() - u_neighbors.begin();
    counters.intersection_size_global += (v_last - v_first) + (u_neighbors.end() - u_neighbors.begin());
    cetric::intersection(
        u_neighbors.begin(),
        u_neighbors.end(),
        v_first,
        v_last,
        [&](graph::RankEncodedNodeId w) {
            counters.type3_triangles++;
            emit(graph::Triangle<graph::RankEncodedNodeId>{v, u, w});
        },
        node_ordering,
        conf
    );
}

/// Handles a received message [v, N+(v)...]: intersects N+(v) with N+(u) for every local vertex u in it.
template <typename Graph, typename It, typename TriangleFunc, typename NodeOrdering>
void process_message(
    Graph const&         G,
    Config const&        conf,
    It                   begin,
    It                   end,
    GlobalPhaseCounters& counters,
    TriangleFunc&        emit,
    NodeOrdering const&  node_ordering
) {
    graph::RankEncodedNodeId const v = *begin;
    KASSERT(!G.is_owned(v));
    auto const neighbors_first = std::next(begin);
    for (auto it = neighbors_first; it != end; ++it) {
        if (G.is_owned(*it)) {
            intersect_with_message(G, conf, v, *it, neighbors_first, end, counters, emit, node_ordering);
        }
    }
}

/// The global phase: sends [v, N+(v)] for every vertex v of `senders` once to each rank holding the head of a cut
/// out-edge of v (consecutive heads on the same rank are deduplicated, as the neighborhoods are sorted by rank), and
/// intersects every received neighborhood with those of its local receivers. `threshold` is the message queue's
/// buffer size.
template <class CommunicationPolicy, typename Graph, typename Senders, typename TriangleFunc, typename NodeOrdering>
void run(
    Graph const&        G,
    Config const&       conf,
    std::size_t         threshold,
    Senders const&      senders,
    TriangleFunc        emit,
    NodeOrdering const& node_ordering
) {
    auto base_queue = message_queue::make_buffered_queue<graph::RankEncodedNodeId>(Merger{}, Splitter{});
    auto queue      = [&]() {
        if constexpr (std::is_same_v<CommunicationPolicy, GridPolicy>) {
            return IndirectMessageQueueAdaptor(base_queue);
        } else {
            return DirectMessageQueueAdaptor(base_queue);
        }
    }();
    queue.set_threshold(threshold);
    GlobalPhaseCounters counters;
    // the rank N+(v) was sent to last, per local vertex
    std::vector<PEID> last_proc(G.local_node_count(), -1);
    // Time spent building and posting messages and handling received ones (the intersections), accumulated per call
    // because a Caliper region per message would cost more than the work it measures. The rest of the phase's
    // exclusive time is polling; the region `termination` is the wait for the other ranks.
    using clock = std::chrono::steady_clock;
    clock::duration send_time{};
    clock::duration handle_time{};
    auto            on_message = [&](SharedVectorSpan<graph::RankEncodedNodeId> span, PEID /*sender*/) {
        auto const start = clock::now();
        process_message(G, conf, span.begin(), span.end(), counters, emit, node_ordering);
        handle_time += clock::now() - start;
    };
    for (graph::RankEncodedNodeId v: senders) {
        counters.interface_nodes++;
        if (conf.pseudo2core && G.outdegree(v) < 2) {
            counters.skipped_nodes++;
        } else {
            auto const start = clock::now();
            auto const v_idx = G.to_local_idx(v);
            for (graph::RankEncodedNodeId u: G.out_adj(v).neighbors()) {
                if (!G.is_owned(u) && last_proc[v_idx] != G.owner(u)) {
                    std::vector<graph::RankEncodedNodeId> message;
                    message.emplace_back(v);
                    std::ranges::copy(G.out_adj(v).neighbors(), std::back_inserter(message));
                    queue.post_message(std::move(message), G.owner(u));
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

} // namespace global_phase
} // namespace cetric
