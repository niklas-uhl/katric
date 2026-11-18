// Pushing per-vertex data to the ranks that hold the vertex as a ghost. Adapted from KaCCv2's
// kacc::push_to_ghosts (src/kacc/graph/ghost_exchange.hpp), a candidate for dSTL. Differences: katric's graph instead
// of KaCCv2's graph concepts, the messages are bucketed by destination with one counting sort instead of a hash map of
// per-rank vectors, the transport is a runtime choice that includes dSTL's sparse all-to-all, and the counters go to
// Caliper.
#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <dstl/grid_alltoallv.hpp>
#include <dstl/grid_comm.hpp>
#include <dstl/sparse_alltoall.hpp>
#include <kamping/v2/collectives/alltoall.hpp>
#include <kamping/v2/collectives/alltoallv.hpp>
#include <kamping/v2/p2p/mrecv.hpp>
#include <kamping/v2/type_pool.hpp>
#include <kamping/v2/views/auto_counts_view.hpp>
#include <kamping/v2/views/auto_displs_view.hpp>
#include <kamping/v2/views/resize_v_view.hpp>
#include <kamping/v2/views/resize_view.hpp>
#include <kamping/v2/views/with_counts_view.hpp>
#include <kamping/v2/views/with_displs_view.hpp>
#include <mpi.h>

#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/message_statistics.h"
#include "cetric/message_tags.h"
#include "cetric/util.h"

namespace cetric {

/// Sends `vertex_to_message(u)` once to every rank that owns a neighbor of `u`, for each `u` in `sources` (local
/// vertices), and returns the messages this rank received. Since the graph is undirected, the receiver gets the
/// message of every ghost it holds. Deduplication: a vertex's neighborhood is scanned at once, so remembering per rank
/// the last vertex sent there suffices. `vertex_to_message` is only called for vertices with a remote neighbor.
///
/// `Message` must have a kamping-v2 datatype (a builtin type, or `kamping::v2::datatype_traits`, e.g. `std::pair` with
/// kamping/v2/types/std/unsafe/utility.hpp, as its bytes); it is registered in a type pool for the call. `tag` must not be in use on `comm`
/// during the call (sparse mode).
/// Reports the message counters of the exchange (sent/received messages and volume) to the open Caliper region, with
/// the volume in units of `volume_per_message`; the grid mode reports no received message count, since its unordered
/// receive does not tell the sources apart. Caliper regions `prepare` (bucketing) and `exchange`.
template <typename Graph, typename Sources, typename VertexToMessage>
[[nodiscard]] auto push_to_ghosts(
    Graph const&      G,
    Sources&&         sources,
    VertexToMessage&& vertex_to_message,
    GhostExchangeMode mode,
    MessageTag        tag,
    std::size_t       volume_per_message = 1
) {
    using Message           = std::remove_cvref_t<decltype(vertex_to_message(*std::ranges::begin(sources)))>;
    auto const        comm  = G.comm();
    auto const        size  = static_cast<std::size_t>(comm.size());
    PEID const        rank  = comm.rank();
    constexpr auto    never = std::numeric_limits<std::size_t>::max();
    std::vector<int>  send_counts(size, 0);
    std::vector<int>  send_displs(size, 0);
    std::vector<Message> send_buf;
    {
        CALI_CXX_MARK_SCOPE("prepare");
        // Append (message, destination) in one pass over the adjacency, then place the messages by destination with a
        // counting sort: no per-rank containers, and no second pass over the adjacency.
        std::vector<std::pair<Message, int>> outgoing;
        std::vector<std::size_t>             last_source_sent_to(size, never);
        for (auto u: sources) {
            auto const           u_local = G.to_local_idx(u);
            std::optional<Message> message;
            for (auto v: G.neighbors(u)) {
                PEID const v_rank = G.owner(v);
                if (v_rank == rank || last_source_sent_to[static_cast<std::size_t>(v_rank)] == u_local) {
                    continue;
                }
                if (!message) {
                    message = vertex_to_message(u);
                }
                outgoing.emplace_back(*message, v_rank);
                last_source_sent_to[static_cast<std::size_t>(v_rank)] = u_local;
            }
        }
        for (auto const& [message, dest]: outgoing) {
            send_counts[static_cast<std::size_t>(dest)]++;
        }
        for (std::size_t r = 1; r < size; ++r) {
            send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        }
        send_buf.resize(outgoing.size());
        std::vector<int> next = send_displs;
        for (auto const& [message, dest]: outgoing) {
            send_buf[static_cast<std::size_t>(next[static_cast<std::size_t>(dest)]++)] = message;
        }
    }

    profiling::MessageStatistics stats;
    for (int count: send_counts) {
        if (count > 0) {
            stats.sent_messages++;
            stats.send_volume += static_cast<std::size_t>(count) * volume_per_message;
        }
    }
    std::vector<Message> recv_buf;
    {
        CALI_CXX_MARK_SCOPE("exchange");
        namespace views = kamping::v2::views;
        kamping::v2::type_pool pool; // commits Message's datatype on first use
        switch (mode) {
            case GhostExchangeMode::sparse: {
                std::vector<std::pair<int, std::span<Message const>>> messages;
                for (std::size_t r = 0; r < size; ++r) {
                    if (send_counts[r] > 0) {
                        messages.emplace_back(
                            static_cast<int>(r),
                            std::span<Message const>(send_buf).subspan(
                                static_cast<std::size_t>(send_displs[r]),
                                static_cast<std::size_t>(send_counts[r])
                            )
                        );
                    }
                }
                std::vector<Message> message;
                dstl::sparse_alltoall(
                    dstl::sparse_alltoall_algorithm::nbx,
                    messages | views::with_auto_value_pool(pool),
                    [&](kamping::v2::probe_result probe) {
                        kamping::v2::mrecv(message | views::with_auto_pool(pool) | views::resize, std::move(probe));
                        stats.received_messages++;
                        recv_buf.insert(recv_buf.end(), message.begin(), message.end());
                    },
                    tag,
                    comm
                );
                break;
            }
            case GhostExchangeMode::dense: {
                // the receive counts are deferred: alltoallv exchanges them first
                auto [sent, received] = kamping::v2::alltoallv(
                    send_buf | views::with_auto_pool(pool) | views::with_counts(send_counts)
                        | views::with_displs(send_displs),
                    recv_buf | views::with_auto_pool(pool) | views::auto_counts() | views::auto_displs() | views::resize_v,
                    comm
                );
                for (int count: received.mpi_counts()) {
                    stats.received_messages += count > 0 ? 1 : 0;
                }
                break;
            }
            case GhostExchangeMode::grid: {
                auto const& grid = G.layer()->grid();
                dstl::grid_alltoallv(
                    send_buf | views::with_auto_pool(pool) | views::with_counts(send_counts)
                        | views::with_displs(send_displs),
                    recv_buf | views::with_auto_pool(pool) | views::resize,
                    grid,
                    dstl::layout::unordered{}
                );
                break;
            }
        }
    }
    stats.receive_volume = recv_buf.size() * volume_per_message;
    if (mode == GhostExchangeMode::grid) {
        profiling::counter_add("sent_messages", stats.sent_messages);
        profiling::counter_add("send_volume", stats.send_volume);
        profiling::counter_add("receive_volume", stats.receive_volume);
    } else {
        stats.report();
    }
    return recv_buf;
}

} // namespace cetric
