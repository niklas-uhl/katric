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
#include <kamping/v2/cache.hpp>
#include <kamping/v2/collectives/alltoall.hpp>
#include <kamping/v2/collectives/alltoallv.hpp>
#include <kamping/v2/p2p/mrecv.hpp>
#include <kamping/v2/views/resize_view.hpp>
#include <kamping/v2/views/with_counts_view.hpp>
#include <kamping/v2/views/with_displs_view.hpp>
#include <kamping/v2/views/with_type_view.hpp>
#include <kamping/v2/views/with_value_type_view.hpp>
#include <mpi.h>

#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/message_statistics.h"
#include "cetric/util.h"

namespace cetric {

/// The grid communicator of `comm` for the grid exchange. Building one splits `comm` (collective, O(p)), so it is built
/// on the first grid exchange and cached on `comm` as an MPI attribute. A static keyval, as kamping-v2 recommends for
/// state without an owner (docs/attribute_caching.md): the value lives as long as `comm`; on MPI_COMM_WORLD that is
/// the whole program, and the static keyval skips freeing its key after MPI_Finalize. Collective on first use for a
/// given `comm`.
inline dstl::grid_comm<dstl::execution_policy::seq> const& cached_grid_comm(MPI_Comm comm) {
    static kamping::v2::keyval<kamping::v2::attribute::comm, dstl::grid_comm<dstl::execution_policy::seq>> grids;
    return grids.get_or_create(comm, [comm] {
        CALI_CXX_MARK_SCOPE("grid_comm");
        return dstl::grid_comm<dstl::execution_policy::seq>{kamping::v2::comm_view{comm}};
    });
}

/// Sends `vertex_to_message(u)` once to every rank that owns a neighbor of `u`, for each `u` in `sources` (local
/// vertices), and returns the messages this rank received. Since the graph is undirected, the receiver gets the
/// message of every ghost it holds. Deduplication: a vertex's neighborhood is scanned at once, so remembering per rank
/// the last vertex sent there suffices. `vertex_to_message` is only called for vertices with a remote neighbor.
///
/// `message_type` is the MPI datatype of `Message`. `tag` must not be in use on `comm` during the call (sparse mode).
/// Reports the message counters of the exchange (sent/received messages and volume) to the open Caliper region, with
/// the volume in units of `volume_per_message`; the grid mode reports no received message count, since its unordered
/// receive does not tell the sources apart. Caliper regions `prepare` (bucketing) and `exchange`.
template <typename Graph, typename Sources, typename VertexToMessage>
[[nodiscard]] auto push_to_ghosts(
    Graph const&      G,
    Sources&&         sources,
    VertexToMessage&& vertex_to_message,
    GhostExchangeMode mode,
    MPI_Datatype      message_type,
    int               tag,
    std::size_t       volume_per_message = 1
) {
    using Message           = std::remove_cvref_t<decltype(vertex_to_message(*std::ranges::begin(sources)))>;
    MPI_Comm const    comm  = G.comm();
    auto const        size  = static_cast<std::size_t>(G.size());
    PEID const        rank  = G.rank();
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
            for (auto v: G.adj(u).neighbors()) {
                PEID const v_rank = v.rank();
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
                    messages | views::with_value_type(message_type),
                    [&](kamping::v2::probe_result probe) {
                        kamping::v2::mrecv(message | views::with_type(message_type) | views::resize, std::move(probe));
                        stats.received_messages++;
                        recv_buf.insert(recv_buf.end(), message.begin(), message.end());
                    },
                    tag,
                    comm
                );
                break;
            }
            case GhostExchangeMode::dense: {
                std::vector<int> recv_counts(size);
                kamping::v2::alltoall(send_counts, recv_counts, comm);
                std::vector<int> recv_displs(size, 0);
                for (std::size_t r = 1; r < size; ++r) {
                    recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
                }
                recv_buf.resize(static_cast<std::size_t>(recv_displs.back() + recv_counts.back()));
                kamping::v2::alltoallv(
                    send_buf | views::with_type(message_type) | views::with_counts(send_counts)
                        | views::with_displs(send_displs),
                    recv_buf | views::with_type(message_type) | views::with_counts(recv_counts)
                        | views::with_displs(recv_displs),
                    comm
                );
                for (int count: recv_counts) {
                    stats.received_messages += count > 0 ? 1 : 0;
                }
                break;
            }
            case GhostExchangeMode::grid: {
                auto const& grid = cached_grid_comm(comm);
                dstl::grid_alltoallv(
                    send_buf | views::with_type(message_type) | views::with_counts(send_counts)
                        | views::with_displs(send_displs),
                    recv_buf | views::with_type(message_type) | views::resize,
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
