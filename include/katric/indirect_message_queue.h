/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

#include <kassert/kassert.hpp>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include "message-queue/buffered_queue.h"

namespace katric {

/// `--communication-policy grid`: routes every message through at most one
/// proxy on a 2D grid of the ranks (rows of round(sqrt(p)) ranks, the last row
/// possibly shorter), so a rank exchanges messages with O(sqrt(p)) others. A
/// message from `from` to `to` goes to the rank in from's row and to's column;
/// if from is in the short last row and that position does not exist, to the
/// rank in from's column-th row and to's column (a full row, since there are at
/// least round(sqrt(p)) rows). The proxy is in to's column, so it sends on
/// directly. Each message carries its original sender and its receiver as two
/// trailing words. To be replaced by BriefKAsten's grid adapter.
template <typename QueueType>
class IndirectMessageQueueAdaptor {
public:
    using value_type = typename QueueType::value_type;

    explicit IndirectMessageQueueAdaptor(QueueType& queue) : queue_(queue) {
        // the grid is laid over the ranks of the queue's communicator, which is MPI_COMM_WORLD (legacy message-queue)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        grid_size_ = static_cast<int>(std::round(std::sqrt(size_)));
    }

    /// Copies `message` (any range of values) once, to append the routing words.
    template <typename Message>
    void post_message(Message const& message, int receiver) {
        std::vector<value_type> routed(std::ranges::begin(message), std::ranges::end(message));
        routed.push_back(static_cast<value_type>(rank_));
        routed.push_back(static_cast<value_type>(receiver));
        queue_.post_message(std::move(routed), get_proxy(rank_, receiver));
    }

    void set_threshold(std::size_t threshold) {
        queue_.set_threshold(threshold);
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        queue_.poll(route(on_message));
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        queue_.terminate(route(on_message));
    }

    auto stats() -> message_queue::MessageStatistics const& {
        return queue_.stats();
    }

private:
    /// The handler for the underlying queue: delivers a message addressed to this rank to `on_message` (without the
    /// routing words, with its original sender), forwards any other as received.
    template <typename MessageHandler>
    auto route(MessageHandler& on_message) {
        return [this, &on_message](std::span<value_type const> message, int /*sender*/) {
            KASSERT(message.size() >= 2);
            auto const receiver        = static_cast<int>(message[message.size() - 1]);
            auto const original_sender = static_cast<int>(message[message.size() - 2]);
            if (receiver == rank_) {
                on_message(message.first(message.size() - 2), original_sender);
            } else {
                auto const proxy = get_proxy(rank_, receiver);
                SPDLOG_DEBUG("Redirecting message to {} via {}", receiver, proxy);
                queue_.post_message(message, proxy);
            }
        };
    }

    struct GridPosition {
        int  row;
        int  column;
        auto operator==(GridPosition const&) const -> bool = default;
    };

    [[nodiscard]] auto get_grid_position(int rank) const -> GridPosition {
        return GridPosition{rank / grid_size_, rank % grid_size_};
    }
    [[nodiscard]] auto get_rank(GridPosition grid_position) const -> int {
        return grid_position.row * grid_size_ + grid_position.column;
    }
    [[nodiscard]] auto get_proxy(int from, int to) const -> int {
        auto const   from_pos = get_grid_position(from);
        auto const   to_pos   = get_grid_position(to);
        GridPosition proxy    = {from_pos.row, to_pos.column};
        if (get_rank(proxy) >= size_) {
            proxy = {from_pos.column, to_pos.column};
        }
        if (proxy == from_pos) {
            proxy = to_pos;
        }
        KASSERT(get_rank(proxy) < size_);
        return get_rank(proxy);
    }

    int        rank_      = 0;
    int        size_      = 0;
    int        grid_size_ = 0;
    QueueType& queue_;
};

} // namespace katric
