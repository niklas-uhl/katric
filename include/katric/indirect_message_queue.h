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
#include <ranges>
#include <span>
#include <vector>
#include <type_traits>

#include <mpi.h>
#include <spdlog/spdlog.h>

#include "message-queue/buffered_queue.h"

namespace katric {
template <typename T>
T rank_to_value(int rank) {
    return rank;
}
template <typename T>
int value_to_rank(T const& rank) {
    return rank;
}

template <typename QueueType>
class IndirectMessageQueueAdaptor {
public:
    using value_type = typename QueueType::value_type;
    IndirectMessageQueueAdaptor(QueueType& queue) : rank_(), size_(), queue_(queue) {
        // the grid is laid over the ranks of the queue's communicator, which is MPI_COMM_WORLD (legacy message-queue)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        grid_size_ = std::round(std::sqrt(size_));
    };
    /// Copies `message` (any range of values) once, to append the original sender and the receiver for the proxies.
    template <typename Message>
    void post_message(Message const& message, int receiver, bool direct_send = false) {
        std::vector<value_type> routed(std::ranges::begin(message), std::ranges::end(message));
        routed.push_back(rank_to_value<value_type>(rank_));
        routed.push_back(rank_to_value<value_type>(receiver));
        int proxy;
        if (direct_send) {
            proxy = receiver;
        } else {
            proxy = get_proxy(rank_, receiver);
        }
        SPDLOG_DEBUG("Redirecting message to {} via {}", receiver, proxy);
        queue_.post_message(std::move(routed), proxy);
    }

    void set_threshold(size_t threshold) {
        queue_.set_threshold(threshold);
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        // static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
        //                                   typename std::vector<T>::iterator, int>);
        queue_.poll([&](std::span<value_type const> span, int /*sender*/) {
            auto end             = span.end();
            int receiver        = value_to_rank(*(end - 1));
            int original_sender = value_to_rank(*(end - 2));
            if (receiver == rank_) {
                auto received_message = span.subspan(0, span.size() - 2);
                on_message(received_message, original_sender);
            } else {
                auto              proxy = get_proxy(rank_, receiver);
                SPDLOG_DEBUG("Redirecting message to {} via {}", receiver, proxy);
                queue_.post_message(span, proxy);  // forwarded as received, with its routing words
            }
        });
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        // static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
        //                                   typename std::vector<T>::iterator, int>);
        queue_.terminate([&](std::span<value_type const> span, int /*sender*/) {
            auto end             = span.end();
            int receiver        = value_to_rank(*(end - 1));
            int original_sender = value_to_rank(*(end - 2));
            if (receiver == rank_) {
                auto received_message = span.subspan(0, span.size() - 2);
                on_message(received_message, original_sender);
            } else {
                auto              proxy = get_proxy(rank_, receiver);
                SPDLOG_DEBUG("Redirecting message to {} via {}", receiver, proxy);
                queue_.post_message(span, proxy);  // forwarded as received, with its routing words
            }
        });
    }

    size_t overflows() const {
        return queue_.overflows();
    }

    void check_for_overflow_and_flush() {
        queue_.check_for_overflow_and_flush();
    }

    const message_queue::MessageStatistics& stats() {
        return queue_.stats();
    }

    void reset() {
        queue_.reset();
    }

private:
    struct GridPosition {
        int  row;
        int  column;
        bool operator==(const GridPosition& rhs) const {
            return row == rhs.row && column == rhs.column;
        }
    };

    GridPosition get_grid_position(int rank) {
        return GridPosition{rank / grid_size_, rank % grid_size_};
    }
    int get_rank(GridPosition grid_position) {
        return grid_position.row * grid_size_ + grid_position.column;
    }
    int get_proxy(int from, int to) {
        auto         from_pos = get_grid_position(from);
        auto         to_pos   = get_grid_position(to);
        GridPosition proxy    = {from_pos.row, to_pos.column};
        if (get_rank(proxy) >= size_) {
            proxy = {from_pos.column, to_pos.column};
        }
        if (proxy == from_pos) {
            proxy = to_pos;
        }
        assert(get_rank(proxy) < size_);
        return get_rank(proxy);
    }
    int       rank_;
    int       size_;
    int       grid_size_;
    QueueType& queue_;
};

template <typename QueueType>
class DirectMessageQueueAdaptor {
public:
    using value_type = typename QueueType::value_type;
    DirectMessageQueueAdaptor(QueueType& queue) : queue_(queue){};
    template <typename Message>
    void post_message(Message const& message, int receiver, bool direct_send [[maybe_unused]] = false) {
        queue_.post_message(message, receiver);
    }

    void set_threshold(size_t threshold) {
        queue_.set_threshold(threshold);
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        // static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
        //                                   typename std::vector<T>::iterator, int>);
        queue_.poll(on_message);
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        queue_.terminate(on_message);
    }

    size_t overflows() const {
        return queue_.overflows();
    }

    void check_for_overflow_and_flush() {
        queue_.check_for_overflow_and_flush();
    }

    const message_queue::MessageStatistics& stats() {
        return queue_.stats();
    }

    void reset() {
        queue_.reset();
    }

private:
    QueueType& queue_;
};
// template <class QueueType, class T, typename Merger, typename Splitter>
// auto make_indirect_queue(Merger&& merger, Splitter&& splitter) {
//     return IndirectMessageQueue<QueueType>(std::forward<Merger>(merger), std::forward<Splitter>(splitter));
// }
} // namespace katric
