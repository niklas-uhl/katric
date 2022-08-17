#pragma once

#include "fmt/core.h"
#include "fmt/ranges.h"
#include "message-queue/debug_print.h"
#include "message-queue/queue.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/spin_rw_mutex.h>

#include "backward.hpp"

namespace message_queue {

template <class T, typename Merger, typename Splitter>
class ConcurrentBufferedMessageQueue {
    // static_assert(std::is_invocable_v<Splitter,
    //                                   std::vector<T>,
    //                                   std::function<void()>,
    //                                   PEID>);
    // static_assert(std::is_invocable_v<Merger, std::vector<T>&, std::vector<T>, int>);

public:
    ConcurrentBufferedMessageQueue(size_t num_threads, Merger&& merge, Splitter&& split)
        : queue_(),
          buffers_(queue_.size()),
          merge(merge),
          split(split),
          num_worker_threads_(num_threads - 1) {}

    void post_message(std::vector<T>&& message, PEID receiver, int tag = 0) {
        num_writing_threads_++;
        {
            if (buffer_ocupacy_ >= threshold_) {
                waiting_threads_++;
                // atomic_debug("Waiting");
                {
                    std::unique_lock lock(mutex_);
                    cv_buffer_full_.wait(lock, [this]() { return buffer_ocupacy_ < threshold_; });
                }
                waiting_threads_--;
                // atomic_debug("Waiting finished");
            }
            auto&             buffer = buffers_[receiver];
            std::stringstream out;
            // out << "message from thread " << tbb::this_task_arena::current_thread_index() << " for rank " << receiver
            //     << ": " << message;
            // atomic_debug(out.str());
            size_t added_elements = merge(buffer, std::forward<std::vector<T>>(message), tag);
            buffer_ocupacy_ += added_elements;
        }
        num_writing_threads_--;
    }

    void check_for_overflow_and_flush() {
        if (buffer_ocupacy_ >= threshold_ && waiting_threads_ != 0 && waiting_threads_ == num_writing_threads_) {
            // assert(buffer_ocupacy_ > threshold_);
            // atomic_debug("Overflow");
            overflows_++;
            flush_all();
        }
        cv_buffer_full_.notify_all();
    }

    void set_threshold(size_t threshold) {
        threshold_ = threshold;
        if (buffer_ocupacy_ > threshold_) {
            overflows_++;
            flush_all();
        }
    }

    size_t flush_impl(PEID receiver) {
        auto&  buffer           = buffers_[receiver];
        size_t removed_elements = 0;
        if (!buffer.empty()) {
            std::vector<T> message;
            message = {buffer.begin(), buffer.end()};
            // atomic_debug(fmt::format("Flushing buffer for {}", receiver));
            buffer.clear();
            removed_elements = message.size();
            queue_.post_message(std::move(message), receiver);
        }
        return removed_elements;
    }

    void flush(PEID receiver) {
        buffer_ocupacy_ -= flush_impl(receiver);
    }

    void flush_all() {
        size_t removed_elements = 0;
        for (size_t i = 0; i < buffers_.size(); ++i) {
            removed_elements += flush_impl(i);
        }
        buffer_ocupacy_ -= removed_elements;
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        queue_.poll([&](std::vector<T> message, PEID sender) { split(std::move(message), on_message, sender); });
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        queue_.terminate_impl(
            [&](std::vector<T> message, PEID sender) {
                // atomic_debug(fmt::format("Got message from {}", sender));
                split(std::move(message), on_message, sender);
            },
            [&]() {
                flush_all();
                cv_buffer_full_.notify_all();
            }
        );
        // cv_buffer_full_.notify_all();
    }

    size_t overflows() const {
        return overflows_;
    }

    const MessageStatistics& stats() {
        return queue_.stats();
    }

    void reset() {
        queue_.reset();
        buffers_.clear();
        buffer_ocupacy_ = 0;
        overflows_      = 0;
    }

private:
    MessageQueue<T>                        queue_;
    std::vector<tbb::concurrent_vector<T>> buffers_;
    std::mutex                             mutex_;
    Merger                                 merge;
    Splitter                               split;
    size_t                                 num_worker_threads_;
    size_t                                 threshold_           = std::numeric_limits<size_t>::max();
    size_t                                 overflows_           = 0;
    std::atomic<size_t>                    num_writing_threads_ = 0;
    std::atomic<size_t>                    buffer_ocupacy_      = 0;
    std::atomic<size_t>                    waiting_threads_     = 0;
    std::condition_variable                cv_buffer_full_;
};

template <class T, typename Merger, typename Splitter>
auto make_concurrent_buffered_queue(size_t num_threads, Merger&& merger, Splitter&& splitter) {
    return ConcurrentBufferedMessageQueue<T, Merger, Splitter>(
        num_threads,
        std::forward<Merger>(merger),
        std::forward<Splitter>(splitter)
    );
}

} // namespace message_queue
