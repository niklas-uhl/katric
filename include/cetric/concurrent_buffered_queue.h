#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <backward.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <message-queue/queue.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/spin_rw_mutex.h>
#include "cetric/datastructures/graph_definitions.h"

namespace cetric {

template <class T, typename Merger, typename Splitter>
class ConcurrentBufferedMessageQueue {
    // static_assert(std::is_invocable_v<Splitter,
    //                                   std::vector<T>,
    //                                   std::function<void()>,
    //                                   PEID>);
    // static_assert(std::is_invocable_v<Merger, std::vector<T>&, std::vector<T>, int>);

public:
    using merger     = Merger;
    using splitter   = Splitter;
    using value_type = T;

    ConcurrentBufferedMessageQueue(size_t num_threads, std::thread::id master, Merger&& merge, Splitter&& split)
        : master_thread_id_(master),
          queue_(),
          buffers_(queue_.size()),
          merge(merge),
          split(split),
          num_worker_threads_(num_threads - 1) {}

    // a thread tries to post a message
    // if the buffer is full it blocks (thread is writing + waiting)
    // main thread polls and sees overflow
    // if no threads are waiting

    void post_message(std::vector<T>&& message, message_queue::PEID receiver, int tag = 0) {
        if (std::this_thread::get_id() == master_thread_id_) {
            check_for_overflow_and_flush();
            auto&  buffer         = buffers_[receiver];
            size_t added_elements = merge(buffer, std::forward<std::vector<T>>(message), tag);
            buffer_ocupacy_ += added_elements;
            return;
        }
        num_writing_threads_.fetch_add(1, std::memory_order_relaxed);
        {
            if (buffer_ocupacy_.load(std::memory_order_relaxed) >= threshold_) {
                waiting_threads_.fetch_add(1, std::memory_order_relaxed);
                {
                    std::unique_lock lock(mutex_);
                    size_t           overflows = overflows_.load(std::memory_order_relaxed);
                    cv_buffer_full_.wait(lock, [this, overflows]() {
                        if (threshold_ == 0) {
                            // we may wake up to early, therefore check if the buffer has already been flushed
                            return !flushing_;
                        } else {
                            return buffer_ocupacy_.load(std::memory_order_relaxed) < threshold_;
                        }
                    });
                    filling_threads_.fetch_add(1, std::memory_order_relaxed);
                    waiting_threads_.fetch_sub(1, std::memory_order_relaxed);
                }
                // atomic_debug("Waiting finished");
            }
            auto&             buffer = buffers_[receiver];
            std::stringstream out;
            // out << "message from thread " << tbb::this_task_arena::current_thread_index() << " for rank " << receiver
            //     << ": " << message;
            // atomic_debug(out.str());
            size_t added_elements = merge(buffer, std::forward<std::vector<T>>(message), tag);
            buffer_ocupacy_.fetch_add(added_elements, std::memory_order_relaxed);
            filling_threads_.fetch_sub(1, std::memory_order_relaxed);
        }
        num_writing_threads_.fetch_sub(1, std::memory_order_relaxed);
    }

    void check_for_overflow_and_flush() {
        if (buffer_ocupacy_.load(std::memory_order_relaxed) >= threshold_
            && waiting_threads_.load(std::memory_order_relaxed) != 0
            && waiting_threads_.load(std::memory_order_relaxed) == num_writing_threads_.load(std::memory_order_relaxed)
            && filling_threads_.load(std::memory_order_relaxed) == 0) {
            // assert(buffer_ocupacy_ > threshold_);
            // atomic_debug("Overflow");
            flush_all();
            overflows_.fetch_add(1, std::memory_order_relaxed);
        }
        cv_buffer_full_.notify_all();
    }

    void set_threshold(size_t threshold) {
        threshold_ = threshold;
        if (buffer_ocupacy_ > threshold_) {
            flush_all();
            overflows_++;
        }
    }

    size_t flush_impl(message_queue::PEID receiver) {
        auto&  buffer           = buffers_[receiver];
        size_t removed_elements = 0;
        if (!buffer.empty()) {
            std::vector<T> message;
            message = {buffer.begin(), buffer.end()};
            // atomic_debug(fmt::format("Flushing buffer for {}", receiver));
            buffer.clear();
            removed_elements = message.size();
            KASSERT(message.back() == cetric::graph::RankEncodedNodeId::sentinel());
            queue_.post_message(std::move(message), receiver);
        }
        return removed_elements;
    }

    void flush(message_queue::PEID receiver) {
        buffer_ocupacy_ -= flush_impl(receiver);
    }

    void flush_all() {
        flushing_ = true;
        size_t removed_elements = 0;
        for (size_t i = 0; i < buffers_.size(); ++i) {
            removed_elements += flush_impl(i);
        }
        buffer_ocupacy_.fetch_sub(removed_elements, std::memory_order_relaxed);
        flushing_ = false;
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        queue_.poll([&](std::vector<T> message, message_queue::PEID sender) {
            split(std::move(message), on_message, sender);
        });
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        queue_.terminate_impl(
            [&](std::vector<T> message, message_queue::PEID sender) {
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
        return overflows_.load();
    }

    const message_queue::MessageStatistics& stats() {
        return queue_.stats();
    }

    void reset() {
        queue_.reset();
        buffers_.clear();
        buffer_ocupacy_ = 0;
        overflows_      = 0;
    }

private:
    std::thread::id                        master_thread_id_;
    message_queue::MessageQueue<T>         queue_;
    std::vector<tbb::concurrent_vector<T>> buffers_;
    std::mutex                             mutex_;
    Merger                                 merge;
    Splitter                               split;
    size_t                                 num_worker_threads_;
    size_t                                 threshold_           = std::numeric_limits<size_t>::max();
    std::atomic<size_t>                    overflows_           = 0;
    std::atomic<size_t>                    num_writing_threads_ = 0;
    std::atomic<size_t>                    buffer_ocupacy_      = 0;
    std::atomic<size_t>                    waiting_threads_     = 0;
    std::atomic<size_t>                    filling_threads_     = 0;
    std::atomic<bool>                      flushing_            = false;
    std::condition_variable                cv_buffer_full_;
};

template <class T, typename Merger, typename Splitter>
auto make_concurrent_buffered_queue(size_t num_threads, std::thread::id master, Merger&& merger, Splitter&& splitter) {
    return ConcurrentBufferedMessageQueue<T, Merger, Splitter>(
        num_threads,
        master,
        std::forward<Merger>(merger),
        std::forward<Splitter>(splitter)
    );
}

} // namespace cetric
