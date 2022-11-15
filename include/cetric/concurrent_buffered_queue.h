#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <backward.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <kassert/kassert.hpp>
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

        {
            std::unique_lock lock(mutex_);
            if (buffer_ocupacy_ > threshold_) {
                cv_buffer_full_.wait(lock, [this]() { return buffer_ocupacy_ <= threshold_; });
            }
            buffer_ocupacy_ += message.size() + 1;
            // shared_mutex_.lock_shared();
            read_write_unlocked_.wait(lock, [this]() { return !flush_lock_; });
            num_writing_threads_++;
        }
        auto& buffer = buffers_[receiver];
        merge(buffer, std::forward<std::vector<T>>(message), tag);
        num_writing_threads_--;
    }

    void check_for_overflow_and_flush() {
        if (buffer_ocupacy_ > threshold_) {
            mutex_.lock();
            if (!flush_lock_ && num_writing_threads_ == 0) {
                flush_lock_ = true;
                mutex_.unlock();
                flush_all();
                overflows_++;
                flush_lock_ = false;
                read_write_unlocked_.notify_all();
            } else {
                mutex_.unlock();
            }
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
        flushing_               = true;
        size_t removed_elements = 0;
        for (size_t i = 0; i < buffers_.size(); ++i) {
            removed_elements += flush_impl(i);
        }
        buffer_ocupacy_.fetch_sub(removed_elements, std::memory_order_seq_cst);
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
    std::shared_mutex                      shared_mutex_;
    Merger                                 merge;
    Splitter                               split;
    size_t                                 num_worker_threads_;
    size_t                                 threshold_           = std::numeric_limits<size_t>::max();
    std::atomic<size_t>                    overflows_           = 0;
    std::atomic<size_t>                    num_writing_threads_ = 0;
    std::atomic<size_t>                    buffer_ocupacy_      = 0;
    std::atomic<bool>                      flushing_            = false;
    std::condition_variable                cv_buffer_full_;
    std::atomic<bool>                      flush_lock_ = false;
    std::condition_variable                read_write_unlocked_;
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
