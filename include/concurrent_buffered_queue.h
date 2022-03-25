#pragma once

#include <tbb/concurrent_vector.h>
#include <tbb/spin_rw_mutex.h>
#include <tbb/task_arena.h>
#include <atomic>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include "backward.hpp"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "message-queue/debug_print.h"
#include "message-queue/queue.h"

namespace message_queue {

template <class T, typename Merger, typename Splitter>
class ConcurrentBufferedMessageQueue {
    static_assert(std::is_invocable_v<Splitter,
                                      std::vector<T>&,
                                      std::function<void(typename tbb::concurrent_vector<T>::iterator,
                                                         typename tbb::concurrent_vector<T>::iterator,
                                                         PEID)>,
                                      PEID>);
    static_assert(std::is_invocable_v<Merger, std::vector<T>&, std::vector<T>, int>);

public:
    ConcurrentBufferedMessageQueue(Merger&& merge, Splitter&& split)
        : queue_(),
          buffers_(),
          buffer_ocupacy_(0),
          threshold_(std::numeric_limits<size_t>::max()),
          overflows_(0),
          merge(merge),
          split(split) {}

    void post_message(std::vector<T>&& message, PEID receiver, int tag = 0) {
        //assert (receiver < queue_.size()) ;
        auto& buffer = buffers_[receiver];
        // atomic_debug(fmt::format("receiver {}", receiver));
        size_t added_elements;
        {
            tbb::spin_rw_mutex::scoped_lock merge_lock(mutexes_[receiver], false);
            std::stringstream out;
            out << "message from thread " << tbb::this_task_arena::current_thread_index() << " for rank " << receiver
                << ": " << message;
            atomic_debug(out.str());
            added_elements = merge(buffer, std::forward<std::vector<T>>(message), tag);
            buffer_ocupacy_ += added_elements;
        }
        // if (buffer_ocupacy_ > threshold_) {
        //     overflows_++;
        //     flush_all();
        // }
        // atomic_debug(buffer);
    }

    void check_for_overflow_and_flush() {
        if (buffer_ocupacy_ > threshold_) {
            atomic_debug("Overflow");
            overflows_++;
            flush_all();
        }
    }

    void set_threshold(size_t threshold) {
        threshold_ = threshold;
        if (buffer_ocupacy_ > threshold_) {
            overflows_++;
            flush_all();
        }
    }

    void flush(PEID receiver) {
        auto& buffer = buffers_[receiver];
        if (!buffer.empty()) {
            std::vector<T> message;
            {
                tbb::spin_rw_mutex::scoped_lock flush_lock(mutexes_[receiver], true);
                message = {buffer.begin(), buffer.end()};
                atomic_debug(fmt::format("Flushing buffer for {} with message {}", receiver, message));
                buffer.clear();
                buffer_ocupacy_ -= message.size();
            }
            atomic_debug(receiver);
            queue_.post_message(std::move(message), receiver);
        }
    }

    void flush_all() {
        for (auto& kv : buffers_) {
            flush(kv.first);
        }
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
                                          typename std::vector<T>::iterator, PEID>);
        queue_.poll([&](PEID sender, std::vector<T> message) { split(message, on_message, sender); });
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
                                          typename std::vector<T>::iterator, PEID>);
        queue_.terminate_impl([&](PEID sender, std::vector<T> message) { split(message, on_message, sender); },
                              [&]() { flush_all(); });
        /* for (auto buffer : buffers_) { */
        /*     atomic_debug(buffer); */
        /* } */
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
        overflows_ = 0;
    }

private:
    MessageQueue<T> queue_;
    std::unordered_map<PEID, tbb::concurrent_vector<T>> buffers_;
    std::unordered_map<PEID, tbb::spin_rw_mutex> mutexes_;
    std::atomic<size_t> buffer_ocupacy_;
    size_t threshold_;
    size_t overflows_;
    Merger merge;
    Splitter split;
};

template <class T, typename Merger, typename Splitter>
auto make_concurrent_buffered_queue(Merger&& merger, Splitter&& splitter) {
    return ConcurrentBufferedMessageQueue<T, Merger, Splitter>(std::forward<Merger>(merger),
                                                               std::forward<Splitter>(splitter));
}

}  // namespace message_queue
