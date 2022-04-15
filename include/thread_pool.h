#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

class ThreadPool {
public:
    using Job = std::function<void()>;
    enum class Priority { high = 0, normal };

private:
    std::array<std::deque<Job>, 2> jobs_;
    std::vector<std::thread> threads_;
    std::unordered_map<std::thread::id, size_t> thread_id_map_;
    std::mutex mutex_;
    std::condition_variable cv_jobs_;
    std::condition_variable cv_finished_;

    std::atomic<size_t> busy_ = 0;
    std::atomic<size_t> idle_ = 0;
    std::atomic<size_t> done_ = 0;
    std::atomic<size_t> enqueued_ = 0;
    std::atomic<bool> terminate_ = false;

public:
    ThreadPool(size_t num_threads) : threads_(num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads_[i] = std::thread(&ThreadPool::worker, this, i);
        }
    }
    ~ThreadPool() {
        std::unique_lock lock(mutex_);
        terminate_ = true;
        cv_jobs_.notify_all();
        lock.unlock();
        for (size_t i = 0; i < threads_.size(); ++i) {
            threads_[i].join();
        }
    }

    void enqueue(Job&& job, Priority priority = Priority::normal) {
        std::unique_lock lock(mutex_);
        jobs_[static_cast<int>(priority)].emplace_back(std::move(job));
        cv_jobs_.notify_one();
        enqueued_++;
    }

    void loop_until_empty() {
        std::unique_lock lock(mutex_);
        cv_finished_.wait(lock, [this]() { return next_queue().empty() && (busy_ == 0); });
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void loop_until_terminate() {
        std::unique_lock lock(mutex_);
        cv_finished_.wait(lock, [this]() { return terminate_ && (busy_ == 0); });
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    size_t get_worker_thread_id() const {
        return thread_id_map_.at(std::this_thread::get_id());
    }
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex_);
        // flag termination
        terminate_ = true;
        // wake up all worker threads and let them terminate.
        cv_jobs_.notify_all();
        // notify LoopUntilTerminate in case all threads are idle.
        cv_finished_.notify_one();
    }

    void worker(size_t p) {
        std::unique_lock lock(mutex_);
        thread_id_map_[std::this_thread::get_id()] = p;
        while (true) {
            auto& job_queue = next_queue();
            if (!terminate_ && job_queue.empty()) {
                ++idle_;
                cv_jobs_.wait(lock, [this]() { return terminate_ || !next_queue().empty(); });
                --idle_;
            }
            if (terminate_) {
                break;
            }
            if (!job_queue.empty()) {
                ++busy_;
                {
                    Job job = std::move(job_queue.front());
                    job_queue.pop_front();
                    lock.unlock();
                    job();
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);

                ++done_;
                --busy_;

                lock.lock();
                cv_finished_.notify_one();
            }
        }
    }

    size_t enqueued() const {
        return enqueued_;
    }

    size_t done() const {
        return done_;
    }

private:
    std::deque<Job>& next_queue() {
        size_t i = 0;
        for (; i < jobs_.size() - 1; ++i) {
            if (!jobs_[i].empty()) {
                return jobs_[i];
            }
        }
        return jobs_[i];
    }
};
