#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tbb/task_arena.h>
#include <tbb/task_group.h>

#include "thread_pool.h"

namespace cetric {

enum class TaskPriority { high = 0, normal };

class TbbTaskPool {
public:
    TbbTaskPool(size_t num_threads) : arena_(num_threads), tg_() {}
    template <typename Job>
    void submit_work(Job&& job, TaskPriority = TaskPriority::normal) {
        arena_.execute([&] { tg_.run(std::forward<Job>(job)); });
    }

    template <typename Job>
    void run_and_wait(Job&& job) {
        arena_.execute([&] { tg_.run_and_wait(std::forward<Job>(job)); });
    }

    void wait() {
        arena_.execute([&] { tg_.wait(); });
    }

private:
    tbb::task_arena arena_;
    tbb::task_group tg_;
};

class ThreadTaskPool {
public:
    ThreadTaskPool(size_t num_threads) : thread_pool_(num_threads - 1) {}
    template <typename Job>
    void submit_work(Job&& job, TaskPriority = TaskPriority::normal) {
        thread_pool_.enqueue(std::forward<Job>(job));
        // arena_.execute([&] { tg_.run(job); });
    }
    template <typename Job>
    void run_and_wait(Job&& job) {
        job();
        thread_pool_.loop_until_empty();
        // arena_.execute([&] { tg_.run_and_wait(job); });
    }
    void wait() {
        thread_pool_.loop_until_empty();
        // arena_.execute([&] { tg_.wait(); });
    }
private:
    ThreadPool thread_pool_;
};

} // namespace cetric
