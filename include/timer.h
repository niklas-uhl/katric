//
// Created by Tim Niklas Uhl on 22.10.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_TIMER_H
#define PARALLEL_TRIANGLE_COUNTER_TIMER_H

#include "util.h"
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

#include <mpi.h>
#include <tlx/multi_timer.hpp>

class Timer {
public:
    explicit Timer(std::string name, bool report = true)
        : name(std::move(name)),
          start(std::chrono::system_clock::now()) {
        restart(report);
    }
    void restart(bool report = true) {
        if (report) {
            std::cout << "Starting " << name << " ..." << std::endl;
        }
        start = std::chrono::system_clock::now();
    }
    double report_elapsed_time() {
        double duration = elapsed_time();
        std::cout << name << " took " << duration << " ms." << std::endl;
        return duration;
    }

    double elapsed_time() {
        auto                                      end            = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> execution_time = end - start;
        return execution_time.count();
    }

private:
    std::string                                        name;
    std::chrono::time_point<std::chrono::system_clock> start;
};

class MPITimer {
public:
    explicit MPITimer(std::string name, PEID rank, PEID size [[maybe_unused]], bool report = true)
        : name(std::move(name)),
          start(MPI_Wtime()),
          rank(rank) {
        restart(report);
    }
    void restart(bool report = true) {
        if (report && rank == 0) {
            std::stringstream out;
            out << "Starting " << name << " ...";
            atomic_debug(out.str());
        }
        start = MPI_Wtime();
    }

    void report_total_elapsed_time() {
        double total_duration = get_total_elapsed_time();
        if (rank == 0) {
            std::stringstream out;
            out << name << " took " << total_duration << " s.";
            atomic_debug(out.str());
        }
    }

    double get_total_elapsed_time() {
        double duration = local_elapsed_time_and_stop();
        double total_duration;
        MPI_Reduce(&duration, &total_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return total_duration;
    }

    double local_elapsed_time_and_stop() {
        auto end       = MPI_Wtime();
        execution_time = end - start;
        return execution_time;
    }

    double aggregate_total_execution_time() {
        double total_execution_time;
        MPI_Reduce(&execution_time, &total_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return execution_time;
    }

private:
    std::string name;
    double      start;
    double      execution_time;
    PEID        rank;
};

template <typename Task>
void report_time(const std::string& name, Task task) {
    Timer t(name);
    task();
    t.report_elapsed_time();
}

namespace cetric {
namespace profiling {
class Timer {
public:
    explicit Timer() : start(MPI_Wtime()) {
        restart();
    }

    void restart() {
        start = MPI_Wtime();
    }

    double elapsed_time() const {
        auto end = MPI_Wtime();
        return end - start;
    }

private:
    double start;
};
} // namespace profiling
} // namespace cetric

#endif // PARALLEL_TRIANGLE_COUNTER_TIMER_H
