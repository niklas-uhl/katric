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

//
// Created by Tim Niklas Uhl on 19.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_UTIL_H
#define PARALLEL_TRIANGLE_COUNTER_UTIL_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <set>
#include <sstream>
#include <vector>

#include <backward.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <graph-io/local_graph_view.h>
#include <mpi.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>
#include <tlx/logger.hpp>
#include <unistd.h>

#include "cetric/atomic_debug.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/mpi_traits.h"

namespace cetric {
using PEID = int;

inline void print_stacktrace(std::ostream& os = std::cerr) {
    backward::StackTrace stacktrace;
    backward::Printer    printer;
    stacktrace.load_here(32);
    printer.print(stacktrace, os);
}

inline std::string get_stacktrace() {
    std::stringstream out;
    print_stacktrace(out);
    return out.str();
}

#ifndef DEBUG_BARRIER
    #ifndef NDEBUG
        #define DEBUG_BARRIER(rank)                                                                    \
            {                                                                                          \
                if (std::getenv("DEBUG_BARRIER") != nullptr) {                                         \
                    std::string      value(std::getenv("DEBUG_BARRIER"));                              \
                    std::string      delimiter = ":";                                                  \
                    size_t           pos       = 0;                                                    \
                    std::string      token;                                                            \
                    std::vector<int> PEs;                                                              \
                    while ((pos = value.find(delimiter)) != std::string::npos) {                       \
                        token = value.substr(0, pos);                                                  \
                        PEs.push_back(std::atoi(token.c_str()));                                       \
                        value.erase(0, pos + delimiter.length());                                      \
                    }                                                                                  \
                    PEs.push_back(std::atoi(value.c_str()));                                           \
                    if (std::find(PEs.begin(), PEs.end(), rank) != PEs.end()) {                        \
                        volatile int i = 0;                                                            \
                        char         hostname[256];                                                    \
                        gethostname(hostname, sizeof(hostname));                                       \
                        printf("PID %d on %s (rank %d) ready for attach\n", getpid(), hostname, rank); \
                        fflush(stdout);                                                                \
                        while (0 == i)                                                                 \
                            sleep(5);                                                                  \
                    }                                                                                  \
                }                                                                                      \
            };
    #else
        #define DEBUG_BARRIER(rank)
    #endif
#endif

struct MPIException : public std::exception {
    MPIException(const std::string& msg) : msg_() {
        PEID rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        msg_ = fmt::format("[R{}] {}", rank, msg);
    }
    const char* what() const throw() {
        return msg_.c_str();
    }

private:
    std::string msg_;
};

inline void check_mpi_error(int errcode, const std::string& file, int line) {
    if (errcode != MPI_SUCCESS) {
        std::array<char, MPI_MAX_ERROR_STRING> buf;
        int                                    resultlen;
        MPI_Error_string(errcode, buf.data(), &resultlen);
        std::string msg(buf.begin(), buf.begin() + resultlen);
        msg = msg + " in " + file + ":" + std::to_string(line);
        throw MPIException(msg);
    }
}

inline void ConditionalBarrier(bool active, MPI_Comm comm = MPI_COMM_WORLD) {
    if (active) {
        MPI_Barrier(comm);
    }
}

constexpr unsigned long long log2(unsigned long long x) {
#if defined                  __has_builtin
    #if __has_builtin(__builtin_clzl)
        #define builtin_clzl(y) __builtin_clzl(y)
    #endif
#endif
#ifdef builtin_clzl
    return 8 * sizeof(unsigned long long) - builtin_clzl(x) - 1;
#else
    int log = 0;
    while (x >>= 1)
        ++log;
    return log;
#endif
}

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    if (v.empty()) {
        out << "[]";
    } else {
        out << "[";
        for (const auto& elem: v) {
            out << elem << ", ";
            ;
        }
        out << "\b\b]";
    }
    return out;
}

template <class T, class V>
std::ostream& operator<<(std::ostream& out, const std::pair<T, V>& p) {
    return out << "<" << p.first << ", " << p.second << ">";
}

namespace execution_policy {
struct sequential {};
struct parallel {
    parallel(size_t num_threads) : num_threads(num_threads) {}
    size_t num_threads;
};
} // namespace execution_policy

template <typename IteratorType>
typename IteratorType::value_type parallel_prefix_sum(IteratorType begin, IteratorType end) {
    size_t size = end - begin;
    return tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, size),
        0,
        [&begin,
         &end,
         &size](tbb::blocked_range<size_t> const& r, typename IteratorType::value_type sum, bool is_final_scan) ->
        typename IteratorType::value_type {
            auto temp = sum;
            for (size_t i = r.begin(); i < r.end(); ++i) {
                auto temp_prev = temp;
                temp += *(begin + i);
                if (is_final_scan) {
                    *(begin + i) = temp_prev;
                }
            }
            return temp;
        },
        std::plus<typename IteratorType::value_type>{}
    );
}
} // namespace cetric

template <>
struct mpi_traits<graphio::LocalGraphView::NodeInfo> {
    static MPI_Datatype register_type() {
        MPI_Datatype mpi_node_info;
        MPI_Type_contiguous(2, MPI_NODE, &mpi_node_info);
        MPI_Type_commit(&mpi_node_info);
        return mpi_node_info;
    }
    static constexpr bool builtin = false;
};

static thread_local int my_tid   = -1;
static std::atomic<int> next_tid = 0;
class ConcurrencyTracker {
public:
    ConcurrencyTracker(size_t max_threads) : tid_regions(3 * max_threads) {}
    void track(const std::string& region) {
        if (my_tid == -1) {
            my_tid = next_tid.fetch_add(1, std::memory_order_relaxed);
        }
        tid_regions[my_tid].insert(region);
    }
    std::string dump() {
        int                        end = next_tid;
        std::map<std::string, int> m;
        for (int i = 0; i < end; i++) {
            for (auto n: tid_regions[i]) {
                m[n] += 1;
            }
        }
        std::stringstream out;
        for (auto& kv : m) {
          out << kv.first << "[" << kv.second << "]; ";
        }
        return out.str();
    }

private:
    std::vector<std::set<std::string>> tid_regions;
};

#endif // PARALLEL_TRIANGLE_COUNTER_UTIL_H
