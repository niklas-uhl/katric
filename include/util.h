//
// Created by Tim Niklas Uhl on 19.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_UTIL_H
#define PARALLEL_TRIANGLE_COUNTER_UTIL_H

#include <atomic_debug.h>
#include <datastructures/graph_definitions.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <graph-io/local_graph_view.h>
#include <mpi.h>
#include <mpi_traits.h>
#include <unistd.h>
#include <array>
#include <backward.hpp>
#include <cstddef>
#include <cstdlib>
#include <debug_assert.hpp>
#include <exception>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <tlx/logger.hpp>
#include <vector>

using PEID = int;

inline void print_stacktrace(std::ostream& os = std::cerr) {
    backward::StackTrace stacktrace;
    backward::Printer printer;
    stacktrace.load_here(32);
    printer.print(stacktrace, os);
}

inline std::string get_stacktrace() {
    std::stringstream out;
    print_stacktrace(out);
    return out.str();
}

#define MODULE_A_LEVEL 0
#define PRINT_TRACE 1
struct debug_module : debug_assert::default_handler, debug_assert::set_level<MODULE_A_LEVEL> {
    static void handle(const debug_assert::source_location& loc, const char* expr, const char* msg = nullptr) noexcept {
        PEID rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::string rank_prefix = fmt::format("[R{}]", rank);
        std::string prefix = "[debug assert]";
        std::stringstream out;
        out << rank_prefix << prefix << " " << loc.file_name << ":" << loc.line_number << ": ";

        if (*expr == '\0') {
            out << "Unreachable code reached";
        } else {
            out << "Assertion " << expr << " failed";
        }
        if (msg) {
            out << " - " << msg << ".";
        } else {
            out << ".";
        }
        out << std::endl;
        if constexpr (PRINT_TRACE) {
            print_stacktrace(out);
        }
        std::cerr << out.str();
    }
};

#ifndef DEBUG_BARRIER
#ifndef NDEBUG
#define DEBUG_BARRIER(rank)                                                                    \
    {                                                                                          \
        if (std::getenv("DEBUG_BARRIER") != nullptr) {                                         \
            std::string value(std::getenv("DEBUG_BARRIER"));                                   \
            std::string delimiter = ":";                                                       \
            size_t pos = 0;                                                                    \
            std::string token;                                                                 \
            std::vector<int> PEs;                                                              \
            while ((pos = value.find(delimiter)) != std::string::npos) {                       \
                token = value.substr(0, pos);                                                  \
                PEs.push_back(std::atoi(token.c_str()));                                       \
                value.erase(0, pos + delimiter.length());                                      \
            }                                                                                  \
            PEs.push_back(std::atoi(value.c_str()));                                           \
            if (std::find(PEs.begin(), PEs.end(), rank) != PEs.end()) {                        \
                volatile int i = 0;                                                            \
                char hostname[256];                                                            \
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
        int resultlen;
        MPI_Error_string(errcode, buf.data(), &resultlen);
        std::string msg(buf.begin(), buf.begin() + resultlen);
        msg = msg + " in " + file + ":" + std::to_string(line);
        throw MPIException(msg);
    }
}

constexpr unsigned long long log2(unsigned long long x) {
#if defined __has_builtin
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
        for (const auto& elem : v) {
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
struct parallel {};
}  // namespace execution_policy

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

#endif  // PARALLEL_TRIANGLE_COUNTER_UTIL_H
