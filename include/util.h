//
// Created by Tim Niklas Uhl on 19.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_UTIL_H
#define PARALLEL_TRIANGLE_COUNTER_UTIL_H

#include <fmt/core.h>
#include <mpi.h>
#include <unistd.h>
#include <backward.hpp>
#include <cstdlib>
#include <debug_assert.hpp>
#include <exception>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <vector>

using PEID = int;

inline void print_stacktrace(std::ostream& os = std::cerr) {
    backward::StackTrace stacktrace;
    backward::Printer printer;
    stacktrace.load_here(32);
    printer.print(stacktrace, os);
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

inline void
check_mpi_error(int errcode) {
    if (errcode != MPI_SUCCESS) {
        std::array<char, MPI_MAX_ERROR_STRING> buf;
        int resultlen;
        MPI_Error_string(errcode, buf.data(), &resultlen);
        std::string msg(buf.begin(), buf.end());
        throw MPIException(msg);
    }
}

template <class MessageType>
inline void atomic_debug(MessageType message, std::ostream& out = std::cout, bool newline = true) {
    std::stringstream sout;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sout << "[R" << rank << "] " << message;
    if (newline) {
        sout << std::endl;
    }
    out << sout.str();
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

#endif  // PARALLEL_TRIANGLE_COUNTER_UTIL_H
