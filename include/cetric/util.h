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

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

namespace cetric {
using PEID = int;

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

} // namespace cetric


#endif // PARALLEL_TRIANGLE_COUNTER_UTIL_H
