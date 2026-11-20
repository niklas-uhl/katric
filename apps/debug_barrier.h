// DEBUG_BARRIER(rank): with assertions on and the environment variable
// DEBUG_BARRIER=r1:r2:..., the listed ranks print their PID and wait for a
// debugger to attach.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

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
