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
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H
#define PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H

#include <cstddef>
#include <iterator>
#include <numeric>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include <spdlog/spdlog.h>
#include "message_statistics.h"
#include <kamping/types/mpi_type_traits.hpp>
#include "tlx/multi_timer.hpp"
#include "util.h"

namespace cetric {
template <class, class = void>
struct has_data : std::false_type {};

template <class T>
struct has_data<T, std::void_t<typename T::data>> : std::true_type {};

enum class MessageTag {
    CostFunction            = 0,
    LoadBalancing           = 5,
    LoadBalancingBoundaries = 9,
    RHGFix                  = 13,
    ReportCost              = 25,
    Orientation             = 31,
    Neighborhood            = 42,
};

inline int as_int(const MessageTag tag) {
    return static_cast<std::underlying_type<MessageTag>::type>(tag);
}

struct CommunicationUtility {
public:
    template <typename T>
    static std::pair<std::vector<T>, std::vector<int>>
    all_gather(std::vector<T>& send_buffer, MPI_Datatype mpi_type, MPI_Comm comm, PEID rank, PEID size) {
        (void)rank;
        int              send_count = send_buffer.size();
        std::vector<int> receive_counts(size);
        std::vector<int> displs(size + 1);
        assert(receive_counts.size() == static_cast<size_t>(size));
        MPI_Allgather(&send_count, 1, MPI_INT, receive_counts.data(), 1, MPI_INT, comm);

        int receive_count = 0;
        for (size_t i = 0; i < receive_counts.size(); ++i) {
            displs[i] = receive_count;
            receive_count += receive_counts[i];
        }
        displs[size] = receive_count;

        std::vector<T> receive_buffer(receive_count);
        MPI_Allgatherv(
            send_buffer.data(),
            send_count,
            mpi_type,
            receive_buffer.data(),
            receive_counts.data(),
            displs.data(),
            mpi_type,
            comm
        );
        return std::make_pair(std::move(receive_buffer), std::move(displs));
    }

    template <typename Container>
    static std::pair<std::vector<typename Container::value_type>, std::vector<int>>
    gather(Container& send_buffer, MPI_Datatype mpi_type, MPI_Comm comm, PEID root, PEID rank, PEID size) {
        int              send_count = send_buffer.size();
        std::vector<int> receive_counts;
        std::vector<int> displs;
        if (rank == root) {
            receive_counts.resize(size);
            displs.resize(size + 1);
        }
        MPI_Gather(&send_count, 1, MPI_INT, receive_counts.data(), 1, MPI_INT, root, comm);

        std::vector<typename Container::value_type> receive_buffer;
        if (rank == root) {
            int receive_count = 0;
            for (size_t i = 0; i < receive_counts.size(); ++i) {
                displs[i] = receive_count;
                receive_count += receive_counts[i];
            }
            displs[size] = receive_count;
            receive_buffer.resize(receive_count);
        }

        MPI_Gatherv(
            send_buffer.data(),
            send_count,
            mpi_type,
            receive_buffer.data(),
            receive_counts.data(),
            displs.data(),
            mpi_type,
            root,
            comm
        );
        return std::make_pair(std::move(receive_buffer), std::move(displs));
    }
};
} // namespace cetric

#endif // PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H
