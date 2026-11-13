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

template <typename T>
class VectorView {
public:
    using iterator_type = typename std::vector<T>::iterator;
    VectorView(std::vector<T>& vector, size_t begin, size_t end) : data_(&vector), begin_(begin), end_(end) {}

    VectorView() : data_(nullptr), begin_(), end_() {}

    iterator_type begin() {
        return data_->begin() + begin_;
    }

    iterator_type end() {
        return data_->begin() + end_;
    }

    size_t size() const {
        return end_ - begin_;
    }

    bool empty() const {
        return size() == 0;
    }

    T* data() {
        return &*begin();
    }

private:
    std::vector<T>* data_;
    size_t          begin_;
    size_t          end_;
};

template <typename T>
class CompactBuffer {
public:
    CompactBuffer(std::vector<T>& data, PEID rank [[maybe_unused]], PEID size) : views_(size), data_(data) {
    }
    VectorView<T>& operator[](PEID rank) {
        return views_[rank];
    };

    void set_extent(PEID rank, size_t begin, size_t end) {
        views_[rank] = VectorView<T>(data_, begin, end);
    };
    std::vector<T>& data() {
        return data_;
    }

private:
    absl::flat_hash_map<PEID, VectorView<T>> views_;
    std::vector<T>&                             data_;
};

template <typename T>
VectorView<T> slice(std::vector<T>& vector, size_t pos, size_t n) {
    return VectorView(vector, pos, pos + n);
}

struct CommunicationUtility {
public:
    template <typename T>
    using hashed_view_buffer = absl::flat_hash_map<PEID, VectorView<T>>;
    template <typename T>
    using HashedBuffer = absl::flat_hash_map<PEID, std::vector<T>>;
    template <class T, typename SendBuf = hashed_view_buffer<T>, typename RecvBuf = HashedBuffer<T>>
    static void sparse_all_to_all(
        SendBuf&                              to_send,
        RecvBuf&                              recv_buffers,
        MPI_Datatype                          mpi_type,
        MPI_Comm comm,
        PEID                                  rank,
        PEID                                  size [[maybe_unused]],
        int                                   message_tag
    ) {
        cetric::profiling::MessageStatistics stats;
        static_assert(std::is_same_v<RecvBuf, HashedBuffer<T>> || std::is_same_v<RecvBuf, CompactBuffer<T>>);
        constexpr bool compact_buffer = std::is_same_v<RecvBuf, CompactBuffer<T>>;
        if constexpr (compact_buffer) {
            absl::flat_hash_map<PEID, std::vector<size_t>> size_buf(size);
            absl::flat_hash_map<PEID, std::vector<size_t>> size_buf_recv(size);
            for (const auto& kv: to_send) {
                if (!kv.second.empty()) {
                    size_buf[kv.first].emplace_back(kv.second.size());
                }
            }
            sparse_all_to_all<size_t, decltype(size_buf), decltype(size_buf_recv)>(
                size_buf, size_buf_recv, kamping::types::mpi_type_traits<size_t>::data_type(), comm, rank, size, message_tag + 1
            );
            std::vector<std::pair<PEID, size_t>> recv_sizes;
            for (const auto& kv: size_buf_recv) {
                recv_sizes.emplace_back(kv.first, kv.second[0]);
            }
            std::sort(recv_sizes.begin(), recv_sizes.end());
            size_t total_recv_volume = 0;
            for (const auto& kv: recv_sizes) {
                recv_buffers.set_extent(kv.first, total_recv_volume, total_recv_volume + kv.second);
                total_recv_volume += kv.second;
            }
            recv_buffers.data().resize(total_recv_volume);
        }
        PEID request_count = 0;
        for (auto& kv: to_send) {
            PEID pe = kv.first;
            if (!kv.second.empty()) {
                if (pe != rank) {
                    request_count++;
                } else {
                    if constexpr (compact_buffer) {
                        std::copy(kv.second.begin(), kv.second.end(), recv_buffers[rank].begin());
                    } else {
                        std::copy(kv.second.begin(), kv.second.end(), std::back_inserter(recv_buffers[rank]));
                    }
                }
            }
        }

        std::vector<MPI_Request> requests(request_count);
        int                      request = 0;
        for (auto& kv: to_send) {
            PEID  pe           = kv.first;
            auto& local_buffer = kv.second;
            if (!local_buffer.empty()) {
                if (pe != rank) {
                    MPI_Issend(
                        local_buffer.data(),
                        local_buffer.size(),
                        mpi_type,
                        pe,
                        message_tag,
                        comm,
                        &requests[request++]
                    );
                    stats.send_volume += local_buffer.size();
                    stats.sent_messages++;
                }
            }
        }

        auto probe = [&]() {
            int iprobe_success = 1;
            while (iprobe_success > 0) {
                iprobe_success = 0;
                MPI_Status status;
                MPI_Iprobe(MPI_ANY_SOURCE, message_tag, comm, &iprobe_success, &status);
                if (iprobe_success > 0) {
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    if constexpr (!compact_buffer) {
                        recv_buffers[status.MPI_SOURCE].resize(message_length);
                    }
                    MPI_Status rst;
                    MPI_Recv(
                        recv_buffers[status.MPI_SOURCE].data(),
                        message_length,
                        mpi_type,
                        status.MPI_SOURCE,
                        status.MPI_TAG,
                        comm,
                        &rst
                    );
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                }
            }
        };

        std::vector<MPI_Status> statuses(request_count);
        int                     isend_done = 0;
        while (isend_done == 0) {
            probe();
            isend_done = 0;
            MPI_Testall(request_count, requests.data(), &isend_done, statuses.data());
        }

        MPI_Request barrier_request;
        MPI_Ibarrier(comm, &barrier_request);

        int ibarrier_done = 0;
        while (ibarrier_done == 0) {
            // TODO We should test after each received message, else we get may get message from the next a2a
            probe();
            MPI_Status test_status;
            MPI_Test(&barrier_request, &ibarrier_done, &test_status);
        }
        stats.report();
    }

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
