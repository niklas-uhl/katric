//
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H
#define PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H

#include <mpi.h>
#include <cstddef>
#include <google/dense_hash_map>
#include <iterator>
#include <sparsehash/dense_hash_map>
#include <type_traits>
#include "message_statistics.h"
#include "mpi_traits.h"
#include "util.h"

template <class, class = void>
struct has_data : std::false_type {};

template <class T>
struct has_data<T, std::void_t<typename T::data>> : std::true_type {};

struct CommunicationStats {
    size_t send_volume = 0;
    size_t recv_volume = 0;
    size_t number_of_sent_messages = 0;
    size_t number_of_recv_messages = 0;
    size_t cut_triangles = 0;
    size_t discarded_triangles = 0;
    double test_all_time = 0;

    void reset() {
        send_volume = 0;
        recv_volume = 0;
        number_of_sent_messages = 0;
        number_of_recv_messages = 0;
        cut_triangles = 0;
        discarded_triangles = 0;
        test_all_time = 0;
    }
};

enum class MessageTag {
    CostFunction = 0,
    LoadBalancing = 5,
    LoadBalancingBoundaries = 9,
    RHGFix = 13,
    ReportCost = 25,
    Orientation = 31,
    Neighborhood = 42,
};

inline int as_int(const MessageTag tag) {
    return static_cast<std::underlying_type<MessageTag>::type>(tag);
}

inline CommunicationStats reduce_stats(const CommunicationStats& stats) {
    size_t local[4];
    size_t global[4];
    local[0] = stats.send_volume;
    local[1] = stats.number_of_sent_messages;
    local[2] = stats.cut_triangles;
    local[3] = stats.discarded_triangles;
    MPI_Reduce(&local, &global, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    CommunicationStats global_stats;
    global_stats.send_volume = global[0];
    global_stats.number_of_sent_messages = global[1];
    global_stats.cut_triangles = global[2];
    global_stats.discarded_triangles = global[3];
    MPI_Reduce(&stats.test_all_time, &global_stats.test_all_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global_stats;
}

template <typename T>
class VectorView {
public:
    using iterator_type = typename std::vector<T>::iterator;
    VectorView(std::vector<T>& vector, iterator_type begin, iterator_type end)
        : begin_(begin), end_(end), data_(&vector) {}

    VectorView(): begin_(), end_(), data_(nullptr) {}

    iterator_type begin() {
        return begin_;
    }

    iterator_type end() {
        return end_;
    }

    size_t size() const {
        return std::distance(begin_, end_);
    }

    bool empty() const {
        return size() == 0;
    }

    T* data() {
        return &*begin_;
    }

private:
    iterator_type begin_;
    iterator_type end_;
    std::vector<T>* data_;
};

template <typename T>
class CompactBuffer {
public:
    CompactBuffer(std::vector<T>& data, PEID rank [[maybe_unused]], PEID size) : views_(size), data_(data) {
        views_.set_empty_key(size + 1);
    }
    VectorView<T>& operator[](PEID rank) {
        return views_[rank];
    };
    void set_extent(PEID rank, size_t begin, size_t end) {
        views_[rank] = VectorView<T>(data_, data_.begin() + begin, data_.begin() + end);
    };
    std::vector<T>& data() {
        return data_;
    }

private:
    google::dense_hash_map<PEID, VectorView<T>> views_;
    std::vector<T>& data_;
};

template <typename T>
VectorView<T> slice(std::vector<T>& vector, size_t pos, size_t n) {
    return VectorView(vector, vector.begin() + pos, vector.begin() + pos + n);
}

struct CommunicationUtility {
public:
    template <typename T>
    using hashed_view_buffer = google::dense_hash_map<PEID, VectorView<T>>;
    template <typename T>
    using HashedBuffer = google::dense_hash_map<PEID, std::vector<T>>;
    template <class T, typename SendBuf = hashed_view_buffer<T>, typename RecvBuf = HashedBuffer<T>>
    static void sparse_all_to_all(SendBuf& to_send,
                           RecvBuf& recv_buffers,
                           PEID rank,
                           PEID size,
                           int message_tag,
                           cetric::profiling::MessageStatistics& stats) {
        static_assert(std::is_same_v<RecvBuf, HashedBuffer<T>> || std::is_same_v<RecvBuf, CompactBuffer<T>>);
        constexpr bool compact_buffer = std::is_same_v<RecvBuf, CompactBuffer<T>>;
        MPI_Datatype mpi_type;
        if constexpr(mpi_traits<T>::builtin) {
            mpi_type = mpi_traits<T>::mpi_type;
        } else {
            mpi_type = mpi_traits<T>::register_type();
        }
        if constexpr (compact_buffer) {
            google::dense_hash_map<PEID, std::vector<size_t>> size_buf(size);
            google::dense_hash_map<PEID, std::vector<size_t>> size_buf_recv(size);
            size_buf.set_empty_key(-1);
            size_buf_recv.set_empty_key(-1);
            for (const auto& kv : to_send) {
                if (!kv.second.empty()) {
                    size_buf[kv.first].emplace_back(kv.second.size());
                }
            }
            sparse_all_to_all<size_t>(size_buf, size_buf_recv, rank, size, message_tag + 1, stats);
            std::vector<std::pair<PEID, size_t>> recv_sizes;
            for (const auto& kv : size_buf_recv) {
                recv_sizes.emplace_back(kv.first, kv.second[0]);
            }
            std::sort(recv_sizes.begin(), recv_sizes.end());
            size_t total_recv_volume = 0;
            for (const auto& kv : recv_sizes) {
                recv_buffers.set_extent(kv.first, total_recv_volume, total_recv_volume + kv.second);
                total_recv_volume += kv.second;
            }
            recv_buffers.data().resize(total_recv_volume);
        }
        PEID request_count = 0;
        for (auto& kv : to_send) {
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
        int request = 0;
        for (auto& kv : to_send) {
            PEID pe = kv.first;
            auto& local_buffer = kv.second;
            if (!local_buffer.empty()) {
                if (pe != rank) {
                    int tag = message_tag * size + pe;
                    MPI_Issend(local_buffer.data(), local_buffer.size(), mpi_type, pe, tag, MPI_COMM_WORLD,
                               &requests[request++]);
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
                int tag = message_tag * size + rank;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
                if (iprobe_success > 0) {
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    if constexpr (!compact_buffer) {
                        recv_buffers[status.MPI_SOURCE].resize(message_length);
                    }
                    MPI_Status rst;
                    MPI_Recv(recv_buffers[status.MPI_SOURCE].data(), message_length, mpi_type, status.MPI_SOURCE,
                             status.MPI_TAG, MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                }
            }
        };

        std::vector<MPI_Status> statuses(request_count);
        int isend_done = 0;
        while (isend_done == 0) {
            probe();
            isend_done = 0;
            MPI_Testall(request_count, requests.data(), &isend_done, statuses.data());
        }

        MPI_Request barrier_request;
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

        int ibarrier_done = 0;
        while (ibarrier_done == 0) {
            probe();
            MPI_Status test_status;
            MPI_Test(&barrier_request, &ibarrier_done, &test_status);
        }
    }

    template <class T>
    static void sparse_all_to_all(const google::dense_hash_map<PEID, std::vector<T>>& send_buffers,
                                  google::dense_hash_map<PEID, std::vector<T>>& recv_buffers,
                                  MPI_Datatype mpi_type,
                                  PEID rank,
                                  PEID size,
                                  cetric::profiling::MessageStatistics& stats,
                                  int message_tag) {
        PEID request_count = 0;
        for (const auto& kv : send_buffers) {
            PEID pe = kv.first;
            if (!kv.second.empty()) {
                if (pe != rank) {
                    request_count++;
                } else {
                    for (const auto& elem : kv.second) {
                        recv_buffers[rank].emplace_back(elem);
                    }
                }
            }
        }

        std::vector<MPI_Request> requests(request_count);
        int request = 0;
        for (const auto& kv : send_buffers) {
            PEID pe = kv.first;
            const std::vector<T>& buffer = kv.second;
            if (!buffer.empty()) {
                if (pe != rank) {
                    int tag = message_tag * size + pe;
                    MPI_Issend(buffer.data(), buffer.size(), mpi_type, pe, tag, MPI_COMM_WORLD, &requests[request++]);
                    stats.send_volume += buffer.size();
                    stats.sent_messages++;
                }
            }
        }

        std::vector<MPI_Status> statuses(request_count);
        int isend_done = 0;
        while (isend_done == 0) {
            int iprobe_success = 1;
            while (iprobe_success > 0) {
                iprobe_success = 0;
                MPI_Status status;
                int tag = message_tag * size + rank;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
                if (iprobe_success > 0) {
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    std::vector<T> message(message_length);
                    MPI_Status rst;
                    MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG,
                             MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                    for (const T& elem : message) {
                        recv_buffers[status.MPI_SOURCE].emplace_back(elem);
                    }
                }
            }
            isend_done = 0;
            MPI_Testall(request_count, requests.data(), &isend_done, statuses.data());
        }

        MPI_Request barrier_request;
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

        int ibarrier_done = 0;
        while (ibarrier_done == 0) {
            int iprobe_success = 1;
            while (iprobe_success > 0) {
                iprobe_success = 0;
                MPI_Status status;
                int tag = message_tag * size + rank;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
                if (iprobe_success > 0) {
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    assert(message_length > 0);
                    std::vector<T> message(message_length);
                    MPI_Status rst;
                    MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG,
                             MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                    for (const T& elem : message) {
                        recv_buffers[status.MPI_SOURCE].emplace_back(elem);
                    }
                }
            }
            MPI_Status test_status;
            MPI_Test(&barrier_request, &ibarrier_done, &test_status);
        }
    }

    template <class T>
    static void sparse_all_to_all(google::dense_hash_map<PEID, std::vector<T>>& send_buffers,
                                  google::dense_hash_map<PEID, std::vector<T>>& recv_buffers,
                                  MPI_Datatype mpi_type,
                                  PEID rank,
                                  PEID size,
                                  int message_tag) {
        cetric::profiling::MessageStatistics stats;
        sparse_all_to_all(send_buffers, recv_buffers, mpi_type, rank, size, stats, message_tag);
    }

    template <typename T>
    static void all_to_all(google::dense_hash_map<PEID, std::vector<T>>& send_buffers,
                           google::dense_hash_map<PEID, std::vector<T>>& recv_buffers,
                           MPI_Datatype mpi_type,
                           PEID rank,
                           PEID size,
                           cetric::profiling::MessageStatistics& stats,
                           int message_tag) {
        (void)rank;
        (void)message_tag;

        std::vector<int> send_counts(size);
        std::vector<int> recv_counts(size);
        std::vector<int> send_displs(size);
        std::vector<int> recv_displs(size);
        int total_send_count = 0;
        for (auto& kv : send_buffers) {
            PEID pe = kv.first;
            const std::vector<T>& buffer = kv.second;
            send_counts[pe] = buffer.size();
            total_send_count += buffer.size();
            stats.send_volume += buffer.size();
            stats.sent_messages++;
        }
        send_displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        recv_displs[0] = 0;
        int total_recv_count = 0;
        for (int i = 1; i < size; ++i) {
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            total_recv_count += recv_counts[i - 1];
        }
        total_recv_count += recv_counts[size - 1];
        std::vector<T> send_buffer;
        std::vector<T> recv_buffer;
        send_buffer.resize(total_send_count);
        for (auto& kv : send_buffers) {
            PEID pe = kv.first;
            std::vector<T>& buffer = kv.second;
            for (size_t i = 0; i < buffer.size(); ++i) {
                send_buffer[send_displs[pe] + i] = buffer[i];
            }
            buffer.clear();
        }
        send_buffers.clear();
        recv_buffer.resize(total_recv_count);
        MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), mpi_type, recv_buffer.data(),
                      recv_counts.data(), recv_displs.data(), mpi_type, MPI_COMM_WORLD);
        send_buffer.clear();
        send_buffer.shrink_to_fit();
        for (int pe = 0; pe < size; ++pe) {
            if (recv_counts[pe] > 0) {
                recv_buffers[pe].insert(recv_buffers[pe].end(), recv_buffer.begin() + recv_displs[pe],
                                        recv_buffer.begin() + recv_displs[pe] + recv_counts[pe]);
                stats.received_messages++;
                stats.receive_volume += recv_counts[pe];
            }
        }
        recv_buffer.clear();
        recv_buffer.shrink_to_fit();
    }

    template <typename T>
    static std::pair<std::vector<T>, std::vector<int>> all_gather(std::vector<T>& send_buffer,
                                                                  MPI_Datatype mpi_type,
                                                                  MPI_Comm comm,
                                                                  PEID rank,
                                                                  PEID size) {
        (void)rank;
        int send_count = send_buffer.size();
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
        MPI_Allgatherv(send_buffer.data(), send_count, mpi_type, receive_buffer.data(), receive_counts.data(),
                       displs.data(), mpi_type, comm);
        return std::make_pair(std::move(receive_buffer), std::move(displs));
    }

    template <typename Container>
    static std::pair<std::vector<typename Container::value_type>, std::vector<int>> gather(Container& send_buffer,
                                                                                           MPI_Datatype mpi_type,
                                                                                           MPI_Comm comm,
                                                                                           PEID root,
                                                                                           PEID rank,
                                                                                           PEID size) {
        int send_count = send_buffer.size();
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

        MPI_Gatherv(send_buffer.data(), send_count, mpi_type, receive_buffer.data(), receive_counts.data(),
                    displs.data(), mpi_type, root, comm);
        return std::make_pair(std::move(receive_buffer), std::move(displs));
    }
};

template <typename T>
class BufferedCommunicator {
public:
    BufferedCommunicator(size_t threshold,
                         MPI_Datatype mpi_type,
                         PEID rank,
                         PEID size,
                         int message_tag,
                         bool empty_pending_buffers_on_overflow = false)
        : message_tag_(message_tag),
          mpi_type(mpi_type),
          rank_(rank),
          size_(size),
          send_buffers_(size),
          overflow_buffers_(size),
          recv_buffers_(size),
          requests_(size, MPI_REQUEST_NULL),
          threshold_(threshold),
          empty_pending_buffers_on_overflow(empty_pending_buffers_on_overflow) {
        send_buffers_.set_empty_key(-1);
        overflow_buffers_.set_empty_key(-1);
        recv_buffers_.set_empty_key(-1);
    }

    template <typename MessageFunc>
    inline void add_message(const std::vector<T>& message,
                            PEID recv,
                            MessageFunc on_message,
                            cetric::profiling::MessageStatistics& stats) {
        if (!empty_pending_buffers_on_overflow) {
            if (!overflow_buffers_[recv].empty()) {
                // we haven't finished sending yet
                wait_and_receive(recv, on_message, stats);
            }
        }
        for (const T& elem : message) {
            send_buffers_[recv].emplace_back(elem);
        }
        if (send_buffers_[recv].size() > threshold_) {
            if (empty_pending_buffers_on_overflow) {
                if (!overflow_buffers_[recv].empty()) {
                    // we haven't finished sending yet
                    wait_and_receive(recv, on_message, stats);
                }
            }
            assert(overflow_buffers_[recv].empty());
            send_buffers_[recv].swap(overflow_buffers_[recv]);
            const std::vector<T>& buffer = overflow_buffers_[recv];
            int tag = message_tag_ * size_ + recv;
            stats.sent_messages++;
            stats.send_volume += buffer.size();
            MPI_Issend(buffer.data(), buffer.size(), mpi_type, recv, tag, MPI_COMM_WORLD, &requests_[recv]);
        }
    }

    template <typename MessageFunc>
    inline void wait_and_receive(PEID recv, MessageFunc on_message, cetric::profiling::MessageStatistics& stats) {
        assert(requests_[recv] != MPI_REQUEST_NULL);
        int buffer_sent = 0;
        while (buffer_sent == 0) {
            check_for_message(on_message, stats);
            MPI_Test(&requests_[recv], &buffer_sent, MPI_STATUS_IGNORE);
        }
        // the overflow buffer has been completely sent, we can clear it
        overflow_buffers_[recv].clear();
        assert(requests_[recv] == MPI_REQUEST_NULL);
    }
    template <typename MessageFunc>
    inline void finish_overflow_sending(MessageFunc on_message, cetric::profiling::MessageStatistics& stats) {
        if (threshold_ == std::numeric_limits<size_t>::max()) {
            return;
        }
        int isend_done = 0;
        while (isend_done == 0) {
            check_for_message(on_message, stats);
            isend_done = 0;
            MPI_Testall(requests_.size(), requests_.data(), &isend_done, MPI_STATUSES_IGNORE);
        }
        // the overflow buffer has been completely sent, we can clear it
        for (auto& kv : overflow_buffers_) {
            kv.second.clear();
        }
    }

    template <typename MessageFunc>
    inline void busy_waiting_for_receival(MessageFunc on_message, cetric::profiling::MessageStatistics& stats) {
        MPI_Request barrier_request;
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

        int ibarrier_done = 0;
        while (ibarrier_done == 0) {
            int iprobe_success = 1;
            while (iprobe_success > 0) {
                iprobe_success = 0;
                MPI_Status status;
                int tag = message_tag_ * size_ + rank_;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
                if (iprobe_success > 0) {
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    assert(message_length > 0);
                    std::vector<T> message(message_length);
                    MPI_Status rst;
                    MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG,
                             MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                    on_message(status.MPI_SOURCE, message);
                }
            }
            MPI_Status test_status;
            MPI_Test(&barrier_request, &ibarrier_done, &test_status);
        }
    }

    template <typename MessageFunc>
    inline void check_for_message(MessageFunc on_message, cetric::profiling::MessageStatistics& stats) {
        if (threshold_ == std::numeric_limits<size_t>::max()) {
            return;
        }
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;
            MPI_Status status;
            int tag = message_tag_ * size_ + rank_;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
            if (iprobe_success > 0) {
                int message_length;
                MPI_Get_count(&status, mpi_type, &message_length);
                assert(message_length > 0);
                std::vector<T> message(message_length);
                MPI_Status rst;
                MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD,
                         &rst);
                stats.receive_volume += message_length;
                stats.received_messages++;
                on_message(status.MPI_SOURCE, message);
            }
        }
    }

    template <typename MessageFunc>
    inline void all_to_all(MessageFunc on_message,
                           cetric::profiling::MessageStatistics& stats,
                           bool full_all_to_all = false) {
        // TODO: ensure that old buffers have been sent

        finish_overflow_sending(on_message, stats);
        if (full_all_to_all) {
            busy_waiting_for_receival(on_message, stats);
            CommunicationUtility::all_to_all(send_buffers_, recv_buffers_, mpi_type, rank_, size_, stats, message_tag_);
        } else {
            CommunicationUtility::sparse_all_to_all(send_buffers_, recv_buffers_, mpi_type, rank_, size_, stats,
                                                    message_tag_);
        }
        for (auto& kv : recv_buffers_) {
            std::vector<T>& buffer = kv.second;
            on_message(kv.first, buffer);
            buffer.clear();
        }
    }

private:
    template <typename V>
    using NodeBuffer = google::dense_hash_map<PEID, std::vector<V>>;
    int message_tag_;
    MPI_Datatype mpi_type;
    PEID rank_;
    PEID size_;
    NodeBuffer<T> send_buffers_;
    NodeBuffer<T> overflow_buffers_;
    NodeBuffer<T> recv_buffers_;
    std::vector<MPI_Request> requests_;
    size_t threshold_;
    bool empty_pending_buffers_on_overflow;
};

#endif  // PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H
