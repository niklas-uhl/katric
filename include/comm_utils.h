//
// Created by Tim Niklas Uhl on 29.11.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H
#define PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H

#include <google/dense_hash_map>
#include <datastructures/graph.h>
#include <statistics.h>
#include "util.h"
#include "prefix_sum.h"


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

enum MessageTag {
    CostFunction = 0,
    LoadBalancing = 5,
    LoadBalancingBoundaries = 9,
    RHGFix = 13,
    ReportCost = 25,
    Orientation = 31,
    Neighborhood = 42,
};

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

struct CommunicationUtility {
public:
    template<class T>
    static void sparse_all_to_all(const google::dense_hash_map<PEID, std::vector<T>>& send_buffers,
                                  google::dense_hash_map<PEID, std::vector<T>>& recv_buffers,
                                  MPI_Datatype mpi_type, PEID rank, PEID size, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer, int message_tag) {
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
                    timer.start("Receiving");
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    std::vector<T> message(message_length);
                    MPI_Status rst;
                    MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                    for (const T& elem : message) {
                        recv_buffers[status.MPI_SOURCE].emplace_back(elem);
                    }
                    timer.start("a2a");
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
                    timer.start("Receiving");
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    assert(message_length > 0);
                    std::vector<T> message(message_length);
                    MPI_Status rst;
                    MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                    for (const T& elem : message) {
                        recv_buffers[status.MPI_SOURCE].emplace_back(elem);
                    }
                    timer.start("a2a");
                }
            }
            MPI_Status test_status;
            MPI_Test(&barrier_request, &ibarrier_done, &test_status);
        }

    }

    template<class T>
    static void sparse_all_to_all(google::dense_hash_map<PEID, std::vector<T>>& send_buffers,
                                  google::dense_hash_map<PEID, std::vector<T>>& recv_buffers,
                                  MPI_Datatype mpi_type, PEID rank, PEID size, int message_tag) {
        cetric::profiling::MessageStatistics stats;
        tlx::MultiTimer dummy_timer;
        sparse_all_to_all(send_buffers, recv_buffers, mpi_type, rank, size, stats, dummy_timer, message_tag);
    }

    template<typename T>
    static void all_to_all(google::dense_hash_map<PEID, std::vector<T>>& send_buffers,
                                  google::dense_hash_map<PEID, std::vector<T>>& recv_buffers,
                                  MPI_Datatype mpi_type, PEID rank, PEID size, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer, int message_tag) {
        (void) rank;
        (void) timer;
        (void) message_tag;

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
        MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), mpi_type, recv_buffer.data(), recv_counts.data(), recv_displs.data(), mpi_type, MPI_COMM_WORLD);
        send_buffer.clear();
        send_buffer.shrink_to_fit();
        for (int pe = 0; pe < size; ++pe) {
            if (recv_counts[pe] > 0) {
                recv_buffers[pe].insert(recv_buffers[pe].end(), recv_buffer.begin() + recv_displs[pe], recv_buffer.begin() + recv_displs[pe] + recv_counts[pe]);
                stats.received_messages++;
                stats.receive_volume += recv_counts[pe];
            }
        }
        recv_buffer.clear();
        recv_buffer.shrink_to_fit();
    }
};

template<typename T>
class BufferedCommunicator {
public:
    BufferedCommunicator(size_t threshold, MPI_Datatype mpi_type, PEID rank, PEID size, int message_tag, bool empty_pending_buffers_on_overflow = false):
        message_tag_(message_tag), mpi_type(mpi_type),
        rank_(rank), size_(size),
        send_buffers_(size), overflow_buffers_(size),
        recv_buffers_(size), requests_(size, MPI_REQUEST_NULL), threshold_(threshold),
        empty_pending_buffers_on_overflow(empty_pending_buffers_on_overflow) {
        send_buffers_.set_empty_key(-1);
        overflow_buffers_.set_empty_key(-1);
        recv_buffers_.set_empty_key(-1);
    }

    template<typename MessageFunc>
    inline void add_message(const std::vector<T>& message, PEID recv, MessageFunc on_message, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer) {
        if (!empty_pending_buffers_on_overflow) {
            if (!overflow_buffers_[recv].empty()) {
                //we haven't finished sending yet
                wait_and_receive(recv, on_message, stats, timer);
            }
        }
        timer.start("Copying message");
        for (const T& elem : message) {
            send_buffers_[recv].emplace_back(elem);
        }
        timer.stop();
        if (send_buffers_[recv].size() > threshold_) {
            if (empty_pending_buffers_on_overflow) {
                if (!overflow_buffers_[recv].empty()) {
                    //we haven't finished sending yet
                    wait_and_receive(recv, on_message, stats, timer);
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

    template<typename MessageFunc>
    inline void wait_and_receive(PEID recv, MessageFunc on_message, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer) {
        assert(requests_[recv] != MPI_REQUEST_NULL);
        int buffer_sent = 0;
        while(buffer_sent == 0) {
            check_for_message(on_message, stats, timer);
            MPI_Test(&requests_[recv], &buffer_sent, MPI_STATUS_IGNORE);
        }
        // the overflow buffer has been completely sent, we can clear it
        overflow_buffers_[recv].clear();
        assert(requests_[recv] == MPI_REQUEST_NULL);
    }
    template<typename MessageFunc>
    inline void finish_overflow_sending(MessageFunc on_message, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer) {
        if (threshold_ == std::numeric_limits<size_t>::max()) {
            return;
        }
        int isend_done = 0;
        while (isend_done == 0) {
            check_for_message(on_message, stats, timer);
            isend_done = 0;
            MPI_Testall(requests_.size(), requests_.data(), &isend_done, MPI_STATUSES_IGNORE);
        }
        // the overflow buffer has been completely sent, we can clear it
        for (auto& kv : overflow_buffers_) {
            kv.second.clear();
        }
    }

    template<typename MessageFunc>
    inline void busy_waiting_for_receival(MessageFunc on_message, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer) {
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
                    timer.start("Receiving");
                    int message_length;
                    MPI_Get_count(&status, mpi_type, &message_length);
                    assert(message_length > 0);
                    std::vector<T> message(message_length);
                    MPI_Status rst;
                    MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &rst);
                    stats.receive_volume += message_length;
                    stats.received_messages++;
                    on_message(status.MPI_SOURCE, message);
                    timer.start("a2a");
                }
            }
            MPI_Status test_status;
            MPI_Test(&barrier_request, &ibarrier_done, &test_status);
        }
    }

    template<typename MessageFunc>
    inline void check_for_message(MessageFunc on_message, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer) {
        if (threshold_ == std::numeric_limits<size_t>::max()) {
            return;
        }
        timer.start("Checking for messages");
        int iprobe_success = 1;
        while (iprobe_success > 0) {
            iprobe_success = 0;
            MPI_Status status;
            int tag = message_tag_ * size_ + rank_;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
            if (iprobe_success > 0) {
                timer.start("Receiving");
                int message_length;
                MPI_Get_count(&status, mpi_type, &message_length);
                assert(message_length > 0);
                std::vector<T> message(message_length);
                MPI_Status rst;
                MPI_Recv(message.data(), message_length, mpi_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &rst);
                stats.receive_volume += message_length;
                stats.received_messages++;
                timer.stop();
                on_message(status.MPI_SOURCE, message);
                timer.start("Checking for messages");
            }
        }
        timer.stop();
    }

    template<typename MessageFunc>
    inline void all_to_all(MessageFunc on_message, cetric::profiling::MessageStatistics& stats, tlx::MultiTimer& timer, bool full_all_to_all = false) {
        //TODO: ensure that old buffers have been sent

        finish_overflow_sending(on_message, stats, timer);
        timer.start("a2a");
        if (full_all_to_all) {
            busy_waiting_for_receival(on_message, stats, timer);
            CommunicationUtility::all_to_all(send_buffers_, recv_buffers_, mpi_type, rank_, size_, stats, timer, message_tag_);
        } else {
            CommunicationUtility::sparse_all_to_all(send_buffers_, recv_buffers_, mpi_type, rank_, size_, stats, timer, message_tag_);
        }
        timer.stop();
        for (auto& kv : recv_buffers_) {
            std::vector<NodeId>& buffer = kv.second;
            on_message(kv.first, buffer);
            buffer.clear();
        }
        timer.stop();
    }
private:
    template<typename V>
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

#endif //PARALLEL_TRIANGLE_COUNTER_COMM_UTILS_H
