#pragma once

#include <mpi.h>
#include <sstream>
#include <type_traits>
#include "counters/cetric_edge_iterator.h"
#include "datastructures/graph_definitions.h"
#include "datastructures/span.h"
#include "message-queue/buffered_queue.h"

namespace cetric {
template <typename T>
T PEID_to_datatype(PEID rank) {
    return rank;
}
template <typename T>
PEID datatype_to_PEID(T const& rank) {
    return rank;
}

template <>
graph::RankEncodedNodeId PEID_to_datatype<graph::RankEncodedNodeId>(PEID rank) {
    return graph::RankEncodedNodeId(0, rank);
}
template <>
PEID datatype_to_PEID<graph::RankEncodedNodeId>(graph::RankEncodedNodeId const& val) {
    return val.rank();
}

using namespace graph;
template <typename T, class Merger, class Splitter>
class IndirectMessageQueue {
public:
    IndirectMessageQueue(Merger&& merger, Splitter&& splitter)
        : rank_(), size_(), queue_(std::move(merger), std::move(splitter)) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        grid_size_ = std::round(std::sqrt(size_));
    };
    void post_message(std::vector<T>&& message, PEID receiver, bool direct_send = false) {
        message.push_back(PEID_to_datatype<T>(rank_));
        message.push_back(PEID_to_datatype<T>(receiver));
        PEID proxy;
        if (direct_send) {
            proxy = receiver;
        } else {
            proxy = get_proxy(rank_, receiver);
        }
        std::stringstream out;
        out << "Redirecting message to " << receiver << " via " << proxy;
        // atomic_debug(out.str());
        queue_.post_message(std::move(message), proxy);
    }

    void set_threshold(size_t threshold) {
        queue_.set_threshold(threshold);
    }

    template <typename MessageHandler>
    void poll(MessageHandler&& on_message) {
        // static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
        //                                   typename std::vector<T>::iterator, PEID>);
        queue_.poll([&](SharedVectorSpan<T> span, PEID /*sender*/) {
            auto begin = span.begin();
            auto end = span.end();
            PEID receiver = datatype_to_PEID(*(end - 1));
            PEID original_sender = datatype_to_PEID(*(end - 2));
            if (receiver == rank_) {
                auto received_message = span.subspan(0, span.size() - 2);
                on_message(received_message, original_sender);
            } else {
                auto proxy = get_proxy(rank_, receiver);
                std::stringstream out;
                out << "Redirecting message to " << receiver << " via " << proxy;
                // atomic_debug(out.str());
                queue_.post_message(std::vector(begin, end), proxy);
            }
        });
    }

    template <typename MessageHandler>
    void terminate(MessageHandler&& on_message) {
        // static_assert(std::is_invocable_v<MessageHandler, typename std::vector<T>::iterator,
        //                                   typename std::vector<T>::iterator, PEID>);
        queue_.terminate([&](SharedVectorSpan<T> span, PEID /*sender*/) {
            auto begin = span.begin();
            auto end = span.end();
            PEID receiver = datatype_to_PEID(*(end - 1));
            PEID original_sender = datatype_to_PEID(*(end - 2));
            if (receiver == rank_) {
                auto received_message = span.subspan(0, span.size() - 2);
                on_message(received_message, original_sender);
            } else {
                auto proxy = get_proxy(rank_, receiver);
                std::stringstream out;
                out << "Redirecting message to " << receiver << " via " << proxy;
                // atomic_debug(out.str());
                queue_.post_message(std::vector(begin, end), proxy);
            }
        });
    }

    size_t overflows() const {
        return queue_.overflows();
    }

    const message_queue::MessageStatistics& stats() {
        return queue_.stats();
    }

    void reset() {
        queue_.reset();
    }

private:
    struct GridPosition {
        int row;
        int column;
        bool operator==(const GridPosition& rhs) {
            return row == rhs.row && column == rhs.column;
        }
    };

    GridPosition get_grid_position(PEID rank) {
        return GridPosition{rank / grid_size_, rank % grid_size_};
    }
    PEID get_rank(GridPosition grid_position) {
        return grid_position.row * grid_size_ + grid_position.column;
    }
    PEID get_proxy(PEID from, PEID to) {
        auto from_pos = get_grid_position(from);
        auto to_pos = get_grid_position(to);
        GridPosition proxy = {from_pos.row, to_pos.column};
        if (get_rank(proxy) >= size_) {
            proxy = {from_pos.column, to_pos.column};
        }
        if (proxy == from_pos) {
            proxy = to_pos;
        }
        assert(get_rank(proxy) < size_);
        return get_rank(proxy);
    }
    PEID rank_;
    PEID size_;
    PEID grid_size_;
    message_queue::BufferedMessageQueue<T, Merger, Splitter> queue_;
};
template <class T, typename Merger, typename Splitter>
auto make_indirect_queue(Merger&& merger, Splitter&& splitter) {
    return IndirectMessageQueue<T, Merger, Splitter>(std::forward<Merger>(merger), std::forward<Splitter>(splitter));
}
}  // namespace cetric
