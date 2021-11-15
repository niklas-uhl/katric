#ifndef MESSAGE_STATISTICS_H
#define MESSAGE_STATISTICS_H

#include <mpi.h>
#include <cereal/cereal.hpp>
#include <cstddef>

namespace cetric {
namespace profiling {

struct MessageStatistics {
    size_t sent_messages;
    size_t received_messages;
    size_t send_volume;
    size_t receive_volume;

    explicit MessageStatistics() : sent_messages(0), received_messages(0), send_volume(0), receive_volume(0) {}

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(sent_messages), CEREAL_NVP(received_messages), CEREAL_NVP(send_volume),
                CEREAL_NVP(receive_volume));
    }

    void add(const MessageStatistics& rhs) {
        sent_messages += rhs.sent_messages;
        received_messages += rhs.received_messages;
        send_volume += rhs.send_volume;
        receive_volume += rhs.receive_volume;
    }
};
}  // namespace profiling
}  // namespace cetric

#endif /* MESSAGE_STATISTICS_H */
