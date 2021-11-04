#ifndef MESSAGE_STATISTICS_H
#define MESSAGE_STATISTICS_H

#include <mpi.h>
#include <nlohmann/json.hpp>

namespace cetric {
    namespace profiling {

        struct MessageStatistics {
            size_t sent_messages;
            size_t received_messages;
            size_t send_volume;
            size_t receive_volume;

            explicit MessageStatistics()
                : sent_messages(0), received_messages(0), send_volume(0),
                receive_volume(0) {}

            template <class Archive> void serialize(Archive &archive) {
                archive(sent_messages, received_messages, send_volume, receive_volume);
            }

            static MPI_Datatype get_mpi_type() {
                MPI_Datatype type;
                MPI_Type_contiguous(4, MPI_UNSIGNED_LONG_LONG, &type);
                return type;
            }
        };

        NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MessageStatistics, sent_messages,
                                           received_messages, send_volume,
                                           receive_volume);

    }
}

#endif /* MESSAGE_STATISTICS_H */
