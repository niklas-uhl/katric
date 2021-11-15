#ifndef STATISTICS_H_PLFYJDX0
#define STATISTICS_H_PLFYJDX0

#include <communicator.h>
#include <message_statistics.h>
#include <mpi.h>
#include <util.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <istream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "cereal/cereal.hpp"

namespace cetric {
namespace profiling {

struct PreprocessingStatistics {
    double orientation_time;
    double sorting_time;
    MessageStatistics message_statistics;

    explicit PreprocessingStatistics() : orientation_time(0), sorting_time(0), message_statistics() {}

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(orientation_time), CEREAL_NVP(sorting_time), CEREAL_NVP(message_statistics));
    }
};

struct LoadBalancingStatistics {
    double cost_function_evaluation_time;
    double cost_function_communication_time;
    double computation_time;
    double redistribution_time;
    double phase_time;
    MessageStatistics message_statistics;

    explicit LoadBalancingStatistics()
        : cost_function_evaluation_time(0),
          cost_function_communication_time(0),
          computation_time(0),
          redistribution_time(0),
          phase_time(0),
          message_statistics() {}

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(cost_function_evaluation_time), CEREAL_NVP(cost_function_communication_time),
                CEREAL_NVP(computation_time), CEREAL_NVP(redistribution_time), CEREAL_NVP(phase_time),
                CEREAL_NVP(message_statistics));
    }
};

struct Statistics {
    struct LocalStatistics {
        double io_time = 0;
        PreprocessingStatistics preprocessing;
        LoadBalancingStatistics primary_load_balancing;
        LoadBalancingStatistics secondary_load_balancing;
        double local_phase_time = 0;
        double contraction_time = 0;
        double global_phase_time = 0;
        double reduce_time = 0;
        MessageStatistics message_statistics;
        size_t local_triangles = 0;
        size_t type3_triangles = 0;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(CEREAL_NVP(io_time), CEREAL_NVP(preprocessing), CEREAL_NVP(primary_load_balancing),
                    CEREAL_NVP(secondary_load_balancing), CEREAL_NVP(local_phase_time), CEREAL_NVP(contraction_time),
                    CEREAL_NVP(global_phase_time), CEREAL_NVP(reduce_time), CEREAL_NVP(message_statistics),
                    CEREAL_NVP(local_triangles), CEREAL_NVP(type3_triangles));
        }
    };
    LocalStatistics local;
    std::vector<LocalStatistics> local_statistics;
    double global_wall_time;
    size_t triangles;
    size_t counted_triangles;
    PEID size;
    PEID rank;

    Statistics() {
        PEID rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        Statistics(rank, size);
    }

    Statistics(PEID rank, PEID size) : local_statistics(size), triangles(0), size(size), rank(rank) {}

    void reduce(PEID root = 0) {
        std::stringstream stream(std::ios::binary | std::ios::in | std::ios::out);
        {
            cereal::BinaryOutputArchive ar(stream);
            ar(local);
        }
        const auto send_buffer = stream.str();
        auto [recv_buffer, displs] = CommunicationUtility::gather(send_buffer, MPI_CHAR, MPI_COMM_WORLD, 0, rank, size);
        if (rank == root) {
            for (PEID i = 0; i < size; i++) {
                std::stringstream ss;
                auto buffer_size = displs[i + 1] - displs[i];
                ss.write(recv_buffer.data() + displs[i], buffer_size);
                {
                    cereal::BinaryInputArchive ar(ss);
                    ar(local_statistics[i]);
                }
                triangles += local_statistics[i].local_triangles;
                triangles += local_statistics[i].type3_triangles;
            }
        }
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(local_statistics), CEREAL_NVP(global_wall_time), CEREAL_NVP(triangles),
                CEREAL_NVP(counted_triangles));
    }
};
} /* end of namespace profiling */
}  // namespace cetric

#endif /* end of include guard: STATISTICS_H_PLFYJDX0 */
