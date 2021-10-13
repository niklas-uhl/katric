#ifndef STATISTICS_H_PLFYJDX0
#define STATISTICS_H_PLFYJDX0

#include <string>
#include <vector>
#include <util.h>
#include <nlohmann/json.hpp>
#include <mpi.h>
#include <numeric>

namespace cetric {
    namespace profiling {

struct MessageStatistics {
    size_t sent_messages;
    size_t received_messages;
    size_t send_volume;
    size_t receive_volume;

    explicit MessageStatistics(): sent_messages(0), received_messages(0), send_volume(0), receive_volume(0) { }

    static MPI_Datatype get_mpi_type() {
        MPI_Datatype type;
        MPI_Type_contiguous(4, MPI_UNSIGNED_LONG_LONG, &type);
        return type;
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MessageStatistics, sent_messages, received_messages, send_volume, receive_volume)

struct PreprocessingStatistics {
    double orientation_time;
    double sorting_time;
    MessageStatistics message_statistics;

    explicit PreprocessingStatistics(): orientation_time(0), sorting_time(0), message_statistics() { }

    static MPI_Datatype get_mpi_type() {
        MPI_Datatype type;
        int lengths[3] = { 1, 1, 1 };

        MPI_Aint displacements[3];

        PreprocessingStatistics dummy;
        MPI_Aint base_address;
        MPI_Get_address(&dummy, &base_address);
        MPI_Get_address(&dummy.orientation_time, &displacements[0]);
        MPI_Get_address(&dummy.sorting_time, &displacements[1]);
        MPI_Get_address(&dummy.message_statistics, &displacements[2]);

        displacements[0] = MPI_Aint_diff(displacements[0], base_address);
        displacements[1] = MPI_Aint_diff(displacements[1], base_address);
        displacements[2] = MPI_Aint_diff(displacements[2], base_address);

        MPI_Datatype stats_type = MessageStatistics::get_mpi_type();
        MPI_Type_commit(&stats_type);
        MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, stats_type };
        MPI_Type_create_struct(3, lengths, displacements, types, &type);;

        return type;
    }

};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PreprocessingStatistics, orientation_time, sorting_time, message_statistics)

struct LoadBalancingStatistics {
    double computation_time;
    double redistribution_time;
    MessageStatistics message_statistics;

    explicit LoadBalancingStatistics(): computation_time(0), redistribution_time(0), message_statistics() { }

    static MPI_Datatype get_mpi_type() {
        MPI_Datatype type;
        int lengths[3] = { 1, 1, 1 };

        MPI_Aint displacements[3];

        LoadBalancingStatistics dummy;
        MPI_Aint base_address;
        MPI_Get_address(&dummy, &base_address);
        MPI_Get_address(&dummy.computation_time, &displacements[0]);
        MPI_Get_address(&dummy.redistribution_time, &displacements[1]);
        MPI_Get_address(&dummy.message_statistics, &displacements[2]);

        displacements[0] = MPI_Aint_diff(displacements[0], base_address);
        displacements[1] = MPI_Aint_diff(displacements[1], base_address);
        displacements[2] = MPI_Aint_diff(displacements[2], base_address);

        MPI_Datatype stats_type = MessageStatistics::get_mpi_type();
        MPI_Type_commit(&stats_type);
        MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, stats_type };
        MPI_Type_create_struct(3, lengths, displacements, types, &type);;

        return type;
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LoadBalancingStatistics, computation_time, redistribution_time, message_statistics)

struct Statistics {
    struct LocalStatistics {
        double io_time = 0;
        PreprocessingStatistics preprocessing;
        LoadBalancingStatistics load_balancing;
        double local_phase_time = 0;
        double contraction_time = 0;
        double global_phase_time = 0;
        MessageStatistics message_statistics;
        size_t local_triangles = 0;
        size_t type3_triangles = 0;

        static MPI_Datatype get_mpi_type() {
            MPI_Datatype type;
            int lengths[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            MPI_Aint displacements[9];

            LocalStatistics dummy;
            MPI_Aint base_address;
            MPI_Get_address(&dummy, &base_address);
            MPI_Get_address(&dummy.io_time, &displacements[0]);
            MPI_Get_address(&dummy.preprocessing, &displacements[1]);
            MPI_Get_address(&dummy.load_balancing, &displacements[2]);
            MPI_Get_address(&dummy.local_phase_time, &displacements[3]);
            MPI_Get_address(&dummy.contraction_time, &displacements[4]);
            MPI_Get_address(&dummy.global_phase_time, &displacements[5]);
            MPI_Get_address(&dummy.message_statistics, &displacements[6]);
            MPI_Get_address(&dummy.local_triangles, &displacements[7]);
            MPI_Get_address(&dummy.type3_triangles, &displacements[8]);

            for (int i = 0; i < 9; ++i) {
                displacements[i] = MPI_Aint_diff(displacements[i], base_address);
            }

            MPI_Datatype preprocessing_type = PreprocessingStatistics::get_mpi_type();
            MPI_Type_commit(&preprocessing_type);
            MPI_Datatype loadbalancing_type = LoadBalancingStatistics::get_mpi_type();
            MPI_Type_commit(&loadbalancing_type);
            MPI_Datatype stats_type = MessageStatistics::get_mpi_type();
            MPI_Type_commit(&stats_type);

            MPI_Datatype types[9] = { 
                MPI_DOUBLE,
                preprocessing_type,
                loadbalancing_type,
                MPI_DOUBLE,
                MPI_DOUBLE,
                MPI_DOUBLE,
                stats_type,
                MPI_UNSIGNED_LONG_LONG,
                MPI_UNSIGNED_LONG_LONG,
            };
            MPI_Type_create_struct(9, lengths, displacements, types, &type);

            return type;
        }
    };
    LocalStatistics local;
    std::vector<double> io_time;
    std::vector<PreprocessingStatistics> preprocessing;
    std::vector<LoadBalancingStatistics> load_balancing;
    std::vector<double> local_phase_time;
    std::vector<double> contraction_time;
    std::vector<double> global_phase_time;
    std::vector<MessageStatistics> message_statistics;
    std::vector<size_t> local_triangles;
    std::vector<size_t> type3_triangles;
    double global_wall_time;
    size_t triangles;
    size_t counted_triangles;
    PEID size;
    PEID rank;

    Statistics(PEID rank, PEID size):      io_time(size), preprocessing(size), load_balancing(size),
                                local_phase_time(size), contraction_time(size), global_phase_time(size),
                                message_statistics(size), local_triangles(size), type3_triangles(size), 
                                triangles(0), size(size), rank(rank) { }

    void reduce(PEID root = 0) {
        MPI_Datatype stats_type = LocalStatistics::get_mpi_type();
        MPI_Type_commit(&stats_type);
        
        std::vector<LocalStatistics> recv_buffer;
        if (rank == root) {
            recv_buffer.resize(size);
        }
        MPI_Gather(&local, 1, stats_type, recv_buffer.data(), 1, stats_type, root, MPI_COMM_WORLD);
        if (rank == root) {
            for (PEID i = 0; i < size; ++i) {
                io_time[i] = recv_buffer[i].io_time;
                preprocessing[i] = recv_buffer[i].preprocessing;
                load_balancing[i] = recv_buffer[i].load_balancing;
                local_phase_time[i] = recv_buffer[i].local_phase_time;
                contraction_time[i] = recv_buffer[i].contraction_time;
                global_phase_time[i] = recv_buffer[i].global_phase_time;
                message_statistics[i] = recv_buffer[i].message_statistics;
                local_triangles[i] = recv_buffer[i].local_triangles;
                type3_triangles[i] = recv_buffer[i].type3_triangles;
            }
        }
        triangles = std::accumulate(local_triangles.begin(), local_triangles.end(), 0);
        triangles += std::accumulate(type3_triangles.begin(), type3_triangles.end(), 0);
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Statistics, io_time, preprocessing, load_balancing, local_phase_time, contraction_time, global_phase_time, message_statistics, local_triangles, type3_triangles, global_wall_time, triangles, counted_triangles)

    } /* end of namespace profiling */
} /* end of namesapce cetric */

#endif /* end of include guard: STATISTICS_H_PLFYJDX0 */
