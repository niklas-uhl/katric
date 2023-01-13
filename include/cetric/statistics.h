#ifndef STATISTICS_H_PLFYJDX0
#define STATISTICS_H_PLFYJDX0

#include <cstddef>
#include <istream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/vector.hpp>
#include <message-queue/message_statistics.h>
#include <mpi.h>
#include <tlx/multi_timer.hpp>

#include "cetric/communicator.h"
#include "cetric/util.h"

namespace cereal {
template <class Archive>
void save(Archive& archive, message_queue::MessageStatistics const& stats) {
    archive(
        cereal::make_nvp("sent_messages", stats.sent_messages.load()),
        cereal::make_nvp("received_messages", stats.received_messages.load()),
        cereal::make_nvp("send_volume", stats.send_volume.load()),
        cereal::make_nvp("receive_volume", stats.receive_volume.load())
    );
}

template <class Archive>
void load(Archive& archive, message_queue::MessageStatistics& stats) {
    size_t sent_messages;
    size_t received_messages;
    size_t send_volume;
    size_t receive_volume;
    archive(
        cereal::make_nvp("sent_messages", sent_messages),
        cereal::make_nvp("received_messages", received_messages),
        cereal::make_nvp("send_volume", send_volume),
        cereal::make_nvp("receive_volume", receive_volume)
    );
    stats.sent_messages.store(sent_messages);
    stats.received_messages.store(received_messages);
    stats.send_volume.store(send_volume);
    stats.receive_volume.store(receive_volume);
}
} // namespace cereal

namespace cetric {
namespace profiling {

struct PreprocessingStatistics {
    double            orientation_time;
    double            sorting_time;
    double            degree_exchange_time;
    MessageStatistics message_statistics;

    explicit PreprocessingStatistics() : orientation_time(0), sorting_time(0), message_statistics() {}

    void ingest(tlx::MultiTimer const& timer) {
        tlx::MultiTimer timer_copy = timer;
        degree_exchange_time       = timer_copy.get("degree_exchange");
        orientation_time           = timer_copy.get("orientation");
        sorting_time               = timer_copy.get("sorting");
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(degree_exchange_time),
            CEREAL_NVP(orientation_time),
            CEREAL_NVP(sorting_time),
            CEREAL_NVP(message_statistics)
        );
    }
};

struct LoadBalancingStatistics {
    double            cost_function_evaluation_time;
    double            cost_function_communication_time;
    double            computation_time;
    double            redistribution_time;
    double            phase_time;
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
        archive(
            CEREAL_NVP(cost_function_evaluation_time),
            CEREAL_NVP(cost_function_communication_time),
            CEREAL_NVP(computation_time),
            CEREAL_NVP(redistribution_time),
            CEREAL_NVP(phase_time),
            CEREAL_NVP(message_statistics)
        );
    }
};

struct Statistics {
    struct LocalStatistics {
        LocalStatistics() {}
        explicit LocalStatistics(const LocalStatistics& rhs) {
            io_time                    = rhs.io_time;
            preprocessing_local_phase  = rhs.preprocessing_local_phase;
            preprocessing_global_phase = rhs.preprocessing_global_phase;
            primary_load_balancing     = rhs.primary_load_balancing;
            secondary_load_balancing   = rhs.secondary_load_balancing;
            local_phase_time           = rhs.local_phase_time;
            contraction_time           = rhs.contraction_time;
            global_phase_time          = rhs.global_phase_time;
            reduce_time                = rhs.reduce_time;
            preprocessing_time         = rhs.preprocessing_time;
            preprocessing_global_time  = rhs.preprocessing_global_time;
            message_statistics         = rhs.message_statistics;
            skipped_nodes              = rhs.skipped_nodes.load();
            wedge_checks               = rhs.wedge_checks.load();
            nodes_parallel2d           = rhs.nodes_parallel2d.load();
            local_triangles            = rhs.local_triangles.load();
            type3_triangles            = rhs.type3_triangles.load();
            global_phase_threshold     = rhs.global_phase_threshold;
            local_time                 = rhs.local_time;
            local_wall_time            = rhs.local_wall_time;
        }
        LocalStatistics& operator=(const LocalStatistics& rhs) {
            io_time                    = rhs.io_time;
            preprocessing_local_phase  = rhs.preprocessing_local_phase;
            preprocessing_global_phase = rhs.preprocessing_global_phase;
            primary_load_balancing     = rhs.primary_load_balancing;
            secondary_load_balancing   = rhs.secondary_load_balancing;
            local_phase_time           = rhs.local_phase_time;
            contraction_time           = rhs.contraction_time;
            global_phase_time          = rhs.global_phase_time;
            reduce_time                = rhs.reduce_time;
            preprocessing_time         = rhs.preprocessing_time;
            preprocessing_global_time  = rhs.preprocessing_global_time;
            message_statistics         = rhs.message_statistics;
            skipped_nodes              = rhs.skipped_nodes.load();
            wedge_checks               = rhs.wedge_checks.load();
            nodes_parallel2d           = rhs.nodes_parallel2d.load();
            local_triangles            = rhs.local_triangles.load();
            type3_triangles            = rhs.type3_triangles.load();
            global_phase_threshold     = rhs.global_phase_threshold;
            local_time                 = rhs.local_time;
            local_wall_time            = rhs.local_wall_time;
            return *this;
        }
        PEID                             rank;
        double                           io_time = 0;
        PreprocessingStatistics          preprocessing_local_phase;
        PreprocessingStatistics          preprocessing_global_phase;
        LoadBalancingStatistics          primary_load_balancing;
        LoadBalancingStatistics          secondary_load_balancing;
        double                           ghost_rank_gather         = 0;
        double                           local_phase_time          = 0;
        double                           contraction_time          = 0;
        double                           global_phase_time         = 0;
        double                           reduce_time               = 0;
        double                           preprocessing_time        = 0;
        double                           preprocessing_global_time = 0;
        message_queue::MessageStatistics message_statistics;
        std::atomic<size_t>              skipped_nodes            = 0;
        std::atomic<size_t>              wedge_checks             = 0;
        std::atomic<size_t>              intersection_size_local  = 0;
        std::atomic<size_t>              intersection_size_global = 0;
        std::atomic<size_t>              nodes_parallel2d         = 0;
        std::atomic<size_t>              local_triangles          = 0;
        std::atomic<size_t>              type3_triangles          = 0;
        size_t                           global_phase_threshold   = 0;
        size_t                           wedges                   = 0;
        double                           local_wall_time          = 0;
        double                           local_time               = 0;

        void ingest(tlx::MultiTimer const& timer) {
            tlx::MultiTimer timer_copy          = timer;
            ghost_rank_gather                   = timer_copy.get("ghost_ranks");
            primary_load_balancing.phase_time   = timer_copy.get("primary_load_balancing");
            local_phase_time                    = timer_copy.get("local_phase");
            contraction_time                    = timer_copy.get("contraction");
            secondary_load_balancing.phase_time = timer_copy.get("secondary_load_balancing");
            global_phase_time                   = timer_copy.get("global_phase");
            reduce_time                         = timer_copy.get("reduce");
            preprocessing_time                  = timer_copy.get("preprocessing");
            preprocessing_global_time           = timer_copy.get("preprocessing_global");
            local_time                          = timer.total();
        }

        template <class Archive>
        void save(Archive& archive) const {
            archive(
                CEREAL_NVP(rank),
                CEREAL_NVP(io_time),
                CEREAL_NVP(ghost_rank_gather),
                CEREAL_NVP(preprocessing_local_phase),
                CEREAL_NVP(preprocessing_global_phase),
                CEREAL_NVP(primary_load_balancing),
                CEREAL_NVP(secondary_load_balancing),
                CEREAL_NVP(local_phase_time),
                CEREAL_NVP(contraction_time),
                CEREAL_NVP(global_phase_time),
                CEREAL_NVP(reduce_time),
                CEREAL_NVP(preprocessing_time),
                CEREAL_NVP(preprocessing_global_time),
                CEREAL_NVP(message_statistics),
                CEREAL_NVP(global_phase_threshold),
                CEREAL_NVP(wedges),
                CEREAL_NVP(local_wall_time),
                CEREAL_NVP(local_time)
            );
            archive(
                cereal::make_nvp("skipped_nodes", skipped_nodes.load()),
                cereal::make_nvp("wedge_checks", wedge_checks.load()),
                cereal::make_nvp("intersection_size_local", intersection_size_local.load()),
                cereal::make_nvp("intersection_size_global", intersection_size_global.load()),
                cereal::make_nvp("nodes_parallel2d", nodes_parallel2d.load()),
                cereal::make_nvp("local_triangles", local_triangles.load()),
                cereal::make_nvp("type3_triangles", type3_triangles.load())
            );
        }

        template <class Archive>
        void load(Archive& archive) {
            archive(
                CEREAL_NVP(rank),
                CEREAL_NVP(io_time),
                CEREAL_NVP(ghost_rank_gather),
                CEREAL_NVP(preprocessing_local_phase),
                CEREAL_NVP(preprocessing_global_phase),
                CEREAL_NVP(primary_load_balancing),
                CEREAL_NVP(secondary_load_balancing),
                CEREAL_NVP(local_phase_time),
                CEREAL_NVP(contraction_time),
                CEREAL_NVP(global_phase_time),
                CEREAL_NVP(reduce_time),
                CEREAL_NVP(preprocessing_time),
                CEREAL_NVP(preprocessing_global_time),
                CEREAL_NVP(message_statistics),
                CEREAL_NVP(global_phase_threshold),
                CEREAL_NVP(wedges),
                CEREAL_NVP(local_wall_time),
                CEREAL_NVP(local_time)
            );
            size_t skipped_nodes_tmp;
            size_t wedge_checks_tmp;
            size_t intersection_size_local_tmp;
            size_t intersection_size_global_tmp;
            size_t nodes_parallel2d_tmp;
            size_t local_triangles_tmp;
            size_t type3_triangles_tmp;
            archive(
                cereal::make_nvp("skipped_nodes", skipped_nodes_tmp),
                cereal::make_nvp("wedge_checks", wedge_checks_tmp),
                cereal::make_nvp("intersection_size_local", intersection_size_local_tmp),
                cereal::make_nvp("intersection_size_global", intersection_size_global_tmp),
                cereal::make_nvp("nodes_parallel2d", nodes_parallel2d_tmp),
                cereal::make_nvp("local_triangles", local_triangles_tmp),
                cereal::make_nvp("type3_triangles", type3_triangles_tmp)
            );
            skipped_nodes.store(skipped_nodes_tmp);
            wedge_checks.store(wedge_checks_tmp);
            intersection_size_local.store(intersection_size_local_tmp);
            intersection_size_global.store(intersection_size_global_tmp);
            nodes_parallel2d.store(nodes_parallel2d_tmp);
            local_triangles.store(local_triangles_tmp);
            type3_triangles.store(type3_triangles_tmp);
        }
    };
    LocalStatistics              local;
    std::vector<LocalStatistics> local_statistics;
    double                       global_wall_time  = 0;
    size_t                       triangles         = 0;
    size_t                       counted_triangles = 0;
    PEID                         size;
    PEID                         rank;

    Statistics() {
        PEID rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        Statistics(rank, size);
    }

    Statistics(PEID rank, PEID size) : local_statistics(), triangles(0), size(size), rank(rank) {}

    void reduce(PEID root = 0) {
        std::stringstream stream(std::ios::binary | std::ios::in | std::ios::out);
        {
            cereal::BinaryOutputArchive ar(stream);
            ar(local);
        }
        const auto send_buffer     = stream.str();
        auto [recv_buffer, displs] = CommunicationUtility::gather(send_buffer, MPI_CHAR, MPI_COMM_WORLD, 0, rank, size);
        if (rank == root) {
            local_statistics.resize(size);
            global_wall_time = 0;
            for (PEID i = 0; i < size; i++) {
                std::stringstream ss;
                auto              buffer_size = displs[i + 1] - displs[i];
                ss.write(recv_buffer.data() + displs[i], buffer_size);
                {
                    cereal::BinaryInputArchive ar(ss);
                    ar(local_statistics[i]);
                }
                local_statistics[i].rank = i;
                triangles += local_statistics[i].local_triangles;
                triangles += local_statistics[i].type3_triangles;
                global_wall_time = std::max(local_statistics[i].local_wall_time, global_wall_time);
            }
        }
    }
    void collapse() {
        local_statistics.push_back(local);
        local_statistics[0].rank = rank;
        triangles = local.local_triangles;
        counted_triangles = triangles;
        global_wall_time = local.local_wall_time;
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(local_statistics),
            CEREAL_NVP(global_wall_time),
            CEREAL_NVP(triangles),
            CEREAL_NVP(counted_triangles)
        );
    }
};
} /* end of namespace profiling */
} // namespace cetric

#endif /* end of include guard: STATISTICS_H_PLFYJDX0 */
