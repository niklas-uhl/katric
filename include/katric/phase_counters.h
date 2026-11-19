// The work counters of the local and global phase, and the message queue's threshold: shared by every graph type.
#pragma once

#include <cstddef>
#include <limits>

#include "katric/caliper.h"
#include "katric/config.h"

namespace katric {

/// Work counters of the local phase.
struct LocalPhaseCounters {
    std::size_t skipped_nodes           = 0;
    std::size_t wedge_checks            = 0;
    std::size_t intersection_size_local = 0;
    std::size_t local_triangles         = 0;

    /// Adds the counts to the open Caliper region.
    void report() const {
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_local", intersection_size_local);
        profiling::counter_add("local_triangles", local_triangles);
    }
};

/// The message queue's buffer threshold (`--threshold`, `--threshold-scale`).
template <typename Graph>
std::size_t get_threshold(Graph const& G, Config const& conf) {
    switch (conf.threshold) {
        case Threshold::local_nodes:
            return conf.threshold_scale * G.local_node_count();
        case Threshold::local_edges:
            return conf.threshold_scale * G.local_edge_count();
        case Threshold::none:
            return std::numeric_limits<std::size_t>::max();
    }
    return 0;
}

/// Work counters of the global phase.
struct GlobalPhaseCounters {
    std::size_t skipped_nodes            = 0;
    std::size_t interface_nodes          = 0;
    std::size_t wedge_checks             = 0;
    std::size_t intersection_size_global = 0;
    std::size_t type3_triangles          = 0;

    /// Adds the counts and the message statistics of the phase's queue to the open Caliper region.
    template <typename MessageStatistics>
    void report(MessageStatistics const& messages) const {
        profiling::counter_add("skipped_nodes", skipped_nodes);
        profiling::counter_add("interface_nodes", interface_nodes);
        profiling::counter_add("wedge_checks", wedge_checks);
        profiling::counter_add("intersection_size_global", intersection_size_global);
        profiling::counter_add("type3_triangles", type3_triangles);
        profiling::counter_add("sent_messages", messages.sent_messages);
        profiling::counter_add("send_volume", messages.send_volume);
        profiling::counter_add("received_messages", messages.received_messages);
        profiling::counter_add("receive_volume", messages.receive_volume);
    }
};

} // namespace katric
