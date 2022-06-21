#pragma once

#include <mpi.h>
#include <tuple>
#include <vector>
#include "datastructures/graph_definitions.h"
namespace cetric {
using NodeId = graph::NodeId;

void gather_PE_ranges(NodeId local_from,
                      NodeId local_to,
                      std::vector<std::pair<NodeId, NodeId>>& ranges,
                      const MPI_Comm& comm,
                      PEID rank,
                      PEID size) {
    (void)rank;
    (void)size;
    MPI_Datatype MPI_RANGE;
    MPI_Type_vector(1, 2, 0, MPI_NODE, &MPI_RANGE);
    MPI_Type_commit(&MPI_RANGE);
    std::pair<NodeId, NodeId> local_range(local_from, local_to);
    MPI_Allgather(&local_range, 1, MPI_RANGE, ranges.data(), 1, MPI_RANGE, comm);
#ifdef CHECK_RANGES
    if (rank == 0) {
        NodeId next_expected = 0;
        for (size_t i = 0; i < ranges.size(); ++i) {
            std::pair<NodeId, NodeId>& range = ranges[i];
            if (range.first == range.second) {
                continue;
            }
            if (range.first > range.second) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " +
                                         std::to_string(range.second) + "] is invalid");
            }
            if (range.first > next_expected) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " +
                                         std::to_string(range.second) + "] has a gap to previous one: [" +
                                         std::to_string(ranges[i - 1].first) + ", " +
                                         std::to_string(ranges[i - 1].second) + "]");
            }
            if (range.first < next_expected) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " +
                                         std::to_string(range.second) + "] overlaps with previous one: [" +
                                         std::to_string(ranges[i - 1].first) + ", " +
                                         std::to_string(ranges[i - 1].second) + "]");
            }
            next_expected = range.second;
        }
    }
#endif
    MPI_Type_free(&MPI_RANGE);
}

PEID get_PE_from_node_ranges(NodeId node, const std::vector<std::pair<NodeId, NodeId>>& ranges) {
    NodeId local_from;
    NodeId local_to;
    for (size_t i = 0; i < ranges.size(); ++i) {
        std::tie(local_from, local_to) = ranges[i];
        if (local_from <= node && node < local_to) {
            return i;
        }
    }
    std::stringstream out;
    out << "Node " << node << " not assigned to any PE";
    throw std::runtime_error(out.str());
}

}  // namespace cetric
