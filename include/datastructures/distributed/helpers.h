#pragma once

#include "datastructures/graph_definitions.h"
#include <tuple>
#include <vector>

#include <mpi.h>
namespace cetric {
using NodeId = graph::NodeId;

void gather_PE_ranges(
    NodeId                                  local_from,
    NodeId                                  local_to,
    std::vector<std::pair<NodeId, NodeId>>& ranges,
    const MPI_Comm&                         comm,
    PEID                                    rank,
    PEID                                    size
) {
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
                throw std::runtime_error(
                    "[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", "
                    + std::to_string(range.second) + "] is invalid"
                );
            }
            if (range.first > next_expected) {
                throw std::runtime_error(
                    "[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", "
                    + std::to_string(range.second) + "] has a gap to previous one: ["
                    + std::to_string(ranges[i - 1].first) + ", " + std::to_string(ranges[i - 1].second) + "]"
                );
            }
            if (range.first < next_expected) {
                throw std::runtime_error(
                    "[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", "
                    + std::to_string(range.second) + "] overlaps with previous one: ["
                    + std::to_string(ranges[i - 1].first) + ", " + std::to_string(ranges[i - 1].second) + "]"
                );
            }
            next_expected = range.second;
        }
    }
#endif
    MPI_Type_free(&MPI_RANGE);
}

template <bool binary_search>
inline PEID get_PE_from_node_ranges(NodeId node, std::vector<std::pair<NodeId, NodeId>> const& ranges) {
    std::vector<std::pair<NodeId, NodeId>>::const_iterator it;
    if constexpr (binary_search) {
        it = std::upper_bound(
            ranges.begin(),
            ranges.end(),
            node,
            [](NodeId const& value, std::pair<NodeId, NodeId> const& elem) { return value < elem.second; }
        );
    } else {
        it = std::find_if(ranges.begin(), ranges.end(), [node](std::pair<NodeId, NodeId> const& elem) {
            return node >= elem.first && node < elem.second;
        });
    }
    KASSERT(it != ranges.end(), "Node " << node << " not assigned to any PE");
    size_t rank = it - ranges.begin();
    return rank;
}

} // namespace cetric
