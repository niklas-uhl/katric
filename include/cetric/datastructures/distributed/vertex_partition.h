// The partition of the global vertex ids into one contiguous range per rank. Copied from KaCCv2's
// kacc::VertexPartition (src/kacc/graph/vertex_partition.hpp) without the split-vertex parts (katric has none), and
// with empty ranks placed at their predecessor's end: katric gives a rank without vertices the range (max, max).
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include <kamping/v2/collectives/allgather.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/sentinels.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>

namespace cetric::graph {

template <typename V>
class VertexPartition {
public:
    using VId         = V;
    using VertexRange = std::pair<VId, VId>;

    VertexPartition() = default;

    /// Collective. `vertex_range` is this rank's half-open range of global ids; an empty one (first >= second) may
    /// have any value.
    VertexPartition(VertexRange vertex_range, MPI_Comm comm) {
        int rank = 0;
        int size = 0;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        comm_rank_ = rank;
        bool const      empty = vertex_range.first >= vertex_range.second;
        std::vector<VId> ranges(2 * static_cast<std::size_t>(size));
        ranges[2 * static_cast<std::size_t>(rank)]     = vertex_range.first;
        ranges[2 * static_cast<std::size_t>(rank) + 1] = empty ? vertex_range.first : vertex_range.second;
        kamping::v2::allgather(kamping::v2::inplace, ranges, kamping::v2::comm_view{comm});
        // vtx_dist_[r] is the first id of rank r, vtx_dist_[size] the number of vertices. An empty rank starts and ends
        // where its predecessor ends.
        vtx_dist_.assign(static_cast<std::size_t>(size) + 1, VId{0});
        for (std::size_t r = 0; r < static_cast<std::size_t>(size); ++r) {
            bool const r_empty = ranges[2 * r] >= ranges[2 * r + 1];
            vtx_dist_[r + 1]   = r_empty ? vtx_dist_[r] : ranges[2 * r + 1];
            KASSERT(r_empty || ranges[2 * r] == vtx_dist_[r], "the vertex ranges have a gap or overlap at rank " << r);
        }
        vertex_range_ = partition_range(rank);
    }

    /// This rank's range of global ids.
    [[nodiscard]] VertexRange vertex_range() const {
        return vertex_range_;
    }

    [[nodiscard]] int num_ranks() const {
        return static_cast<int>(vtx_dist_.size()) - 1;
    }

    /// The range of `rank`.
    [[nodiscard]] VertexRange partition_range(int rank) const {
        KASSERT(rank >= 0 && static_cast<std::size_t>(rank) + 1 < vtx_dist_.size());
        return {vtx_dist_[static_cast<std::size_t>(rank)], vtx_dist_[static_cast<std::size_t>(rank) + 1]};
    }

    [[nodiscard]] bool is_local(VId v) const {
        return vertex_range_.first <= v && v < vertex_range_.second;
    }

    /// The rank owning `v`: this rank without a lookup, otherwise a binary search over the ranges.
    [[nodiscard]] int get_rank(VId v) const {
        if (is_local(v)) {
            return comm_rank_;
        }
        auto it = std::ranges::upper_bound(vtx_dist_, v);
        KASSERT(it != vtx_dist_.begin() && it != vtx_dist_.end(), "vertex " << v << " out of global bounds");
        return static_cast<int>(std::distance(vtx_dist_.begin(), it)) - 1;
    }

    /// The rank owning `v`, searching only from rank `from` on (`v` must not lie before rank `from`'s range): for
    /// ascending lookups, e.g. along a sorted neighborhood.
    [[nodiscard]] int get_rank_from(VId v, int from) const {
        KASSERT(from >= 0 && static_cast<std::size_t>(from) < vtx_dist_.size() && vtx_dist_[static_cast<std::size_t>(from)] <= v);
        auto it = std::upper_bound(vtx_dist_.begin() + from, vtx_dist_.end(), v);
        KASSERT(it != vtx_dist_.end(), "vertex " << v << " out of global bounds");
        return static_cast<int>(std::distance(vtx_dist_.begin(), it)) - 1;
    }

private:
    VertexRange      vertex_range_{};
    std::vector<VId> vtx_dist_;
    int              comm_rank_ = 0;
};

} // namespace cetric::graph
