// The triangles one rank finds over the phases of a run.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <tuple>

#include <caliper/cali.h>
#include <kamping/types/mpi_type_traits.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_vector.h>

#include "katric/datastructures/graph_definitions.h"

namespace katric {

/// Counts the triangles of a rank and, with assertions enabled, records them to check for duplicates (verify()).
class TriangleCounter {
public:
    /// Calls `find(emit)`, where `emit(Triangle)` reports one triangle by global ids (thread-safe), and adds them to the
    /// count.
    template <typename Find>
    void operator()(Find&& find) {
        tbb::combinable<std::size_t> found{0};
        find([this, &found](graph::Triangle<graph::NodeId> t) {
            SPDLOG_DEBUG("{}", t);
            if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
                t.normalize();
                triangles_.push_back(t);
            } else {
                (void)t;
            }
            found.local()++;
        });
        count_ += found.combine(std::plus<>{});
    }

    /// The triangles found on this rank.
    [[nodiscard]] std::size_t count() const {
        return count_;
    }

    /// Checks that no triangle was emitted twice (only with assertions enabled).
    void verify() {
        if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
            KASSERT(triangles_.size() == count_);
            std::sort(triangles_.begin(), triangles_.end(), [](auto const& lhs, auto const& rhs) {
                return std::tuple(lhs.x, lhs.y, lhs.z) < std::tuple(rhs.x, rhs.y, rhs.z);
            });
            auto duplicate = std::adjacent_find(triangles_.begin(), triangles_.end());
            KASSERT(duplicate == triangles_.end(), "Duplicate triangle " << *duplicate);
        }
    }

private:
    std::size_t                                                       count_ = 0;
    tbb::concurrent_vector<graph::Triangle<graph::NodeId>> triangles_;
};

namespace phases {

/// The sum of the ranks' triangle counts, on rank 0.
inline std::size_t reduce(TriangleCounter& triangles, MPI_Comm comm) {
    std::size_t total = 0;
    {
        CALI_CXX_MARK_SCOPE("reduce");
        std::size_t const local = triangles.count();
        MPI_Reduce(&local, &total, 1, kamping::types::mpi_type_traits<std::size_t>::data_type(), MPI_SUM, 0, comm);
    }
    triangles.verify();
    return total;
}

} // namespace phases
} // namespace katric
