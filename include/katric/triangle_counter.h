// The triangles one rank finds over the phases of a run.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <tuple>

#include <caliper/cali.h>
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <tbb/combinable.h>
#include <tbb/concurrent_vector.h>
#include <kamping/v2/collectives/reduce.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/views/ref_single_view.hpp>
#include <kassert/kassert.hpp>

#include "katric/datastructures/graph_definitions.h"

namespace katric {

/// Counts the triangles of a rank and, with assertions enabled, records them to
/// check for duplicates (verify()).
class TriangleCounter {
public:
  /// Calls `find(emit)`, where `emit(Triangle)` reports one triangle by global
  /// ids (thread-safe), and adds them to the count.
  template <typename Find>
  void operator()(Find&& find) {
    tbb::combinable<std::size_t> found{0};
    find([this, &found](graph::Triangle<graph::NodeId> triangle) -> void {
      SPDLOG_DEBUG("{}", triangle);
      if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::normal) {
        triangle.normalize();
        triangles_.push_back(triangle);
      } else {
        (void)triangle;
      }
      found.local()++;
    });
    count_ += found.combine(std::plus<>{});
  }

  /// The triangles found on this rank.
  [[nodiscard]] auto count() const -> std::size_t { return count_; }

  /// Checks that no triangle was emitted twice (only with assertions enabled).
  void verify() {
    if constexpr (KASSERT_ENABLED(kassert::assert::normal)) {
      KASSERT(triangles_.size() == count_);
      std::ranges::sort(triangles_,
                        [](auto const& lhs, auto const& rhs) -> bool {
                          return std::tuple(lhs.x, lhs.y, lhs.z) <
                                 std::tuple(rhs.x, rhs.y, rhs.z);
                        });
      auto duplicate = std::ranges::adjacent_find(triangles_);
      KASSERT(duplicate == triangles_.end(),
              "Duplicate triangle " << *duplicate);
    }
  }

private:
  std::size_t count_ = 0;
  tbb::concurrent_vector<graph::Triangle<graph::NodeId>> triangles_;
};

namespace phases {

/// The sum of the ranks' triangle counts, on rank 0.
inline auto reduce(TriangleCounter& triangles, kamping::v2::comm_view comm)
    -> std::size_t {
  std::size_t total = 0;
  {
    CALI_CXX_MARK_SCOPE("reduce");
    auto const local = triangles.count();
    kamping::v2::reduce(std::views::single(local),
                        kamping::v2::views::ref_single(total), comm);
  }
  triangles.verify();
  return total;
}

}  // namespace phases
}  // namespace katric
