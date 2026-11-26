// Tom2D behind AbstractAlgorithm. Fork: niklas-uhl/2DTriangleCounting, branch
// katric (a static C library with the calls of tom2d.h, mpitc_MapJIK2D split
// at its timer boundary; extern/CMakeLists.txt).
#include "./tom2d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>
#include <mpi.h>
#include <sys/types.h>

#include "./convert.h"
#include "katric/config.h"

#include <tom2d.h>

namespace katric {
namespace {

/// A malloc'd copy of `values` (at least one byte, malloc(0) may give null),
/// for Tom2D, which frees its input.
template <typename T>
auto malloc_copy(std::vector<T> const& values) -> T* {
  auto* copy = static_cast<T*>(
      std::malloc(std::max<std::size_t>(1, values.size() * sizeof(T))));
  if (copy == nullptr) {
    throw std::bad_alloc();
  }
  std::memcpy(copy, values.data(), values.size() * sizeof(T));
  return copy;
}

/// The native input is what Tom2D's readers produce: a 1D-distributed CSR
/// (vtxdist on every rank, local xadj, 64-bit global ids in adjncy) with both
/// directions of every edge. KaGen's own distribution is kept. Tom2D sorts the
/// neighbourhoods itself (in its preprocessing).
///
/// Each run hands Tom2D fresh malloc'd copies (reset(), untimed: it takes
/// ownership and frees them), then tom2d_preprocess() (region
/// `preprocessing`: cyclic redistribution, degree ordering, upper/lower split,
/// transposes; its timer_1) and tom2d_count() (region `counting`: Cannon's
/// sqrt(p) shifts with the kernel and the global sum; its timer_2).
class Tom2dAlgorithm final : public AbstractAlgorithm {
public:
  Tom2dAlgorithm(Config const& /* conf */, kamping::v2::comm_view comm) {
    int result = MPI_UNEQUAL;
    MPI_Comm_compare(comm.mpi_handle(), MPI_COMM_WORLD, &result);
    if (result != MPI_IDENT && result != MPI_CONGRUENT) {
      throw std::runtime_error("--algorithm tom2d runs only on MPI_COMM_WORLD");
    }
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto const root = static_cast<int>(std::lround(std::sqrt(size)));
    if (root * root != size) {
      throw std::runtime_error(
          "--algorithm tom2d needs a perfect square number of ranks, not " +
          std::to_string(size));
    }
  }

  Tom2dAlgorithm(Tom2dAlgorithm const&) = delete;
  Tom2dAlgorithm(Tom2dAlgorithm&&) = delete;
  auto operator=(Tom2dAlgorithm const&) -> Tom2dAlgorithm& = delete;
  auto operator=(Tom2dAlgorithm&&) -> Tom2dAlgorithm& = delete;
  ~Tom2dAlgorithm() override { free_copies(); }

  void ingest(kagen::Graph&& input) override {
    auto const distribution =
        competitors::vertex_distribution(input, MPI_COMM_WORLD);
    vtxdist_.assign(distribution.begin(), distribution.end());
    auto const local_vertices = input.xadj.empty() ? 0 : input.xadj.size() - 1;
    if (local_vertices >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::runtime_error(
          "--algorithm tom2d: Tom2D counts local vertices in 32 bits");
    }
    nvtxs_ = static_cast<std::int32_t>(local_vertices);
    xadj_.assign(input.xadj.begin(), input.xadj.end());
    if (xadj_.empty()) {
      xadj_.push_back(0);
    }
    adjncy_.assign(input.adjncy.begin(), input.adjncy.end());
    input = kagen::Graph{};
  }

  void reset() override {
    free_copies();
    vtxdist_copy_ = malloc_copy(vtxdist_);
    xadj_copy_ = malloc_copy(xadj_);
    adjncy_copy_ = malloc_copy(adjncy_);
    triangle_count_ = 0;
  }

  void run() override {
    tom2d_state* state = nullptr;
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      // takes ownership of the three arrays
      state = tom2d_preprocess(nvtxs_, vtxdist_copy_, xadj_copy_, adjncy_copy_);
      vtxdist_copy_ = nullptr;
      xadj_copy_ = nullptr;
      adjncy_copy_ = nullptr;
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      triangle_count_ = static_cast<std::size_t>(tom2d_count(state));
    }
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  void free_copies() {
    std::free(vtxdist_copy_);
    std::free(xadj_copy_);
    std::free(adjncy_copy_);
    vtxdist_copy_ = nullptr;
    xadj_copy_ = nullptr;
    adjncy_copy_ = nullptr;
  }

  std::int32_t nvtxs_ = 0;
  std::vector<ssize_t> vtxdist_;
  std::vector<ssize_t> xadj_;
  std::vector<std::int64_t> adjncy_;
  ssize_t* vtxdist_copy_ = nullptr;
  ssize_t* xadj_copy_ = nullptr;
  std::int64_t* adjncy_copy_ = nullptr;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_tom2d(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<Tom2dAlgorithm>(conf, comm);
}

}  // namespace katric
