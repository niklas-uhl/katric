// LCC behind AbstractAlgorithm. Forks: niklas-uhl/lcc and niklas-uhl/CLaMPI,
// branches katric (LCC as a library with the calls of lcc_api.h, built as in
// the paper with SIMD and CLaMPI; extern/CMakeLists.txt).
#include "./lcc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>
#include <mpi.h>

#include "./convert.h"
#include "katric/config.h"

#include <lcc_api.h>

namespace katric {
namespace {

/// The native input is what LCC's reader produces: on each rank some edges
/// (u, v) as pairs of 64-bit ids, here with both directions of every edge
/// (KaGen's), so LCC's mirror() step is not needed. LCC distributes them
/// itself, a block of N/p consecutive vertices per rank (1 x p mesh).
///
/// Each run hands LCC a fresh copy of the edges (reset(), untimed: LCC takes
/// ownership and reallocates them), then lcc_setup() + lcc_preprocess()
/// (region `preprocessing`: communicators, partitioning, sorting and
/// deduplication, CSC) and lcc_count() (region `counting`: the kernel, which
/// creates the RMA windows and CLaMPI caches, and the global sum).
/// lcc_free() runs at the end of the run.
class LccAlgorithm final : public AbstractAlgorithm {
public:
  LccAlgorithm(Config const& conf, kamping::v2::comm_view comm) : conf_(conf) {
    int result = MPI_UNEQUAL;
    MPI_Comm_compare(comm.mpi_handle(), MPI_COMM_WORLD, &result);
    if (result != MPI_IDENT && result != MPI_CONGRUENT) {
      throw std::runtime_error("--algorithm lcc runs only on MPI_COMM_WORLD");
    }
  }

  LccAlgorithm(LccAlgorithm const&) = delete;
  LccAlgorithm(LccAlgorithm&&) = delete;
  auto operator=(LccAlgorithm const&) -> LccAlgorithm& = delete;
  auto operator=(LccAlgorithm&&) -> LccAlgorithm& = delete;
  ~LccAlgorithm() override { std::free(lcc_edges_); }

  void ingest(kagen::Graph&& input) override {
    auto const distribution =
        competitors::vertex_distribution(input, MPI_COMM_WORLD);
    n_ = static_cast<std::uint64_t>(distribution.back());
    if (n_ > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error(
          "--algorithm lcc: LCC uses 32-bit vertex ids, the input has too "
          "many vertices");
    }
    auto const first = static_cast<std::uint64_t>(input.vertex_range.first);
    edges_.clear();
    edges_.reserve(2 * input.adjncy.size());
    for (std::size_t vertex = 0; vertex + 1 < input.xadj.size(); ++vertex) {
      for (auto edge = input.xadj[vertex]; edge < input.xadj[vertex + 1];
           ++edge) {
        edges_.push_back(first + vertex);
        edges_.push_back(static_cast<std::uint64_t>(input.adjncy[edge]));
      }
    }
    input = kagen::Graph{};
  }

  void reset() override {
    std::free(lcc_edges_);
    // at least one byte: LCC reallocates the array, malloc(0) may give null
    lcc_edges_ = static_cast<std::uint64_t*>(
        std::malloc(std::max<std::size_t>(1, edges_.size() * sizeof(std::uint64_t))));
    if (lcc_edges_ == nullptr) {
      throw std::bad_alloc();
    }
    std::memcpy(lcc_edges_, edges_.data(), edges_.size() * sizeof(std::uint64_t));
    triangle_count_ = 0;
  }

  void run() override {
    lcc_options const options{
        .clampi_cache_size = conf_.lcc_cache_size,
        .degree_reduction = conf_.lcc_degree_reduction ? 1 : 0};
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      lcc_setup(n_, &options);
      lcc_preprocess(lcc_edges_, edges_.size() / 2);  // takes ownership
      lcc_edges_ = nullptr;
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      triangle_count_ = static_cast<std::size_t>(lcc_count());
    }
    lcc_free();
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  Config const& conf_;
  std::uint64_t n_ = 0;
  std::vector<std::uint64_t> edges_;  // (u, v) pairs
  std::uint64_t* lcc_edges_ = nullptr;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_lcc(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<LccAlgorithm>(conf, comm);
}

}  // namespace katric
