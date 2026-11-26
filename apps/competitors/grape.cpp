// GRAPE behind AbstractAlgorithm. Fork: niklas-uhl/libgrape-lite, branch
// katric (library target without global CMake effects; LCCOpt with a global
// count switch; extern/CMakeLists.txt).
#include "./grape.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>
#include <mpi.h>

#include "katric/config.h"

#include <grape/fragment/basic_fragment_loader.h>
#include <grape/fragment/basic_local_fragment_loader.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/parallel/parallel_engine_spec.h>
#include <grape/worker/comm_spec.h>
#include <grape/worker/worker.h>
#include <lcc/lcc_opt.h>

namespace katric {
namespace {

/// As GRAPE's --opt driver (run_app_opt.h): int64_t ids outside, uint32_t
/// inside (so the SIMD intersection applies), no vertex or edge data.
using fragment_t =
    grape::ImmutableEdgecutFragment<std::int64_t, std::uint32_t,
                                    grape::EmptyType, grape::EmptyType,
                                    grape::LoadStrategy::kOnlyOut>;
/// 64-bit counters (its driver picks 32 bits by --edge_num, which RMAT hubs
/// could overflow)
using app_t = grape::LCCOpt<fragment_t, std::uint64_t>;
using worker_t = app_t::worker_t;

/// The native input is what GRAPE's vertex and edge file readers hand its
/// loader for an undirected graph: on each rank some vertex ids and some
/// edges, each undirected edge once (here u < v from KaGen's CSR, which has
/// both directions; the fragment adds the reverse direction).
///
/// Each run builds the fragment with GRAPE's own loader (region
/// `preprocessing`; also the worker's Init: message routing per vertex,
/// message pool), then runs the LCC query (region `counting`: degree
/// exchange, sending each N+(v) once to each rank holding an out-neighbour,
/// intersection, and unless --grape-global-count the per-vertex counts and
/// the superstep that sends the ghosts' counts to their owners). GRAPE's own
/// benchmarks time only the query. Everything is rebuilt per run, so reset()
/// has nothing to do.
///
/// Loading, as in GRAPE's Graphalytics runs (--opt, defaults): a global vertex
/// map (every rank holds every fragment's id index, O(n) per rank, built in
/// O(p) ring rounds) and the map partitioner (all vertex ids gathered on every
/// rank and cut into equal contiguous chunks of sorted ids: a balanced 1D
/// distribution). With --grape-local-vertex-map instead its local loader:
/// O(local) index, but vertices hashed to ranks (no locality).
class GrapeAlgorithm final : public AbstractAlgorithm {
public:
  GrapeAlgorithm(Config const& conf, kamping::v2::comm_view comm)
      : conf_(conf) {
    int result = MPI_UNEQUAL;
    MPI_Comm_compare(comm.mpi_handle(), MPI_COMM_WORLD, &result);
    if (result != MPI_IDENT && result != MPI_CONGRUENT) {
      throw std::runtime_error("--algorithm grape runs only on MPI_COMM_WORLD");
    }
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      throw std::runtime_error(
          "--algorithm grape needs MPI_THREAD_MULTIPLE (katric requests it "
          "when --algorithm grape is given on the command line)");
    }
    comm_spec_.Init(MPI_COMM_WORLD);
    engine_spec_ = grape::MultiProcessSpec(comm_spec_, false);
    if (conf.grape_threads > 0) {
      engine_spec_.thread_num = static_cast<std::uint32_t>(conf.grape_threads);
    }
  }

  void ingest(kagen::Graph&& input) override {
    auto const first = static_cast<std::int64_t>(input.vertex_range.first);
    auto const local_vertices = input.xadj.empty() ? 0 : input.xadj.size() - 1;
    vertices_.clear();
    vertices_.reserve(local_vertices);
    edges_.clear();
    edges_.reserve(input.adjncy.size() / 2);
    for (std::size_t vertex = 0; vertex < local_vertices; ++vertex) {
      auto const u = first + static_cast<std::int64_t>(vertex);
      vertices_.push_back(u);
      for (auto edge = input.xadj[vertex]; edge < input.xadj[vertex + 1];
           ++edge) {
        auto const v = static_cast<std::int64_t>(input.adjncy[edge]);
        if (u < v) {
          edges_.emplace_back(u, v);
        }
      }
    }
    input = kagen::Graph{};
  }

  void reset() override { triangle_count_ = 0; }

  void run() override {
    std::shared_ptr<fragment_t> fragment;
    std::shared_ptr<worker_t> worker;
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      fragment = conf_.grape_local_vertex_map ? load<true>() : load<false>();
      auto app = std::make_shared<app_t>();
      worker = app_t::CreateWorker(app, fragment);
      worker->Init(comm_spec_, engine_spec_);
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      worker->Query(conf_.grape_global_count);
      auto const& context = *worker->GetContext();
      std::uint64_t local = 0;
      if (conf_.grape_global_count) {
        local = context.triangles;
      } else {
        // each triangle is added to all three of its vertices
        for (auto v : fragment->InnerVertices()) {
          local += context.tricnt[v];
        }
      }
      std::uint64_t global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
      triangle_count_ = static_cast<std::size_t>(
          conf_.grape_global_count ? global : global / 3);
    }
    worker->Finalize();
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  template <bool kLocal>
  auto load() -> std::shared_ptr<fragment_t> {
    grape::LoadGraphSpec spec = grape::DefaultLoadGraphSpec();
    spec.set_directed(false);
    spec.set_rebalance(false, 0);
    using loader_t =
        std::conditional_t<kLocal, grape::BasicLocalFragmentLoader<fragment_t>,
                           grape::BasicFragmentLoader<fragment_t>>;
    if constexpr (kLocal) {
      spec.partitioner_type = grape::PartitionerType::kHashPartitioner;
      spec.idxer_type = grape::IdxerType::kLocalIdxer;
    } else {
      spec.partitioner_type = grape::PartitionerType::kMapPartitioner;
      spec.idxer_type = grape::IdxerType::kHashMapIdxer;
    }
    loader_t loader(comm_spec_, spec);
    for (auto const vertex : vertices_) {
      loader.AddVertex(vertex, grape::EmptyType{});
    }
    loader.ConstructVertices();
    for (auto const& [u, v] : edges_) {
      loader.AddEdge(u, v, grape::EmptyType{});
    }
    std::shared_ptr<fragment_t> fragment;
    loader.ConstructFragment(fragment);
    return fragment;
  }

  Config const& conf_;
  grape::CommSpec comm_spec_;
  grape::ParallelEngineSpec engine_spec_;
  std::vector<std::int64_t> vertices_;
  std::vector<std::pair<std::int64_t, std::int64_t>> edges_;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_grape(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<GrapeAlgorithm>(conf, comm);
}

}  // namespace katric
