// TriC behind AbstractAlgorithm. Fork: niklas-uhl/tric, branch katric (TriC as
// a header-only library target without KaGen and main; extern/CMakeLists.txt).
#include "./tric.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>
#include <mpi.h>

#include "./convert.h"
#include "katric/config.h"

// TriC's headers last: they define macros such as MIN, MAX and PI.
#include <bufastric.hpp>
#include <cfastric.hpp>
#include <hashfastric2.hpp>
#include <inbufastric.hpp>

namespace katric {
namespace {

using tric::Edge;
using tric::Graph;
using tric::GraphElem;
using tric::MPI_GRAPH_TYPE;

/// The native input is TriC's own Graph as its binary file reader
/// (BinaryEdgeList::read) builds it: a 1D vertex distribution, local CSR with
/// global ids, sorted neighbourhoods (check_edgelist() relies on that).
///
/// Each run constructs the variant's Triangulate* object (region
/// `preprocessing`: TriC sizes its buffers and exchanges edge ranges) and calls
/// count() (region `counting`). count() is not re-entrant (it accumulates into
/// the object), so every run constructs a new one. The graph itself stays
/// untouched except for the buffered variants' per-edge state, which reset()
/// rebuilds.
class TricAlgorithm final : public AbstractAlgorithm {
public:
  TricAlgorithm(Config const& conf, kamping::v2::comm_view comm)
      : conf_(conf), comm_(comm.mpi_handle()) {}

  void ingest(kagen::Graph&& input) override {
    auto const distribution = competitors::vertex_distribution(input, comm_);
    auto csr = competitors::sorted_csr(input);
    input = kagen::Graph{};
    auto const local_vertices = static_cast<GraphElem>(csr.xadj.size() - 1);
    auto const local_edges = static_cast<GraphElem>(csr.adjncy.size());
    GraphElem global_edges = 0;
    MPI_Allreduce(&local_edges, &global_edges, 1, MPI_GRAPH_TYPE, MPI_SUM,
                  comm_);
    graph_ = std::make_unique<Graph>(local_vertices, local_edges,
                                     distribution.back(), global_edges, comm_);
    graph_->repart(distribution);
    graph_->edge_indices_ = std::move(csr.xadj);
    for (std::size_t i = 0; i < csr.adjncy.size(); ++i) {
      graph_->edge_list_[i].tail_ = csr.adjncy[i];
    }
  }

  void reset() override {
    // the buffered variants mark edges as done in the graph's edge_stat_
    if (conf_.tric_variant != TricVariant::plain) {
      graph_->edge_stat_.clear();
      graph_->edge_stat_.reserve(graph_->edge_list_.size());
      for (Edge const& edge : graph_->edge_list_) {
        graph_->edge_stat_.emplace_back(&edge);
      }
    }
    triangle_count_ = 0;
  }

  void run() override {
    auto const buffer_size = static_cast<GraphElem>(conf_.tric_buffer_size);
    switch (conf_.tric_variant) {
      case TricVariant::plain:
        count<tric::TriangulateAggrFatCompressed>();
        break;
      case TricVariant::buffered:
        count<tric::TriangulateAggrBuffered>(buffer_size, conf_.tric_sparse_ranges);
        break;
      case TricVariant::hash2:
        count<tric::TriangulateAggrBufferedHash2>(buffer_size,
                                             conf_.tric_sparse_ranges);
        break;
      case TricVariant::inrecv:
        count<tric::TriangulateAggrBufferedInrecv>(buffer_size);
        break;
    }
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  template <typename Triangulate, typename... Args>
  void count(Args... args) {
    std::optional<Triangulate> triangulate;
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      triangulate.emplace(graph_.get(), args...);
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      triangle_count_ = static_cast<std::size_t>(triangulate->count());
    }
    triangulate->clear();  // frees its buffers (its destructor does not)
  }

  Config const& conf_;
  MPI_Comm comm_;
  std::unique_ptr<Graph> graph_;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_tric(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<TricAlgorithm>(conf, comm);
}

}  // namespace katric
