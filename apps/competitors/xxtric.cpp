// xxtric behind AbstractAlgorithm. Fork: niklas-uhl/xxtric, branch katric (the
// CPU path as a header-only library target in namespace xxtric;
// extern/CMakeLists.txt).
#include "./xxtric.h"

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

// xxtric's headers last: they define macros such as MIN, MAX and PI.
#include <xxtric.hpp>

namespace katric {
namespace {

using xxtric::Edge;
using xxtric::Graph;
using xxtric::GraphElem;
using xxtric::MPI_GRAPH_TYPE;

/// The native input is xxtric's Graph as its binary reader
/// (BinaryEdgeList::read) builds it, the same as TriC's: a 1D vertex
/// distribution, local CSR with global ids, sorted neighbourhoods, and the
/// per-edge state edge_stat_.
///
/// Each run constructs TriangleCounting (region `preprocessing`: edge ranges,
/// the local counting, buffer sizing) and calls count() (region `counting`).
/// count() accumulates into its object, so every run constructs a new one. It
/// marks edges as done in the graph's edge_stat_, which reset() rebuilds.
class XxtricAlgorithm final : public AbstractAlgorithm {
public:
  XxtricAlgorithm(Config const& conf, kamping::v2::comm_view comm)
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
    graph_->edge_stat_.clear();
    graph_->edge_stat_.reserve(graph_->edge_list_.size());
    for (Edge const& edge : graph_->edge_list_) {
      graph_->edge_stat_.emplace_back(&edge);
    }
    triangle_count_ = 0;
  }

  void run() override {
    std::optional<xxtric::TriangleCounting> counting;
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      counting.emplace(graph_.get(),
                       static_cast<GraphElem>(conf_.xxtric_buffer_size));
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      triangle_count_ = static_cast<std::size_t>(counting->count());
    }
    counting->clear();  // frees its buffers (its destructor does not)
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  Config const& conf_;
  MPI_Comm comm_;
  std::unique_ptr<Graph> graph_;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_xxtric(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<XxtricAlgorithm>(conf, comm);
}

}  // namespace katric
