// HavoqGT behind AbstractAlgorithm. Fork: niklas-uhl/havoqgt, branch katric
// (header-only library target, triangle count split into orientation and
// count; extern/CMakeLists.txt).
#include "./havoqgt.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>
#include <mpi.h>

#include "./convert.h"
#include "katric/config.h"

#include <havoqgt/delegate_partitioned_graph.hpp>
#include <havoqgt/new_triangle_count.hpp>

namespace katric {
namespace {

/// HavoqGT's mailbox routes through node-local and node-remote communicators
/// and maps rank r to node r / ranks_per_node, so every node must hold the
/// same number of ranks, placed in contiguous blocks. Collective.
void check_rank_placement(MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &node_comm);
  int node_rank = 0;
  int node_size = 0;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);
  MPI_Comm_free(&node_comm);
  int const blocked = (rank - node_rank) % node_size == 0 &&
                      (rank - node_rank) / node_size == rank / node_size;
  // [min size, -max size, blocked everywhere]
  std::vector<int> values{node_size, -node_size, blocked};
  MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                MPI_INT, MPI_MIN, comm);
  if (values[0] != -values[1] || values[2] == 0) {
    throw std::runtime_error(
        "--algorithm havoqgt needs the same number of ranks on every node, "
        "placed in contiguous blocks (ranks per node: " +
        std::to_string(values[0]) + " to " + std::to_string(-values[1]) + ")");
  }
}

/// The native input is an edge list (its driver's ingest_edge_list / KaGen
/// path): each rank passes some edges, both directions of every undirected
/// edge, and HavoqGT distributes them itself (vertices cyclically, delegates
/// for high degrees).
///
/// Each run constructs the delegate-partitioned graph and orients it (region
/// `preprocessing`; HavoqGT's driver timed only what follows the
/// construction), then counts (region `counting`). Both are rebuilt per run,
/// nothing outlives it except the edge list, which HavoqGT only reads, so
/// reset() has nothing to do.
class HavoqgtAlgorithm final : public AbstractAlgorithm {
public:
  using graph_type =
      havoqgt::delegate_partitioned_graph<std::allocator<std::byte>>;

  HavoqgtAlgorithm(Config const& conf, kamping::v2::comm_view comm)
      : conf_(conf) {
    int is_world = MPI_UNEQUAL;
    MPI_Comm_compare(comm.mpi_handle(), MPI_COMM_WORLD, &is_world);
    if (is_world != MPI_IDENT && is_world != MPI_CONGRUENT) {
      throw std::runtime_error(
          "--algorithm havoqgt runs only on MPI_COMM_WORLD");
    }
    check_rank_placement(MPI_COMM_WORLD);
    // MPI is initialized already, so this only sets up HavoqGT's node-local
    // and node-remote communicators (rank by rank through shared memory,
    // O(p)), once and untimed, as its driver does before loading the graph.
    havoqgt::init(nullptr, nullptr);
  }

  void ingest(kagen::Graph&& input) override {
    auto const distribution =
        competitors::vertex_distribution(input, MPI_COMM_WORLD);
    max_vertex_ = static_cast<std::uint64_t>(distribution.back()) - 1;
    auto const first = static_cast<std::uint64_t>(input.vertex_range.first);
    edges_.clear();
    edges_.reserve(input.adjncy.size());
    for (std::size_t vertex = 0; vertex + 1 < input.xadj.size(); ++vertex) {
      for (auto edge = input.xadj[vertex]; edge < input.xadj[vertex + 1];
           ++edge) {
        edges_.emplace_back(first + vertex,
                            static_cast<std::uint64_t>(input.adjncy[edge]),
                            0.0);
      }
    }
    input = kagen::Graph{};
  }

  void reset() override { triangle_count_ = 0; }

  void run() override {
    std::optional<graph_type> graph;
    std::optional<havoqgt::directed_core2_data<graph_type>> directed;
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      std::allocator<std::byte> allocator;
      graph.emplace(allocator, MPI_COMM_WORLD, edges_, max_vertex_,
                    conf_.havoqgt_delegate_threshold,
                    conf_.havoqgt_partition_passes, conf_.havoqgt_chunk_size);
      directed.emplace(*graph);
      havoqgt::triangle_count_orient(*graph, *directed, conf_.havoqgt_stats);
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      triangle_count_ = havoqgt::triangle_count_wedges(*graph, *directed,
                                                       conf_.havoqgt_stats);
    }
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  Config const& conf_;
  std::vector<std::tuple<std::uint64_t, std::uint64_t, double>> edges_;
  std::uint64_t max_vertex_ = 0;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_havoqgt(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<HavoqgtAlgorithm>(conf, comm);
}

}  // namespace katric
