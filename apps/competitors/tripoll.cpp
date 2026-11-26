// TriPoll behind AbstractAlgorithm. Fork: niklas-uhl/tripoll, branch katric
// (header-only library target on YGM master, its drivers and RMAT generator
// removed; extern/CMakeLists.txt).
#include "./tripoll.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/config.h"

#include <algorithm.hpp>
#include <ordered_directed_graph.hpp>
#include <undirected_graph.hpp>
#include <ygm/comm.hpp>
#include <ygm/detail/collective.hpp>

namespace katric {
namespace {

/// Triangles found on this rank. TriPoll calls its per-triangle callback
/// through a pointer to the callback's type (no object), so the callback has
/// to be stateless and count into a global, as its own drivers do.
std::uint64_t local_triangles = 0;

/// The callback gets the metadata of the three vertices and three edges.
constexpr auto count_triangle = [](auto const&... /* metadata */) {
  ++local_triangles;
};

/// The native input is an edge list fed to TriPoll's `undirected_graph`
/// (`async_add_edge`, its drivers' input path), which stores both directions
/// and places each vertex on the rank YGM's hash partitioner picks.
///
/// KaGen's CSR holds every edge in both directions, so ingest() keeps only
/// u < v: `async_add_edge` adds both directions itself, and the drivers'
/// `uniquify_edges` (sort + dedup, needed for RMAT's duplicate edges) is
/// skipped, as LCC's `mirror()`.
///
/// Each run builds the undirected graph and the degree-ordered directed graph
/// (`make_dodgr`, region `preprocessing`; TriPoll's driver reports this as
/// preprocessing too) and counts (region `counting`). Both graphs live only
/// in run(); the edge list is only read, so reset() has nothing to rebuild.
class TripollAlgorithm final : public AbstractAlgorithm {
public:
  using undirected_type = tripoll::undirected_graph<std::uint64_t, bool, bool>;
  /// order = degree, as uint32_t in TriPoll's drivers
  using dodgr_type =
      tripoll::ordered_directed_graph<std::uint64_t, bool, bool, std::uint32_t>;

  TripollAlgorithm(Config const& conf, kamping::v2::comm_view comm)
      : conf_(conf) {
    // YGM reads its configuration from the environment when the communicator
    // is created; NONE is its default.
    setenv("YGM_COMM_ROUTING", routing_name(conf.tripoll_routing), 1);
    // Duplicates the MPI communicator; once and untimed, as in TriPoll's
    // drivers, which create it in main().
    ygm_.emplace(comm.mpi_handle());
  }

  void ingest(kagen::Graph&& input) override {
    auto const first = static_cast<std::uint64_t>(input.vertex_range.first);
    edges_.clear();
    edges_.reserve(input.adjncy.size() / 2);
    for (std::size_t vertex = 0; vertex + 1 < input.xadj.size(); ++vertex) {
      auto const u = first + vertex;
      for (auto edge = input.xadj[vertex]; edge < input.xadj[vertex + 1];
           ++edge) {
        auto const v = static_cast<std::uint64_t>(input.adjncy[edge]);
        if (u < v) {
          edges_.emplace_back(u, v);
        }
      }
    }
    input = kagen::Graph{};
  }

  void reset() override { triangle_count_ = 0; }

  void run() override {
    local_triangles = 0;
    std::optional<dodgr_type> dodgr;
    {
      CALI_CXX_MARK_SCOPE("preprocessing");
      undirected_type graph(*ygm_);
      for (auto const& [u, v] : edges_) {
        graph.async_add_edge(u, v, true);
      }
      graph.barrier();
      dodgr.emplace(*ygm_);
      tripoll::make_dodgr(graph, *dodgr);
    }
    {
      CALI_CXX_MARK_SCOPE("counting");
      switch (conf_.tripoll_variant) {
        case TripollVariant::push_only:
          tripoll::tc_push_only(*dodgr, count_triangle);
          break;
        case TripollVariant::push_pull:
          tripoll::tc_push_pull(*dodgr, count_triangle);
          break;
      }
      triangle_count_ = ygm::sum(local_triangles, *ygm_);
    }
  }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  static auto routing_name(TripollRouting routing) -> char const* {
    switch (routing) {
      case TripollRouting::none:
        return "NONE";
      case TripollRouting::nr:
        return "NR";
      case TripollRouting::nlnr:
        return "NLNR";
    }
    return "NONE";
  }

  Config const& conf_;
  std::optional<ygm::comm> ygm_;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges_;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_tripoll(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  return std::make_unique<TripollAlgorithm>(conf, comm);
}

}  // namespace katric
