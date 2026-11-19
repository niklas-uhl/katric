#include "./algorithm.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/algorithms.h"
#include "katric/caliper.h"
#include "katric/config.h"

namespace katric {

/// KaTric (DiTriC, or CETRIC with --contraction) with either communication
/// policy. ingest() keeps the input; every run only reads it to build its
/// graphs, so nothing is copied per iteration.
namespace {
class KatricAlgorithm final : public AbstractAlgorithm {
public:
  using RunFunction = std::function<
      std::size_t(kagen::Graph const&, Config const&, kamping::v2::comm_view)>;

  KatricAlgorithm(Config const& conf,
                  kamping::v2::comm_view comm,
                  RunFunction run)
      : conf_(conf), comm_(comm), run_(std::move(run)) {}

  void ingest(kagen::Graph&& graph) override {
    pristine_ = std::move(graph);
    report_input_shape();
  }

  void reset() override { triangle_count_ = 0; }

  void run() override { triangle_count_ = run_(pristine_, conf_, comm_); }

  [[nodiscard]] auto triangle_count() const -> std::size_t override {
    return triangle_count_;
  }

private:
  /// Per-rank shape of the input partition, reported once into the region
  /// `ingest` (min/max/sum across ranks). Relates phase costs to locality: cut
  /// edges are the edges whose head is on another rank.
  void report_input_shape() const {
    auto const [first, last] = pristine_.vertex_range;
    std::size_t wedges = 0;
    std::size_t edges = 0;
    std::size_t cut_edges = 0;
    std::size_t interface_nodes = 0;
    std::size_t max_degree = 0;
    for (std::size_t i = 0; i < last - first; ++i) {
      std::size_t const degree = pristine_.xadj[i + 1] - pristine_.xadj[i];
      auto const heads =
          std::span(pristine_.adjncy).subspan(pristine_.xadj[i], degree);
      std::size_t const cut = std::ranges::count_if(
          heads, [&](auto w) -> bool { return w < first || w >= last; });
      wedges += degree * (degree - 1) / 2;
      edges += degree;
      cut_edges += cut;
      interface_nodes += static_cast<std::size_t>(cut > 0);
      max_degree = std::max(max_degree, degree);
    }
    auto add = [](char const* name, std::size_t value) -> void {
      profiling::counter_add(name, static_cast<std::int64_t>(value));
    };
    add("wedges", wedges);
    add("local_nodes", last - first);
    add("local_edges", edges);
    add("cut_edges", cut_edges);
    add("interface_nodes", interface_nodes);
    add("max_degree", max_degree);
  }

  Config const& conf_;
  kamping::v2::comm_view comm_;
  RunFunction run_;
  kagen::Graph pristine_;
  std::size_t triangle_count_ = 0;
};

}  // namespace

auto make_algorithm(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm> {
  if (conf.communication_policy == "new") {
    return std::make_unique<KatricAlgorithm>(
        conf, comm,
        [](kagen::Graph const& input, Config const& run_conf,
           kamping::v2::comm_view run_comm) {
          return count_triangles(input, run_conf, run_comm,
                                 MessageQueuePolicy{});
        });
  }
  if (conf.communication_policy == "grid") {
    return std::make_unique<KatricAlgorithm>(
        conf, comm,
        [](kagen::Graph const& input, Config const& run_conf,
           kamping::v2::comm_view run_comm) {
          return count_triangles(input, run_conf, run_comm, GridPolicy{});
        });
  }
  throw std::invalid_argument("unknown communication policy " +
                              conf.communication_policy);
}

}  // namespace katric
