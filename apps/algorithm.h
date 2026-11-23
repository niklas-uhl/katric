// The benchmark interface (like KaCCv2's benchmark/): the input is generated
// once and ingested, and each iteration resets the algorithm's state and runs
// it.
#pragma once

#include <cstddef>
#include <memory>

#include <kagen.h>
#include <kamping/v2/comm.hpp>

#include "katric/config.h"

namespace katric {

class AbstractAlgorithm {
public:
  AbstractAlgorithm() = default;
  AbstractAlgorithm(AbstractAlgorithm const&) = delete;
  AbstractAlgorithm(AbstractAlgorithm&&) = delete;
  auto operator=(AbstractAlgorithm const&) -> AbstractAlgorithm& = delete;
  auto operator=(AbstractAlgorithm&&) -> AbstractAlgorithm& = delete;
  virtual ~AbstractAlgorithm() = default;

  /// Takes the input graph (the local part on each rank). Called once per run.
  virtual void ingest(kagen::Graph&& graph) = 0;
  /// Restores the state right after ingest(). Called before each run().
  virtual void reset() = 0;
  virtual void run() = 0;
  /// The global triangle count of the last run(); only valid on rank 0.
  [[nodiscard]] virtual auto triangle_count() const -> std::size_t = 0;
};

/// The algorithm selected by `conf.algorithm`, configured by `conf`, running
/// on the ranks of `comm`. `conf` and `comm` must outlive it. A competitor must
/// be compiled in (is_compiled_in()).
auto make_algorithm(Config const& conf, kamping::v2::comm_view comm)
    -> std::unique_ptr<AbstractAlgorithm>;

/// Whether `algorithm` is part of this build: KaTric always, a competitor only
/// with its CMake option KATRIC_COMPETITOR_<NAME>.
auto is_compiled_in(Algorithm algorithm) -> bool;

}  // namespace katric
