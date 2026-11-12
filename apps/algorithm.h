// The benchmark interface (like KaCCv2's benchmark/): the input is generated once and ingested, and each iteration
// resets the algorithm's state and runs it.
#pragma once

#include <cstddef>
#include <memory>

#include <kagen.h>

#include "cetric/config.h"

namespace cetric {

class AbstractAlgorithm {
public:
    AbstractAlgorithm()                                    = default;
    AbstractAlgorithm(AbstractAlgorithm const&)            = delete;
    AbstractAlgorithm(AbstractAlgorithm&&)                 = delete;
    AbstractAlgorithm& operator=(AbstractAlgorithm const&) = delete;
    AbstractAlgorithm& operator=(AbstractAlgorithm&&)      = delete;
    virtual ~AbstractAlgorithm()                           = default;

    /// Takes the input graph (the local part on each rank). Called once per run.
    virtual void ingest(kagen::Graph&& graph) = 0;
    /// Restores the state right after ingest(). Called before each run().
    virtual void reset() = 0;
    virtual void run()   = 0;
    /// The global triangle count of the last run(); only valid on rank 0.
    [[nodiscard]] virtual std::size_t triangle_count() const = 0;
};

/// Builds the algorithm selected by --algorithm and --communication-policy. `conf` must outlive the algorithm.
std::unique_ptr<AbstractAlgorithm> make_algorithm(Config const& conf);

} // namespace cetric
