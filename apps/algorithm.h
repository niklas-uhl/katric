// The benchmark interface (like KaCCv2's benchmark/): the input is generated once and ingested, and each iteration
// resets the algorithm's state and runs it.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

#include <caliper/cali.h>
#include <kagen.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/counters/cetric.h"
#include "cetric/input.h"
#include "cetric/datastructures/distributed/distributed_graph.h"

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

/// katric's own algorithms (cetric, patric, cetric-x with either communication policy). They modify the graph
/// (orientation, load balancing, ghost expansion), so ingest() keeps an unmodified copy and reset() restores the
/// working graph from it.
class KatricAlgorithm final : public AbstractAlgorithm {
public:
    using RunFunction = std::function<std::size_t(graph::DistributedGraph<>&, Config const&)>;

    KatricAlgorithm(Config const& conf, RunFunction run) : conf_(conf), run_(std::move(run)) {}

    void ingest(kagen::Graph&& graph) override {
        auto input = to_local_input(std::move(graph));
        pristine_  = graph::DistributedGraph<>(std::move(input.G), input.node_range, conf_.rank, conf_.PEs);
        report_input_shape();
    }

    void reset() override {
        G_              = pristine_;
        triangle_count_ = 0;
    }

    void run() override {
        triangle_count_ = run_(G_, conf_);
    }

    [[nodiscard]] std::size_t triangle_count() const override {
        return triangle_count_;
    }

private:
    /// Per-rank shape of the input partition, reported once into the region `ingest` (min/max/sum across ranks).
    /// Relates phase costs to locality: cut edges are the edges whose head is on another rank.
    void report_input_shape() const {
        auto [first, last] = pristine_.node_range();
        struct Shape {
            std::size_t wedges = 0, edges = 0, cut_edges = 0, interface_nodes = 0, max_degree = 0;
        };
        tbb::enumerable_thread_specific<Shape> shapes;
        tbb::parallel_for(
            tbb::blocked_range(pristine_.local_nodes().begin(), pristine_.local_nodes().end()),
            [&](auto const& r) {
                auto& shape = shapes.local();
                for (auto node: r) {
                    std::size_t degree = pristine_.degree(node);
                    std::size_t cut    = std::ranges::count_if(pristine_.adj(node).neighbors(), [&](auto w) {
                        return w.id() < first || w.id() >= last;
                    });
                    shape.wedges += degree * (degree - 1) / 2;
                    shape.edges += degree;
                    shape.cut_edges += cut;
                    shape.interface_nodes += cut > 0;
                    shape.max_degree = std::max(shape.max_degree, degree);
                }
            }
        );
        auto total = shapes.combine([](Shape a, Shape const& b) {
            a.wedges += b.wedges;
            a.edges += b.edges;
            a.cut_edges += b.cut_edges;
            a.interface_nodes += b.interface_nodes;
            a.max_degree = std::max(a.max_degree, b.max_degree);
            return a;
        });
        auto add = [](char const* name, std::size_t value) {
            profiling::counter_add(name, static_cast<std::int64_t>(value));
        };
        add("wedges", total.wedges);
        add("local_nodes", last - first);
        add("local_edges", total.edges);
        add("cut_edges", total.cut_edges);
        add("interface_nodes", total.interface_nodes);
        add("max_degree", total.max_degree);
    }

    Config const&             conf_;
    RunFunction               run_;
    graph::DistributedGraph<> pristine_;
    graph::DistributedGraph<> G_;
    std::size_t               triangle_count_ = 0;
};

namespace detail {
template <typename CommunicationPolicy>
KatricAlgorithm::RunFunction select_run(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::cetric:
            return [](graph::DistributedGraph<>& G, Config const& conf) {
                return run_cetric(G, conf, conf.rank, conf.PEs, CommunicationPolicy{}, conf.id_node_ordering);
            };
        case Algorithm::patric:
            return [](graph::DistributedGraph<>& G, Config const& conf) {
                return run_patric(G, conf, conf.rank, conf.PEs, CommunicationPolicy{});
            };
        case Algorithm::cetric_x:
            return [](graph::DistributedGraph<>& G, Config const& conf) {
                return run_cetric_new(G, conf, conf.rank, conf.PEs, CommunicationPolicy{});
            };
    }
    throw std::invalid_argument("unknown algorithm");
}
} // namespace detail

/// Builds the algorithm selected by --algorithm and --communication-policy. `conf` must outlive the algorithm.
inline std::unique_ptr<AbstractAlgorithm> make_algorithm(Config const& conf) {
    if (conf.communication_policy == "new") {
        return std::make_unique<KatricAlgorithm>(conf, detail::select_run<MessageQueuePolicy>(conf.algorithm));
    }
    if (conf.communication_policy == "grid") {
        return std::make_unique<KatricAlgorithm>(conf, detail::select_run<GridPolicy>(conf.algorithm));
    }
    throw std::invalid_argument("unknown communication policy " + conf.communication_policy);
}

} // namespace cetric
