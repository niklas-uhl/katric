#include "./algorithm.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

#include <kagen.h>

#include "cetric/algorithms.h"
#include "cetric/caliper.h"
#include "cetric/config.h"
#include "cetric/datastructures/distributed/distributed_graph.h"
#include "cetric/input.h"

namespace cetric {

/// katric's own algorithms (cetric, patric, cetric-x with either communication policy). ingest() keeps the input;
/// reset() hands each run a fresh copy, as the view the algorithm builds its graphs from (`on_input`) or, for the
/// algorithms not yet moved to compact ids, as a DistributedGraph (`on_graph`).
namespace {
class KatricAlgorithm final : public AbstractAlgorithm {
public:
    struct RunFunction {
        std::function<std::size_t(graph::DistributedGraph<>&, Config const&)>              on_graph;
        std::function<std::size_t(graph::LocalInput&&, Config const&, MPI_Comm)> on_input;
    };

    KatricAlgorithm(Config const& conf, MPI_Comm comm, RunFunction run) : conf_(conf), comm_(comm), run_(std::move(run)) {}

    void ingest(kagen::Graph&& graph) override {
        pristine_ = to_local_input(std::move(graph));
        report_input_shape();
    }

    void reset() override {
        if (run_.on_input) {
            input_ = pristine_;
        } else {
            auto copy = pristine_;
            G_        = graph::DistributedGraph<>(std::move(copy.G), copy.node_range, comm_);
        }
        triangle_count_ = 0;
    }

    void run() override {
        triangle_count_ = run_.on_input ? run_.on_input(std::move(input_), conf_, comm_) : run_.on_graph(G_, conf_);
    }

    [[nodiscard]] std::size_t triangle_count() const override {
        return triangle_count_;
    }

private:
    /// Per-rank shape of the input partition, reported once into the region `ingest` (min/max/sum across ranks).
    /// Relates phase costs to locality: cut edges are the edges whose head is on another rank.
    void report_input_shape() const {
        auto const [first, last] = pristine_.node_range;
        auto const& view         = pristine_.G;
        std::size_t wedges = 0, edges = 0, cut_edges = 0, interface_nodes = 0, max_degree = 0;
        std::size_t next = 0;
        for (auto const& info: view.node_info) {
            std::size_t const degree = info.degree;
            auto const        heads  = std::span(view.edge_heads).subspan(next, degree);
            next += degree;
            std::size_t const cut = std::ranges::count_if(heads, [&](auto w) { return w < first || w >= last; });
            wedges += degree * (degree - 1) / 2;
            edges += degree;
            cut_edges += cut;
            interface_nodes += cut > 0;
            max_degree = std::max(max_degree, degree);
        }
        auto add = [](char const* name, std::size_t value) {
            profiling::counter_add(name, static_cast<std::int64_t>(value));
        };
        add("wedges", wedges);
        add("local_nodes", last - first);
        add("local_edges", edges);
        add("cut_edges", cut_edges);
        add("interface_nodes", interface_nodes);
        add("max_degree", max_degree);
    }

    Config const&             conf_;
    MPI_Comm                  comm_;
    RunFunction               run_;
    graph::LocalInput         pristine_;
    graph::LocalInput         input_;
    graph::DistributedGraph<> G_;
    std::size_t               triangle_count_ = 0;
};

template <typename CommunicationPolicy>
KatricAlgorithm::RunFunction select_run(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::cetric:
            return {.on_graph = [](graph::DistributedGraph<>& G, Config const& conf) {
                return run_cetric(G, conf, CommunicationPolicy{});
            }};
        case Algorithm::patric:
            return {.on_input = [](graph::LocalInput&& input, Config const& conf, MPI_Comm comm) {
                return run_patric(std::move(input), conf, comm, CommunicationPolicy{});
            }};
        case Algorithm::cetric_x:
            return {.on_graph = [](graph::DistributedGraph<>& G, Config const& conf) {
                return run_cetric_x(G, conf, CommunicationPolicy{});
            }};
    }
    throw std::invalid_argument("unknown algorithm");
}
} // namespace

std::unique_ptr<AbstractAlgorithm> make_algorithm(Config const& conf, MPI_Comm comm) {
    if (conf.communication_policy == "new") {
        return std::make_unique<KatricAlgorithm>(conf, comm, select_run<MessageQueuePolicy>(conf.algorithm));
    }
    if (conf.communication_policy == "grid") {
        return std::make_unique<KatricAlgorithm>(conf, comm, select_run<GridPolicy>(conf.algorithm));
    }
    throw std::invalid_argument("unknown communication policy " + conf.communication_policy);
}

} // namespace cetric
