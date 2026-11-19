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

#include "katric/algorithms.h"
#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/input.h"

namespace katric {

/// KaTric (DiTriC, or CETRIC with --contraction) with either communication policy. ingest() keeps the input;
/// every run only reads it to build its graphs, so nothing is copied per iteration.
namespace {
class KatricAlgorithm final : public AbstractAlgorithm {
public:
    using RunFunction = std::function<std::size_t(kagen::Graph const&, Config const&, MPI_Comm)>;

    KatricAlgorithm(Config const& conf, MPI_Comm comm, RunFunction run) : conf_(conf), comm_(comm), run_(std::move(run)) {}

    void ingest(kagen::Graph&& graph) override {
        pristine_ = std::move(graph);
        report_input_shape();
    }

    void reset() override {
        triangle_count_ = 0;
    }

    void run() override {
        triangle_count_ = run_(pristine_, conf_, comm_);
    }

    [[nodiscard]] std::size_t triangle_count() const override {
        return triangle_count_;
    }

private:
    /// Per-rank shape of the input partition, reported once into the region `ingest` (min/max/sum across ranks).
    /// Relates phase costs to locality: cut edges are the edges whose head is on another rank.
    void report_input_shape() const {
        auto const [first, last] = pristine_.vertex_range;
        std::size_t wedges = 0, edges = 0, cut_edges = 0, interface_nodes = 0, max_degree = 0;
        for (std::size_t i = 0; i < last - first; ++i) {
            std::size_t const degree = pristine_.xadj[i + 1] - pristine_.xadj[i];
            auto const        heads  = std::span(pristine_.adjncy).subspan(pristine_.xadj[i], degree);
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
    kagen::Graph              pristine_;
    std::size_t               triangle_count_ = 0;
};

} // namespace

std::unique_ptr<AbstractAlgorithm> make_algorithm(Config const& conf, MPI_Comm comm) {
    if (conf.communication_policy == "new") {
        return std::make_unique<KatricAlgorithm>(conf, comm, [](kagen::Graph const& input, Config const& run_conf, MPI_Comm run_comm) {
            return count_triangles(input, run_conf, run_comm, MessageQueuePolicy{});
        });
    }
    if (conf.communication_policy == "grid") {
        return std::make_unique<KatricAlgorithm>(conf, comm, [](kagen::Graph const& input, Config const& run_conf, MPI_Comm run_comm) {
            return count_triangles(input, run_conf, run_comm, GridPolicy{});
        });
    }
    throw std::invalid_argument("unknown communication policy " + conf.communication_policy);
}

} // namespace katric
