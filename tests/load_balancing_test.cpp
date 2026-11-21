// Copyright (c) 2020-2023 Tim Niklas Uhl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kagen.h>
#include <mpi.h>

#include "katric/phases/load_balancing.h"
#include "test_graphs.h"

namespace {
using katric::graph::VertexId;

class LoadBalancing : public ::testing::TestWithParam<const char*> {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        G_full = katric::testing::read_full_graph(GetParam());
        G      = katric::testing::read_local_graph(GetParam());
    }

    /// Balances `G` by `cost_of(global id)` and checks that every vertex arrives once, with all its edges, on the rank
    /// whose share [r C/p, (r+1) C/p) of the total cost C contains the cost before it.
    template <typename CostOf>
    void balance_and_check(CostOf&& cost_of) {
        std::vector<std::size_t> cost;
        for (auto v = G.vertex_range.first; v < G.vertex_range.second; ++v) {
            cost.push_back(cost_of(v));
        }
        std::vector<std::size_t> cost_before(G_full.size());
        std::size_t              total = 0;
        for (VertexId v = 0; v < G_full.size(); ++v) {
            cost_before[v] = total;
            total += cost_of(v);
        }
        std::size_t const per_pe_cost = std::max<std::size_t>((total + size - 1) / size, 1);

        G = katric::redistribute(std::move(G), cost, MPI_COMM_WORLD);
        auto const [first, last] = G.vertex_range;
        ASSERT_EQ(G.xadj.size(), last - first + 1);
        ASSERT_EQ(G.xadj.back(), G.adjncy.size());
        for (auto v = first; v < last; ++v) {
            auto const i = v - first;
            EXPECT_EQ(G.xadj[i + 1] - G.xadj[i], G_full[v].size());
            for (auto e = G.xadj[i]; e < G.xadj[i + 1]; ++e) {
                EXPECT_EQ(G.adjncy[e], G_full[v][e - G.xadj[i]].head);
            }
            EXPECT_EQ(std::min<std::size_t>(cost_before[v] / per_pe_cost, size - 1), static_cast<std::size_t>(rank));
        }
        std::uint64_t const local = last - first;
        std::uint64_t       global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(global, G_full.size());
    }

    std::vector<std::vector<katric::testing::Edge>> G_full;
    kagen::Graph                                  G;
    int                                           rank = 0, size = 0;
};

TEST_P(LoadBalancing, BalancingWithConstCost) {
    balance_and_check([](VertexId) { return std::size_t{1}; });
}

TEST_P(LoadBalancing, BalancingWithDegree) {
    balance_and_check([&](VertexId v) { return G_full[v].size(); });
}

// vertex_costs() against a count on the full graph, with the orientation by (degree, global id): DPD over the
// out-neighbors, IDPD over the in-neighbors, IDD as IDPD with a ghost's out-degree replaced by its degree.
TEST_P(LoadBalancing, VertexCosts) {
    auto const key        = [&](VertexId v) { return std::pair(G_full[v].size(), v); };
    auto const out_degree = [&](VertexId v) {
        std::size_t d = 0;
        for (auto const& e: G_full[v]) {
            d += key(v) < key(e.head);
        }
        return d;
    };
    auto const [first, last] = G.vertex_range;
    auto const is_local      = [&](VertexId v) { return first <= v && v < last; };
    for (auto const function: {katric::CostFunction::DPD, katric::CostFunction::IDPD, katric::CostFunction::IDD}) {
        katric::Config conf;
        conf.primary_cost_function = function;
        auto       graph           = katric::graph::Graph::from_kagen(G, MPI_COMM_WORLD);
        auto const cost            = katric::vertex_costs(graph, conf);
        ASSERT_EQ(cost.size(), last - first);
        for (auto v = first; v < last; ++v) {
            std::size_t expected = 0;
            for (auto const& e: G_full[v]) {
                VertexId const u   = e.head;
                bool const     out = key(v) < key(u);
                if (out != (function == katric::CostFunction::DPD)) {
                    continue;
                }
                bool const ghost_degree = function == katric::CostFunction::IDD && !is_local(u);
                expected += out_degree(v) + (ghost_degree ? G_full[u].size() : out_degree(u));
            }
            EXPECT_EQ(cost[v - first], expected) << "vertex " << v << ", cost function " << static_cast<int>(function);
        }
    }
}

// --rebalance-threshold: above the cost's max/avg the input graph comes back as it was, with its degrees (so that
// preprocessing does not exchange them again); at 0 the graph is always redistributed (a new graph, no degrees yet).
TEST_P(LoadBalancing, RebalanceThreshold) {
    for (auto const function: {katric::CostFunction::IDPD, katric::CostFunction::IDD}) {
        katric::Config conf;
        conf.primary_cost_function = function;
        conf.rebalance_threshold   = 1e9;
        auto kept = katric::primary_load_balancing(katric::graph::Graph::from_kagen(G, MPI_COMM_WORLD), conf);
        EXPECT_TRUE(kept.has_undirected_degrees());
        EXPECT_EQ(kept.numbering().local_range().first, G.vertex_range.first);
        EXPECT_EQ(kept.num_local_vertices(), G.vertex_range.second - G.vertex_range.first);
        EXPECT_EQ(kept.num_local_edges(), G.adjncy.size());
        katric::preprocessing(kept, conf); // the degrees are reused
        EXPECT_TRUE(kept.has_undirected_degrees());

        conf.rebalance_threshold = 0.0;
        auto moved = katric::primary_load_balancing(katric::graph::Graph::from_kagen(G, MPI_COMM_WORLD), conf);
        EXPECT_FALSE(moved.has_undirected_degrees());
        std::uint64_t const local  = moved.num_local_vertices();
        std::uint64_t       global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(global, G_full.size());
    }
}

const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis", "examples/com-amazon.metis"};
INSTANTIATE_TEST_SUITE_P(SimpleGraphs, LoadBalancing, ::testing::ValuesIn(graphs));
} // namespace
