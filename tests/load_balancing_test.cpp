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
#include <vector>

#include <gtest/gtest.h>
#include <kagen.h>
#include <mpi.h>

#include "katric/datastructures/graph_definitions.h"
#include "katric/load_balancing.h"
#include "test_graphs.h"

namespace {
using katric::graph::NodeId;

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
        for (NodeId v = 0; v < G_full.size(); ++v) {
            cost_before[v] = total;
            total += cost_of(v);
        }
        std::size_t const per_pe_cost = std::max<std::size_t>((total + size - 1) / size, 1);

        G = katric::load_balancing::run(std::move(G), cost, MPI_COMM_WORLD);
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

    std::vector<std::vector<katric::graph::Edge>> G_full;
    kagen::Graph                                  G;
    int                                           rank = 0, size = 0;
};

TEST_P(LoadBalancing, BalancingWithConstCost) {
    balance_and_check([](NodeId) { return std::size_t{1}; });
}

TEST_P(LoadBalancing, BalancingWithDegree) {
    balance_and_check([&](NodeId v) { return G_full[v].size(); });
}

const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis", "examples/com-amazon.metis"};
INSTANTIATE_TEST_SUITE_P(SimpleGraphs, LoadBalancing, ::testing::ValuesIn(graphs));
} // namespace
