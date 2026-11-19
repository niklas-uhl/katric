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
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "katric/message_tags.h"
#include "katric/config.h"
#include "katric/ghost_exchange.h"
#include "katric/graph/build.h"
#include "katric/graph/graph.h"
#include "test_graphs.h"

namespace {
using katric::graph::Degree;
using katric::graph::VertexHandle;
using katric::graph::VertexId;

class Orientation : public ::testing::TestWithParam<const char*> {
protected:
    void SetUp() override {
        G_full     = katric::testing::read_full_graph(GetParam());
        local      = katric::testing::read_local_graph(GetParam());
    }

    /// The input graph of this rank's part (each orientation consumes one).
    [[nodiscard]] katric::graph::Graph input_graph() const {
        return katric::graph::build_input_graph(local, MPI_COMM_WORLD);
    }

    [[nodiscard]] std::pair<Degree, VertexId> key(VertexId v) const {
        return {G_full[v].size(), v};
    }

    /// The global ids of a neighborhood of `graph`, sorted.
    [[nodiscard]] static std::vector<VertexId>
    globals(katric::graph::Graph const& graph, std::span<VertexHandle const> neighbors) {
        std::vector<VertexId> ids;
        for (VertexHandle const u: neighbors) {
            ids.push_back(graph.global_id(u));
        }
        std::ranges::sort(ids);
        return ids;
    }

    std::vector<std::vector<katric::testing::Edge>> G_full;
    kagen::Graph                               local;
};

TEST_P(Orientation, GlobalDegree) {
    auto G = input_graph();
    // every transport of the degree exchange delivers the true degree of every ghost
    std::map<katric::GhostExchangeMode, std::vector<std::uint64_t>> received;
    for (auto mode: {katric::GhostExchangeMode::sparse, katric::GhostExchangeMode::dense, katric::GhostExchangeMode::grid}) {
        received[mode] = katric::graph::exchange_ghost_values(
            G,
            [&](VertexHandle v) { return G.degree(v); },
            mode,
            katric::MessageTag::Orientation
        );
        ASSERT_EQ(received[mode].size(), G.num_ghosts());
        for (VertexHandle const g: G.ghosts()) {
            EXPECT_EQ(received[mode][G.layer().to_ghost_index(g)], G_full[G.global_id(g)].size());
        }
    }
    auto const& ghost_degree = received[katric::GhostExchangeMode::sparse];
    auto const  G_oriented   = katric::graph::orient(
        std::move(G),
        [&](VertexHandle c) {
            Degree const degree = G.is_local(c) ? G.degree(c) : ghost_degree[G.layer().to_ghost_index(c)];
            return std::pair(degree, G.global_id(c));
        },
        true
    );
    // out-neighbors: the neighbors with a larger (degree, id); a ghost's out-neighbors: its local neighbors with a larger key
    for (VertexHandle const v: G_oriented.local_vertices()) {
        VertexId const        v_global = G_oriented.global_id(v);
        std::vector<VertexId> expected;
        for (auto const& e: G_full[v_global]) {
            if (key(v_global) < key(e.head)) {
                expected.push_back(e.head);
            }
        }
        std::ranges::sort(expected);
        EXPECT_EQ(globals(G_oriented, G_oriented.neighbors(v)), expected);
    }
    for (VertexHandle const g: G_oriented.ghosts()) {
        VertexId const        g_global = G_oriented.global_id(g);
        std::vector<VertexId> expected;
        for (auto const& e: G_full[g_global]) {
            if (G_oriented.layer().is_local_id(e.head) && key(g_global) < key(e.head)) {
                expected.push_back(e.head);
            }
        }
        std::ranges::sort(expected);
        EXPECT_EQ(globals(G_oriented, G_oriented.ghost_out_neighbors(g)), expected);
        EXPECT_TRUE(std::ranges::is_sorted(G_oriented.ghost_out_neighbors(g)));
    }
}

TEST_P(Orientation, LocalDegreeOutward) {
    auto G = input_graph();
    // an arbitrary key: cut edges outward, local edges by (degree, id), no exchange
    auto const G_oriented = katric::graph::orient(
        std::move(G),
        [&](VertexHandle c) {
            Degree const degree = G.is_local(c) ? G.degree(c) : std::numeric_limits<Degree>::max();
            return std::pair(degree, G.global_id(c));
        },
        false
    );
    for (VertexHandle const v: G_oriented.local_vertices()) {
        VertexId const        v_global = G_oriented.global_id(v);
        std::vector<VertexId> expected;
        for (auto const& e: G_full[v_global]) {
            if (!G_oriented.layer().is_local_id(e.head) || key(v_global) < key(e.head)) {
                expected.push_back(e.head);
            }
        }
        std::ranges::sort(expected);
        EXPECT_EQ(globals(G_oriented, G_oriented.neighbors(v)), expected);
    }
}

const char* graphs[] = {"examples/metis-sample.metis", "examples/triangle.metis", "examples/hiv.metis"};
INSTANTIATE_TEST_SUITE_P(SimpleGraphs, Orientation, ::testing::ValuesIn(graphs));
} // namespace
