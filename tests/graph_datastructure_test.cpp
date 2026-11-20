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


#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "katric/graph/graph.h"
#include "test_graphs.h"

namespace {
using katric::testing::Edge;
using katric::graph::VertexHandle;
using katric::graph::VertexId;

TEST(InputGraphTest, neighbors_ghosts_and_handles) {
    auto const input  = "examples/hiv.metis";
    auto const G_full = katric::testing::read_full_graph(input);
    auto const local  = katric::testing::read_local_graph(input);
    auto const G      = katric::graph::Graph::from_kagen(local, MPI_COMM_WORLD);
    auto const& numbering = G.numbering();

    // every local vertex has its full neighborhood, in global ids
    for (VertexHandle const v: G.local_vertices()) {
        VertexId const      v_global = G.global_id(v);
        std::vector<Edge> edges;
        for (VertexHandle const u: G.local_neighbors(v)) {
            edges.emplace_back(v_global, G.global_id(u));
            EXPECT_EQ(numbering.handle(G.global_id(u)), u);
        }
        EXPECT_THAT(edges, testing::UnorderedElementsAreArray(G_full[v_global]));
    }
    // the ghosts are sorted by global id, owned elsewhere, and their owners' ranges contain them
    for (VertexHandle const g: G.ghosts()) {
        EXPECT_FALSE(G.is_local(g));
        EXPECT_TRUE(numbering.distribution().is_local(G.global_id(g), G.owner(g)));
    }
    for (std::size_t i = 1; i < G.num_ghosts(); ++i) {
        EXPECT_LT(G.global_id(numbering.ghost_handle(i - 1)), G.global_id(numbering.ghost_handle(i)));
    }
    // every vertex is local on exactly one rank
    std::size_t const local_count = G.num_local_vertices();
    std::size_t       total       = 0;
    MPI_Allreduce(&local_count, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total, G_full.size());
}
} // namespace
