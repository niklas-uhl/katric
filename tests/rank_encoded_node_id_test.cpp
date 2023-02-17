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

#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "cetric/datastructures/graph_definitions.h"

TEST_CASE("Rank encoding works") {
    cetric::graph::RankEncodedNodeId node_id{42};
    REQUIRE(node_id.id() == 42);
    node_id.set_rank(255);
    REQUIRE(node_id.id() == 42);
    REQUIRE(node_id.rank() == 255);
    REQUIRE(node_id.data() != 42);
    node_id.set_rank(0);
    REQUIRE(node_id.id() == 42);
    REQUIRE(node_id.rank() == 0);
    REQUIRE(node_id.data() == 42);
    REQUIRE(cetric::graph::RankEncodedNodeId::sentinel().data() == std::numeric_limits<uint64_t>::max());
    // node_id.set_rank(1'000'000'000);
}
