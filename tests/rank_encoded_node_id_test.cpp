#include "datastructures/graph_definitions.h"
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

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
