#include <algorithm>
#include <iterator>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "cetric/counters/intersection.h"

TEST_CASE("intersection works", "[simple]") {
    bool swap = GENERATE(false, true);
    // bool swap = false;

    // std::vector<size_t> v1 = {0, 1, 3, 6, 9, 478};
    // std::vector<size_t> v2 = {1, 6, 7, 8, 478};
    // std::vector<size_t> v1 = {13, 46, 54, 58, 64};
    // std::vector<size_t> v2 = {6, 21, 43, 47, 75};
    std::vector<size_t> v1       = {};
    std::vector<size_t> v2       = {6, 21, 43, 47, 75};
    std::vector<size_t> expected = {};
    if (swap) {
        v1.swap(v2);
    }
    std::vector<size_t> result;
    SECTION("with merge intersection") {
        cetric::merge_intersection(
            v1.begin(),
            v1.end(),
            v2.begin(),
            v2.end(),
            std::back_inserter(result),
            std::less<>{}
        );
        REQUIRE_THAT(result, Catch::Matchers::Equals(expected));
    }
    SECTION("with binary search intersection") {
        cetric::binary_search_intersection(
            v1.begin(),
            v1.end(),
            v2.begin(),
            v2.end(),
            std::back_inserter(result),
            std::less<>{}
        );
        REQUIRE_THAT(result, Catch::Matchers::Equals(expected));
    }
    SECTION("with binary intersection") {
        cetric::binary_intersection(
            v1.begin(),
            v1.end(),
            v2.begin(),
            v2.end(),
            std::back_inserter(result),
            std::less<>{}
        );
        REQUIRE_THAT(result, Catch::Matchers::Equals(expected));
    }
}

TEST_CASE("intersection works with random values", "[random]") {
    std::vector<size_t> v1 = GENERATE(take(100, chunk(5, random(size_t{0}, size_t{100}))));
    std::sort(v1.begin(), v1.end());
    auto v1_last = std::unique(v1.begin(), v1.end());
    v1.erase(v1_last, v1.end());
    std::vector<size_t> v2 = GENERATE(take(100, chunk(5, random(size_t{0}, size_t{100}))));
    std::sort(v2.begin(), v2.end());
    auto v2_last = std::unique(v2.begin(), v2.end());
    v2.erase(v2_last, v2.end());
    CAPTURE(v1, v2);

    std::vector<size_t> expected;
    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(expected));

    std::vector<size_t> result;
    SECTION("with merge intersection") {
        cetric::merge_intersection(
            v1.begin(),
            v1.end(),
            v2.begin(),
            v2.end(),
            std::back_inserter(result),
            std::less<>{}
        );
        REQUIRE_THAT(result, Catch::Matchers::Equals(expected));
    }
    SECTION("with binary search intersection") {
        cetric::binary_search_intersection(
            v1.begin(),
            v1.end(),
            v2.begin(),
            v2.end(),
            std::back_inserter(result),
            std::less<>{}
        );
        REQUIRE_THAT(result, Catch::Matchers::Equals(expected));
    }
    SECTION("with binary intersection") {
        cetric::binary_intersection(
            v1.begin(),
            v1.end(),
            v2.begin(),
            v2.end(),
            std::back_inserter(result),
            std::less<>{}
        );
        REQUIRE_THAT(result, Catch::Matchers::Equals(expected));
    }
}
