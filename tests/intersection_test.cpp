#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <random>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "catch2/generators/catch_generators.hpp"
#include "cetric/counters/intersection.h"

TEST_CASE("intersection works", "[simple]") {
    auto constexpr make_input =
        [](std::initializer_list<size_t> v1, std::initializer_list<size_t> v2, std::initializer_list<size_t> expected) {
            return std::make_tuple(std::vector(v1), std::vector(v2), std::vector(expected));
        };

    auto inputs = GENERATE_REF(
        make_input({0, 1, 3, 6, 9, 478}, {1, 6, 7, 8, 478}, {1, 6, 478}),
        make_input({}, {1, 6, 7, 8, 478}, {}),
        make_input({}, {}, {}),
        make_input({1}, {}, {}),
        make_input({1}, {1}, {1}),
        make_input({1}, {42}, {}),
        make_input({13, 46, 54, 58, 64}, {6, 21, 43, 47, 75}, {})
    );

    auto v1       = std::get<0>(inputs);
    auto v2       = std::get<1>(inputs);
    auto expected = std::get<2>(inputs);

    bool swap = GENERATE(false, true);
    if (swap) {
        v1.swap(v2);
    }

    CAPTURE(v1, v2);

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
    constexpr size_t    max_length   = 10000;
    std::vector<size_t> vector_sizes = GENERATE(take(100, chunk(2, random(size_t{0}, max_length))));
    auto                v1_size      = vector_sizes[0];
    auto                v2_size      = vector_sizes[1];

    std::random_device                    dev;
    std::default_random_engine            eng(dev());
    std::uniform_int_distribution<size_t> dist(1, max_length / 10);

    auto gen = [&dist, &eng]() {
        return dist(eng);
    };
    auto gen_input = [&](size_t size) {
        std::vector<size_t> v(size);
        std::generate(v.begin(), v.end(), gen);
        std::sort(v.begin(), v.end());
        auto v_last = std::unique(v.begin(), v.end());
        v.erase(v_last, v.end());
        return v;
    };

    auto v1 = gen_input(v1_size);
    auto v2 = gen_input(v2_size);

    bool v2_empty = GENERATE(false, true);
    if (v2_empty) {
        v2 = {};
    }

    bool swap = GENERATE(false, true);
    if (swap) {
        v1.swap(v2);
    }
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
