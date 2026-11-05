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
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "cetric/counters/intersection.h"

namespace {

using Vec = std::vector<size_t>;

std::string to_string(Vec const& v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i == 0 ? "" : ", ") + std::to_string(v[i]);
    }
    return s + "}";
}

// Runs all intersection variants on (v1, v2) and (v2, v1) and compares against expected.
void check_all_intersections(Vec v1, Vec v2, Vec const& expected) {
    using ::testing::ElementsAreArray;
    using ::testing::UnorderedElementsAreArray;
    for (bool swap: {false, true}) {
        if (swap) {
            v1.swap(v2);
        }
        SCOPED_TRACE("v1 = " + to_string(v1) + ", v2 = " + to_string(v2));
        auto run = [&](auto&& intersect) {
            Vec result;
            intersect(
                v1.begin(),
                v1.end(),
                v2.begin(),
                v2.end(),
                [&result](auto const& val) { result.push_back(val); },
                std::less<>{}
            );
            return result;
        };
        auto merge = [](auto&&... args) {
            cetric::merge_intersection(std::forward<decltype(args)>(args)...);
        };
        auto binary_search = [](auto&&... args) {
            cetric::binary_search_intersection(std::forward<decltype(args)>(args)...);
        };
        auto binary = [](auto&&... args) {
            cetric::binary_intersection(std::forward<decltype(args)>(args)...);
        };
        EXPECT_THAT(run(merge), ElementsAreArray(expected)) << "merge_intersection";
        EXPECT_THAT(run(binary_search), ElementsAreArray(expected)) << "binary_search_intersection";
        EXPECT_THAT(run(binary), UnorderedElementsAreArray(expected)) << "binary_intersection";
    }
}

} // namespace

TEST(IntersectionTest, Simple) {
    struct Input {
        Vec v1, v2, expected;
    };
    std::vector<Input> const inputs = {
        {{0, 1, 3, 6, 9, 478}, {1, 6, 7, 8, 478}, {1, 6, 478}},
        {{}, {1, 6, 7, 8, 478}, {}},
        {{}, {}, {}},
        {{1}, {}, {}},
        {{1}, {1}, {1}},
        {{1}, {42}, {}},
        {{13, 46, 54, 58, 64}, {6, 21, 43, 47, 75}, {}},
        {{1, 2, 5, 9, 10}, {1, 2, 4, 5, 6, 7, 8}, {1, 2, 5}},
        {{2, 3, 4, 7, 10}, {3, 5, 6}, {3}},
        {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11, 12}, {}},
    };
    for (auto const& input: inputs) {
        check_all_intersections(input.v1, input.v2, input.expected);
    }
}

TEST(IntersectionTest, Random) {
    constexpr size_t                      max_length = 1000;
    constexpr size_t                      iterations = 100;
    std::mt19937_64                       eng(42);
    std::uniform_int_distribution<size_t> size_dist(0, max_length);
    std::uniform_int_distribution<size_t> value_dist(1, std::max(size_t{10}, max_length / 10));

    auto gen_input = [&](size_t size) {
        Vec v(size);
        std::generate(v.begin(), v.end(), [&] { return value_dist(eng); });
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    };

    for (size_t i = 0; i < iterations; ++i) {
        Vec v1 = gen_input(size_dist(eng));
        Vec v2 = gen_input(size_dist(eng));
        for (bool v2_empty: {false, true}) {
            Vec v2_used = v2_empty ? Vec{} : v2;
            Vec expected;
            std::set_intersection(
                v1.begin(),
                v1.end(),
                v2_used.begin(),
                v2_used.end(),
                std::back_inserter(expected)
            );
            check_all_intersections(v1, v2_used, expected);
        }
    }
}
