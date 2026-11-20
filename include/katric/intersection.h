/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Intersections of two sorted neighborhoods (ascending, no duplicates), by the
// method of `--intersection-method`.
#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>

#include <fmt/ranges.h>
#include <kassert/kassert.hpp>

#include "katric/config.h"

namespace katric {

/// A range the intersections read (as const): random access and sized, for
/// the binary searches and the methods' size tests. Sortedness is a
/// precondition, checked with assertions on.
template <typename R>
concept IntersectableRange = std::ranges::random_access_range<R const> &&
                             std::ranges::sized_range<R const>;

/// Two ranges to intersect: elements comparable with each other.
template <typename R1, typename R2>
concept IntersectableRanges =
    IntersectableRange<R1> && IntersectableRange<R2> &&
    std::totally_ordered_with<std::ranges::range_reference_t<R1 const>,
                              std::ranges::range_reference_t<R2 const>>;

/// Called with each common element, taken from either range.
template <typename F, typename R1, typename R2>
concept IntersectionCallback =
    std::invocable<F&, std::ranges::range_reference_t<R1 const>> &&
    std::invocable<F&, std::ranges::range_reference_t<R2 const>>;

/// Walks both ranges once: O(|left| + |right|).
template <typename R1, typename R2, typename F>
  requires IntersectableRanges<R1, R2> && IntersectionCallback<F, R1, R2>
void merge_intersection(R1 const& left, R2 const& right, F on_intersection) {
  auto first1 = std::ranges::begin(left);
  auto const last1 = std::ranges::end(left);
  auto first2 = std::ranges::begin(right);
  auto const last2 = std::ranges::end(right);
  while (first1 != last1 && first2 != last2) {
    if (*first1 < *first2) {
      ++first1;
    } else if (*first2 < *first1) {
      ++first2;
    } else {
      on_intersection(*first1);
      ++first1;
      ++first2;
    }
  }
}

/// Looks up each element of the smaller range in the larger one:
/// O(min · log max).
template <typename R1, typename R2, typename F>
  requires IntersectableRanges<R1, R2> && IntersectionCallback<F, R1, R2>
void binary_search_intersection(R1 const& left,
                                R2 const& right,
                                F on_intersection) {
  auto const search = [&](auto&& smaller, auto&& larger) -> void {
    for (auto&& element : smaller) {
      if (std::ranges::binary_search(larger, element)) {
        on_intersection(element);
      }
    }
  };
  if (std::ranges::size(left) <= std::ranges::size(right)) {
    search(left, right);
  } else {
    search(right, left);
  }
}

namespace internal {

/// binary_intersection on subranges, so that the recursion keeps one type per
/// side. The pivot (middle of the smaller side) is excluded from both halves,
/// so each call shrinks a side: the depth is at most log |left| + log |right|.
template <typename I1, typename I2, typename F>
void binary_intersection(std::ranges::subrange<I1> left,
                         std::ranges::subrange<I2> right,
                         F& on_intersection,
                         std::size_t cutoff) {
  if (left.size() <= cutoff && right.size() <= cutoff) {
    merge_intersection(left, right, on_intersection);
    return;
  }
  if (left.empty() || right.empty()) {
    return;
  }
  if (left.size() <= right.size()) {
    auto const pivot1 = left.begin() + (left.size() / 2);
    auto pivot2 = std::ranges::lower_bound(right, *pivot1);
    binary_intersection(std::ranges::subrange(left.begin(), pivot1),
                        std::ranges::subrange(right.begin(), pivot2),
                        on_intersection, cutoff);
    if (pivot2 != right.end() && !(*pivot1 < *pivot2)) {
      on_intersection(*pivot1);
      ++pivot2;
    }
    binary_intersection(std::ranges::subrange(pivot1 + 1, left.end()),
                        std::ranges::subrange(pivot2, right.end()),
                        on_intersection, cutoff);
  } else {
    auto const pivot2 = right.begin() + (right.size() / 2);
    auto pivot1 = std::ranges::lower_bound(left, *pivot2);
    binary_intersection(std::ranges::subrange(left.begin(), pivot1),
                        std::ranges::subrange(right.begin(), pivot2),
                        on_intersection, cutoff);
    if (pivot1 != left.end() && !(*pivot2 < *pivot1)) {
      on_intersection(*pivot2);
      ++pivot1;
    }
    binary_intersection(std::ranges::subrange(pivot1, left.end()),
                        std::ranges::subrange(pivot2 + 1, right.end()),
                        on_intersection, cutoff);
  }
}

}  // namespace internal

/// Divide and conquer: splits the smaller range at its middle, finds the
/// pivot in the larger one by binary search, and recurses on both sides; parts
/// with both sides at most `cutoff` long are merged.
template <typename R1, typename R2, typename F>
  requires IntersectableRanges<R1, R2> && IntersectionCallback<F, R1, R2>
void binary_intersection(R1 const& left,
                         R2 const& right,
                         F on_intersection,
                         std::size_t cutoff) {
  internal::binary_intersection(std::ranges::subrange(left),
                                std::ranges::subrange(right), on_intersection,
                                cutoff);
}

/// Intersects two sorted ranges with the method of `--intersection-method`;
/// calls `on_intersection` per common element.
template <typename R1, typename R2, typename F>
  requires IntersectableRanges<R1, R2> && IntersectionCallback<F, R1, R2>
void intersection(R1 const& left,
                  R2 const& right,
                  F on_intersection,
                  Config const& conf) {
  KASSERT(std::ranges::is_sorted(left), fmt::format("{}", left));
  KASSERT(std::ranges::adjacent_find(left) == std::ranges::end(left));
  KASSERT(std::ranges::is_sorted(right), fmt::format("{}", right));
  KASSERT(std::ranges::adjacent_find(right) == std::ranges::end(right));

  switch (conf.intersection_method) {
    case IntersectionMethod::merge:
      merge_intersection(left, right, on_intersection);
      break;
    case IntersectionMethod::binary_search:
      binary_search_intersection(left, right, on_intersection);
      break;
    case IntersectionMethod::binary:
      binary_intersection(left, right, on_intersection, conf.binary_intersection_cutoff);
      break;
    case IntersectionMethod::hybrid: {
      // binary search if min · log max < scale · (min + max), the two costs
      // above; log2 as the bit width (an integer, enough for left cutoff)
      std::size_t const size_left = std::ranges::size(left);
      std::size_t const size_right = std::ranges::size(right);
      std::size_t const smaller = std::min(size_left, size_right);
      std::size_t const larger = std::max(size_left, size_right);
      auto const search_cost = static_cast<double>(
          smaller * static_cast<std::size_t>(std::bit_width(larger)));
      if (search_cost < conf.hybrid_cutoff_scale *
                            static_cast<double>(smaller + larger)) {
        binary_search_intersection(left, right, on_intersection);
      } else {
        merge_intersection(left, right, on_intersection);
      }
      break;
    }
  }
}

}  // namespace katric
