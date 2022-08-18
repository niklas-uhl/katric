#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stack>

namespace cetric {

template <class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt
merge_intersection(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp) {
    return std::set_intersection(first1, last1, first2, last2, d_first, comp);
}

template <class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt binary_search_intersection(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp
) {
    auto dist1 = std::distance(first1, last1);
    auto dist2 = std::distance(first2, last2);
    if (dist1 <= dist2) {
        for (auto current = first1; current != last1; ++current) {
            if (std::binary_search(first2, last2, *current, comp)) {
                *d_first = *current;
                d_first++;
            }
        }
    } else {
        for (auto current = first2; current != last2; ++current) {
            if (std::binary_search(first1, last1, *current, comp)) {
                *d_first = *current;
                d_first++;
            }
        }
    }
    return d_first;
}

template <class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt binary_intersection(
    InputIt1 _first1,
    InputIt1 _last1,
    InputIt2 _first2,
    InputIt2 _last2,
    OutputIt d_first,
    Compare  comp,
    size_t   recursion_threshold = 1000
) {
#if !defined(CETRIC_BINARY_INTERSECTION_RECURSIVE)
    struct Task {
        size_t first1 = 0, last1 = 0, first2 = 0, last2 = 0;
        size_t dist1() const {
            return last1 - first1;
        }
        size_t dist2() const {
            return last2 - first2;
        }
    };
    std::stack<Task> stack;
    {
        size_t dist1 = std::distance(_first1, _last1);
        size_t dist2 = std::distance(_first2, _last2);
        stack.emplace(Task{0, dist1, 0, dist2});
    }

    while (!stack.empty()) {
        Task task = stack.top();
        stack.pop();
        auto first1 = _first1 + task.first1;
        auto last1  = _first1 + task.last1;

        auto first2 = _first2 + task.first2;
        auto last2  = _first2 + task.last2;

        if (task.dist1() <= recursion_threshold && task.dist2() <= recursion_threshold) {
            d_first = merge_intersection(first1, last1, first2, last2, d_first, comp);
            continue;
        }
        if (task.dist1() == 0 || task.dist2() == 0) {
            continue;
        }
        if (task.dist1() <= task.dist2()) {
            auto pivot1 = first1 + task.dist1() / 2;
            auto pivot2 = std::lower_bound(first2, last2, *pivot1, comp);

            // avoid endless recursion
            if (!(pivot1 == last1 && pivot2 == last2)) {
                stack.emplace(Task{
                    task.first1,
                    task.first1 + std::distance(first1, pivot1),
                    task.first2,
                    task.first2 + std::distance(first2, pivot2)});
                /* d_first = binary_intersection(first1, pivot1, first2, pivot2, d_first, comp); */
            }
            if (!(pivot2 == last2) && !comp(*pivot1, *pivot2)) {
                // duplicates may lead to undefined behavior
                *d_first = *pivot1;
                d_first++;
                pivot1++;
                pivot2++;
            }
            if (!(pivot1 == first1 && pivot2 == first2)) {
                stack.emplace(Task{
                    task.first1 + std::distance(first1, pivot1),
                    task.last1,
                    task.first2 + std::distance(first2, pivot2),
                    task.last2});
                /* d_first = binary_intersection(pivot1, last1, pivot2, last2,
                 * d_first, comp); */
            }
        } else {
            auto pivot2 = first2 + task.dist2() / 2;
            auto pivot1 = std::lower_bound(first1, last1, *pivot2, comp);
            if (!(pivot2 == last2 && pivot1 == last1)) {
                stack.emplace(Task{
                    task.first1,
                    task.first1 + std::distance(first1, pivot1),
                    task.first2,
                    task.first2 + std::distance(first2, pivot2)});
            }
            if (!(pivot1 == last1) && !comp(*pivot2, *pivot1)) {
                *d_first = *pivot2;
                pivot1++;
                pivot2++;
                d_first++;
            }
            if (!(pivot2 == first2 && pivot1 == first1)) {
                stack.emplace(Task{
                    task.first1 + std::distance(first1, pivot1),
                    task.last1,
                    task.first2 + std::distance(first2, pivot2),
                    task.last2});
            }
        }
    }

#else
  size_t dist1 = std::distance(_first1, _last1);
  size_t dist2 = std::distance(_first2, _last2);

  // if both lists get too small, we use merge intersection
  if (dist1 <= recursion_threshold && dist2 <= recursion_threshold) {
    return merge_intersection(_first1, _last1, _first2, _last2, d_first, comp);
    }
    if (dist1 == 0 || dist2 == 0) {
        return d_first;
    }

    // use smaller list for splitting
    if (dist1 <= dist2) {
        auto pivot1 = _first1 + dist1 / 2;
        auto pivot2 = std::lower_bound(_first2, _last2, *pivot1, comp);

        // avoid endless recursion
        if (!(pivot1 == _last1 && pivot2 == _last2)) {
            d_first = binary_intersection(_first1, pivot1, _first2, pivot2, d_first, comp);
        }
        if (!(pivot2 == _last2) && !comp(*pivot1, *pivot2)) {
            // duplicates may lead to undefined behavior
            *d_first = *pivot1;
            pivot1++;
            pivot2++;
            d_first++;
        }
        if (!(pivot1 == _first1 && pivot2 == _first2)) {
            d_first = binary_intersection(pivot1, _last1, pivot2, _last2, d_first, comp);
        }
    } else {
        auto pivot2 = _first2 + dist2 / 2;
        auto pivot1 = std::lower_bound(_first1, _last1, *pivot2, comp);
        if (!(pivot2 == _last2 && pivot1 == _last1)) {
            d_first = binary_intersection(_first2, pivot2, _first1, pivot1, d_first, comp);
        }
        if (!(pivot1 == _last1) && !comp(*pivot2, *pivot1)) {
            *d_first = *pivot2;
            pivot1++;
            pivot2++;
            d_first++;
        }
        if (!(pivot2 == _first2 && pivot1 == _first1)) {
            d_first = binary_intersection(pivot2, _last2, pivot1, _last1, d_first, comp);
        }
    }
#endif
    return d_first;
}

} // namespace cetric
