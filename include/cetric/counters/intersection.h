#pragma once

#include <algorithm>
#include <iostream>

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
OutputIt
binary_intersection(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp) {
    constexpr size_t recursion_threshold = 5;

    size_t dist1 = std::distance(first1, last1);
    size_t dist2 = std::distance(first2, last2);

    // if both lists get too small, we use merge intersection
    if (dist1 <= recursion_threshold && dist2 <= recursion_threshold) {
        return merge_intersection(first1, last1, first2, last2, d_first, comp);
    }
    if (dist1 == 0 || dist2 == 0) {
        return d_first;
    }

    // use smaller list for splitting
    if (dist1 <= dist2) {
        auto pivot1 = first1 + dist1 / 2;
        auto pivot2 = std::lower_bound(first2, last2, *pivot1, comp);

        // avoid endless recursion
        if (!(pivot1 == last1 && pivot2 == last2)) {
            d_first = binary_intersection(first1, pivot1, first2, pivot2, d_first, comp);
        }
        if (!(pivot2 == last2) && !comp(*pivot1, *pivot2)) {
            // duplicates may lead to undefined behavior
            *d_first = *pivot1;
            pivot1++;
            pivot2++;
            d_first++;
        }
        if (!(pivot1 == first1 && pivot2 == first2)) {
            d_first = binary_intersection(pivot1, last1, pivot2, last2, d_first, comp);
        }
    } else {
        auto pivot2 = first2 + dist2 / 2;
        auto pivot1 = std::lower_bound(first1, last1, *pivot2, comp);
        if (!(pivot2 == last2 && pivot1 == last1)) {
            d_first = binary_intersection(first2, pivot2, first1, pivot1, d_first, comp);
        }
        if (!(pivot1 == last1) && !comp(*pivot2, *pivot1)) {
            *d_first = *pivot2;
            pivot1++;
            pivot2++;
            d_first++;
        }
        if (!(pivot2 == first2 && pivot1 == first1)) {
            d_first = binary_intersection(pivot2, last2, pivot1, last1, d_first, comp);
        }
    }
    return d_first;
}

} // namespace cetric
