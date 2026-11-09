// Copyright (c) 2026 Tim Niklas Uhl
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

// Hash table types used throughout katric: Abseil's flat_hash_map/set if CETRIC_USE_ABSEIL is defined, sparsehash's
// dense_hash_map/set otherwise.

#pragma once

#ifdef CETRIC_USE_ABSEIL
    #include <absl/container/flat_hash_map.h>
    #include <absl/container/flat_hash_set.h>
    #include <absl/hash/hash.h>
#else
    #include <functional>

    #include <sparsehash/dense_hash_map>
    #include <sparsehash/dense_hash_set>
#endif

namespace cetric {

#ifdef CETRIC_USE_ABSEIL
template <typename Key, typename Value, typename Hash = absl::Hash<Key>>
using hash_map = absl::flat_hash_map<Key, Value, Hash>;

template <typename Key, typename Hash = absl::Hash<Key>>
using hash_set = absl::flat_hash_set<Key, Hash>;
#else
template <typename Key, typename Value, typename Hash = std::hash<Key>>
using hash_map = google::dense_hash_map<Key, Value, Hash>;

template <typename Key, typename Hash = std::hash<Key>>
using hash_set = google::dense_hash_set<Key, Hash>;
#endif

/// sparsehash marks empty slots with a key that is never inserted and requires it to be set before the first use.
/// Abseil needs no such key, so this does nothing there.
template <typename HashTable, typename Key>
inline void set_empty_key([[maybe_unused]] HashTable& table, [[maybe_unused]] Key const& key) {
#ifndef CETRIC_USE_ABSEIL
    table.set_empty_key(key);
#endif
}

} // namespace cetric
