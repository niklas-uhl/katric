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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <adiak.hpp>
#include <cereal/cereal.hpp>

namespace cetric {

/// cereal output archive that records every serialized value as an Adiak name/value pair, so that anything with a
/// cereal `serialize()` (e.g. Config) ends up as run metadata in the Caliper profile. Nested names are
/// joined with '.', unnamed container elements are numbered (e.g. `config.list.0`).
/// It is a text archive, i.e. enums are written as their names (see config.h).
class AdiakOutputArchive : public cereal::OutputArchive<AdiakOutputArchive>, public cereal::traits::TextArchive {
public:
    explicit AdiakOutputArchive(std::string root_prefix = "")
        : cereal::OutputArchive<AdiakOutputArchive>(this),
          prefix_(std::move(root_prefix)) {}

    void setNextName(char const* name) {
        next_name_ = name;
    }

    void startNode() {
        std::string name = next_name_ ? std::string(*next_name_) : std::to_string(counters_.empty() ? 0 : counters_.back()++);
        next_name_.reset();
        names_.push_back(std::move(name));
        counters_.push_back(0);
    }

    void finishNode() {
        names_.pop_back();
        counters_.pop_back();
    }

    template <typename T>
    void record(T const& value) {
        startNode();
        adiak::value(full_name(), value);
        finishNode();
    }

private:
    std::string full_name() const {
        std::string result = prefix_;
        for (auto const& n: names_) {
            if (!result.empty()) {
                result += '.';
            }
            result += n;
        }
        return result;
    }

    std::string                prefix_;
    std::vector<std::string>   names_;
    std::vector<std::size_t>   counters_;
    std::optional<std::string> next_name_;
};

template <typename T>
constexpr bool is_adiak_leaf_v = std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || std::is_enum_v<T>;

// nodes for everything that is not a leaf value or a name/size tag
template <typename T, std::enable_if_t<!is_adiak_leaf_v<T>, int> = 0>
inline void prologue(AdiakOutputArchive& ar, T const&) {
    ar.startNode();
}
template <typename T, std::enable_if_t<!is_adiak_leaf_v<T>, int> = 0>
inline void epilogue(AdiakOutputArchive& ar, T const&) {
    ar.finishNode();
}
template <typename T, std::enable_if_t<is_adiak_leaf_v<T>, int> = 0>
inline void prologue(AdiakOutputArchive&, T const&) {}
template <typename T, std::enable_if_t<is_adiak_leaf_v<T>, int> = 0>
inline void epilogue(AdiakOutputArchive&, T const&) {}
template <typename T>
inline void prologue(AdiakOutputArchive&, cereal::NameValuePair<T> const&) {}
template <typename T>
inline void epilogue(AdiakOutputArchive&, cereal::NameValuePair<T> const&) {}
template <typename T>
inline void prologue(AdiakOutputArchive&, cereal::SizeTag<T> const&) {}
template <typename T>
inline void epilogue(AdiakOutputArchive&, cereal::SizeTag<T> const&) {}

template <typename T>
inline void CEREAL_SAVE_FUNCTION_NAME(AdiakOutputArchive& ar, cereal::NameValuePair<T> const& nvp) {
    ar.setNextName(nvp.name);
    ar(nvp.value);
}
template <typename T>
inline void CEREAL_SAVE_FUNCTION_NAME(AdiakOutputArchive&, cereal::SizeTag<T> const&) {}

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
inline void CEREAL_SAVE_FUNCTION_NAME(AdiakOutputArchive& ar, T const& value) {
    if constexpr (std::is_same_v<T, bool>) {
        ar.record(static_cast<long>(value)); // Adiak has no bool type
    } else if constexpr (std::is_floating_point_v<T>) {
        ar.record(static_cast<double>(value));
    } else if constexpr (std::is_unsigned_v<T>) {
        ar.record(static_cast<unsigned long>(value));
    } else {
        ar.record(static_cast<long>(value));
    }
}
inline void CEREAL_SAVE_FUNCTION_NAME(AdiakOutputArchive& ar, std::string const& value) {
    ar.record(value);
}

} // namespace cetric
