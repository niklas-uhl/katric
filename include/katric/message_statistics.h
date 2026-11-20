/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "katric/caliper.h"

namespace katric {
namespace profiling {

/// Messages and volume (in elements) of one communication step, counted locally
/// and reported once at its end.
struct MessageStatistics {
  std::size_t sent_messages = 0;
  std::size_t received_messages = 0;
  std::size_t send_volume = 0;
  std::size_t receive_volume = 0;

  /// Adds the counts to the open Caliper region (see counter_add()); call it
  /// from the thread that opened it.
  void report() const {
    counter_add("sent_messages", static_cast<std::int64_t>(sent_messages));
    counter_add("send_volume", static_cast<std::int64_t>(send_volume));
    counter_add("received_messages",
                static_cast<std::int64_t>(received_messages));
    counter_add("receive_volume", static_cast<std::int64_t>(receive_volume));
  }
};
}  // namespace profiling
}  // namespace katric
