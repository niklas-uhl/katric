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

#ifndef MESSAGE_STATISTICS_H
#define MESSAGE_STATISTICS_H

#include <cstddef>

#include "cetric/caliper.h"

namespace cetric {
namespace profiling {

/// Messages and volume (in elements) of one communication step, counted locally and reported once at its end.
struct MessageStatistics {
    size_t sent_messages     = 0;
    size_t received_messages = 0;
    size_t send_volume       = 0;
    size_t receive_volume    = 0;

    /// Adds the counts to the open Caliper region (see counter_add()); call it from the thread that opened it.
    void report() const {
        counter_add("sent_messages", sent_messages);
        counter_add("send_volume", send_volume);
        counter_add("received_messages", received_messages);
        counter_add("receive_volume", receive_volume);
    }
};
} // namespace profiling
} // namespace cetric

#endif /* MESSAGE_STATISTICS_H */
