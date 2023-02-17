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

#include <cereal/cereal.hpp>
#include <mpi.h>

namespace cetric {
namespace profiling {

struct MessageStatistics {
    size_t sent_messages;
    size_t received_messages;
    size_t send_volume;
    size_t receive_volume;

    explicit MessageStatistics() : sent_messages(0), received_messages(0), send_volume(0), receive_volume(0) {}

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(sent_messages),
            CEREAL_NVP(received_messages),
            CEREAL_NVP(send_volume),
            CEREAL_NVP(receive_volume)
        );
    }

    void add(const MessageStatistics& rhs) {
        sent_messages += rhs.sent_messages;
        received_messages += rhs.received_messages;
        send_volume += rhs.send_volume;
        receive_volume += rhs.receive_volume;
    }
};
} // namespace profiling
} // namespace cetric

#endif /* MESSAGE_STATISTICS_H */
