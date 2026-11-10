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

#ifndef STATISTICS_H_PLFYJDX0
#define STATISTICS_H_PLFYJDX0

#include <cereal/cereal.hpp>

#include "cetric/message_statistics.h"

namespace cetric {
namespace profiling {

struct PreprocessingStatistics {
    double            orientation_time;
    double            sorting_time;
    double            degree_exchange_time;
    MessageStatistics message_statistics;

    explicit PreprocessingStatistics() : orientation_time(0), sorting_time(0), message_statistics() {}

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(degree_exchange_time),
            CEREAL_NVP(orientation_time),
            CEREAL_NVP(sorting_time),
            CEREAL_NVP(message_statistics)
        );
    }
};

struct LoadBalancingStatistics {
    double            cost_function_evaluation_time;
    double            cost_function_communication_time;
    double            computation_time;
    double            redistribution_time;
    double            phase_time;
    MessageStatistics message_statistics;

    explicit LoadBalancingStatistics()
        : cost_function_evaluation_time(0),
          cost_function_communication_time(0),
          computation_time(0),
          redistribution_time(0),
          phase_time(0),
          message_statistics() {}

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(cost_function_evaluation_time),
            CEREAL_NVP(cost_function_communication_time),
            CEREAL_NVP(computation_time),
            CEREAL_NVP(redistribution_time),
            CEREAL_NVP(phase_time),
            CEREAL_NVP(message_statistics)
        );
    }
};

} /* end of namespace profiling */
} // namespace cetric

#endif /* end of include guard: STATISTICS_H_PLFYJDX0 */
