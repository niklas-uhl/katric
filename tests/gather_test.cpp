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

#include <numeric>

#include "cetric/communicator.h"
#include <gtest/gtest.h>
#include <mpi.h>

class CommunicatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
  cetric::PEID rank, size;
};

TEST_F(CommunicatorTest, AllGatherRegularWorks) {
    std::vector<int> result(3 * size);
    std::iota(result.begin(), result.end(), 0);
    std::vector<int> values    = {rank * 3, rank * 3 + 1, rank * 3 + 2};
    auto [recv_buffer, displs] = cetric::CommunicationUtility::all_gather(values, MPI_INT, MPI_COMM_WORLD, rank, size);
    ASSERT_EQ(recv_buffer, result);
}

TEST_F(CommunicatorTest, AllGatherIrregularWorks) {
    std::vector<int> result((size * (size - 1)) / 2);
    std::iota(result.begin(), result.end(), 0);
    std::vector<int> values(rank);
    std::iota(values.begin(), values.end(), ((rank - 1) * rank) / 2);
    auto [recv_buffer, displs] = cetric::CommunicationUtility::all_gather(values, MPI_INT, MPI_COMM_WORLD, rank, size);
    ASSERT_EQ(recv_buffer, result);
}

TEST_F(CommunicatorTest, AllGatherEmptyWorks) {
    std::vector<int> result;
    std::vector<int> values;
    auto [recv_buffer, displs] = cetric::CommunicationUtility::all_gather(values, MPI_INT, MPI_COMM_WORLD, rank, size);
    ASSERT_EQ(recv_buffer, result);
}
