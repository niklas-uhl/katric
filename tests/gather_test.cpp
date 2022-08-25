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
