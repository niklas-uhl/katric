/******************************************************************************
 *
 * Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC
 * and other gtest-mpi-listener developers. See the COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 ******************************************************************************/

#include "gtest/gtest.h"
#include "mpi.h"

// Simple-minded functions for some testing

namespace {
// Always passes out == rank
int getMpiRank(MPI_Comm comm) {
    int out;
    MPI_Comm_rank(comm, &out);
    return out;
}

// Always fails out == rank
int getMpiRankPlusOne(MPI_Comm comm) {
    int out;
    MPI_Comm_rank(comm, &out);
    return (out + 1);
}

// Passes out == rank when rank is zero, fails otherwise
int getZero(MPI_Comm comm) {
    return 0;
}

// Passes out == rank except on rank zero, fails otherwise
int getNonzeroMpiRank(MPI_Comm comm) {
    int out;
    MPI_Comm_rank(comm, &out);
    return (out ? out : 1);
}

} // end anonymous namespace

// These tests could be made shorter with a fixture, but a fixture
// deliberately isn't used in order to make the test harness extremely simple
TEST(BasicMPI, PassOnAllRanks) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int      rank;
    MPI_Comm_rank(comm, &rank);
    EXPECT_EQ(rank, getMpiRank(comm));
}

TEST(BasicMPI, FailOnAllRanks) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int      rank;
    MPI_Comm_rank(comm, &rank);
    EXPECT_EQ(rank, getMpiRankPlusOne(comm));
}

TEST(BasicMPI, FailExceptOnRankZero) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int      rank;
    MPI_Comm_rank(comm, &rank);
    EXPECT_EQ(rank, getZero(comm));
}

TEST(BasicMPI, PassExceptOnRankZero) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int      rank;
    MPI_Comm_rank(comm, &rank);
    EXPECT_EQ(rank, getNonzeroMpiRank(comm));
}
