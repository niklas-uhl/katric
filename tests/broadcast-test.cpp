#include <catch2/catch.hpp>
#include <mpi.h>

TEST_CASE("Broadcast works", "[broadcast]") {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int buffer;
    if (rank == 0) {
        buffer = 42;
    }
    MPI_Bcast(&buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
    REQUIRE(buffer == 42);
}
