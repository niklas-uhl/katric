#define CATCH_CONFIG_RUNNER
#include <iostream>
#include <sstream>

#include <catch2/catch_session.hpp>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // std::stringstream out;
    // auto cout_buf = std::cout.rdbuf(out.rdbuf());
    int result = Catch::Session().run(argc, argv);

    // std::cout.rdbuf(cout_buf);

    int global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    // if (!global_result) {
    //     if (rank == 0) {
    //         std::cout << out.str();
    //     }
    //     MPI_Finalize();
    //     return result;
    // }

    // std::stringstream print_rank;
    // print_rank << "Rank ";
    // print_rank.width(2);
    // print_rank << std::right << rank << ":\n";

    // for (int i = 0; i < size; ++i) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (i == rank) {
    //         if (out.str().rfind("All tests passed") == std::string::npos) {
    //             std::cout << print_rank.str() + out.str();
    //         }
    //     }
    // }
    MPI_Finalize();
    return result;
}
