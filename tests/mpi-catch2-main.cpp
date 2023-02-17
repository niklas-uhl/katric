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
