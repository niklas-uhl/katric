#pragma once

#include <mpi.h>
#include <iostream>
#include <sstream>

template <class MessageType>
inline void atomic_debug(MessageType message, std::ostream& out = std::cout, bool newline = true) {
    std::stringstream sout;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sout << "[R" << rank << "] " << message;
    if (newline) {
        sout << std::endl;
    }
    out << sout.str();
}
