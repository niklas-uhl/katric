#ifndef MPI_TRAITS_H
#define MPI_TRAITS_H

#include <mpi.h>
#include <cstdint>

template <typename T>
struct mpi_traits {
};

template<>
struct mpi_traits<int> {
    static constexpr MPI_Datatype mpi_type = MPI_INT;
    static constexpr bool builtin = true;
};

template <>
struct mpi_traits<std::uint64_t> {
    static constexpr MPI_Datatype mpi_type = MPI_UINT64_T;
    static constexpr bool builtin = true;
};

#endif /* MPI_TRAITS_H */
