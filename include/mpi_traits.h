#ifndef MPI_TRAITS_H
#define MPI_TRAITS_H

#include <mpi.h>
#include <cstdint>

template <typename T>
struct mpi_traits {
};

template<>
struct mpi_traits<int> {
    inline static MPI_Datatype mpi_type = MPI_INT;
    static constexpr bool builtin = true;
};

//inline MPI_Datatype mpi_traits<int>::mpi_type = MPI_INT;

template <>
struct mpi_traits<std::uint64_t> {
    inline static MPI_Datatype mpi_type = MPI_UINT64_T;
    static constexpr bool builtin = true;
};

template <>
struct mpi_traits<std::size_t> {
    static_assert(sizeof(std::size_t) == 8);
    inline static MPI_Datatype mpi_type = MPI_UINT64_T;
    static constexpr bool builtin = true;
};


#endif /* MPI_TRAITS_H */
