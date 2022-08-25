#ifndef MPI_TRAITS_H
#define MPI_TRAITS_H

#include <mpi.h>
#include <cstdint>
#include <type_traits>

template <typename T, class Enable = void>
struct mpi_traits {
};

template<>
struct mpi_traits<int> {
    inline static MPI_Datatype mpi_type = MPI_INT;
    static constexpr bool builtin = true;
};

template <typename T>
struct mpi_traits<T, typename std::enable_if<sizeof(T) == 8>::type> {
    inline static MPI_Datatype mpi_type = MPI_UINT64_T;
    static constexpr bool builtin = true;
};

#endif /* MPI_TRAITS_H */
