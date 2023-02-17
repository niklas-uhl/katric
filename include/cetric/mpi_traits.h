/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef MPI_TRAITS_H
#define MPI_TRAITS_H

#include <cstdint>
#include <type_traits>

#include <mpi.h>

template <typename T, class Enable = void>
struct mpi_traits {};

template <>
struct mpi_traits<int> {
    inline static MPI_Datatype mpi_type = MPI_INT;
    static constexpr bool      builtin  = true;
};

template <typename T>
struct mpi_traits<T, typename std::enable_if<sizeof(T) == 8>::type> {
    inline static MPI_Datatype mpi_type = MPI_UINT64_T;
    static constexpr bool      builtin  = true;
};

#endif /* MPI_TRAITS_H */
