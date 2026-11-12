// --verify: a sequential reference triangle count on the input gathered to rank 0.
#pragma once

#include <cstddef>

#include <kagen.h>
#include <mpi.h>

namespace cetric {

/// Gathers `graph` (KaGen's CSR representation, contiguous vertex ranges in rank order) to rank 0 and counts its
/// triangles sequentially. Self-loops and parallel edges are ignored and the edges are treated as undirected. The
/// result is only valid on rank 0.
std::size_t reference_triangle_count(kagen::Graph const& graph, MPI_Comm comm);

} // namespace cetric
