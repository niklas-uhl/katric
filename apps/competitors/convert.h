// Conversions of KaGen's input into what a competitor's own file reader would
// produce. They run in ingest(), untimed (doc/competitors.md, decision 2).
// Neighbourhoods are never assumed to be sorted: KaGen gives no such guarantee.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <kagen.h>
#include <mpi.h>

namespace katric::competitors {

/// The first vertex of every rank, and the global vertex count at the end
/// (|comm| + 1 entries). Ranks without vertices are handled.
inline auto vertex_distribution(kagen::Graph const& graph, MPI_Comm comm)
    -> std::vector<std::int64_t> {
  return kagen::BuildVertexDistribution<std::int64_t>(graph, MPI_INT64_T, comm);
}

/// The local part of the input as a CSR with 64-bit vertex ids (`xadj` indexed
/// by local vertex, `adjncy` holds global ids), each neighbourhood sorted by
/// id.
struct SortedCsr {
  std::vector<std::int64_t> xadj;
  std::vector<std::int64_t> adjncy;
};

/// Takes KaGen's CSR representation (katric::generate_input()).
inline auto sorted_csr(kagen::Graph const& graph) -> SortedCsr {
  SortedCsr csr;
  csr.xadj.assign(graph.xadj.begin(), graph.xadj.end());
  csr.adjncy.assign(graph.adjncy.begin(), graph.adjncy.end());
  for (std::size_t vertex = 0; vertex + 1 < csr.xadj.size(); ++vertex) {
    std::sort(csr.adjncy.begin() + csr.xadj[vertex],
              csr.adjncy.begin() + csr.xadj[vertex + 1]);
  }
  return csr;
}

}  // namespace katric::competitors
