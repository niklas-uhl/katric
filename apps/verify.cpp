// --verify: a sequential reference triangle count on the graph gathered to rank
// 0 (small inputs only).
#include "./verify.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <kamping/v2/collectives/gatherv.hpp>
#include <kamping/v2/collectives/reduce.hpp>
#include <kamping/v2/comm.hpp>
#include <kamping/v2/views/auto_counts_view.hpp>
#include <kamping/v2/views/auto_displs_view.hpp>
#include <kamping/v2/views/ref_single_view.hpp>
#include <kamping/v2/views/resize_v_view.hpp>
#include <utility>
#include <vector>

#include <kagen.h>
#include <mpi.h>

namespace katric {

/// Gathers `graph` (KaGen's CSR representation, contiguous vertex ranges in
/// rank order) to rank 0 and counts its triangles sequentially. Self-loops and
/// parallel edges are ignored and the edges are treated as undirected. The
/// result is only valid on rank 0. Limits: the whole graph must fit into rank
/// 0's memory, and the gather counts are `int` (at most 2^31 - 1 words per
/// rank, i.e. about 10^9 edges).
auto reference_triangle_count(kagen::Graph const& graph,
                              kamping::v2::comm_view comm) -> std::size_t {
  // local edges as (min, max) pairs
  std::vector<std::uint64_t> local_edges;
  auto const [first, last] = graph.vertex_range;
  for (kagen::SInt v = 0; v < last - first; ++v) {
    for (auto edge_id = graph.xadj[v]; edge_id < graph.xadj[v + 1]; ++edge_id) {
      std::uint64_t const u = first + v;
      std::uint64_t const w = graph.adjncy[edge_id];
      if (u != w) {
        local_edges.push_back(std::min(u, w));
        local_edges.push_back(std::max(u, w));
      }
    }
  }
  std::uint64_t const local_num_vertices = last - first;
  std::uint64_t num_vertices = 0;
  kamping::v2::reduce(kamping::v2::views::ref_single(local_num_vertices),
                      kamping::v2::views::ref_single(num_vertices), comm);

  std::vector<std::uint64_t> flat;
  kamping::v2::gatherv(local_edges,
                       flat | kamping::v2::views::auto_counts |
                           kamping::v2::views::auto_displs |
                           kamping::v2::views::resize_v,
                       comm);
  if (comm.rank() != 0) {
    return 0;
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  edges.reserve(flat.size() / 2);
  for (std::size_t i = 0; i < flat.size(); i += 2) {
    edges.emplace_back(flat[i], flat[i + 1]);
  }
  flat = {};
  std::ranges::sort(edges);
  auto const duplicates = std::ranges::unique(edges);
  edges.erase(duplicates.begin(), duplicates.end());

  std::vector<std::uint64_t> degree(num_vertices, 0);
  for (auto const& [u, v] : edges) {
    ++degree[u];
    ++degree[v];
  }
  // orient each edge towards the endpoint of higher (degree, id), then
  // intersect out-neighborhoods
  auto const before = [&](std::uint64_t u, std::uint64_t v) -> bool {
    return std::pair(degree[u], u) < std::pair(degree[v], v);
  };
  std::vector<std::uint64_t> first_out(num_vertices + 1, 0);
  for (auto const& [u, v] : edges) {
    ++first_out[(before(u, v) ? u : v) + 1];
  }
  for (std::size_t u = 0; u < num_vertices; ++u) {
    first_out[u + 1] += first_out[u];
  }
  std::vector<std::uint64_t> head(edges.size());
  std::vector<std::uint64_t> pos(first_out.begin(), first_out.end() - 1);
  for (auto const& [u, v] : edges) {
    auto const [tail, target] =
        before(u, v) ? std::pair(u, v) : std::pair(v, u);
    head[pos[tail]++] = target;
  }
  auto out = [&](std::uint64_t u) -> auto {
    return std::ranges::subrange(head.begin() + static_cast<std::ptrdiff_t>(first_out[u]),
                                 head.begin() + static_cast<std::ptrdiff_t>(first_out[u + 1]));
  };
  for (std::uint64_t u = 0; u < num_vertices; ++u) {
    std::ranges::sort(out(u));
  }
  std::size_t triangles = 0;
  for (std::uint64_t u = 0; u < num_vertices; ++u) {
    for (auto v : out(u)) {
      auto u_it = out(u).begin();
      auto v_it = out(v).begin();
      while (u_it != out(u).end() && v_it != out(v).end()) {
        if (*u_it < *v_it) {
          ++u_it;
        } else if (*v_it < *u_it) {
          ++v_it;
        } else {
          ++triangles;
          ++u_it;
          ++v_it;
        }
      }
    }
  }
  return triangles;
}

}  // namespace katric
