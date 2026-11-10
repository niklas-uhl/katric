// --verify: a sequential reference triangle count on the graph gathered to rank 0 (small inputs only).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <kagen.h>
#include <mpi.h>

namespace cetric {

/// Gathers `graph` (KaGen's CSR representation, contiguous vertex ranges in rank order) to rank 0 and counts its
/// triangles sequentially. Self-loops and parallel edges are ignored and the edges are treated as undirected. The
/// result is only valid on rank 0.
inline std::size_t reference_triangle_count(kagen::Graph const& graph, MPI_Comm comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // local edges as (min, max) pairs
    std::vector<std::uint64_t> local_edges;
    auto const [first, last] = graph.vertex_range;
    for (kagen::SInt v = 0; v < last - first; ++v) {
        for (auto e = graph.xadj[v]; e < graph.xadj[v + 1]; ++e) {
            std::uint64_t const u = first + v;
            std::uint64_t const w = graph.adjncy[e];
            if (u != w) {
                local_edges.push_back(std::min(u, w));
                local_edges.push_back(std::max(u, w));
            }
        }
    }
    std::uint64_t const local_n = last - first;
    std::uint64_t       n       = 0;
    MPI_Reduce(&local_n, &n, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    int const        send_count = static_cast<int>(local_edges.size());
    std::vector<int> counts(rank == 0 ? size : 0);
    MPI_Gather(&send_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    std::vector<int> displs(counts.size());
    std::size_t      total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        displs[i] = static_cast<int>(total);
        total += static_cast<std::size_t>(counts[i]);
    }
    std::vector<std::uint64_t> flat(total);
    MPI_Gatherv(
        local_edges.data(),
        send_count,
        MPI_UINT64_T,
        flat.data(),
        counts.data(),
        displs.data(),
        MPI_UINT64_T,
        0,
        comm
    );
    if (rank != 0) {
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

    std::vector<std::uint64_t> degree(n, 0);
    for (auto const& [u, v]: edges) {
        ++degree[u];
        ++degree[v];
    }
    // orient each edge towards the endpoint of higher (degree, id), then intersect out-neighborhoods
    auto const before = [&](std::uint64_t u, std::uint64_t v) {
        return std::pair(degree[u], u) < std::pair(degree[v], v);
    };
    std::vector<std::uint64_t> first_out(n + 1, 0);
    for (auto const& [u, v]: edges) {
        ++first_out[(before(u, v) ? u : v) + 1];
    }
    for (std::size_t u = 0; u < n; ++u) {
        first_out[u + 1] += first_out[u];
    }
    std::vector<std::uint64_t> head(edges.size());
    std::vector<std::uint64_t> pos(first_out.begin(), first_out.end() - 1);
    for (auto const& [u, v]: edges) {
        auto const [tail, target] = before(u, v) ? std::pair(u, v) : std::pair(v, u);
        head[pos[tail]++]         = target;
    }
    auto out = [&](std::uint64_t u) {
        return std::ranges::subrange(head.begin() + first_out[u], head.begin() + first_out[u + 1]);
    };
    for (std::uint64_t u = 0; u < n; ++u) {
        std::ranges::sort(out(u));
    }
    std::size_t triangles = 0;
    for (std::uint64_t u = 0; u < n; ++u) {
        for (auto v: out(u)) {
            auto a = out(u).begin();
            auto b = out(v).begin();
            while (a != out(u).end() && b != out(v).end()) {
                if (*a < *b) {
                    ++a;
                } else if (*b < *a) {
                    ++b;
                } else {
                    ++triangles;
                    ++a;
                    ++b;
                }
            }
        }
    }
    return triangles;
}

} // namespace cetric
