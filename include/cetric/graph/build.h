// Building the graphs of graph/graph.h: the input graph from a LocalGraphView, the exchange of per-vertex values to the
// ghosts, the orientation, and the way back to a LocalGraphView for load balancing.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <kamping/types/contiguous_type.hpp>
#include <kamping/types/mpi_type_traits.hpp>
#include <kamping/types/scoped_datatype.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>
#include <tlx/vector_free.hpp>

#include "cetric/datastructures/local_graph_view.h"
#include "cetric/ghost_exchange.h"
#include "cetric/graph/graph.h"

namespace cetric::graph {

/// The undirected graph of a rank's part of the input, with ghosts. Collective. Rows keep the input's neighbor order.
/// The cut edges are sorted once by (head, position): that gives the ghosts in global-id order and their handles without
/// a lookup per edge.
inline CsrGraph build_input_graph(LocalGraphView&& view, std::pair<NodeId, NodeId> range, MPI_Comm comm) {
    auto const [first, last] = range;
    std::size_t const n      = first < last ? last - first : 0;
    auto const        owned  = [first, n](NodeId gid) {
        return gid >= first && gid - first < n;
    };
    std::vector<Handle>                    heads(view.edge_heads.size());
    std::vector<std::pair<NodeId, EdgeId>> cut; // (head, position in heads)
    for (EdgeId e = 0; e < view.edge_heads.size(); ++e) {
        NodeId const gid = view.edge_heads[e];
        if (owned(gid)) {
            heads[e] = gid - first;
        } else {
            cut.emplace_back(gid, e);
        }
    }
    std::ranges::sort(cut);
    std::vector<NodeId> ghost_ids;
    for (auto const& [gid, e]: cut) {
        if (ghost_ids.empty() || ghost_ids.back() != gid) {
            ghost_ids.push_back(gid);
        }
        heads[e] = n + ghost_ids.size() - 1;
    }
    tlx::vector_free(cut);
    auto layer = std::make_shared<VertexLayer const>(range, std::move(ghost_ids), comm);
    KASSERT(layer->n() == n);

    // vertices of the range without a NodeInfo have no edges; node_info is sorted by global id, so its adjacency lists
    // are already in row order
    std::vector<EdgeId> offsets(n + 1, 0);
    for (auto const& info: view.node_info) {
        KASSERT(layer->owns_global(info.global_id));
        offsets[info.global_id - first + 1] = info.degree;
    }
    for (std::size_t c = 0; c < n; ++c) {
        offsets[c + 1] += offsets[c];
    }
    KASSERT(offsets.back() == heads.size());
    tlx::vector_free(view.edge_heads);
    tlx::vector_free(view.node_info);
    return CsrGraph(std::move(layer), std::move(offsets), std::move(heads));
}

/// The rows of `G` as a LocalGraphView (global ids), every local vertex included.
inline LocalGraphView to_local_graph_view(CsrGraph const& G) {
    LocalGraphView view;
    view.node_info.reserve(G.local_node_count());
    view.edge_heads.reserve(G.local_edge_count());
    for (Handle const v: G.local_nodes()) {
        view.node_info.emplace_back(G.global_id(v), G.degree(v));
        for (Handle const u: G.neighbors(v)) {
            view.edge_heads.push_back(G.global_id(u));
        }
    }
    return view;
}

/// Sends `value(v)` of every local vertex v to the ranks holding v as a ghost and returns the received values, indexed
/// by ghost slot (handle - n). Message counters as push_to_ghosts(), two words per entry (vertex and value).
template <typename Graph, typename ValueFunc>
std::vector<std::uint64_t> exchange_ghost_values(Graph const& G, ValueFunc&& value, GhostExchange exchange, int tag) {
    struct Entry {
        NodeId        node = 0;
        std::uint64_t data = 0;
    };
    static_assert(sizeof(Entry) == 2 * sizeof(std::uint64_t));
    kamping::types::ScopedDatatype const entry_type{
        kamping::types::mpi_type_traits<std::array<std::uint64_t, 2>>::data_type()};
    auto const received = push_to_ghosts(
        G,
        G.local_nodes(),
        [&](Handle v) { return Entry{G.global_id(v), static_cast<std::uint64_t>(value(v))}; },
        exchange,
        entry_type.data_type(),
        tag,
        2
    );
    auto const&                layer = *G.layer();
    std::vector<std::uint64_t> values(layer.ghost_count());
    for (auto const& [node, data]: received) {
        values[layer.handle(node) - layer.n()] = data;
    }
    return values;
}

/// Orients the edges of the undirected graph `G` from the smaller to the larger `key` and returns the out-edges, rows
/// unsorted (in `G`'s order). With `ghost_rows`, also the ghosts' out-edges to local vertices (CETRIC), sorted.
template <typename Key>
CsrGraph orient(CsrGraph const& G, Key&& key, bool ghost_rows) {
    auto const&         layer = *G.layer();
    std::size_t const   n     = layer.n();
    auto const          out   = [&](Handle v, Handle u) {
        return key(v) < key(u);
    };
    std::vector<EdgeId> offsets(n + 1, 0);
    std::vector<EdgeId> ghost_offsets(ghost_rows ? layer.ghost_count() + 1 : 0, 0);
    for (Handle const v: G.local_nodes()) {
        for (Handle const u: G.neighbors(v)) {
            if (out(v, u)) {
                offsets[v + 1]++;
            } else if (ghost_rows && !layer.is_owned(u)) {
                ghost_offsets[u - n + 1]++;
            }
        }
    }
    for (std::size_t c = 0; c < n; ++c) {
        offsets[c + 1] += offsets[c];
    }
    for (std::size_t s = 0; s + 1 < ghost_offsets.size(); ++s) {
        ghost_offsets[s + 1] += ghost_offsets[s];
    }
    std::vector<Handle> heads(offsets.back());
    std::vector<Handle> ghost_heads(ghost_rows ? ghost_offsets.back() : 0);
    std::vector<EdgeId> ghost_next(ghost_offsets);
    for (Handle const v: G.local_nodes()) {
        EdgeId next = offsets[v];
        for (Handle const u: G.neighbors(v)) {
            if (out(v, u)) {
                heads[next++] = u;
            } else if (ghost_rows && !layer.is_owned(u)) {
                // v ascending: each ghost row comes out sorted
                ghost_heads[ghost_next[u - n]++] = v;
            }
        }
    }
    return CsrGraph(G.layer(), std::move(offsets), std::move(heads), std::move(ghost_offsets), std::move(ghost_heads));
}

/// The orientation of DiTriC and CETRIC: by (degree, global id), with the ghosts' degrees from `ghost_degree`.
inline CsrGraph orient_by_degree(CsrGraph const& G, std::vector<std::uint64_t> const& ghost_degree, bool ghost_rows) {
    auto const& layer = *G.layer();
    return orient(
        G,
        [&](Handle c) {
            Degree const degree = layer.is_owned(c) ? G.degree(c) : ghost_degree[c - layer.n()];
            return std::pair(degree, layer.global_id(c));
        },
        ghost_rows
    );
}

} // namespace cetric::graph
