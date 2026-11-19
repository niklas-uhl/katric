// Building the graphs of graph/graph.h: the input graph from KaGen's CSR, the exchange of per-vertex values to the
// ghosts, the orientation, and the way back to KaGen's CSR for load balancing.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include <kamping/v2/types/std/unsafe/utility.hpp>
#include <kagen.h>
#include <kassert/kassert.hpp>
#include <mpi.h>
#include <tlx/vector_free.hpp>

#include "cetric/ghost_exchange.h"
#include "cetric/graph/graph.h"

namespace cetric::graph {

/// The undirected graph of a rank's part of the input (KaGen's CSR: vertex range, xadj, adjncy in global ids), with
/// ghosts. Collective. Neighborhoods keep the input's order. The cut edges are sorted once by (head, position): that
/// gives the ghosts in global-id order and their handles without a lookup per edge.
inline CsrGraph build_input_graph(kagen::Graph const& input, kamping::v2::comm_view comm) {
    KASSERT(input.representation == kagen::GraphRepresentation::CSR);
    std::pair<NodeId, NodeId> const range{input.vertex_range.first, input.vertex_range.second};
    auto const [first, last] = range;
    std::size_t const n      = first < last ? last - first : 0;
    auto const        owned  = [first, n](NodeId gid) {
        return gid >= first && gid - first < n;
    };
    std::vector<Handle>                    adjncy(input.adjncy.size());
    std::vector<std::pair<NodeId, EdgeId>> cut; // (head, position in adjncy)
    for (EdgeId e = 0; e < input.adjncy.size(); ++e) {
        NodeId const gid = input.adjncy[e];
        if (owned(gid)) {
            adjncy[e] = gid - first;
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
        adjncy[e] = n + ghost_ids.size() - 1;
    }
    tlx::vector_free(cut);
    // copied: the input is only read (the harness reuses it for every iteration), and KaGen's SInt need not be our
    // EdgeId type anyway (unsigned long long vs. unsigned long)
    std::vector<EdgeId> xadj(input.xadj.begin(), input.xadj.end());
    if (xadj.empty()) {
        xadj.push_back(0);
    }
    KASSERT(xadj.size() == n + 1);
    auto layer = std::make_shared<VertexLayer const>(range, std::move(ghost_ids), comm);
    KASSERT(layer->n() == n);
    return CsrGraph(std::move(layer), std::move(xadj), std::move(adjncy));
}

/// `G` in KaGen's CSR representation (global ids).
inline kagen::Graph to_kagen_graph(CsrGraph const& G) {
    auto const&  layer = *G.layer();
    kagen::Graph input;
    input.representation = kagen::GraphRepresentation::CSR;
    input.vertex_range   = {layer.first(), layer.first() + layer.n()};
    input.xadj.reserve(G.local_node_count() + 1);
    input.adjncy.reserve(G.local_edge_count());
    input.xadj.push_back(0);
    for (Handle const v: G.local_nodes()) {
        for (Handle const u: G.neighbors(v)) {
            input.adjncy.push_back(G.global_id(u));
        }
        input.xadj.push_back(input.adjncy.size());
    }
    return input;
}

/// Sends `value(v)` of every local vertex v to the ranks holding v as a ghost and returns the received values, indexed
/// by ghost slot (handle - n). Message counters as push_to_ghosts(), two words per entry (vertex and value).
template <typename Graph, typename ValueFunc>
std::vector<std::uint64_t>
exchange_ghost_values(Graph const& G, ValueFunc&& value, GhostExchangeMode mode, MessageTag tag) {
    // (vertex, value)
    auto const received = push_to_ghosts(
        G,
        G.local_nodes(),
        [&](Handle v) { return std::pair<NodeId, std::uint64_t>{G.global_id(v), static_cast<std::uint64_t>(value(v))}; },
        mode,
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

/// Orients the edges of the undirected graph `G` from the smaller to the larger `key` and returns the out-edges,
/// neighborhoods unsorted (in `G`'s order). With `ghost_out_edges`, also the ghosts' out-edges to local vertices
/// (CETRIC), sorted.
template <typename Key>
CsrGraph orient(CsrGraph const& G, Key&& key, bool ghost_out_edges) {
    auto const&         layer = *G.layer();
    std::size_t const   n     = layer.n();
    auto const          out   = [&](Handle v, Handle u) {
        return key(v) < key(u);
    };
    std::vector<EdgeId> xadj(n + 1, 0);
    std::vector<EdgeId> ghost_xadj(ghost_out_edges ? layer.ghost_count() + 1 : 0, 0);
    for (Handle const v: G.local_nodes()) {
        for (Handle const u: G.neighbors(v)) {
            if (out(v, u)) {
                xadj[v + 1]++;
            } else if (ghost_out_edges && !layer.is_owned(u)) {
                ghost_xadj[u - n + 1]++;
            }
        }
    }
    for (std::size_t c = 0; c < n; ++c) {
        xadj[c + 1] += xadj[c];
    }
    for (std::size_t s = 0; s + 1 < ghost_xadj.size(); ++s) {
        ghost_xadj[s + 1] += ghost_xadj[s];
    }
    std::vector<Handle> adjncy(xadj.back());
    std::vector<Handle> ghost_adjncy(ghost_out_edges ? ghost_xadj.back() : 0);
    std::vector<EdgeId> ghost_next(ghost_xadj);
    for (Handle const v: G.local_nodes()) {
        EdgeId next = xadj[v];
        for (Handle const u: G.neighbors(v)) {
            if (out(v, u)) {
                adjncy[next++] = u;
            } else if (ghost_out_edges && !layer.is_owned(u)) {
                // v ascending: each ghost's neighborhood comes out sorted
                ghost_adjncy[ghost_next[u - n]++] = v;
            }
        }
    }
    return CsrGraph(G.layer(), std::move(xadj), std::move(adjncy), std::move(ghost_xadj), std::move(ghost_adjncy));
}

/// DiTriC: the out-neighborhoods of `G` (sorted by handle) as global ids, in place. A neighborhood sorted by handle is
/// [local vertices][ghosts below the local range][ghosts above]; translated and with the ghosts below rotated to the
/// front it is ascending.
inline GlobalIdGraph to_global_ids(CsrGraph&& G) {
    auto [layer_ptr, xadj, adjncy] = std::move(G).release();
    auto const& layer              = *layer_ptr;
    for (std::size_t v = 0; v < layer.n(); ++v) {
        auto const first     = adjncy.begin() + static_cast<std::ptrdiff_t>(xadj[v]);
        auto const last      = adjncy.begin() + static_cast<std::ptrdiff_t>(xadj[v + 1]);
        auto const local_end = std::lower_bound(first, last, Handle{layer.n()});
        auto const below_end = std::lower_bound(local_end, last, layer.ghosts_above());
        for (auto it = first; it != last; ++it) {
            *it = layer.global_id(*it);
        }
        std::rotate(first, local_end, below_end);
    }
    return GlobalIdGraph(std::move(layer_ptr), std::move(xadj), std::move(adjncy));
}

/// Contracts `G` (neighborhoods sorted by handle) to its cut edges, in place: of each neighborhood only the ghost part
/// is kept ([ghosts below][ghosts above], already in global order), as global ids. The kept vertices are those with a
/// cut out-edge.
inline GlobalIdGraph contract_to_global_ids(CsrGraph&& G) {
    auto [layer_ptr, xadj, adjncy] = std::move(G).release();
    auto const&         layer      = *layer_ptr;
    auto const          n          = static_cast<Handle>(layer.n());
    std::vector<Handle> kept;
    EdgeId              out   = 0;
    EdgeId              begin = 0; // xadj[v] before it was overwritten
    for (Handle v = 0; v < n; ++v) {
        EdgeId const end        = xadj[v + 1];
        auto const   last       = adjncy.begin() + static_cast<std::ptrdiff_t>(end);
        auto const   ghost_part = std::lower_bound(adjncy.begin() + static_cast<std::ptrdiff_t>(begin), last, n);
        if (ghost_part != last) {
            kept.push_back(v);
        }
        for (auto it = ghost_part; it != last; ++it) {
            adjncy[out++] = layer.global_id(*it); // out <= it: writes never overtake reads
        }
        xadj[v + 1] = out;
        begin       = end;
    }
    adjncy.resize(out);
    return GlobalIdGraph(std::move(layer_ptr), std::move(xadj), std::move(adjncy), std::move(kept));
}

} // namespace cetric::graph
