// The graphs the algorithms run on: CSR over compact ids. A rank numbers its local vertices 0..n-1 (in global-id order,
// so local index = global id - first) and its ghosts n..n+g-1 (in global-id order); adjacency lists hold these compact
// ids ("neighbor handles", as in KaCCv2), not global ids. Ownership, owner and global id of a handle are a comparison or
// an array lookup. The graphs of one run share their vertex layer (VertexLayer) and are immutable once built
// (graph/build.h builds them).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "cetric/datastructures/distributed/vertex_partition.h"
#include "cetric/datastructures/graph_definitions.h"
#include "cetric/util.h"

namespace cetric::graph {

/// A vertex as seen by one rank: [0, n) a local vertex, [n, n + g) a ghost.
using Handle = std::uint64_t;

/// This rank's vertices and ghosts: the local vertex range, the ghosts' global ids and owners (both sorted by global
/// id), and the partition table of all ranks.
class VertexLayer {
public:
    /// Collective (builds the partition table). `range` is this rank's half-open range of global ids, `ghost_ids` the
    /// sorted, distinct global ids of the ghosts.
    VertexLayer(std::pair<NodeId, NodeId> range, std::vector<NodeId> ghost_ids, MPI_Comm comm)
        : partition_(range, comm),
          comm_(comm),
          rank_(comm_rank(comm)),
          size_(comm_size(comm)),
          first_(partition_.vertex_range().first),
          n_(partition_.vertex_range().second - partition_.vertex_range().first),
          ghost_ids_(std::move(ghost_ids)) {
        KASSERT(std::ranges::is_sorted(ghost_ids_));
        KASSERT(std::ranges::adjacent_find(ghost_ids_) == ghost_ids_.end());
        owners_.reserve(ghost_ids_.size());
        for (NodeId const gid: ghost_ids_) {
            KASSERT(!owns_global(gid));
            owners_.push_back(partition_.get_rank(gid));
        }
        ghosts_below_ = static_cast<std::size_t>(std::ranges::lower_bound(ghost_ids_, first_) - ghost_ids_.begin());
    }

    /// Number of local vertices.
    [[nodiscard]] std::size_t n() const {
        return n_;
    }
    /// Number of ghosts.
    [[nodiscard]] std::size_t ghost_count() const {
        return ghost_ids_.size();
    }
    /// The global id of local vertex 0.
    [[nodiscard]] NodeId first() const {
        return first_;
    }

    [[nodiscard]] bool is_owned(Handle c) const {
        return c < n_;
    }
    [[nodiscard]] PEID owner(Handle c) const {
        return c < n_ ? rank_ : owners_[c - n_];
    }
    [[nodiscard]] NodeId global_id(Handle c) const {
        return c < n_ ? first_ + c : ghost_ids_[c - n_];
    }

    /// Whether this rank owns the vertex with global id `gid`.
    [[nodiscard]] bool owns_global(NodeId gid) const {
        return gid >= first_ && gid - first_ < n_;
    }
    /// The handle of a local vertex or ghost (binary search for a ghost).
    [[nodiscard]] Handle handle(NodeId gid) const {
        if (owns_global(gid)) {
            return gid - first_;
        }
        auto const it = std::ranges::lower_bound(ghost_ids_, gid);
        KASSERT(it != ghost_ids_.end() && *it == gid, "vertex " << gid << " is not a ghost of rank " << rank_);
        return n_ + static_cast<Handle>(it - ghost_ids_.begin());
    }
    /// The first ghost handle above the local range: ghosts [n, ghosts_above()) have smaller global ids than the local
    /// vertices, ghosts [ghosts_above(), n + g) larger ones.
    [[nodiscard]] Handle ghosts_above() const {
        return n_ + ghosts_below_;
    }

    [[nodiscard]] MPI_Comm comm() const {
        return comm_;
    }
    [[nodiscard]] PEID rank() const {
        return rank_;
    }
    [[nodiscard]] PEID size() const {
        return size_;
    }
    [[nodiscard]] VertexPartition<NodeId> const& partition() const {
        return partition_;
    }

private:
    VertexPartition<NodeId> partition_;
    MPI_Comm                comm_;
    PEID                    rank_;
    PEID                    size_;
    NodeId                  first_;
    std::size_t             n_;
    std::vector<NodeId>     ghost_ids_;
    std::vector<PEID>       owners_;
    std::size_t             ghosts_below_ = 0;
};

/// An orientation: v -> u iff key(v) < key(u), with key(c) = (degree[c], global id of c) and a degree per local vertex
/// and ghost. Kept by the oriented graphs, so the global phase can compare keys (message trimming).
class Orientation {
public:
    Orientation(std::shared_ptr<VertexLayer const> layer, std::vector<Degree> degree)
        : layer_(std::move(layer)),
          degree_(std::move(degree)) {
        KASSERT(degree_.size() == layer_->n() + layer_->ghost_count());
    }
    [[nodiscard]] std::pair<Degree, NodeId> key(Handle c) const {
        return {degree_[c], layer_->global_id(c)};
    }
    [[nodiscard]] bool before(Handle a, Handle b) const {
        return key(a) < key(b);
    }

private:
    std::shared_ptr<VertexLayer const> layer_;
    std::vector<Degree>                degree_;
};

/// Rows sorted by handle in global ids, ascending. A sorted row is [local vertices][ghosts below the local range][ghosts
/// above], so in global order it is [ghosts below][local vertices][ghosts above].
inline std::vector<NodeId>
global_rows(VertexLayer const& layer, std::vector<EdgeId> const& offsets, std::vector<Handle> const& heads) {
    std::vector<NodeId> global_heads(heads.size());
    for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
        auto const first     = heads.begin() + static_cast<std::ptrdiff_t>(offsets[row]);
        auto const last      = heads.begin() + static_cast<std::ptrdiff_t>(offsets[row + 1]);
        auto const local_end = std::lower_bound(first, last, Handle{layer.n()});
        auto const below_end = std::lower_bound(local_end, last, layer.ghosts_above());
        auto       out       = global_heads.begin() + static_cast<std::ptrdiff_t>(offsets[row]);
        for (auto it = local_end; it != below_end; ++it) {
            *out++ = layer.global_id(*it);
        }
        for (auto it = first; it != local_end; ++it) {
            *out++ = layer.first() + *it;
        }
        for (auto it = below_end; it != last; ++it) {
            *out++ = layer.global_id(*it);
        }
    }
    return global_heads;
}

/// A CSR graph with a row for every local vertex and, optionally (CETRIC), a row for every ghost: the ghost's
/// out-edges to local vertices.
class CsrGraph {
public:
    CsrGraph() = default;
    CsrGraph(
        std::shared_ptr<VertexLayer const> layer,
        std::vector<EdgeId>                offsets,
        std::vector<Handle>                heads,
        std::vector<EdgeId>                ghost_offsets = {},
        std::vector<Handle>                ghost_heads   = {},
        std::shared_ptr<Orientation const> orientation   = {}
    )
        : layer_(std::move(layer)),
          orientation_(std::move(orientation)),
          offsets_(std::move(offsets)),
          heads_(std::move(heads)),
          ghost_offsets_(std::move(ghost_offsets)),
          ghost_heads_(std::move(ghost_heads)) {
        KASSERT(offsets_.size() == layer_->n() + 1);
        KASSERT(offsets_.back() == heads_.size());
        KASSERT(ghost_offsets_.empty() || ghost_offsets_.size() == layer_->ghost_count() + 1);
    }

    [[nodiscard]] std::shared_ptr<VertexLayer const> const& layer() const {
        return layer_;
    }
    /// The orientation the rows follow (none for the input graph).
    [[nodiscard]] std::shared_ptr<Orientation const> const& orientation() const {
        return orientation_;
    }

    /// The vertices with a row: all local vertices.
    [[nodiscard]] auto local_nodes() const {
        return std::views::iota(Handle{0}, Handle{layer_->n()});
    }
    [[nodiscard]] std::size_t local_node_count() const {
        return layer_->n();
    }
    [[nodiscard]] auto ghosts() const {
        return std::views::iota(Handle{layer_->n()}, Handle{layer_->n() + layer_->ghost_count()});
    }
    [[nodiscard]] std::size_t ghost_count() const {
        return layer_->ghost_count();
    }
    /// Number of edges in the local vertices' rows.
    [[nodiscard]] EdgeId local_edge_count() const {
        return heads_.size();
    }

    /// The row of local vertex `c` (its index among the rows).
    [[nodiscard]] std::size_t to_local_idx(Handle c) const {
        KASSERT(c < layer_->n());
        return c;
    }
    /// Whether `c` has a row: every local vertex.
    [[nodiscard]] bool is_local(Handle c) const {
        return c < layer_->n();
    }
    [[nodiscard]] std::span<Handle const> neighbors(Handle c) const {
        KASSERT(c < layer_->n());
        return {heads_.data() + offsets_[c], offsets_[c + 1] - offsets_[c]};
    }
    [[nodiscard]] Degree degree(Handle c) const {
        KASSERT(c < layer_->n());
        return offsets_[c + 1] - offsets_[c];
    }

    [[nodiscard]] bool has_ghost_rows() const {
        return !ghost_offsets_.empty();
    }
    /// The row of ghost `c`: its out-edges to local vertices.
    [[nodiscard]] std::span<Handle const> ghost_neighbors(Handle c) const {
        KASSERT(has_ghost_rows() && c >= layer_->n());
        auto const slot = c - layer_->n();
        return {ghost_heads_.data() + ghost_offsets_[slot], ghost_offsets_[slot + 1] - ghost_offsets_[slot]};
    }
    [[nodiscard]] Degree ghost_degree(Handle c) const {
        return ghost_neighbors(c).size();
    }

    /// The row of local vertex `c` in global ids, ascending (after add_global_rows()). The global phase works on these:
    /// messages carry global ids, and merging them needs no lookup per entry.
    [[nodiscard]] std::span<NodeId const> global_neighbors(Handle c) const {
        KASSERT(c < layer_->n());
        KASSERT(global_heads_.size() == heads_.size(), "add_global_rows() was not called");
        return {global_heads_.data() + offsets_[c], offsets_[c + 1] - offsets_[c]};
    }

    /// Building step (after sort_rows()): the rows in global ids.
    void add_global_rows() {
        global_heads_ = global_rows(*layer_, offsets_, heads_);
    }

    /// Building step: sorts every row by handle (local vertices first, then ghosts, each by global id).
    void sort_rows() {
        for (std::size_t c = 0; c + 1 < offsets_.size(); ++c) {
            std::sort(heads_.begin() + static_cast<std::ptrdiff_t>(offsets_[c]),
                      heads_.begin() + static_cast<std::ptrdiff_t>(offsets_[c + 1]));
        }
    }

    [[nodiscard]] bool is_owned(Handle c) const {
        return layer_->is_owned(c);
    }
    [[nodiscard]] PEID owner(Handle c) const {
        return layer_->owner(c);
    }
    [[nodiscard]] NodeId global_id(Handle c) const {
        return layer_->global_id(c);
    }
    [[nodiscard]] MPI_Comm comm() const {
        return layer_->comm();
    }
    [[nodiscard]] PEID rank() const {
        return layer_->rank();
    }
    [[nodiscard]] PEID size() const {
        return layer_->size();
    }

private:
    std::shared_ptr<VertexLayer const> layer_;
    std::shared_ptr<Orientation const> orientation_;
    std::vector<EdgeId>                offsets_;
    std::vector<Handle>                heads_;
    std::vector<NodeId>                global_heads_;
    std::vector<EdgeId>                ghost_offsets_;
    std::vector<Handle>                ghost_heads_;
};

/// The contracted graph of CETRIC and pre-CETRIC: rows only for some local vertices (those with cut edges), sorted by
/// handle, together with their rows in global ids. Shares the vertex layer (handles) with the graph it came from.
class ContractedGraph {
public:
    static constexpr std::size_t no_row = std::numeric_limits<std::size_t>::max();

    ContractedGraph() = default;
    /// `vertices` ascending; row i belongs to vertices[i].
    ContractedGraph(
        std::shared_ptr<VertexLayer const> layer,
        std::vector<Handle>                vertices,
        std::vector<EdgeId>                offsets,
        std::vector<Handle>                heads,
        std::shared_ptr<Orientation const> orientation
    )
        : layer_(std::move(layer)),
          orientation_(std::move(orientation)),
          vertices_(std::move(vertices)),
          row_of_(layer_->n(), no_row),
          offsets_(std::move(offsets)),
          heads_(std::move(heads)) {
        KASSERT(offsets_.size() == vertices_.size() + 1);
        KASSERT(offsets_.back() == heads_.size());
        KASSERT(std::ranges::is_sorted(vertices_));
        for (std::size_t row = 0; row < vertices_.size(); ++row) {
            row_of_[vertices_[row]] = row;
        }
        global_heads_ = global_rows(*layer_, offsets_, heads_);
    }

    [[nodiscard]] std::shared_ptr<VertexLayer const> const& layer() const {
        return layer_;
    }
    /// The orientation the rows follow.
    [[nodiscard]] std::shared_ptr<Orientation const> const& orientation() const {
        return orientation_;
    }
    /// The vertices with a row, ascending.
    [[nodiscard]] std::span<Handle const> local_nodes() const {
        return vertices_;
    }
    [[nodiscard]] std::size_t local_node_count() const {
        return vertices_.size();
    }
    [[nodiscard]] EdgeId local_edge_count() const {
        return heads_.size();
    }
    [[nodiscard]] std::size_t to_local_idx(Handle c) const {
        KASSERT(is_local(c));
        return row_of_[c];
    }
    /// Whether local vertex or ghost `c` has a row.
    [[nodiscard]] bool is_local(Handle c) const {
        return c < layer_->n() && row_of_[c] != no_row;
    }
    [[nodiscard]] std::span<Handle const> neighbors(Handle c) const {
        auto const row = to_local_idx(c);
        return {heads_.data() + offsets_[row], offsets_[row + 1] - offsets_[row]};
    }
    [[nodiscard]] Degree degree(Handle c) const {
        return neighbors(c).size();
    }
    [[nodiscard]] std::span<NodeId const> global_neighbors(Handle c) const {
        auto const row = to_local_idx(c);
        return {global_heads_.data() + offsets_[row], offsets_[row + 1] - offsets_[row]};
    }

    [[nodiscard]] bool is_owned(Handle c) const {
        return layer_->is_owned(c);
    }
    [[nodiscard]] PEID owner(Handle c) const {
        return layer_->owner(c);
    }
    [[nodiscard]] NodeId global_id(Handle c) const {
        return layer_->global_id(c);
    }
    [[nodiscard]] MPI_Comm comm() const {
        return layer_->comm();
    }
    [[nodiscard]] PEID rank() const {
        return layer_->rank();
    }
    [[nodiscard]] PEID size() const {
        return layer_->size();
    }

private:
    std::shared_ptr<VertexLayer const> layer_;
    std::shared_ptr<Orientation const> orientation_;
    std::vector<Handle>                vertices_;
    std::vector<std::size_t>           row_of_; // per local vertex
    std::vector<EdgeId>                offsets_;
    std::vector<Handle>                heads_;
    std::vector<NodeId>                global_heads_;
};

} // namespace cetric::graph
