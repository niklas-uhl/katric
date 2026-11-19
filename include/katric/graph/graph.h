// The graphs the algorithms run on: CSR (xadj, adjncy) over compact ids. A rank numbers its local vertices 0..n-1
// (in global-id order, so local index = global id - first) and its ghosts n..n+g-1 (in global-id order); adjncy holds
// these compact ids ("neighbor handles", as in KaCCv2), not global ids. Ownership, owner and global id of a handle are
// a comparison or an array lookup. The graphs of one run share their vertex layer (VertexLayer) and are immutable once
// built (graph/build.h builds them).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <dstl/grid_comm.hpp>
#include <kamping/v2/comm.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "katric/caliper.h"
#include "katric/datastructures/distributed/vertex_partition.h"
#include "katric/datastructures/graph_definitions.h"
#include "katric/util.h"

namespace katric::graph {

/// A vertex as seen by one rank: [0, n) a local vertex, [n, n + g) a ghost.
using Handle = std::uint64_t;

/// This rank's vertices and ghosts: the local vertex range, the ghosts' global ids and owners (both sorted by global
/// id), and the partition table of all ranks.
///
/// The graphs of a run (input, oriented, contracted) use the same handles and share one layer through a
/// `std::shared_ptr`: copying it per graph would duplicate the ghost table, and the graphs' lifetimes do not nest (the
/// input graph is dropped after preprocessing; load balancing replaces the input graph together with its layer). A
/// layer thus lives until its last graph is gone, at the latest at the end of the run. Its destruction is collective
/// if a grid exchange ran on it (it frees the grid communicator); all ranks drop their graphs at the same point of the
/// driver, so this is safe.
class VertexLayer {
public:
    /// Collective (builds the partition table). `range` is this rank's half-open range of global ids, `ghost_ids` the
    /// sorted, distinct global ids of the ghosts.
    VertexLayer(std::pair<NodeId, NodeId> range, std::vector<NodeId> ghost_ids, kamping::v2::comm_view comm)
        : partition_(range, kamping::v2::handle(comm)),
          comm_(comm),
          rank_(comm.rank()),
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

    [[nodiscard]] kamping::v2::comm_view comm() const {
        return comm_;
    }
    [[nodiscard]] VertexPartition<NodeId> const& partition() const {
        return partition_;
    }
    /// The 2D grid over the communicator's ranks for the grid exchange, built on first use (a collective split: the
    /// first call must be made by all ranks together, as every exchange is). Each run builds its own, as it builds its
    /// vertex layer; it is freed with the layer.
    [[nodiscard]] dstl::grid_comm<dstl::execution_policy::seq> const& grid() const {
        if (!grid_) {
            CALI_CXX_MARK_SCOPE("grid_comm");
            grid_.emplace(comm_);
        }
        return *grid_;
    }

private:
    VertexPartition<NodeId> partition_;
    kamping::v2::comm_view  comm_;
    PEID                    rank_; // for owner()
    NodeId                  first_;
    std::size_t             n_;
    std::vector<NodeId>     ghost_ids_;
    std::vector<PEID>       owners_;
    std::size_t             ghosts_below_ = 0;
    mutable std::optional<dstl::grid_comm<dstl::execution_policy::seq>> grid_;
};

/// A CSR graph (xadj, adjncy) over the local vertices and, optionally (CETRIC), a second one over the ghosts: each
/// ghost's out-edges to local vertices.
class CsrGraph {
public:
    CsrGraph() = default;
    CsrGraph(
        std::shared_ptr<VertexLayer const> layer,
        std::vector<EdgeId>                xadj,
        std::vector<Handle>                adjncy,
        std::vector<EdgeId>                ghost_xadj = {},
        std::vector<Handle>                ghost_adjncy      = {}
    )
        : layer_(std::move(layer)),
          xadj_(std::move(xadj)),
          adjncy_(std::move(adjncy)),
          ghost_xadj_(std::move(ghost_xadj)),
          ghost_adjncy_(std::move(ghost_adjncy)) {
        KASSERT(xadj_.size() == layer_->n() + 1);
        KASSERT(xadj_.back() == adjncy_.size());
        KASSERT(ghost_xadj_.empty() || ghost_xadj_.size() == layer_->ghost_count() + 1);
    }

    [[nodiscard]] std::shared_ptr<VertexLayer const> const& layer() const {
        return layer_;
    }

    /// All local vertices.
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
    /// Number of edges of the local vertices.
    [[nodiscard]] EdgeId local_edge_count() const {
        return adjncy_.size();
    }

    /// The index of local vertex `c` among the local vertices: `c` itself.
    [[nodiscard]] std::size_t to_local_idx(Handle c) const {
        KASSERT(c < layer_->n());
        return c;
    }
    /// Whether `c` is a vertex of this graph: every local vertex.
    [[nodiscard]] bool is_local(Handle c) const {
        return c < layer_->n();
    }
    [[nodiscard]] std::span<Handle const> neighbors(Handle c) const {
        KASSERT(c < layer_->n());
        return {adjncy_.data() + xadj_[c], xadj_[c + 1] - xadj_[c]};
    }
    [[nodiscard]] Degree degree(Handle c) const {
        KASSERT(c < layer_->n());
        return xadj_[c + 1] - xadj_[c];
    }

    [[nodiscard]] bool has_ghost_out_edges() const {
        return !ghost_xadj_.empty();
    }
    /// The out-neighbors of ghost `c`: local vertices.
    [[nodiscard]] std::span<Handle const> ghost_neighbors(Handle c) const {
        KASSERT(has_ghost_out_edges() && c >= layer_->n());
        auto const slot = c - layer_->n();
        return {ghost_adjncy_.data() + ghost_xadj_[slot], ghost_xadj_[slot + 1] - ghost_xadj_[slot]};
    }
    [[nodiscard]] Degree ghost_degree(Handle c) const {
        return ghost_neighbors(c).size();
    }

    /// Building step: sorts every neighborhood by handle (local vertices first, then ghosts, each by global id).
    void sort_neighborhoods() {
        for (std::size_t c = 0; c + 1 < xadj_.size(); ++c) {
            std::sort(
                adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[c]),
                adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[c + 1])
            );
        }
    }

    /// Hands over the vertex layer and the arrays of the local vertices (for a conversion that reuses them); the
    /// ghosts' out-edges are dropped.
    [[nodiscard]] std::tuple<std::shared_ptr<VertexLayer const>, std::vector<EdgeId>, std::vector<Handle>> release() && {
        return {std::move(layer_), std::move(xadj_), std::move(adjncy_)};
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
    [[nodiscard]] kamping::v2::comm_view comm() const {
        return layer_->comm();
    }

private:
    std::shared_ptr<VertexLayer const> layer_;
    std::vector<EdgeId>                xadj_;
    std::vector<Handle>                adjncy_;
    std::vector<EdgeId>                ghost_xadj_;
    std::vector<Handle>                ghost_adjncy_;
};

/// The graph of the global phase: the out-neighbors of the local vertices as global ids, each neighborhood ascending.
/// Converted from an oriented CsrGraph after the local phase (graph/build.h), reusing its arrays: DiTriC keeps every
/// neighborhood, CETRIC and pre-CETRIC keep (contract to) the cut edges of the vertices with any. Messages carry
/// global ids, so the global phase works on these directly.
class GlobalIdGraph {
public:
    GlobalIdGraph() = default;
    /// Every local vertex is part of the graph (DiTriC).
    GlobalIdGraph(std::shared_ptr<VertexLayer const> layer, std::vector<EdgeId> xadj, std::vector<NodeId> adjncy)
        : layer_(std::move(layer)),
          xadj_(std::move(xadj)),
          adjncy_(std::move(adjncy)) {
        KASSERT(xadj_.size() == layer_->n() + 1);
        KASSERT(xadj_.back() == adjncy_.size());
    }
    /// Contracted: only the local vertices `kept` (ascending) are part of the graph; the others have no neighbors.
    GlobalIdGraph(
        std::shared_ptr<VertexLayer const> layer,
        std::vector<EdgeId>                xadj,
        std::vector<NodeId>                adjncy,
        std::vector<Handle>                kept
    )
        : GlobalIdGraph(std::move(layer), std::move(xadj), std::move(adjncy)) {
        KASSERT(std::ranges::is_sorted(kept));
        contracted_ = true;
        kept_       = std::move(kept);
        is_kept_.assign(layer_->n(), false);
        for (Handle const v: kept_) {
            is_kept_[v] = true;
        }
    }

    [[nodiscard]] std::shared_ptr<VertexLayer const> const& layer() const {
        return layer_;
    }
    [[nodiscard]] kamping::v2::comm_view comm() const {
        return layer_->comm();
    }
    /// Whether local vertex `c` is part of the graph (contracted: it had cut edges).
    [[nodiscard]] bool is_kept(Handle c) const {
        return !contracted_ || is_kept_[c];
    }
    /// The kept local vertices, ascending (contracted graphs only).
    [[nodiscard]] std::span<Handle const> kept() const {
        return kept_;
    }
    /// The out-neighbors of local vertex `c` as global ids, ascending.
    [[nodiscard]] std::span<NodeId const> neighbors(Handle c) const {
        KASSERT(c < layer_->n());
        return {adjncy_.data() + xadj_[c], xadj_[c + 1] - xadj_[c]};
    }
    [[nodiscard]] EdgeId local_edge_count() const {
        return adjncy_.size();
    }

private:
    std::shared_ptr<VertexLayer const> layer_;
    std::vector<EdgeId>                xadj_;
    std::vector<NodeId>                adjncy_;
    bool                               contracted_ = false;
    std::vector<Handle>                kept_;
    std::vector<bool>                  is_kept_;
};

} // namespace katric::graph
