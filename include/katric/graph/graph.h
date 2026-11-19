// The graphs the algorithms run on: CSR (xadj, adjncy) over vertex handles. A
// rank numbers its local vertices 0..n-1 (in global-id order, so local index =
// global id - first) and its ghosts n..n+g-1 (in global-id order); adjncy holds
// these handles ("neighbor handles", as in KaCCv2), not global ids. A handle is
// a type of its own without arithmetic: the vertex layer converts it to a local
// or ghost index and back, to its global id and to its owner (a comparison or
// an array lookup). The graphs of one run share their vertex layer
// (VertexLayer) and are immutable once built (graph/build.h builds them).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <dstl/grid_comm.hpp>
#include <kamping/v2/comm.hpp>
#include <kassert/kassert.hpp>

#include "katric/datastructures/distributed/vertex_partition.h"
#include "katric/util.h"

namespace katric::graph {

/// A vertex's global id: its position in the input (KaGen's numbering).
using VertexId = std::uint64_t;
/// The number of neighbors of a vertex.
using Degree = std::size_t;

/// A vertex as seen by one rank: a local vertex or a ghost. Ordered like the
/// vertices' global ids within each of the two groups, local vertices first;
/// no arithmetic (VertexLayer converts).
enum class VertexHandle : std::uint64_t {};

/// For fmt (assertion and log messages): the handle's number.
inline auto format_as(VertexHandle vertex) -> std::uint64_t {
  return static_cast<std::uint64_t>(vertex);
}

/// This rank's vertices and ghosts: the local vertex range, the ghosts' global
/// ids and owners (both sorted by global id), and the partition table of all
/// ranks.
///
/// The graphs of a run (input, oriented, contracted) use the same handles and
/// share one layer through a `std::shared_ptr`: copying it per graph would
/// duplicate the ghost table, and the graphs' lifetimes do not nest (the input
/// graph is dropped after preprocessing; load balancing replaces the input
/// graph together with its layer). A layer thus lives until its last graph is
/// gone, at the latest at the end of the run. Its destruction is collective if
/// a grid exchange ran on it (it frees the grid communicator); all ranks drop
/// their graphs at the same point of the driver, so this is safe.
class VertexLayer {
public:
  /// Collective (builds the partition table). `range` is this rank's half-open
  /// range of global ids, `ghost_ids` the sorted, distinct global ids of the
  /// ghosts.
  VertexLayer(std::pair<VertexId, VertexId> range,
              std::vector<VertexId> ghost_ids,
              kamping::v2::comm_view comm)
      : partition_(range, kamping::v2::handle(comm)),
        comm_(comm),
        rank_(comm.rank()),
        first_(partition_.vertex_range().first),
        num_local_vertices_(partition_.vertex_range().second -
                            partition_.vertex_range().first),
        ghost_ids_(std::move(ghost_ids)) {
    KASSERT(std::ranges::is_sorted(ghost_ids_));
    KASSERT(std::ranges::adjacent_find(ghost_ids_) == ghost_ids_.end());
    owners_.reserve(ghost_ids_.size());
    for (VertexId const gid : ghost_ids_) {
      KASSERT(!is_local_id(gid));
      owners_.push_back(partition_.get_rank(gid));
    }
    num_ghosts_below_ = static_cast<std::size_t>(
        std::ranges::lower_bound(ghost_ids_, first_) - ghost_ids_.begin());
  }

  /// The handle of the local vertex with local index `index` (needs no layer:
  /// local handles do not depend on the number of local vertices).
  [[nodiscard]] static auto local_handle(std::size_t index) -> VertexHandle {
    return VertexHandle{index};
  }
  /// The handle of the ghost with ghost index `index`.
  [[nodiscard]] auto ghost_handle(std::size_t index) const -> VertexHandle {
    KASSERT(index < num_ghosts());
    return VertexHandle{num_local_vertices_ + index};
  }
  /// The local index of local vertex `vertex`: 0..n-1, in global-id order.
  [[nodiscard]] auto to_local_index(VertexHandle vertex) const -> std::size_t {
    KASSERT(is_local(vertex));
    return static_cast<std::size_t>(vertex);
  }
  /// The ghost index of ghost `vertex`: 0..g-1, in global-id order.
  [[nodiscard]] auto to_ghost_index(VertexHandle vertex) const -> std::size_t {
    KASSERT(is_ghost(vertex));
    return static_cast<std::size_t>(vertex) - num_local_vertices_;
  }

  [[nodiscard]] auto num_local_vertices() const -> std::size_t {
    return num_local_vertices_;
  }
  [[nodiscard]] auto num_ghosts() const -> std::size_t {
    return ghost_ids_.size();
  }
  /// The local vertices, ascending.
  [[nodiscard]] auto local_vertices() const {
    return std::views::iota(std::size_t{0}, num_local_vertices_) |
           std::views::transform(&VertexLayer::local_handle);
  }
  /// The ghosts, ascending.
  [[nodiscard]] auto ghosts() const {
    return std::views::iota(std::size_t{0}, num_ghosts()) |
           std::views::transform([this](std::size_t index) -> VertexHandle {
             return ghost_handle(index);
           });
  }
  /// The first ghost: every handle below it is a local vertex.
  [[nodiscard]] auto first_ghost() const -> VertexHandle {
    return VertexHandle{num_local_vertices_};
  }
  /// The first ghost above the local range: the ghosts [first_ghost(),
  /// first_ghost_above()) have smaller global ids than the local vertices, the
  /// ghosts from first_ghost_above() on larger ones.
  [[nodiscard]] auto first_ghost_above() const -> VertexHandle {
    return VertexHandle{num_local_vertices_ + num_ghosts_below_};
  }
  /// This rank's half-open range of global ids.
  [[nodiscard]] auto local_range() const -> std::pair<VertexId, VertexId> {
    return {first_, first_ + num_local_vertices_};
  }

  [[nodiscard]] auto is_local(VertexHandle vertex) const -> bool {
    return vertex < first_ghost();
  }
  [[nodiscard]] auto is_ghost(VertexHandle vertex) const -> bool {
    return !is_local(vertex);
  }
  [[nodiscard]] auto owner(VertexHandle vertex) const -> PEID {
    return is_local(vertex) ? rank_ : owners_[to_ghost_index(vertex)];
  }
  [[nodiscard]] auto global_id(VertexHandle vertex) const -> VertexId {
    return is_local(vertex) ? first_ + to_local_index(vertex)
                            : ghost_ids_[to_ghost_index(vertex)];
  }

  /// Whether the vertex with global id `gid` is a local vertex of this rank.
  [[nodiscard]] auto is_local_id(VertexId gid) const -> bool {
    return gid >= first_ && gid - first_ < num_local_vertices_;
  }
  /// The handle of a local vertex or ghost (binary search for a ghost).
  [[nodiscard]] auto handle(VertexId gid) const -> VertexHandle {
    if (is_local_id(gid)) {
      return local_handle(gid - first_);
    }
    auto const it = std::ranges::lower_bound(ghost_ids_, gid);
    KASSERT(it != ghost_ids_.end() && *it == gid,
            "vertex " << gid << " is not a ghost of rank " << rank_);
    return ghost_handle(static_cast<std::size_t>(it - ghost_ids_.begin()));
  }

  [[nodiscard]] auto comm() const -> kamping::v2::comm_view { return comm_; }
  [[nodiscard]] auto partition() const -> VertexPartition<VertexId> const& {
    return partition_;
  }
  /// The 2D grid over the communicator's ranks for the grid exchange, built on
  /// first use (a collective split: the first call must be made by all ranks
  /// together, as every exchange is). Each run builds its own, as it builds its
  /// vertex layer; it is freed with the layer.
  [[nodiscard]] auto grid() const
      -> dstl::grid_comm<dstl::execution_policy::seq> const& {
    if (!grid_) {
      CALI_CXX_MARK_SCOPE("grid_comm");
      grid_.emplace(comm_);
    }
    return *grid_;
  }

private:
  VertexPartition<VertexId> partition_;
  kamping::v2::comm_view comm_;
  PEID rank_;  // for owner()
  VertexId first_;
  std::size_t num_local_vertices_;
  std::vector<VertexId> ghost_ids_;
  std::vector<PEID> owners_;
  std::size_t num_ghosts_below_ = 0;
  mutable std::optional<dstl::grid_comm<dstl::execution_policy::seq>> grid_;
};

/// A rank's graph on handles: the neighbors of the local vertices (CSR: xadj,
/// adjncy) and, optionally (CETRIC), each ghost's out-edges to local vertices
/// (a second CSR over the ghosts). The vertex queries forward to the layer.
class Graph {
public:
  Graph() = default;
  Graph(std::shared_ptr<VertexLayer const> layer,
        std::vector<std::size_t> xadj,
        std::vector<VertexHandle> adjncy,
        std::vector<std::size_t> ghost_xadj = {},
        std::vector<VertexHandle> ghost_adjncy = {})
      : layer_(std::move(layer)),
        xadj_(std::move(xadj)),
        adjncy_(std::move(adjncy)),
        ghost_xadj_(std::move(ghost_xadj)),
        ghost_adjncy_(std::move(ghost_adjncy)) {
    KASSERT(xadj_.size() == layer_->num_local_vertices() + 1);
    KASSERT(xadj_.back() == adjncy_.size());
    KASSERT(ghost_xadj_.empty() ||
            ghost_xadj_.size() == layer_->num_ghosts() + 1);
    KASSERT(ghost_xadj_.empty() || ghost_xadj_.back() == ghost_adjncy_.size());
  }

  [[nodiscard]] auto layer() const
      -> std::shared_ptr<VertexLayer const> const& {
    return layer_;
  }
  [[nodiscard]] auto comm() const -> kamping::v2::comm_view {
    return layer_->comm();
  }

  [[nodiscard]] auto num_local_vertices() const -> std::size_t {
    return layer_->num_local_vertices();
  }
  [[nodiscard]] auto num_ghosts() const -> std::size_t {
    return layer_->num_ghosts();
  }
  /// The number of edges of the local vertices.
  [[nodiscard]] auto num_local_edges() const -> std::size_t {
    return adjncy_.size();
  }
  [[nodiscard]] auto local_vertices() const { return layer_->local_vertices(); }
  [[nodiscard]] auto ghosts() const { return layer_->ghosts(); }

  [[nodiscard]] auto is_local(VertexHandle vertex) const -> bool {
    return layer_->is_local(vertex);
  }
  [[nodiscard]] auto is_ghost(VertexHandle vertex) const -> bool {
    return layer_->is_ghost(vertex);
  }
  [[nodiscard]] auto owner(VertexHandle vertex) const -> PEID {
    return layer_->owner(vertex);
  }
  [[nodiscard]] auto global_id(VertexHandle vertex) const -> VertexId {
    return layer_->global_id(vertex);
  }
  [[nodiscard]] auto to_local_index(VertexHandle vertex) const -> std::size_t {
    return layer_->to_local_index(vertex);
  }

  /// The neighbors of local vertex `vertex`.
  [[nodiscard]] auto neighbors(VertexHandle vertex) const
      -> std::span<VertexHandle const> {
    auto const index = to_local_index(vertex);
    return std::span(adjncy_).subspan(xadj_[index],
                                      xadj_[index + 1] - xadj_[index]);
  }
  [[nodiscard]] auto degree(VertexHandle vertex) const -> Degree {
    auto const index = to_local_index(vertex);
    return xadj_[index + 1] - xadj_[index];
  }

  [[nodiscard]] auto has_ghost_out_edges() const -> bool {
    return !ghost_xadj_.empty();
  }
  /// The out-neighbors of ghost `ghost`: local vertices, ascending.
  [[nodiscard]] auto ghost_out_neighbors(VertexHandle ghost) const
      -> std::span<VertexHandle const> {
    KASSERT(has_ghost_out_edges());
    auto const index = layer_->to_ghost_index(ghost);
    return std::span(ghost_adjncy_)
        .subspan(ghost_xadj_[index],
                 ghost_xadj_[index + 1] - ghost_xadj_[index]);
  }
  [[nodiscard]] auto ghost_out_degree(VertexHandle ghost) const -> Degree {
    return ghost_out_neighbors(ghost).size();
  }

  /// Building step: sorts the neighbors of every local vertex by handle (local
  /// vertices first, then ghosts, each by global id). The ghosts' out-neighbors
  /// need no sorting: `orient` writes them in ascending order.
  void sort_neighborhoods() {
    for (std::size_t index = 0; index + 1 < xadj_.size(); ++index) {
      std::sort(
          adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[index]),
          adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[index + 1]));
    }
  }

  /// Hands over the vertex layer and the arrays of the local vertices (for a
  /// conversion that reuses them); the ghosts' out-edges are dropped.
  [[nodiscard]] auto
  release() && -> std::tuple<std::shared_ptr<VertexLayer const>,
                             std::vector<std::size_t>,
                             std::vector<VertexHandle>> {
    return {std::move(layer_), std::move(xadj_), std::move(adjncy_)};
  }

private:
  std::shared_ptr<VertexLayer const> layer_;
  std::vector<std::size_t> xadj_;
  std::vector<VertexHandle> adjncy_;
  std::vector<std::size_t> ghost_xadj_;
  std::vector<VertexHandle> ghost_adjncy_;
};

/// The graph of the global phase: the out-neighbors of the local vertices as
/// global ids, each neighborhood ascending. Converted from an oriented Graph
/// after the local phase (graph/build.h): without contraction every
/// neighborhood is kept, with contraction (CETRIC) only the cut edges of the
/// vertices with any. Messages carry global ids, so the global phase works on
/// these directly.
class GlobalIdGraph {
public:
  GlobalIdGraph() = default;
  /// Every local vertex is part of the graph (DiTriC).
  GlobalIdGraph(std::shared_ptr<VertexLayer const> layer,
                std::vector<std::size_t> xadj,
                std::vector<VertexId> adjncy)
      : layer_(std::move(layer)),
        xadj_(std::move(xadj)),
        adjncy_(std::move(adjncy)) {
    KASSERT(xadj_.size() == layer_->num_local_vertices() + 1);
    KASSERT(xadj_.back() == adjncy_.size());
  }
  /// Contracted: only the local vertices `kept` (ascending) are part of the
  /// graph; the others have no neighbors.
  GlobalIdGraph(std::shared_ptr<VertexLayer const> layer,
                std::vector<std::size_t> xadj,
                std::vector<VertexId> adjncy,
                std::vector<VertexHandle> kept)
      : GlobalIdGraph(std::move(layer), std::move(xadj), std::move(adjncy)) {
    KASSERT(std::ranges::is_sorted(kept));
    contracted_ = true;
    kept_ = std::move(kept);
    is_kept_.assign(layer_->num_local_vertices(), false);
    for (VertexHandle const vertex : kept_) {
      is_kept_[layer_->to_local_index(vertex)] = true;
    }
  }

  [[nodiscard]] auto layer() const
      -> std::shared_ptr<VertexLayer const> const& {
    return layer_;
  }
  [[nodiscard]] auto comm() const -> kamping::v2::comm_view {
    return layer_->comm();
  }
  /// The number of edges of the local vertices.
  [[nodiscard]] auto num_local_edges() const -> std::size_t {
    return adjncy_.size();
  }
  /// Whether local vertex `vertex` is part of the graph (contracted: it had cut
  /// edges).
  [[nodiscard]] auto is_kept(VertexHandle vertex) const -> bool {
    return !contracted_ || is_kept_[layer_->to_local_index(vertex)];
  }
  /// The kept local vertices, ascending (contracted graphs only).
  [[nodiscard]] auto kept() const -> std::span<VertexHandle const> {
    return kept_;
  }
  /// The out-neighbors of local vertex `vertex` as global ids, ascending.
  [[nodiscard]] auto neighbors(VertexHandle vertex) const
      -> std::span<VertexId const> {
    auto const index = layer_->to_local_index(vertex);
    return std::span(adjncy_).subspan(xadj_[index],
                                      xadj_[index + 1] - xadj_[index]);
  }

private:
  std::shared_ptr<VertexLayer const> layer_;
  std::vector<std::size_t> xadj_;
  std::vector<VertexId> adjncy_;
  bool contracted_ = false;
  std::vector<VertexHandle> kept_;
  std::vector<bool> is_kept_;
};

}  // namespace katric::graph
