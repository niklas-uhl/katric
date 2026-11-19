// The graphs the algorithms run on: CSR (xadj, adjncy) over vertex handles. A
// rank numbers its local vertices 0..n-1 (in global-id order, so local index =
// global id - first) and its ghosts n..n+g-1 (in global-id order); adjncy holds
// these handles ("neighbor handles", as in KaCCv2), not global ids. A handle is
// a type of its own without arithmetic: the vertex numbering (VertexNumbering)
// converts it to a local or ghost index and back, to its global id and to its
// owner (a comparison or an array lookup). The graphs are immutable once built
// (graph/build.h builds them).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <dstl/distribution.hpp>
#include <dstl/grid_comm.hpp>
#include <kamping/v2/comm.hpp>
#include <kassert/kassert.hpp>

#include "katric/util.h"

namespace katric::graph {

/// A vertex's global id: its position in the input (KaGen's vertex ids).
using VertexId = std::uint64_t;
/// The number of neighbors of a vertex.
using Degree = std::size_t;

/// A vertex as seen by one rank: a local vertex or a ghost. Ordered like the
/// vertices' global ids within each of the two groups, local vertices first;
/// no arithmetic (VertexNumbering converts).
enum class VertexHandle : std::uint64_t {};

/// For fmt (assertion and log messages): the handle's number.
inline auto format_as(VertexHandle vertex) -> std::uint64_t {
  return static_cast<std::uint64_t>(vertex);
}

/// The numbering of this rank's local vertices: the partition table over the
/// communicator, and handles 0..n-1 for the local vertices (local index =
/// global id - first). Known before the ghosts are (the input graph's scan
/// writes local handles with it); VertexNumbering adds the ghosts. The queries
/// taking a vertex (`handle`, `global_id`, `owner`) cover the local vertices
/// only here; VertexNumbering extends them to the ghosts.
class LocalNumbering {
public:
  /// `distribution` is the partition table over `comm`; this rank's local
  /// vertices are its range there.
  LocalNumbering(dstl::distribution distribution, kamping::v2::comm_view comm)
      : distribution_(std::move(distribution)),
        comm_(comm),
        rank_(comm.rank()),
        first_(distribution_.index_range_begin(rank_)),
        num_local_vertices_(distribution_.local_size(rank_)) {}

  /// The handle of the local vertex with local index `index`.
  [[nodiscard]] auto local_handle(std::size_t index) const -> VertexHandle {
    KASSERT(index < num_local_vertices_);
    return VertexHandle{index};
  }
  /// The local index of local vertex `vertex`: 0..n-1, in global-id order.
  [[nodiscard]] auto to_local_index(VertexHandle vertex) const -> std::size_t {
    KASSERT(is_local(vertex));
    return static_cast<std::size_t>(vertex);
  }

  [[nodiscard]] auto num_local_vertices() const -> std::size_t {
    return num_local_vertices_;
  }
  /// The local vertices, ascending.
  [[nodiscard]] auto local_vertices() const {
    return std::views::iota(std::size_t{0}, num_local_vertices_) |
           std::views::transform([this](std::size_t index) -> VertexHandle {
             return local_handle(index);
           });
  }
  /// This rank's half-open range of global ids.
  [[nodiscard]] auto local_range() const -> std::pair<VertexId, VertexId> {
    return {first_, first_ + num_local_vertices_};
  }

  [[nodiscard]] auto is_local(VertexHandle vertex) const -> bool {
    return static_cast<std::size_t>(vertex) < num_local_vertices_;
  }
  /// Whether the vertex with global id `gid` is a local vertex of this rank.
  [[nodiscard]] auto is_local_id(VertexId gid) const -> bool {
    return gid >= first_ && gid - first_ < num_local_vertices_;
  }
  /// The handle of local vertex `gid`.
  [[nodiscard]] auto handle(VertexId gid) const -> VertexHandle {
    KASSERT(is_local_id(gid));
    return local_handle(gid - first_);
  }
  /// The global id of local vertex `vertex`.
  [[nodiscard]] auto global_id(VertexHandle vertex) const -> VertexId {
    return first_ + to_local_index(vertex);
  }
  /// The owner of local vertex `vertex`: this rank.
  [[nodiscard]] auto owner(VertexHandle vertex) const -> PEID {
    KASSERT(is_local(vertex));
    return rank_;
  }

  /// The partition table: the vertex range of every rank.
  [[nodiscard]] auto distribution() const -> dstl::distribution const& {
    return distribution_;
  }
  [[nodiscard]] auto comm() const -> kamping::v2::comm_view { return comm_; }

private:
  dstl::distribution distribution_;
  kamping::v2::comm_view comm_;
  PEID rank_;  // for owner()
  VertexId first_;
  std::size_t num_local_vertices_;
};

/// The numbering of every vertex this rank knows: the local vertices as in
/// LocalNumbering (handles 0..n-1), then the ghosts (handles n..n+g-1), each
/// in global-id order; the ghosts' global ids and owners.
///
/// The graphs of a run (input, oriented, global ids) use the same handles: each
/// graph holds its numbering by value, and each step consumes its input graph
/// and moves the numbering on to the graph it builds, so a numbering has one
/// owner at a time and is never copied (moving it moves its vectors). Load
/// balancing replaces the input graph together with its numbering. A
/// numbering's destruction is collective if a grid exchange ran on it (it
/// frees the grid communicator); all ranks drop their graphs at the same point
/// of the driver, so this is safe.
class VertexNumbering : public LocalNumbering {
public:
  /// Adds the ghosts to `local`: `ghost_ids` are their sorted, distinct global
  /// ids.
  VertexNumbering(LocalNumbering local, std::vector<VertexId> ghost_ids)
      : LocalNumbering(std::move(local)), ghost_ids_(std::move(ghost_ids)) {
    KASSERT(std::ranges::is_sorted(ghost_ids_));
    KASSERT(std::ranges::adjacent_find(ghost_ids_) == ghost_ids_.end());
    owners_.reserve(ghost_ids_.size());
    for (VertexId const gid : ghost_ids_) {
      KASSERT(!is_local_id(gid));
      owners_.push_back(distribution().get_owner(gid));
    }
    num_ghosts_below_ = static_cast<std::size_t>(
        std::ranges::lower_bound(ghost_ids_, local_range().first) -
        ghost_ids_.begin());
  }

  /// The handle of the ghost with ghost index `index`.
  [[nodiscard]] auto ghost_handle(std::size_t index) const -> VertexHandle {
    KASSERT(index < num_ghosts());
    return VertexHandle{num_local_vertices() + index};
  }
  /// The ghost index of ghost `vertex`: 0..g-1, in global-id order.
  [[nodiscard]] auto to_ghost_index(VertexHandle vertex) const -> std::size_t {
    KASSERT(is_ghost(vertex));
    return static_cast<std::size_t>(vertex) - num_local_vertices();
  }

  [[nodiscard]] auto num_ghosts() const -> std::size_t {
    return ghost_ids_.size();
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
    return VertexHandle{num_local_vertices()};
  }
  /// The first ghost above the local range: the ghosts [first_ghost(),
  /// first_ghost_above()) have smaller global ids than the local vertices, the
  /// ghosts from first_ghost_above() on larger ones.
  [[nodiscard]] auto first_ghost_above() const -> VertexHandle {
    return VertexHandle{num_local_vertices() + num_ghosts_below_};
  }

  [[nodiscard]] auto is_ghost(VertexHandle vertex) const -> bool {
    return !is_local(vertex);
  }
  /// The handle of a local vertex or ghost (binary search for a ghost).
  [[nodiscard]] auto handle(VertexId gid) const -> VertexHandle {
    if (is_local_id(gid)) {
      return LocalNumbering::handle(gid);
    }
    auto const it = std::ranges::lower_bound(ghost_ids_, gid);
    KASSERT(it != ghost_ids_.end() && *it == gid,
            "vertex " << gid << " is not a ghost of rank " << comm().rank());
    return ghost_handle(static_cast<std::size_t>(it - ghost_ids_.begin()));
  }
  [[nodiscard]] auto global_id(VertexHandle vertex) const -> VertexId {
    return is_local(vertex) ? LocalNumbering::global_id(vertex)
                            : ghost_ids_[to_ghost_index(vertex)];
  }
  [[nodiscard]] auto owner(VertexHandle vertex) const -> PEID {
    return is_local(vertex) ? LocalNumbering::owner(vertex)
                            : owners_[to_ghost_index(vertex)];
  }

  /// The 2D grid over the communicator's ranks for the grid exchange, built on
  /// first use (a collective split: the first call must be made by all ranks
  /// together, as every exchange is). Each run builds its own, as it builds its
  /// numbering; it is freed with the numbering.
  [[nodiscard]] auto grid() const
      -> dstl::grid_comm<dstl::execution_policy::seq> const& {
    if (!grid_) {
      CALI_CXX_MARK_SCOPE("grid_comm");
      grid_.emplace(comm());
    }
    return *grid_;
  }

private:
  std::vector<VertexId> ghost_ids_;
  std::vector<PEID> owners_;
  std::size_t num_ghosts_below_ = 0;
  mutable std::optional<dstl::grid_comm<dstl::execution_policy::seq>> grid_;
};

/// The neighborhoods of the vertices 0..n-1 (by index), stored as offsets
/// `xadj` (n + 1 entries) and the concatenated neighborhoods `adjncy`. The
/// graphs below use it for each set of neighborhoods they hold; `Neighbor` is
/// what a neighborhood lists (handles or global ids).
template <typename Neighbor>
class AdjacencyArray {
public:
  AdjacencyArray() : xadj_(1, 0) {}
  AdjacencyArray(std::vector<std::size_t> xadj, std::vector<Neighbor> adjncy)
      : xadj_(std::move(xadj)), adjncy_(std::move(adjncy)) {
    KASSERT(!xadj_.empty());
    KASSERT(xadj_.front() == 0U);
    KASSERT(xadj_.back() == adjncy_.size());
  }

  [[nodiscard]] auto num_vertices() const -> std::size_t {
    return xadj_.size() - 1;
  }
  [[nodiscard]] auto num_edges() const -> std::size_t { return adjncy_.size(); }
  /// The neighbors of the vertex with index `index`.
  [[nodiscard]] auto neighbors(std::size_t index) const
      -> std::span<Neighbor const> {
    KASSERT(index < num_vertices());
    return std::span(adjncy_).subspan(xadj_[index], degree(index));
  }
  [[nodiscard]] auto degree(std::size_t index) const -> Degree {
    KASSERT(index < num_vertices());
    return xadj_[index + 1] - xadj_[index];
  }

  /// Sorts every neighborhood ascending.
  void sort_neighborhoods() {
    for (std::size_t index = 0; index < num_vertices(); ++index) {
      std::sort(
          adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[index]),
          adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[index + 1]));
    }
  }

  /// Hands over the arrays (for a conversion that reuses them).
  [[nodiscard]] auto
  release() && -> std::pair<std::vector<std::size_t>, std::vector<Neighbor>> {
    return {std::move(xadj_), std::move(adjncy_)};
  }

private:
  std::vector<std::size_t> xadj_;
  std::vector<Neighbor> adjncy_;
};

/// A rank's graph on handles: the neighbors of the local vertices and,
/// optionally (CETRIC), each ghost's out-edges to local vertices. The vertex
/// queries forward to the numbering.
class Graph {
public:
  Graph(VertexNumbering numbering,
        AdjacencyArray<VertexHandle> local,
        std::optional<AdjacencyArray<VertexHandle>> ghost_out = std::nullopt)
      : numbering_(std::move(numbering)),
        local_(std::move(local)),
        has_ghost_out_edges_(ghost_out.has_value()),
        ghost_out_(
            std::move(ghost_out).value_or(AdjacencyArray<VertexHandle>{})) {
    KASSERT(local_.num_vertices() == numbering_.num_local_vertices());
    KASSERT(!has_ghost_out_edges_ ||
            ghost_out_.num_vertices() == numbering_.num_ghosts());
  }

  [[nodiscard]] auto numbering() const -> VertexNumbering const& {
    return numbering_;
  }
  [[nodiscard]] auto comm() const -> kamping::v2::comm_view {
    return numbering_.comm();
  }

  [[nodiscard]] auto num_local_vertices() const -> std::size_t {
    return numbering_.num_local_vertices();
  }
  [[nodiscard]] auto num_ghosts() const -> std::size_t {
    return numbering_.num_ghosts();
  }
  /// The number of edges of the local vertices.
  [[nodiscard]] auto num_local_edges() const -> std::size_t {
    return local_.num_edges();
  }
  [[nodiscard]] auto local_vertices() const {
    return numbering_.local_vertices();
  }
  [[nodiscard]] auto ghosts() const { return numbering_.ghosts(); }

  [[nodiscard]] auto is_local(VertexHandle vertex) const -> bool {
    return numbering_.is_local(vertex);
  }
  [[nodiscard]] auto is_ghost(VertexHandle vertex) const -> bool {
    return numbering_.is_ghost(vertex);
  }
  [[nodiscard]] auto owner(VertexHandle vertex) const -> PEID {
    return numbering_.owner(vertex);
  }
  [[nodiscard]] auto global_id(VertexHandle vertex) const -> VertexId {
    return numbering_.global_id(vertex);
  }
  [[nodiscard]] auto to_local_index(VertexHandle vertex) const -> std::size_t {
    return numbering_.to_local_index(vertex);
  }

  /// The neighbors of local vertex `vertex`.
  [[nodiscard]] auto neighbors(VertexHandle vertex) const
      -> std::span<VertexHandle const> {
    return local_.neighbors(to_local_index(vertex));
  }
  [[nodiscard]] auto degree(VertexHandle vertex) const -> Degree {
    return local_.degree(to_local_index(vertex));
  }

  [[nodiscard]] auto has_ghost_out_edges() const -> bool {
    return has_ghost_out_edges_;
  }
  /// The out-neighbors of ghost `ghost`: local vertices, ascending.
  [[nodiscard]] auto ghost_out_neighbors(VertexHandle ghost) const
      -> std::span<VertexHandle const> {
    KASSERT(has_ghost_out_edges());
    return ghost_out_.neighbors(numbering_.to_ghost_index(ghost));
  }
  [[nodiscard]] auto ghost_out_degree(VertexHandle ghost) const -> Degree {
    KASSERT(has_ghost_out_edges());
    return ghost_out_.degree(numbering_.to_ghost_index(ghost));
  }

  /// Building step: sorts the neighbors of every local vertex by handle (local
  /// vertices first, then ghosts, each by global id). The ghosts' out-neighbors
  /// need no sorting: `orient` writes them in ascending order.
  void sort_neighborhoods() { local_.sort_neighborhoods(); }

  /// Hands over the vertex numbering and the neighborhoods of the local
  /// vertices (for a conversion that reuses them); the ghosts' out-edges are
  /// dropped.
  [[nodiscard]] auto
  release() && -> std::pair<VertexNumbering, AdjacencyArray<VertexHandle>> {
    return {std::move(numbering_), std::move(local_)};
  }

private:
  VertexNumbering numbering_;
  AdjacencyArray<VertexHandle> local_;
  bool has_ghost_out_edges_ = false;
  AdjacencyArray<VertexHandle> ghost_out_;  // empty without
};

/// The graph of the global phase: the out-neighbors of the local vertices as
/// global ids, each neighborhood ascending. Converted from an oriented Graph
/// after the local phase (graph/build.h): without contraction every
/// neighborhood is kept, with contraction (CETRIC) only the cut edges of the
/// vertices with any. Messages carry global ids, so the global phase works on
/// these directly.
class GlobalIdGraph {
public:
  /// Every local vertex is part of the graph (DiTriC).
  GlobalIdGraph(VertexNumbering numbering, AdjacencyArray<VertexId> local)
      : numbering_(std::move(numbering)), local_(std::move(local)) {
    KASSERT(local_.num_vertices() == numbering_.num_local_vertices());
  }
  /// Contracted: only the local vertices `kept` (ascending) are part of the
  /// graph; the others have no neighbors.
  GlobalIdGraph(VertexNumbering numbering,
                AdjacencyArray<VertexId> local,
                std::vector<VertexHandle> kept)
      : GlobalIdGraph(std::move(numbering), std::move(local)) {
    KASSERT(std::ranges::is_sorted(kept));
    contracted_ = true;
    kept_ = std::move(kept);
    is_kept_.assign(numbering_.num_local_vertices(), false);
    for (VertexHandle const vertex : kept_) {
      is_kept_[numbering_.to_local_index(vertex)] = true;
    }
  }

  [[nodiscard]] auto numbering() const -> VertexNumbering const& {
    return numbering_;
  }
  [[nodiscard]] auto comm() const -> kamping::v2::comm_view {
    return numbering_.comm();
  }
  /// The number of edges of the local vertices.
  [[nodiscard]] auto num_local_edges() const -> std::size_t {
    return local_.num_edges();
  }
  /// Whether local vertex `vertex` is part of the graph (contracted: it had cut
  /// edges).
  [[nodiscard]] auto is_kept(VertexHandle vertex) const -> bool {
    return !contracted_ || is_kept_[numbering_.to_local_index(vertex)];
  }
  /// The kept local vertices, ascending (contracted graphs only).
  [[nodiscard]] auto kept() const -> std::span<VertexHandle const> {
    return kept_;
  }
  /// The out-neighbors of local vertex `vertex` as global ids, ascending.
  [[nodiscard]] auto neighbors(VertexHandle vertex) const
      -> std::span<VertexId const> {
    return local_.neighbors(numbering_.to_local_index(vertex));
  }

private:
  VertexNumbering numbering_;
  AdjacencyArray<VertexId> local_;
  bool contracted_ = false;
  std::vector<VertexHandle> kept_;
  std::vector<bool> is_kept_;
};

}  // namespace katric::graph
