// The graphs the algorithms run on: CSR (xadj, adjncy) over vertex handles. A
// rank numbers its local vertices 0..n-1 (in global-id order, so local index =
// global id - first) and its ghosts n..n+g-1 (in global-id order); adjncy holds
// these handles ("neighbor handles", as in KaCCv2), not global ids. A handle is
// a type of its own without arithmetic: the vertex numbering (VertexNumbering)
// converts it to a local or ghost index and back, to its global id and to its
// owner (a comparison or an array lookup). Graph's member functions build it
// step by step (from KaGen's CSR, ghost degrees, orientation, sorting); the
// algorithm decides the steps and their order (katric/phases/).
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
#include <kagen.h>
#include <dstl/distribution.hpp>
#include <dstl/grid_comm.hpp>
#include <kamping/v2/comm.hpp>
#include <kassert/kassert.hpp>

#include "katric/config.h"
#include "katric/ghost_exchange.h"
#include "katric/message_tags.h"

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

/// The handle with number `index`: for the numbering's vertex ranges (a named
/// type, so that all of them have the same type).
struct ToHandle {
  auto operator()(std::size_t index) const -> VertexHandle {
    return VertexHandle{index};
  }
};

/// A contiguous range of handles, ascending (`local_vertices()`, `ghosts()`,
/// `vertices()`).
using VertexRange = std::ranges::transform_view<
    std::ranges::iota_view<std::size_t, std::size_t>, ToHandle>;

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
  [[nodiscard]] auto local_vertices() const -> VertexRange {
    return handles(0, num_local_vertices_);
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
  [[nodiscard]] auto owner(VertexHandle vertex) const -> int {
    KASSERT(is_local(vertex));
    return rank_;
  }

  /// The partition table: the vertex range of every rank.
  [[nodiscard]] auto distribution() const -> dstl::distribution const& {
    return distribution_;
  }
  /// The rank whose range contains global id `gid` (binary search over the
  /// partition table; O(1) for equal ranges).
  [[nodiscard]] auto owner_of_id(VertexId gid) const -> int {
    return distribution_.get_owner(gid);
  }
  /// Rank `rank`'s half-open range of global ids.
  [[nodiscard]] auto range_of(int rank) const -> std::pair<VertexId, VertexId> {
    return {distribution_.index_range_begin(rank),
            distribution_.index_range_end(rank)};
  }
  [[nodiscard]] auto comm() const -> kamping::v2::comm_view { return comm_; }

protected:
  /// The handles [first, last).
  static auto handles(std::size_t first, std::size_t last) -> VertexRange {
    return std::views::iota(first, last) | std::views::transform(ToHandle{});
  }

private:
  dstl::distribution distribution_;
  kamping::v2::comm_view comm_;
  int rank_;  // for owner()
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
  [[nodiscard]] auto ghosts() const -> VertexRange {
    return handles(num_local_vertices(), num_local_vertices() + num_ghosts());
  }
  /// The local vertices, then the ghosts: all handles, ascending.
  [[nodiscard]] auto vertices() const -> VertexRange {
    return handles(0, num_local_vertices() + num_ghosts());
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
  [[nodiscard]] auto owner(VertexHandle vertex) const -> int {
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
  std::vector<int> owners_;
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

  /// Building step: the neighbors of the vertex with index `index`, to reorder
  /// in place.
  [[nodiscard]] auto mutable_neighbors(std::size_t index)
      -> std::span<Neighbor> {
    KASSERT(index < num_vertices());
    return std::span(adjncy_).subspan(xadj_[index], degree(index));
  }
  /// Building step: keeps the first `counts[index]` neighbors of every vertex
  /// and drops the rest, in place (the capacity stays).
  void truncate_neighborhoods(std::span<std::size_t const> counts) {
    KASSERT(counts.size() == num_vertices());
    std::size_t write = 0;
    for (std::size_t index = 0; index < num_vertices(); ++index) {
      KASSERT(counts[index] <= degree(index));
      std::size_t const read = xadj_[index];
      if (read != write) {
        // moves left: std::copy allows it (the target starts before the source)
        auto const source = adjncy_.begin() + static_cast<std::ptrdiff_t>(read);
        std::copy(source, source + static_cast<std::ptrdiff_t>(counts[index]),
                  adjncy_.begin() + static_cast<std::ptrdiff_t>(write));
      }
      xadj_[index] = write;
      write += counts[index];
    }
    xadj_.back() = write;
    adjncy_.resize(write);
  }

  /// Sorts every neighborhood ascending.
  void sort_neighborhoods() {
    for (std::size_t index = 0; index < num_vertices(); ++index) {
      std::sort(
          adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[index]),
          adjncy_.begin() + static_cast<std::ptrdiff_t>(xadj_[index + 1]));
    }
  }

private:
  std::vector<std::size_t> xadj_;
  std::vector<Neighbor> adjncy_;
};

/// A rank's graph on handles: the neighbors of the local vertices and the
/// parts the building steps add, each available from its step on (checked by
/// assertions): the undirected degrees (exchange_ghost_degrees()), the
/// orientation and, optionally (CETRIC), each ghost's out-edges to local
/// vertices (orient()), the split of each sorted neighborhood into local and
/// ghost heads (sort_neighborhoods()). The vertex queries forward to the
/// numbering.
class Graph {
public:
  /// An undirected graph.
  Graph(VertexNumbering numbering, AdjacencyArray<VertexHandle> local)
      : numbering_(std::move(numbering)), local_(std::move(local)) {
    KASSERT(local_.num_vertices() == numbering_.num_local_vertices());
  }

  /// The undirected graph of a rank's part of the input (KaGen's CSR: vertex
  /// range, xadj, adjncy in global ids), with ghosts. Collective (builds the
  /// partition table). Neighborhoods keep the input's order. The cut edges are
  /// sorted once by (head, position): that gives the ghosts in global-id order,
  /// and each run of equal heads one ghost, so their handles need no lookup per
  /// edge.
  static auto from_kagen(kagen::Graph const& input, kamping::v2::comm_view comm)
      -> Graph {
    KASSERT(input.representation == kagen::GraphRepresentation::CSR);
    auto const [first, last] = input.vertex_range;
    // an empty range may have any first and last
    LocalNumbering local(
        dstl::distribution(first < last ? last - first : 0, comm), comm);
    KASSERT(first >= last || local.local_range().first == first,
            "the vertex ranges are not contiguous in rank order");
    std::vector<VertexHandle> adjncy(input.adjncy.size());
    // (head, position in adjncy), sorted
    std::vector<std::pair<VertexId, std::size_t>> cut;
    for (std::size_t position = 0; position < input.adjncy.size();
         ++position) {
      VertexId const gid = input.adjncy[position];
      if (local.is_local_id(gid)) {
        adjncy[position] = local.handle(gid);
      } else {
        cut.emplace_back(gid, position);
      }
    }
    std::ranges::sort(cut);
    std::vector<VertexId> ghost_ids;
    std::ranges::unique_copy(cut | std::views::keys,
                             std::back_inserter(ghost_ids));
    VertexNumbering numbering(std::move(local), std::move(ghost_ids));
    std::size_t ghost_index = 0;
    for (std::size_t i = 0; i < cut.size(); ++i) {
      if (i > 0 && cut[i].first != cut[i - 1].first) {
        ++ghost_index;  // the next run: the next ghost
      }
      adjncy[cut[i].second] = numbering.ghost_handle(ghost_index);
    }
    decltype(cut){}.swap(cut);  // frees the cut pairs before the copy of xadj
    // copied: the input is only read (the harness reuses it for every
    // iteration), and KaGen's SInt need not be std::size_t (unsigned long long
    // vs. unsigned long)
    std::vector<std::size_t> xadj(input.xadj.begin(), input.xadj.end());
    if (xadj.empty()) {
      xadj.push_back(0);
    }
    KASSERT(xadj.size() == numbering.num_local_vertices() + 1);
    return {std::move(numbering), {std::move(xadj), std::move(adjncy)}};
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

  /// The local vertices, then the ghosts: all handles, ascending.
  [[nodiscard]] auto vertices() const { return numbering_.vertices(); }

  [[nodiscard]] auto is_local(VertexHandle vertex) const -> bool {
    return numbering_.is_local(vertex);
  }
  [[nodiscard]] auto is_ghost(VertexHandle vertex) const -> bool {
    return numbering_.is_ghost(vertex);
  }
  [[nodiscard]] auto owner(VertexHandle vertex) const -> int {
    return numbering_.owner(vertex);
  }
  [[nodiscard]] auto global_id(VertexHandle vertex) const -> VertexId {
    return numbering_.global_id(vertex);
  }
  [[nodiscard]] auto to_local_index(VertexHandle vertex) const -> std::size_t {
    return numbering_.to_local_index(vertex);
  }
  /// Whether global id `gid` is one of this rank's vertices.
  [[nodiscard]] auto is_local_id(VertexId gid) const -> bool {
    return numbering_.is_local_id(gid);
  }
  /// The handle of global id `gid` (a local vertex or a ghost).
  [[nodiscard]] auto handle(VertexId gid) const -> VertexHandle {
    return numbering_.handle(gid);
  }
  /// The rank owning global id `gid`.
  [[nodiscard]] auto owner_of_id(VertexId gid) const -> int {
    return numbering_.owner_of_id(gid);
  }
  /// Rank `rank`'s half-open range of global ids.
  [[nodiscard]] auto range_of(int rank) const -> std::pair<VertexId, VertexId> {
    return numbering_.range_of(rank);
  }

  /// The neighbors of local vertex `vertex` (before to_global_ids()).
  [[nodiscard]] auto local_neighbors(VertexHandle vertex) const
      -> std::span<VertexHandle const> {
    KASSERT(!has_global_neighbors_, "converted to global ids");
    return local_.neighbors(to_local_index(vertex));
  }

  /// The number of neighbors of local vertex `vertex` in this graph: after
  /// orient() its out-degree (before to_global_ids()).
  [[nodiscard]] auto degree(VertexHandle vertex) const -> Degree {
    KASSERT(!has_global_neighbors_, "converted to global ids");
    return local_.degree(to_local_index(vertex));
  }

  /// After exchange_ghost_degrees(): the degree of local vertex or ghost
  /// `vertex` in the undirected graph (before the orientation).
  [[nodiscard]] auto undirected_degree(VertexHandle vertex) const -> Degree {
    KASSERT(has_undirected_degrees(), "ghost degrees not exchanged");
    return is_local(vertex)
               ? local_undirected_degrees_[to_local_index(vertex)]
               : ghost_undirected_degrees_[numbering_.to_ghost_index(vertex)];
  }
  [[nodiscard]] auto has_undirected_degrees() const -> bool {
    return degrees_exchanged_;
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

  [[nodiscard]] auto neighbors(VertexHandle vertex) const
      -> std::span<VertexHandle const> {
    if (is_local(vertex)) {
      return local_neighbors(vertex);
    }
    return ghost_out_neighbors(vertex);
  }

  /// After sort_neighborhoods(): the neighbors of `vertex` that are local
  /// vertices, the prefix of its sorted neighborhood (a ghost's neighbors are
  /// all local).
  [[nodiscard]] auto local_heads(VertexHandle vertex) const
      -> std::span<VertexHandle const> {
    if (is_local(vertex)) {
      return local_neighbors(vertex).first(num_local_heads(vertex));
    }
    return ghost_out_neighbors(vertex);
  }
  /// After sort_neighborhoods(): the neighbors of `vertex` that are ghosts
  /// (its cut edges), the rest of its sorted neighborhood; none for a ghost.
  [[nodiscard]] auto ghost_heads(VertexHandle vertex) const
      -> std::span<VertexHandle const> {
    if (is_local(vertex)) {
      return local_neighbors(vertex).subspan(num_local_heads(vertex));
    }
    return {};
  }

  /// Building step on the undirected graph: records the degrees of the local
  /// vertices and receives those of the ghosts (push_to_ghosts() with `mode`
  /// and `tag`; collective). Reports the exchange's counters to the open
  /// Caliper region.
  void exchange_ghost_degrees(GhostExchangeMode mode, MessageTag tag) {
    KASSERT(!oriented_, "the degree exchange needs the undirected graph");
    local_undirected_degrees_.resize(num_local_vertices());
    for (VertexHandle const v : local_vertices()) {
      local_undirected_degrees_[to_local_index(v)] = degree(v);
    }
    auto const ghost_degrees = push_to_ghosts(
        *this, [&](VertexHandle v) { return degree(v); }, mode, tag);
    ghost_undirected_degrees_.assign(ghost_degrees.begin(),
                                     ghost_degrees.end());
    degrees_exchanged_ = true;
  }

  /// Building step: orients the edges of the undirected graph from the smaller
  /// to the larger `key`, in place: each neighborhood is partitioned
  /// (out-neighbors first) and cut to its out-neighbors, keeping the arrays and
  /// their capacity; neighborhoods in no particular order afterwards. `key` is
  /// evaluated once per vertex and once per edge; it may read the degrees and
  /// the numbering, not the neighborhoods. With `ghost_out_edges`, also the
  /// ghosts' out-edges to local vertices (CETRIC), sorted.
  template <typename Key>
  void orient(Key const& key, bool ghost_out_edges) {
    KASSERT(!oriented_);
    std::vector<std::size_t> out_degree(num_local_vertices());
    for (VertexHandle const v : local_vertices()) {
      auto const key_v = key(v);
      auto const neighbors = local_.mutable_neighbors(to_local_index(v));
      auto const in_begin = std::partition(
          neighbors.begin(), neighbors.end(),
          [&](VertexHandle u) -> bool { return key_v < key(u); });
      out_degree[to_local_index(v)] =
          static_cast<std::size_t>(in_begin - neighbors.begin());
    }
    // the ghosts' out-edges: a local vertex's in-neighbors that are ghosts
    if (ghost_out_edges) {
      auto const in_neighbors = [&](VertexHandle v) {
        return local_neighbors(v).subspan(out_degree[to_local_index(v)]);
      };
      std::vector<std::size_t> ghost_xadj(num_ghosts() + 1, 0);
      for (VertexHandle const v : local_vertices()) {
        for (VertexHandle const u : in_neighbors(v)) {
          if (is_ghost(u)) {
            ghost_xadj[numbering_.to_ghost_index(u) + 1]++;
          }
        }
      }
      for (std::size_t index = 0; index + 1 < ghost_xadj.size(); ++index) {
        ghost_xadj[index + 1] += ghost_xadj[index];
      }
      std::vector<VertexHandle> ghost_adjncy(ghost_xadj.back());
      std::vector<std::size_t> next(ghost_xadj.begin(), ghost_xadj.end() - 1);
      for (VertexHandle const v : local_vertices()) {
        for (VertexHandle const u : in_neighbors(v)) {
          if (is_ghost(u)) {
            // v ascending: each ghost's neighborhood comes out sorted
            ghost_adjncy[next[numbering_.to_ghost_index(u)]++] = v;
          }
        }
      }
      ghost_out_ = {std::move(ghost_xadj), std::move(ghost_adjncy)};
      has_ghost_out_edges_ = true;
    }
    local_.truncate_neighborhoods(out_degree);
    oriented_ = true;
  }
  /// Building step: sorts the neighbors of every local vertex by handle (local
  /// vertices first, then ghosts, each by global id) and records where the
  /// ghosts start (local_heads() / ghost_heads()); the neighborhoods must not
  /// change afterwards. The ghosts' out-neighbors need no sorting: `orient`
  /// writes them in ascending order.
  void sort_neighborhoods() {
    local_.sort_neighborhoods();
    num_local_heads_.resize(num_local_vertices());
    for (VertexHandle const v : local_vertices()) {
      auto const neighbors = local_neighbors(v);
      num_local_heads_[to_local_index(v)] = static_cast<std::size_t>(
          std::ranges::lower_bound(neighbors, numbering_.first_ghost()) -
          neighbors.begin());
    }
  }

  /// Building step after the local phase: replaces the neighborhoods of the
  /// local vertices by their global ids, for the global phase, which compares
  /// them with received global ids; with `cut_edges_only` (CETRIC) only the
  /// ghost heads are kept. Each neighborhood comes out ascending: a sorted
  /// neighborhood is [local heads][ghosts below the local range][ghosts above]
  /// and is translated as [ghosts below][local heads][ghosts above]. The handle
  /// neighborhoods and the ghosts' out-edges are freed: afterwards only
  /// global_neighbors() and the vertex queries are available.
  void to_global_ids(bool cut_edges_only) {
    KASSERT(!has_global_neighbors_);
    std::vector<std::size_t> xadj(num_local_vertices() + 1, 0);
    std::size_t num_entries = 0;
    for (VertexHandle const v : local_vertices()) {
      num_entries += cut_edges_only ? ghost_heads(v).size() : degree(v);
    }
    std::vector<VertexId> ids;
    ids.reserve(num_entries);
    auto const translate = [&](std::span<VertexHandle const> heads) {
      for (VertexHandle const u : heads) {
        ids.push_back(global_id(u));
      }
    };
    for (VertexHandle const v : local_vertices()) {
      auto const ghosts = ghost_heads(v);
      if (cut_edges_only) {
        translate(ghosts);  // [ghosts below][ghosts above]: already ascending
      } else {
        auto const above = static_cast<std::size_t>(
            std::ranges::lower_bound(ghosts, numbering_.first_ghost_above()) -
            ghosts.begin());
        translate(ghosts.first(above));
        translate(local_heads(v));
        translate(ghosts.subspan(above));
      }
      xadj[to_local_index(v) + 1] = ids.size();
    }
    global_neighbors_ = {std::move(xadj), std::move(ids)};
    has_global_neighbors_ = true;
    local_ = {};
    ghost_out_ = {};
    has_ghost_out_edges_ = false;
    num_local_heads_ = {};
  }
  /// After to_global_ids(): the neighbors of local vertex `vertex` (all, or
  /// only the cut edges) as global ids, ascending.
  [[nodiscard]] auto global_neighbors(VertexHandle vertex) const
      -> std::span<VertexId const> {
    KASSERT(has_global_neighbors_, "not converted to global ids");
    return global_neighbors_.neighbors(to_local_index(vertex));
  }
  /// After to_global_ids(): the number of entries of all global neighborhoods.
  [[nodiscard]] auto num_global_neighbors() const -> std::size_t {
    KASSERT(has_global_neighbors_, "not converted to global ids");
    return global_neighbors_.num_edges();
  }
  /// After to_global_ids(): whether local vertex `vertex` has a cut out-edge:
  /// its neighborhood (ascending) starts below or ends above this rank's range.
  [[nodiscard]] auto has_cut_edges(VertexHandle vertex) const -> bool {
    auto const neighbors_v = global_neighbors(vertex);
    auto const [first, last] = numbering_.local_range();
    return !neighbors_v.empty() &&
           (neighbors_v.front() < first || neighbors_v.back() >= last);
  }

private:
  [[nodiscard]] auto num_local_heads(VertexHandle vertex) const -> std::size_t {
    KASSERT(num_local_heads_.size() == num_local_vertices(),
            "neighborhoods not sorted");
    return num_local_heads_[to_local_index(vertex)];
  }

  VertexNumbering numbering_;
  AdjacencyArray<VertexHandle> local_;
  std::vector<Degree> local_undirected_degrees_;  // once exchanged
  std::vector<Degree> ghost_undirected_degrees_;  // once exchanged
  bool degrees_exchanged_ = false;
  bool oriented_ = false;
  bool has_ghost_out_edges_ = false;
  AdjacencyArray<VertexHandle> ghost_out_;  // empty without
  std::vector<std::size_t> num_local_heads_;  // per local vertex, once sorted
  bool has_global_neighbors_ = false;
  AdjacencyArray<VertexId> global_neighbors_;  // empty before to_global_ids()
};

/// `G` in KaGen's CSR representation (global ids), for the load balancing.
/// Consumes `G`: its arrays are freed on return.
inline auto to_kagen_graph(Graph&& G) -> kagen::Graph {
  Graph const graph = std::move(G);
  kagen::Graph input;
  input.representation = kagen::GraphRepresentation::CSR;
  input.vertex_range = graph.numbering().local_range();
  input.xadj.reserve(graph.num_local_vertices() + 1);
  input.adjncy.reserve(graph.num_local_edges());
  input.xadj.push_back(0);
  for (VertexHandle const v : graph.local_vertices()) {
    for (VertexHandle const u : graph.local_neighbors(v)) {
      input.adjncy.push_back(graph.global_id(u));
    }
    input.xadj.push_back(input.adjncy.size());
  }
  return input;
}

}  // namespace katric::graph
