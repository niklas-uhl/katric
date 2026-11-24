/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// The run configuration, filled from the command line (apps/parse_config.cpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <magic_enum/magic_enum.hpp>

namespace katric {

// Enum values are named after their command line spelling: `_` is written as
// `-` (both are accepted).
enum class Threshold { local_nodes, local_edges, none };
enum class IntersectionMethod { merge, binary_search, binary, hybrid };
enum class TaskPoolType { std_thread, task_group };
/// The cost of a vertex for the primary load balancing,
/// `--primary-cost-function` (PATRIC's cost functions N, DPD, IDPD, and IDD;
/// `none` turns the load balancing off). See vertex_costs()
/// (katric/phases/load_balancing.h).
enum class CostFunction { none, N, DPD, IDPD, IDD };
/// How push_to_ghosts() (katric/ghost_exchange.h) moves its messages,
/// `--degree-exchange`.
enum class GhostExchangeMode {
  sparse,  ///< dSTL's sparse all-to-all (NBX): point-to-point messages to the
           ///< ranks with data only
  dense,   ///< one MPI_Alltoallv (after an MPI_Alltoall of the counts)
  grid,    ///< dSTL's two-dimensional grid all-to-all: O(sqrt p) messages per
           ///< rank instead of O(p)
};

/// How the global phase sends its messages, `--communication-policy` (`new` is
/// accepted for `direct`, the spelling before 2026-11-20).
enum class CommunicationPolicy {
  direct,  ///< through the buffered message queue to the receiver
  grid,    ///< through a 2D grid of indirections (one proxy per message)
};

/// What runs, `--algorithm`: KaTric, or a competitor behind the same
/// interface (apps/competitors/, each only with its CMake option
/// KATRIC_COMPETITOR_<NAME>, doc/competitors.md).
enum class Algorithm { katric, tric, havoqgt, xxtric, lcc };
/// TriC's counting variant, `--tric-variant`: the classes from TriC's main.cpp
/// (plain is its default build, buffered is `-DAGGR_BUFR`, hash2 is
/// `-DAGGR_HASH2`, inrecv is `-DAGGR_BUFR_INRECV`).
enum class TricVariant {
  plain,     ///< TriangulateAggrFatCompressed (cfastric.hpp)
  buffered,  ///< TriangulateAggrBuffered (bufastric.hpp)
  hash2,     ///< TriangulateAggrBufferedHash2 (hashfastric2.hpp)
  inrecv,    ///< TriangulateAggrBufferedInrecv (inbufastric.hpp): counts
             ///< each triangle once (precedence by vertex id), the variant
             ///< TriC's README recommends
};

struct Config {
  Algorithm algorithm = Algorithm::katric;
  std::string kagen_option_string;
  std::size_t iterations = 1;
  bool verify = false;
  CostFunction primary_cost_function = CostFunction::N;
  /// The primary load balancing redistributes only if the cost's max/avg over
  /// the ranks exceeds this (`--rebalance-threshold`; 1 = whenever it is not
  /// perfectly balanced)
  double rebalance_threshold = 1.0;
  /// CETRIC instead of DiTriC: the ghosts' out-edges in the local phase, only
  /// the cut edges in the global phase
  bool contraction = false;
  Threshold threshold = Threshold::local_nodes;
  double threshold_scale = 1.0;
  CommunicationPolicy communication_policy = CommunicationPolicy::direct;
  bool local_parallel = false;
  bool global_parallel = false;
  bool parallel_compact = false;
  TaskPoolType task_pool_type = TaskPoolType::std_thread;
  bool task_priorities = false;
  bool edge_partitioning = false;
  std::size_t local_degree_of_parallelism = 1;
  std::size_t global_degree_of_parallelism = 1;
  IntersectionMethod intersection_method = IntersectionMethod::merge;
  std::size_t binary_intersection_cutoff = 1000;
  double hybrid_cutoff_scale = 1.0;

  GhostExchangeMode degree_exchange = GhostExchangeMode::sparse;

  // Competitors (only used with their --algorithm)
  TricVariant tric_variant = TricVariant::plain;
  /// Per-rank buffer size of the buffered TriC variants (TriC's `-s`; its
  /// DEFAULT_BUF_SIZE is 8000)
  std::size_t tric_buffer_size = 8000;
  /// Buffered variants: the neighbour id ranges only of the local vertices and
  /// ghosts, fetched from their owners, instead of TriC's global array of all
  /// 2 n ranges on every rank (katric fork, vertex_ranges.hpp). Same pruning.
  bool tric_sparse_ranges = false;
  /// HavoqGT: vertices with more edges are delegates (split over all ranks);
  /// the default of the fork's old driver
  std::uint64_t havoqgt_delegate_threshold = 1048576;
  /// HavoqGT: edge partitioning passes within a node (capped at ranks/node)
  std::uint64_t havoqgt_partition_passes = 1;
  /// HavoqGT: edges per chunk during graph construction
  std::uint64_t havoqgt_chunk_size = 8192;
  /// HavoqGT: compute and print its diagnostics (allreduces, degree
  /// histograms) between orientation and count, as its own driver does
  bool havoqgt_stats = false;
  /// xxtric's per-rank buffer size (its `-s`); -1, its default, means twice
  /// the number of ghost entries, i.e. no cap
  std::int64_t xxtric_buffer_size = -1;
  /// LCC: bytes per rank for CLaMPI's RMA caches (LCC hard-coded 8 GiB)
  std::uint64_t lcc_cache_size = 8589934592ULL;
  /// LCC's -H 1: remove degree-1 vertices first (off by default, as in LCC)
  bool lcc_degree_reduction = false;

  bool pseudo2core = false;
  /// Global phase: each rank gets only the part of N+(v) from its first
  /// receiver on (in orientation order), and no message if nothing follows it.
  bool message_trimming = false;

  // Caliper ConfigManager configuration (e.g. "spot"); empty disables Caliper.
  // Output file via caliper_output.
  std::string caliper_config = "";
  std::string caliper_output = "";
  std::size_t num_threads = 1;

  std::string git_commit;

  // name and parsed value of every command line option, recorded as run
  // metadata
  std::vector<std::pair<std::string, std::string>> options;
};

/// Command line spelling of an enum value: its name with `_` written as `-`.
template <typename Enum>
  requires std::is_enum_v<Enum>
std::string cli_name(Enum value) {
  std::string name(magic_enum::enum_name(value));
  std::replace(name.begin(), name.end(), '_', '-');
  return name;
}

}  // namespace katric
