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
/// `--primary-cost-function` (PATRIC's cost functions N, DPD and IDPD; `none`
/// turns the load balancing off). See vertex_costs()
/// (katric/phases/load_balancing.h).
enum class CostFunction { none, N, DPD, IDPD };
/// What the primary load balancing checks before computing the cost,
/// `--rebalance-precheck`. See primary_load_balancing().
enum class RebalancePrecheck {
  degree_bound,  ///< first the cost with each ghost's out-degree bounded by its
                 ///< degree (IDD for IDPD), which needs only the degree
                 ///< exchange; keep the graph if its max/avg does not exceed
                 ///< `--rebalance-threshold`
  none,          ///< compute the cost (with the ghosts' out-degrees) always
};
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

/// The message queue of the global phase, `--message-queue`
/// (doc/briefkasten-migration.md).
enum class MessageQueue {
  legacy,       ///< the old message-queue: one global threshold (`--threshold`),
                ///< flushes every buffer on overflow; grid through katric's
                ///< own adaptor. Kept for reproducing earlier results.
  briefkasten,  ///< BriefKAsten: a capacity per destination buffer
                ///< (`--buffer-size`), bounded outstanding sends, credit-based
                ///< flow control, its own grid indirection
};
/// BriefKAsten's receive strategy, `--receiver`.
enum class Receiver {
  persistent,  ///< persistent receive requests (BriefKAsten's default)
  probe,       ///< MPI_Improbe, then a receive of the probed size
  posted,      ///< preposted receives
};

struct Config {
  std::string kagen_option_string;
  std::size_t iterations = 1;
  bool verify = false;
  CostFunction primary_cost_function = CostFunction::IDPD;
  /// The primary load balancing redistributes only if the cost's max over the
  /// ranks exceeds this times the achievable, max(avg, the largest vertex cost)
  /// (`--rebalance-threshold`; 1 = whenever it can be improved at all)
  double rebalance_threshold = 1.5;
  RebalancePrecheck rebalance_precheck = RebalancePrecheck::degree_bound;
  /// CETRIC instead of DiTriC: the ghosts' out-edges in the local phase, only
  /// the cut edges in the global phase
  bool contraction = false;
  Threshold threshold = Threshold::local_nodes;
  double threshold_scale = 1.0;
  CommunicationPolicy communication_policy = CommunicationPolicy::direct;
  MessageQueue message_queue = MessageQueue::legacy;
  /// BriefKAsten: capacity of a destination's buffer in bytes (local
  /// threshold); a message larger than it is sent on its own
  std::size_t buffer_size = 4096;
  /// BriefKAsten: sends in flight at once
  std::size_t request_slots = 8;
  /// BriefKAsten: credit-based flow control (window per peer)
  bool flow_control = true;
  Receiver receiver = Receiver::persistent;
  /// BriefKAsten: poll only on every n-th sender vertex (1: on every one, as
  /// the legacy queue)
  std::size_t poll_skip = 1;
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

  bool pseudo2core = false;
  /// Global phase: each rank gets only the part of N+(v) from its first
  /// receiver on (in orientation order), and no message if nothing follows it.
  /// (`--message-trimming`, the default; `--no-message-trimming` turns it off)
  bool message_trimming = true;

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
