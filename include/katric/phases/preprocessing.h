// Preprocessing: the degree exchange, the orientation by (degree, global id)
// and the sorting of the neighborhoods. Each phase is a Caliper region; there
// are no barriers between the phases, so a phase's time on a rank includes
// waiting for other ranks inside the next phase's first collective.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <caliper/cali.h>
#include <kamping/v2/comm.hpp>
#include <kassert/kassert.hpp>

#include "katric/caliper.h"
#include "katric/config.h"
#include "katric/ghost_exchange.h"
#include "katric/graph/graph.h"
#include "katric/message_tags.h"

namespace katric {

/// The orientation key of DiTriC and CETRIC: (undirected degree, global id),
/// for local vertices and ghosts (after `G.exchange_ghost_degrees`).
inline auto degree_key(graph::Graph const& G) {
  return [&G](graph::VertexHandle vertex) {
    return std::pair(G.undirected_degree(vertex), G.global_id(vertex));
  };
}

/// Degree exchange (region `degree_exchange`) and orientation by (degree,
/// global id) (region `orientation`; taken from the primary load balancing's
/// cost function if it kept the graph), then the neighborhoods sorted by handle
/// (`sorting`), in place. With `--contraction`, also the ghosts' out-edges to
/// local vertices, so that the local phase finds every triangle with two local
/// vertices. Counters (region `preprocessing`) `local_nodes`, `out_edges`,
/// `cut_out_edges`, `max_out_degree` of the oriented graph.
inline void preprocessing(graph::Graph& G, Config const& conf) {
  CALI_CXX_MARK_SCOPE("preprocessing");
  {
    CALI_CXX_MARK_SCOPE("degree_exchange");
    G.exchange_ghost_degrees(conf.degree_exchange, MessageTag::Orientation);
  }
  {
    CALI_CXX_MARK_SCOPE("orientation");
    if (G.has_kept_orientation()) {  // decided by the cost function, same key
      G.orient_by_kept(conf.contraction);
    } else {
      G.orient(degree_key(G), conf.contraction);
    }
  }
  {
    CALI_CXX_MARK_SCOPE("sorting");
    G.sort_neighborhoods();
  }
  // the shape of the oriented graph per rank (after any redistribution; the
  // input's is in `ingest`): what the global phase sends
  std::size_t cut_out_edges = 0;
  std::size_t max_out_degree = 0;
  for (auto const v : G.local_vertices()) {
    cut_out_edges += G.ghost_heads(v).size();
    max_out_degree = std::max(max_out_degree, G.local_neighbors(v).size());
  }
  profiling::counter_add("local_nodes",
                         static_cast<std::int64_t>(G.num_local_vertices()));
  profiling::counter_add("out_edges",
                         static_cast<std::int64_t>(G.num_local_edges()));
  profiling::counter_add("cut_out_edges",
                         static_cast<std::int64_t>(cut_out_edges));
  profiling::counter_add("max_out_degree",
                         static_cast<std::int64_t>(max_out_degree));
}

}  // namespace katric
