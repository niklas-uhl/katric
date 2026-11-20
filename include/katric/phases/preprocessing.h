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
/// global id) (region `orientation`), then the neighborhoods sorted by handle
/// (`sorting`), in place. With `--contraction`, also the ghosts' out-edges to
/// local vertices, so that the local phase finds every triangle with two local
/// vertices.
inline void preprocessing(graph::Graph& G, Config const& conf) {
  CALI_CXX_MARK_SCOPE("preprocessing");
  {
    CALI_CXX_MARK_SCOPE("degree_exchange");
    G.exchange_ghost_degrees(conf.degree_exchange, MessageTag::Orientation);
  }
  {
    CALI_CXX_MARK_SCOPE("orientation");
    G.orient(degree_key(G), conf.contraction);
  }
  {
    CALI_CXX_MARK_SCOPE("sorting");
    G.sort_neighborhoods();
  }
}

}  // namespace katric
