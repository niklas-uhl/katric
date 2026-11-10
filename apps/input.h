// Input graph generation and reading via KaGen.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <graph-io/local_graph_view.h>
#include <kagen.h>
#include <mpi.h>

#include "cetric/datastructures/graph_definitions.h"

namespace cetric {

/// Generates or reads the input graph as described by a KaGen option string, e.g.
/// "rgg2d;n=16384;m=131072;seed=7" or "file;filename=graph.metis;input_format=metis;distribution=balance_edges".
inline kagen::Graph generate_input(std::string const& kagen_option_string, MPI_Comm comm) {
    kagen::KaGen gen(comm);
    gen.UseCSRRepresentation();
    return gen.GenerateFromOptionString(kagen_option_string);
}

/// The local part of a KaGen graph as the view katric's DistributedGraph is built from.
struct LocalInput {
    graphio::LocalGraphView                 G;
    std::pair<graph::NodeId, graph::NodeId> node_range; // [first, last) global ids of the local vertices
};

inline LocalInput to_local_input(kagen::Graph&& graph) {
    auto [first, last] = graph.vertex_range;
    auto xadj          = graph.TakeXadj<>();
    auto adjncy        = graph.TakeAdjncy<>();
    graph.Clear();

    LocalInput input;
    input.node_range = {static_cast<graph::NodeId>(first), static_cast<graph::NodeId>(last)};
    input.G.node_info.reserve(last - first);
    for (kagen::SInt v = 0; v < last - first; ++v) {
        input.G.node_info.emplace_back(
            static_cast<graph::NodeId>(first + v),
            static_cast<graph::Degree>(xadj[v + 1] - xadj[v])
        );
    }
    input.G.edge_heads.assign(adjncy.begin(), adjncy.end());
    return input;
}

} // namespace cetric
