// Input graph generation and reading via KaGen.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <kagen.h>
#include <mpi.h>

#include "katric/datastructures/graph_definitions.h"

namespace katric {

/// Generates or reads the input graph as described by a KaGen option string, e.g.
/// "rgg2d;n=16384;m=131072;seed=7" or "file;filename=graph.metis;input_format=metis;distribution=balance-edges".
inline kagen::Graph generate_input(std::string const& kagen_option_string, MPI_Comm comm) {
    kagen::KaGen gen(comm);
    gen.UseCSRRepresentation();
    return gen.GenerateFromOptionString(kagen_option_string);
}

} // namespace katric
