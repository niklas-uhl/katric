// Input graph generation and reading via KaGen.
#pragma once

#include <kamping/v2/comm.hpp>
#include <string>

#include <kagen.h>
#include <mpi.h>

namespace katric {

/// Generates or reads the input graph as described by a KaGen option string,
/// e.g. "rgg2d;n=16384;m=131072;seed=7" or
/// "file;filename=graph.metis;input_format=metis;distribution=balance-edges".
inline auto generate_input(std::string const& kagen_option_string,
                           kamping::v2::comm_view comm) -> kagen::Graph {
  kagen::KaGen gen(comm.mpi_handle());
  gen.UseCSRRepresentation();
  return gen.GenerateFromOptionString(kagen_option_string);
}

}  // namespace katric
