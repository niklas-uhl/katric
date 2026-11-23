#pragma once

#include "katric/config.h"
#include <span>
#include <kamping/v2/comm.hpp>

/// Parses the command line and --config files. Exits (after MPI_Finalize) on --help or a parse error.
auto parse_config(std::span<char*> argv, kamping::v2::comm_view comm) -> katric::Config;
