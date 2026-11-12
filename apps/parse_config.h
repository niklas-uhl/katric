#pragma once

#include "cetric/config.h"

/// Parses the command line and --config files. Exits (after MPI_Finalize) on --help or a parse error.
cetric::Config parse_config(int argc, char* argv[], cetric::PEID rank, cetric::PEID size);
