#pragma once

#include "katric/config.h"

/// Parses the command line and --config files. Exits (after MPI_Finalize) on --help or a parse error.
katric::Config parse_config(int argc, char* argv[], int rank, int size);
