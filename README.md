# KaTric: Scalable Distributed-Memory Triangle Counting
[![GitHub](https://img.shields.io/github/license/niklas-uhl/katric)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7665533.svg)](https://doi.org/10.5281/zenodo.7665533)

![katric logo](./doc/katric.svg)

This is the code to accompany our paper:
_Sanders, P. and Uhl, T.N., 2023. Engineering a Distributed-Memory Triangle Counting Algorithm._ published in the proceedings of the 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS).

If you use this code in the context of an academic publication, we kindly ask you to [cite it](https://doi.org/10.1109/IPDPS54959.2023.00076):

```bibtex
@inproceedings{sanders2023
  author       = {Peter Sanders and
                  Tim Niklas Uhl},
  title        = {Engineering a Distributed-Memory Triangle Counting Algorithm},
  booktitle    = {{IEEE} International Parallel and Distributed Processing Symposium,
                  {IPDPS} 2023, St. Petersburg, FL, USA, May 15-19, 2023},
  pages        = {702--712},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/IPDPS54959.2023.00076},
  doi          = {10.1109/IPDPS54959.2023.00076},
}
```

You can also find a [freely accessible postprint in the arXiv](https://arxiv.org/abs/2302.11443).

## Introduction
Counting triangles in a graph and incident to each vertex is a
fundamental and frequently considered task of graph analysis.  We
consider how to efficiently do this for huge graphs using massively
parallel distributed-memory machines. Unsurprisingly, the main
issue is to reduce communication between processors. We achieve this
by counting locally whenever possible and reducing the amount of
information that needs to be sent in order to handle (possible)
nonlocal triangles.  We also achieve linear memory requirements
despite superlinear communication volume by introducing a new
asynchronous sparse-all-to-all operation. Furthermore, we
dramatically reduce startup overheads by allowing this communication
to use indirect routing.  Our algorithms scale (at least) up to 32 768 cores 
and are up to 18 times faster than the previous
state of the art.

## Building

### Requirements
To compile this project you need:
- A C++20 compiler (tested with `g++` 16).
- [OpenMPI](https://www.open-mpi.org/) or [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.pr0oht)
- oneTBB: from the system (load the module on a cluster); `-DCETRIC_FETCH_TBB=ON` builds the pinned version from source
- CMake 3.25 or higher and network access during configuration: all other dependencies (Abseil, fmt, Caliper, KaGen, tlx, ...) are fetched via `FetchContent`, pinned to a release or commit.

### Compiling

The project ships two CMake presets, both building into `build/`:

```shell
cmake --preset experiments      # Release, -march=native
cmake --build --preset experiments
# binary: build/apps/katric

cmake --workflow --preset dev   # Debug, warnings, assertions; configure, build and run the tests
```

For machine-specific settings (compiler, local checkouts of dependencies), copy `CMakeUserPresets.json.template` to
`CMakeUserPresets.json` and adapt it; that file is not tracked.

## Running
For reproducing our experiments with kaval, see [experiments/README.md](./experiments/README.md).

We provide two base configurations of our algorithm:
- DiTriC, which uses the asynchronous buffered messaging approach from our publication 
- CeTriC, which employs our communication reduction technique

While we provide many command-line parameters to tune our algorithms' behavior, we provide the variants from the publication as config presets:
```shell
# running DiTriC
mpiexec -np <NUM_PES> build/apps/katric --kagen_option_string <KAGEN_OPTIONS> --config configs/ditric.toml

# running CeTriC
mpiexec -np <NUM_PES> build/apps/katric --kagen_option_string <KAGEN_OPTIONS> --config configs/cetric.toml
```

If you want to enable the grid-based 2D indirect routing of messages, load the additional config
`--config configs/2d-grid.toml` after loading one of the base configs.

To use the hybrid implementation add `--config configs/hybrid.toml` and control the number of threads used per MPI rank via the `--num-threads` parameter.

By default, this only logs the running time and the number of triangles of each iteration.

### Profiling
Phase times and counters (wedge checks, intersection sizes, messages and volume, ...) are measured with
[Caliper](https://github.com/LLNL/Caliper), aggregated across ranks (min/max/avg or sum), with the configuration and
system information recorded via [Adiak](https://github.com/LLNL/Adiak):

```shell
mpiexec -np <NUM_PES> build/apps/katric --kagen_option_string <KAGEN_OPTIONS> --config configs/cetric.toml --caliper-output run
# run.cali:          spot profile (region times, run metadata) for Thicket; read by eval/katric_eval.py
# run.timers.json:   per region (e.g. "0/total/local_phase" in iteration 0) min/max/avg time in s and calls
# run.counters.json: per region and counter min/max/sum across ranks
# the JSON files contain the run metadata ("globals"); scripts/caliper_to_json.py merges them into one document
```

The JSON files are for looking at a single run; evaluations load the `.cali` profiles with
[Thicket](https://github.com/LLNL/thicket) (see [`eval/`](./eval)). `--caliper-config` selects other Caliper
configs instead, e.g. `katric-timers,katric-counters` (JSON only) or `runtime-report` (prints a region tree). Caliper's `CALI_CONFIG`
environment variable works independently of these options.

For additional parameters see `--help`.

### Providing input graphs
Input graphs are read or generated by [KaGen](https://github.com/KarlsruheGraphGeneration/KaGen), selected with
`--kagen_option_string`:

```shell
# read a METIS file, distributed so that every rank gets about the same number of edges
mpiexec -np 4 build/apps/katric --kagen_option_string "file;filename=examples/com-amazon.metis;input_format=metis;distribution=balance-edges" --config configs/cetric.toml
# generate a random geometric graph
mpiexec -np 4 build/apps/katric --kagen_option_string "rgg2d;n=16384;m=131072;seed=7" --config configs/cetric.toml
```

KaGen reads METIS, ParHIP (binary, much faster for large graphs) and edge lists, among others; `distribution` is
`balance-vertices` (default), `balance-edges` or `explicit` (vertex ranges per rank from a file). The input is
generated once per run; every iteration starts from an unmodified copy. Toy graphs are in `./examples`. See the KaGen
documentation for generators and options.

------------------------------
Licensed under [MIT](./LICENSE).
