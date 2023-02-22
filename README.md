# KaTric: Scalable Distributed-Memory Triangle Counting
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7665533.svg)](https://doi.org/10.5281/zenodo.7665533)

![katric logo](./doc/katric.svg)

This is the code to accompany our (soon to be published) paper:
_Sanders, P. and Uhl, T.N., 2023. Engineering a Distributed-Memory Triangle Counting Algorithm._

If you use this code in the context of an academic publication, please cite:
```bibtex
// TODO
```
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
- A modern, C++17-ready compiler such as `g++` version 9 or higher or `clang` version 11 or higher.
- [OpenMPI](https://www.open-mpi.org/) or [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.pr0oht)
- OpenMP and [oneTBB](https://oneapi-src.github.io/oneTBB/)
- [Google Sparsehash](https://github.com/sparsehash/sparsehash)
- Boost

### Compiling

```shell
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```


## Running
For reproducing our experiments see [the corresponding document](./experiments/README.md).

We provide two base configurations of our algorithm:
- DiTriC, which uses the asynchronous buffered messaging approach from our publication 
- CeTriC, which employs our communication reduction technique

While we provide many command-line parameters to tune our algorithms' behavior, we provide the variants from the publication as config presets:
```shell
# running DiTriC
mpiexec -np <NUM_PES> build/apps/katric <GRAPH> --config configs/ditric.toml

# running CeTriC
mpiexec -np <NUM_PES> build/apps/katric <GRAPH> --config configs/cetric.toml
```

If you want to enable the grid-based 2D indirect routing of messages, load the additional config
`--config configs/2d-grid.toml` after loading one of the base configs.

To use the hybrid implementation add `--config configs/hybrid.toml` and control the number of threads used per MPI rank via the `--num-threads` parameter.

By default, this only prints the running time and the total number of triangles to stdout.
For detailed metrics of all algorithm phases and communication, use the `--json-ouput` flag with `stdout` or a outfile as parameter.

For additional parameters see `--help`.

### Providing input graphs
You can read undirected graphs from input files represented in `METIS` or `binary` format. The `binary` is a lot faster to read. You can find some toy graphs in `./examples`.
To convert other graph represenations to supported formats see [our graph converter suite](https://github.com/niklas-uhl/graph-converter).

Select the input format using `--input-format`.

You can also use random graphs generated via [KaGen](https://github.com/sebalamm/kagen).
Choose a generator with the `--gen` flag and set the desired parameters (`--gen_*`). See the KaGen documentation for details.

------------------------------
Licensed under [MIT](./LICENSE).
