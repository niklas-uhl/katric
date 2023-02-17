# KaTric: Scalable Distributed-Memory Triangle Counting
![katric logo](./doc/katric.svg)

This is the code to accompany our paper:
_Sanders, P. and Uhl, T.N., 2023. Engineering a Distributed-Memory Triangle Counting Algorithm._

If you use this code in the context of an academic publication, please cite:
```bibtex
// TODO
```
## Introduction

## Building

### Requirements
To compile this project you need:
- A modern, C++17-ready compiler such as `g++` version 9 or higher or `clang` version 11 or higher.
- [https://www.open-mpi.org/](OpenMPI) or [https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.pr0oht](Intel MPI)
- OpenMP and oneTBB [https://oneapi-src.github.io/oneTBB/](oneTBB)
- [https://github.com/sparsehash/sparsehash](Google Sparsehash)

### Compiling

```shell
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```


## Running
For reproducing our experiments see [./experiments/README.md](the corresponding document).

We provide two base configurations of our algorithm:
- DiTriC, which uses the asynchronous buffered messaging approach from our publication 
- CeTriC, which employs our communication reduction technique

While we provide many command-line parameters to tune our algorithms' behavior, we provide the variants from the publication as config presets:
```shell
# running DiTriC
mpiexec -np <NUM_PES> build/apps/cetric <GRAPH> --config configs/ditric.conf

# running CeTriC
mpiexec -np <NUM_PES> build/apps/cetric <GRAPH> --config configs/cetric.conf
```

If you want to enable the grid-based 2D indirect routing of messages, load the additional config
`--config configs/2d-grid` after loading one of the base configs.

To use the hybrid implementation add `--config configs/hybrid` and control the number of threads used per MPI rank via the `--num-threads` parameter.

By default, this only prints the running time and the total number of triangles to stdout.
For detailed metrics of all algorithm phases and communication, use the `--json-ouput` flag with `stdout` or a outfile as parameter.

For additional parameters see `--help`.

### Providing input graphs
You can read undirected graphs from input files represented in `METIS` or `binary` format. The `binary` is a lot faster to read.
To convert other graph represenations to supported formats see [https://github.com/niklas-uhl/graph-converters](our graph converter suite).
Select the input format using `--input-format`.

You can also use random graphs generated via [https://github.com/sebalamm/kagen](KaGen).
Choose a generator with the `--gen` flag and set the desired parameters (`--gen_*`). See the KaGen documentation for details.

------------------------------
Licensed under [./LICENSE](MIT). 
