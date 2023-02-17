# Karlsruhe Triangle Counting

![katric logo](./katric.svg)

This repository contains our experimental implementation of CETRIC.

The code is based on the following thesis:

T. N. Uhl. Communication Efficient Triangle Counting. Karlsruhe Institute of Technology. 2021

# Building

```shell
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make
```

# Running
```shell
cd build

# running CETRIC
mpiexec -np NUM_PROCS apps/parallel-counter GRAPH --ghost --k3-intersection --compress_more

# running PATRIC
mpiexec -np NUM_PROCS apps/parallel-counter GRAPH 

# for more details run
mpiexec -np NUM_PROCS apps/parallel-counter --help
# or refer to the example scripts under experiments
```
