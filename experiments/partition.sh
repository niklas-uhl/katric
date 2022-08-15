#!/bin/sh

kampinpar_binary=~/Code/KaMinPar/build/apps/KaMinPar
threads=$(nproc --all)

function kaminpar() {
    graph=$1
    partitions=$2
    graph_name=$(basename $graph)
    partition_name="${graph_name%.metis}_k${partitions}"
    cmd="$kampinpar_binary -G $graph -k $partitions --save-partition --partition-name=$partition_name --threads=$threads --degree-weight"
    eval $cmd
    
}

mpi_ranks=(
2
4
8
16
32
64
128
256
512
1024
2048
4096
8192
16384
)

graphs=(
~/Data/graphs/com-amazon.metis
)

for graph in ${graphs[@]}; do
    for k in ${mpi_ranks[@]}; do
        kaminpar $graph $k
    done
done

