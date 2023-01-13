#!/bin/bash

allThreads=(
1
2
4
8
16
32
64
)

declare -A configs
base_config="--id-node-ordering --intersection-method=merge"
configs["edge-partitioning"]="$base_config --edge-partitioning"
configs["edge-partitioning-static"]="$base_config --edge-partitioning --edge-partitioning-static"
configs["2d"]="$base_config --local-degree-of-parallelism=3"
configs["2d-hybrid"]="$base_config --local-degree-of-parallelism=2"
configs["1d"]="$base_config --local-degree-of-parallelism=1"

input_file=$1

if [ -z "$2" ]; then
    keys=${!configs[@]}
else
    keys=$2
fi
mkdir -p ./shmem-results
for key in $keys; do
    mkdir -p ./shmem-results/$key
    for threads in ${allThreads[@]}; do
        inputname=$(basename $input_file)
        outfile=./shmem-results/$key/$inputname.$key.$threads.json
        echo ../build/apps/shmetric $input_file --input-format=binary --num-threads=$threads ${configs[${key}]} --json-output $outfile
    done
done
