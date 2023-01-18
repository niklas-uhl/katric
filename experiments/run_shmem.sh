#!/bin/bash

export LD_LIBRARY_PATH=/software/gcc/12.1.0/lib64

outputDir=./shmem-results-132-new
allThreads=(
1
2
4
8
16
32
64
)

declare -A configs base_config="--id-node-ordering --intersection-method=merge"
configs["edge-partitioning"]="$base_config --edge-partitioning"
configs["edge-partitioning-static"]="$base_config --edge-partitioning --edge-partitioning-static"
configs["2d"]="$base_config --local-degree-of-parallelism=3"
configs["2d-hybrid"]="$base_config --local-degree-of-parallelism=2"
configs["1d"]="$base_config --local-degree-of-parallelism=1"
configs["omp-2d"]="$base_config --local-degree-of-parallelism=3 --parallelization-method=omp_for"

input_file=$1
echo Using input $input_file

if [ -z "$2" ]; then
    keys=${!configs[@]}
    keys=( $keys )
else
    keys=( $@ )
    keys=("${keys[@]:1}")
fi
echo Using configs "${keys[@]}"
mkdir -p $outputDir
for key in ${keys[@]}; do
    mkdir -p $outputDir/$key
    for threads in ${allThreads[@]}; do
        inputname=$(basename $input_file)
        outfile=$outputDir/$key/$inputname.$key.$threads.json
	input_type=${input_file##*.}
	if [ $input_type == $input_file ]; then
		input_type="binary"
	else
		input_type="metis"
	fi
        ../build/apps/shmetric $input_file --input-format=$input_type --num-threads=$threads --iterations=3 ${configs[${key}]} --json-output $outfile
    done
done
