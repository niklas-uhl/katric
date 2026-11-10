# Experiments through kaval (https://github.com/niklas-uhl/kaval), checked out into ./kaval at a pinned commit.
# Build first with the `experiments` preset (binary at build/apps/katric). `just --list` shows the recipes.

set dotenv-load := true

export KATRIC_DIR := justfile_directory()

# kaval main (merge of #32, feat/horeka2)
kaval_rev := "9bc527c0f94a5bb3167a3b3a37735d4b1ba61e68"

kaval := "uv run kaval/run-experiments.py --search-dirs ./experiments"

# Clone kaval into ./kaval, or check out the pinned commit in an existing clone
kaval:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d kaval ]; then git clone https://github.com/niklas-uhl/kaval.git kaval; fi
    git -C kaval fetch --quiet origin
    git -C kaval -c advice.detachedHead=false checkout --quiet {{ kaval_rev }}

# List the experiment suites
list: kaval
    {{ kaval }} --list

# Run the smoke suite (small inputs, --verify) locally with mpiexec
run-local suite="smoke" +args="": kaval
    uv run kaval/run-experiments.py --search-dirs ./experiments/local --machine shared {{ suite }} {{ args }}

# Generate (and submit) SLURM jobs for SuperMUC-NG
run-supermuc-experiment experiment +args="": kaval
    {{ kaval }} --machine supermuc --command-template ./kaval/command-templates/supermuc-IntelMPI-OpenMP.txt {{ experiment }} {{ args }}

# Generate (and submit) SLURM jobs for HoreKa
run-horeka-experiment experiment +args="": kaval
    {{ kaval }} --machine horeka --command-template ./kaval/command-templates/horeka-IntelMPI.txt {{ experiment }} {{ args }}

# Copy experiment output from a cluster into eval/data/<cluster> (job binaries excluded)
fetch-from-supermuc:
    rsync -aP --exclude 'bin/' supermuc:katric/experiment_data/ eval/data/supermuc

fetch-from-horeka:
    rsync -aP --exclude 'bin/' horeka:katric/experiment_data/ eval/data/horeka

# Summarize experiment directories (kaval's <suite>_<date>): mean of the maximum "total" time per configuration
summary +dirs:
    uv run --group eval eval/katric_eval.py {{ dirs }}
