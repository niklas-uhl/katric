# Experiments through kaval (https://github.com/niklas-uhl/kaval), checked out into ./kaval at a pinned commit.
# Build first with the `experiments` preset (binary at build/apps/katric). `just --list` shows the recipes.

set dotenv-load := true

export KATRIC_DIR := justfile_directory()

# kaval main (merge of #33, suites with unset env vars are skipped until used)
kaval_rev := "616cb4c1a4a6f8badc1f198539747bb76685480f"

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

# Generate (and submit) SLURM jobs for SuperMUC-NG (as KaCCv2; at least one node, the suite's max_cores applies)
run-supermuc-experiment experiment +args="": kaval
    {{ kaval }} --machine supermuc --module-config katric --command-template ./kaval/command-templates/supermuc-IntelMPI-OpenMP.txt --min-cores 48 {{ experiment }} {{ args }}

# Generate (and submit) SLURM jobs for HoreKa (as KaCCv2: power-of-two multiples of a node)
run-horeka-experiment experiment +args="": kaval
    {{ kaval }} --machine horeka --module-config katric --command-template ./kaval/command-templates/horeka-IntelMPI.txt --cores node-size-pow2 --min-cores 64 {{ experiment }} {{ args }}

# Generate (and submit) SLURM jobs for HoreKa2
run-horeka2-experiment experiment +args="": kaval
    {{ kaval }} --machine horeka2 --module-config katric --cores node-size-pow2 --min-cores 64 {{ experiment }} {{ args }}

# Copy experiment output from a cluster into eval/data/<cluster> (job binaries excluded)
fetch-from-supermuc:
    rsync -aP --exclude 'bin/' supermuc:cetric/experiment_data/ eval/data/supermuc

fetch-from-horeka:
    rsync -aP --exclude 'bin/' horeka:katric/experiment_data/ eval/data/horeka

fetch-from-horeka2:
    rsync -aP --exclude 'bin/' horeka2:cetric/experiment_data/ eval/data/horeka2

# Summarize experiment directories (kaval's <suite>_<date>): mean of the maximum "total" time per configuration
summary +dirs:
    uv run --group eval eval/katric_eval.py {{ dirs }}

# Open a marimo notebook from eval/ (default explore.py: scaling and phase breakdown of chosen experiment directories)
notebook name="explore":
    uv run --group eval marimo edit --watch eval/{{ name }}.py
