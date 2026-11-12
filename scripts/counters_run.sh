#!/bin/zsh
# usage: scripts/counters_run.sh <katric binary> <out dir>
# Refactoring check (doc/refactoring-plan.md): runs a fixed set of inputs x configurations x ranks and stores the
# counters JSON and the log (triangle count) of each run. Compare two outputs with scripts/counters_compare.py.
# zsh, because macOS's bash 3.2 has no associative arrays.
set -u
setopt sh_word_split
BIN=$1; OUT=$2; mkdir -p "$OUT"
R=${0:A:h:h}
typeset -A G
G=(
  rgg "rgg2d;n=32768;m=524288;seed=7"
  rggp "rgg2d;n=32768;m=524288;seed=7;permute=1"
  rhg "rhg;n=32768;m=524288;gamma=2.8;seed=7"
  rhgp "rhg;n=32768;m=524288;gamma=2.8;seed=7;permute=1"
  amazon "file;filename=$R/examples/com-amazon.metis;input_format=metis;distribution=balance-edges"
)
typeset -A C
C=(
  ditric "--config $R/configs/ditric.toml"
  ditric-grid "--config $R/configs/ditric.toml --config $R/configs/2d-grid.toml"
  cetric "--config $R/configs/cetric.toml"
  cetric-grid "--config $R/configs/cetric.toml --config $R/configs/2d-grid.toml"
  pre "--config $R/configs/cetric.toml --algorithm cetric"
  pre-grid "--config $R/configs/cetric.toml --config $R/configs/2d-grid.toml --algorithm cetric"
  ditric-dpd "--config $R/configs/ditric.toml --primary-cost-function DPD"
)
for g in ${(k)G}; do for c in ${(k)C}; do for p in 2 4; do
  name=$g.$c.$p
  mpiexec -n $p --oversubscribe "$BIN" --kagen_option_string "${G[$g]}" ${C[$c]} --iterations 1 \
    --caliper-config katric-counters --caliper-output "$OUT/$name" > "$OUT/$name.log" 2>&1 \
    || echo "FAILED $name"
done; done; done
