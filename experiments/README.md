# Experiments

Experiments run through [kaval](https://github.com/niklas-uhl/kaval), which expands a suite (`*.suite.yaml`) into one
`katric` call per input, core count, threads-per-rank and configuration, and either runs them directly or writes
SLURM job files. Requires [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/).

```shell
cmake --preset experiments && cmake --build --preset experiments   # binary: build/apps/katric
just kaval                                  # clone kaval into ./kaval at the commit pinned in the justfile
just list                                   # suites in experiments/
just run-local                              # the smoke suite on the example graphs, 2 and 4 cores, with --verify
just run-supermuc-experiment paper-real-world   # module config `katric`, SLURM account from PROJECT (.env)
just fetch-from-supermuc                    # rsync experiment_data/ into eval/data/supermuc
just summary eval/data/supermuc/<suite>_<date>   # mean run time per input, configuration and core count
```

kaval passes `--caliper-output <run>` to every run, so each run writes `<run>.cali` (spot profile: region times and
the run metadata from Adiak), `<run>.timers.json` and `<run>.counters.json` into the suite's output directory. The
JSON files are for looking at single runs (`scripts/caliper_to_json.py <run>`). For evaluation,
[`eval/katric_eval.py`](../eval/katric_eval.py) loads experiment directories with
[Thicket](https://github.com/LLNL/thicket) into polars tables (runs with all options as columns, timers, counters);
explore them in marimo notebooks with altair plots: `just notebook` opens the example
[`eval/scaling.py`](../eval/scaling.py), which imports the module and plots time over cores; put further notebooks next
to it. `uv sync --group eval` installs these.

## Suites

| Suite | What |
| --- | --- |
| `paper-real-world` | Strong scaling of DiTriC (`patric`) and CETRIC (`cetric-x`), 512 to 32768 cores |
| `paper-weak-scaling` | Weak scaling on RGG 2D, RHG and GNM, 512 to 32768 cores |
| `paper-hybrid-real-world` | Hybrid CETRIC (MPI + TBB), 1 to 48 threads per rank |
| `phases` | Phase breakdown of DiTriC and CETRIC (one thread per rank) on RGG 2D / RHG with and without vertex permutation (locality x skew), 48 / 1536 / 3072 cores; `just notebook phases` |
| `local/smoke` | The paper configurations on small inputs, every count verified |
| `local/phases` | `phases` on a laptop (1 to 8 cores, smaller graphs) |

Inputs are KaGen option strings: generated graphs, or files described in `*.instances.yaml`
(`paper-real-world` imports `real-world-core`). The real-world graphs are ParHIP files under
`${WORK_pn72pu}/graphs`; set that variable, or override `root` with an import's `with:`. Their sources are listed in
the paper.

The old suites (including the TriC and HavoqGT competitor runs, which kaval cannot express yet) and the paper suites
in their old format are kept in [`legacy/`](./legacy) for reference; the runner they were written for is gone.
