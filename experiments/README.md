# Experiments

Experiments run through [kaval](https://github.com/niklas-uhl/kaval), which expands a suite (`*.suite.yaml`) into one
`katric` call per input, core count, threads-per-rank and configuration, and either runs them directly or writes
SLURM job files. Requires [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/).

```shell
cmake --preset experiments && cmake --build --preset experiments   # binary: build/apps/katric
just kaval                                  # clone kaval into ./kaval at the commit pinned in the justfile
just list                                   # suites in experiments/
just run-local                              # the smoke suite on the example graphs, 2 and 4 cores, with --verify
just run-supermuc-experiment paper-real-world --module-config <modules> --account <project>
just collect experiment_data/<suite>_<date>/output > results.jsonl
```

kaval passes `--caliper-output <run>` to every run, so each run writes `<run>.timers.json` and
`<run>.counters.json` into the suite's output directory. `just collect` (`scripts/caliper_to_json.py`) merges them
into one JSON document per run: the options, the triangle count, per-iteration timers and counters.

## Suites

| Suite | What |
| --- | --- |
| `paper-real-world` | Strong scaling of DiTriC (`patric`) and CETRIC (`cetric-x`), 512 to 32768 cores |
| `paper-weak-scaling` | Weak scaling on RGG 2D, RHG and GNM, 512 to 32768 cores |
| `paper-hybrid-real-world` | Hybrid CETRIC (MPI + TBB), 1 to 48 threads per rank |
| `local/smoke` | The paper configurations on small inputs, every count verified |

Inputs are KaGen option strings: generated graphs, or files described in `*.instances.yaml`
(`paper-real-world` imports `real-world-core`). The real-world graphs are ParHIP files under
`${WORK_pn72pu}/graphs`; set that variable, or override `root` with an import's `with:`. Their sources are listed in
the paper.

The old suites (including the TriC and HavoqGT competitor runs, which kaval cannot express yet) and the paper suites
in their old format are kept in [`legacy/`](./legacy) for reference; the runner they were written for is gone.
