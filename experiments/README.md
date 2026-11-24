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
explore them in the marimo notebook [`eval/explore.py`](../eval/explore.py) (`just notebook`): select one or more
experiment directories (or a directory holding several, e.g. `eval/data/supermuc`) to see time over cores and where the
time goes per phase, across machines; details (imbalance per phase, input shape) are in collapsed sections below. Put
further notebooks next to it (`just notebook <name>`). `uv sync --group eval` installs these.

## Suites

The variants: DiTriC (default) and CETRIC (`--contraction`). The suites below, except `baseline` and `baseline-rmat`, were written
before the rename of 2026-11-19 and still use `--algorithm patric` (DiTriC) / `cetric-x` (CETRIC), which stay accepted; `--algorithm
cetric` (pre-CETRIC) was removed and now fails, run it from the tag `pre-refactor`. `eval/katric_eval.py` gives every
run a `variant` column from either spelling.

Defaults changed 2026-11-23: `--primary-cost-function IDPD`, `--rebalance-threshold 1.5`, `--rebalance-precheck
degree-bound` and message trimming on (`no-message-trimming: true` turns it off, as kaval omits false flags). The
suites below pin what they relied on before; IDD was removed, so `rebalance` / `rebalance-rmat`'s IDD configurations
no longer parse.

| Suite | What |
| --- | --- |
| `baseline` | The current code before the next steps (2026-11-20): DiTriC / CETRIC x direct / grid x load balancing none / IDPD x message trimming off / on, on the `phases` inputs and GNM (run at 192 to 3072 cores). New option names |
| `baseline-rmat` | `baseline` on RMAT with IDPD only (up to 768 cores) |
| `rebalance` | `--rebalance-threshold` and the cost function IDD (e13a2de): DiTriC, direct, trimming; none / IDPD at 1 (as `baseline`) / IDPD and IDD at 1.5, on the `phases` inputs and GNM (run at 192 to 3072 cores, as `baseline`) |
| `rebalance-rmat` | IDPD vs IDD at threshold 1.5 on RMAT (up to 768 cores) |
| `briefkasten` | The global phase on BriefKAsten (`--message-queue briefkasten`, buffer 4 / 16 / 64 KiB, at 16 KiB also `--poll-skip 100`) vs the legacy queue, direct / grid: DiTriC, trimming, no load balancing, on permuted RGG, GNM and permuted RHG (192 to 3072 cores; doc/briefkasten-migration.md step 6) |
| `briefkasten-rmat` | `briefkasten` on RMAT with the default load balancing (large messages; up to 768 cores) |
| `straggler` | Per-rank times (Caliper recorder via environment variables) of the ghost exchanges on permuted RGG at 1536 cores, IDPD / none, after fadbd12; read with `eval/perrank.py`. Since 2026-11-23 also per-rank resource counters of the pattern build (`eval/perrank.py --counters`) |
| `rmat-volume` | Per-rank send volume and oriented-graph shape on RMAT with the defaults (IDPD at 1.5), up to 768 cores: where the volume imbalance after balancing comes from (Caliper recorder; `eval/perrank.py --counters`) |
| `rhg-seeds` | RHG (with and without locality) at 1536 / 3072 with the `phases` seed and two others, none vs IDPD at 1.5: is the 3072 hub an outlier, and does the achievable-balance threshold keep the graph there |
| `paper-real-world` | Strong scaling of DiTriC (`patric`) and CETRIC (`cetric-x`), 512 to 32768 cores |
| `paper-weak-scaling` | Weak scaling on RGG 2D, RHG and GNM, 512 to 32768 cores |
| `paper-hybrid-real-world` | Hybrid CETRIC (MPI + TBB), 1 to 48 threads per rank |
| `phases` | Phase breakdown of DiTriC and CETRIC (one thread per rank) on RGG 2D / RHG with and without vertex permutation (locality x skew), node multiples up to 3072 cores |
| `lb-cost` | DiTriC on RHG with primary load balancing `none` / `DPD` / `IDPD`, 48 to 768 cores (idea 9) |
| `lb-cost-rmat` | `lb-cost` on RMAT (the worst imbalance), 48 and 192 cores |
| `trimming` | DiTriC and CETRIC with and without message trimming on permuted RGG, RHG and GNM, 48 to 1536 cores |
| `degree-exchange` | DiTriC on permuted RGG, RHG, RMAT and GNM with `--degree-exchange dense` / `sparse` / `grid`, 48 to 3072 cores |
| `local/smoke` | The paper configurations on small inputs, every count verified |
| `local/phases` | `phases` on a laptop (1 to 8 cores, smaller graphs) |

Inputs are KaGen option strings: generated graphs, or files described in `*.instances.yaml`
(`paper-real-world` imports `real-world-core`). The real-world graphs are ParHIP files under
`${WORK_pn72pu}/graphs`; set that variable, or override `root` with an import's `with:`. Their sources are listed in
the paper.

The old suites (including the TriC and HavoqGT competitor runs, which kaval cannot express yet) and the paper suites
in their old format are kept in [`legacy/`](./legacy) for reference; the runner they were written for is gone.
