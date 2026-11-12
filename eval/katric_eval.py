"""Load katric experiment output into polars DataFrames.

Every run writes <run>.cali (spot profile with the region times and the Adiak run metadata) and <run>.counters.json
(counters per region, which spot cannot hold) into the output/ directory of a kaval experiment directory. Thicket
reads the profiles; this module flattens them into three tables, joined on `run` (<experiment directory>/<the run's
file name without extension>, so the same suite run on several machines can be loaded together):

  runs      one row per run: every CLI option (Adiak's `config.<option>`, under its option name), `triangles`,
            `ranks`, `threads`, `cores`, `graph` and the KaGen options split into `kagen.<key>` columns, `git_commit`,
            `launchdate`, `cluster` (Adiak's cluster name, e.g. hkn = HoreKa, ircs = SuperMUC-NG), and `dir` (the
            experiment directory)
  timers    one row per run, iteration and region: `iteration` (null outside the iterations, e.g. `input`),
            `region` (path below the iteration, e.g. "total/local_phase"), and spot's metrics in seconds:
            `min`, `max`, `avg` (across ranks), `total` (sum over ranks), `*_exc` (exclusive), `calls_max`
  counters  one row per run, iteration, region and counter: `counter`, `min`, `max`, `sum` (across ranks)

The run time of an iteration is the maximum across ranks of the region "total", so a scaling plot is

    r = load("eval/data/supermuc/paper-weak-scaling_26_11_12")
    t = r.timers.filter(region="total").join(r.runs, on="run")
    t.group_by("graph", "algorithm", "communication-policy", "cores").agg(pl.col("max").mean())

altair takes these DataFrames directly (see the marimo notebook eval/explore.py). `load_thicket` gives the Thicket
itself, e.g. for its tree view.
"""

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# spot's metric columns as hatchet names them
METRICS = {
    "Min time/rank": "min",
    "Max time/rank": "max",
    "Avg time/rank": "avg",
    "Total time": "total",
    "Min time/rank (exc)": "min_exc",
    "Max time/rank (exc)": "max_exc",
    "Avg time/rank (exc)": "avg_exc",
    "Total time (exc)": "total_exc",
    "Calls/rank (max)": "calls_max",
}

# Adiak values that are not CLI options but worth a column
METADATA = {"triangles": "triangles", "jobsize": "ranks", "git_commit": "git_commit", "launchdate": "launchdate",
            "cluster": "cluster"}


@dataclass
class Results:
    runs: pl.DataFrame
    timers: pl.DataFrame
    counters: pl.DataFrame


def profiles(dirs):
    """The non-empty .cali files of experiment directories (kaval's <suite>_<date>, or their output/ directory)."""
    files = []
    for d in map(Path, dirs):
        out = d / "output" if (d / "output").is_dir() else d
        found = sorted(f for f in out.glob("*.cali") if f.stat().st_size > 0)  # killed runs leave empty files
        if not found:
            warnings.warn(f"{d}: no .cali files")
        files += found
    return files


def load_thicket(dirs):
    import thicket

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # hatchet's pandas concat
        return thicket.Thicket.from_caliperreader([str(f) for f in profiles(dirs)], disable_tqdm=True)


def split_path(path):
    """'0/total/local_phase' -> (0, 'total/local_phase'); '0' -> (0, 'iteration'); 'input' -> (None, 'input')."""
    first, _, rest = path.partition("/")
    if first.isdigit():
        return int(first), rest or "iteration"
    return None, path


def kagen_options(option_string):
    """'type=rgg2d;n=1024;permute' -> {'type': 'rgg2d', 'n': '1024', 'permute': 'true'} (values stay strings)."""
    options = {}
    for part in filter(None, option_string.split(";")):
        key, sep, value = part.partition("=")
        options[key] = value if sep else "true"
    if "type" not in options and options:  # KaGen also takes the type as the first, bare element
        first = option_string.split(";")[0]
        if "=" not in first:
            options.pop(first)
            options["type"] = first
    return options


def graph_label(options, cores):
    """The file name for file inputs, else type(log2 n, log2 m) per core, as kaval's N and M (plus gamma for rhg),
    so the instances of a weak-scaling series get the same label."""
    if options.get("type") == "file":
        return Path(options.get("filename", "?")).name.split(".")[0]

    def per_core(key):
        if key not in options:
            return None
        x = int(options[key]) / cores
        e = math.log2(x)
        return str(int(e)) if e.is_integer() else f"{x:g}"

    args = [a for a in (per_core("n"), per_core("m")) if a is not None]
    if "gamma" in options:
        args.append(f"gamma={options['gamma']}")
    if options.get("permute") in ("1", "true"):
        args.append("permuted")
    return f"{options.get('type', '?')}({','.join(args)})"


def _runs(tk, run_of_profile, dir_of_profile):
    meta = tk.metadata.reset_index()
    columns = {"profile": meta["profile"]}
    for name in meta.columns:
        if name.startswith("config."):
            columns[name.removeprefix("config.")] = meta[name].astype(str)
        elif name in METADATA:
            columns[METADATA[name]] = meta[name]
    runs = pl.DataFrame(columns).with_columns(
        pl.col("profile").replace_strict(run_of_profile).alias("run"),
        pl.col("profile").replace_strict(dir_of_profile).alias("dir"),
        pl.col("num-threads").cast(pl.Int64).alias("threads"),
    )
    runs = runs.with_columns((pl.col("ranks") * pl.col("threads")).alias("cores"))
    for option in ("iterations", "num-threads"):
        runs = runs.with_columns(pl.col(option).cast(pl.Int64))
    kagen = [kagen_options(s) for s in runs["kagen_option_string"]]
    keys = sorted({k for options in kagen for k in options})
    runs = runs.with_columns(
        pl.Series("graph", [graph_label(o, c) for o, c in zip(kagen, runs["cores"])]),
        *[pl.Series(f"kagen.{k}", [o.get(k) for o in kagen]) for k in keys],
    )
    return runs.drop("profile")


def _timers(tk, run_of_profile):
    df = tk.dataframe.reset_index()
    df = df[df["Max time/rank"].notna()]  # hatchet adds rows for regions a run does not have
    paths = ["/".join(str(n.frame["name"]) for n in node.path()) for node in df["node"]]
    split = [split_path(p) for p in paths]
    timers = pl.DataFrame(
        {
            "run": [run_of_profile[p] for p in df["profile"]],
            "iteration": pl.Series([i for i, _ in split], dtype=pl.Int64),
            "region": [r for _, r in split],
            **{short: df[name].to_numpy() for name, short in METRICS.items() if name in df},
        }
    )
    return timers


def _experiment_dir(f):
    """The kaval experiment directory of a profile (output/ files live one level below it)."""
    f = Path(f)
    return f.parent.parent if f.parent.name == "output" else f.parent


def _run_id(f):
    return f"{_experiment_dir(f)}/{Path(f).stem}"


def _counters(files):
    rows = []
    for f in files:
        counters = f.with_suffix(".counters.json")
        if not counters.exists():
            continue
        with open(counters) as file:
            records = json.load(file)["records"]
        for r in records:
            iteration, region = split_path(r.get("path") or "")
            rows.append({"run": _run_id(f), "iteration": iteration, "region": region, "counter": r["counter.name"],
                         "min": r["min"], "max": r["max"], "sum": r["sum"]})
    schema = {"run": pl.String, "iteration": pl.Int64, "region": pl.String, "counter": pl.String,
              "min": pl.Int64, "max": pl.Int64, "sum": pl.Int64}
    return pl.DataFrame(rows, schema=schema)


def load(*dirs):
    """Runs, timers and counters of the given experiment directories (see the module docstring)."""
    files = profiles(dirs)
    if not files:
        raise ValueError(f"no profiles in {dirs}")
    tk = load_thicket(dirs)
    file_of_profile = {p: Path(f) for p, f in tk.profile_mapping.items()}
    run_of_profile = {p: _run_id(f) for p, f in file_of_profile.items()}
    dir_of_profile = {p: str(_experiment_dir(f)) for p, f in file_of_profile.items()}
    if len(set(run_of_profile.values())) != len(run_of_profile):
        raise ValueError("a profile was loaded twice; pass each experiment directory once")
    return Results(_runs(tk, run_of_profile, dir_of_profile), _timers(tk, run_of_profile), _counters(files))


def config_options(runs):
    """The CLI options that differ between the runs, i.e. what distinguishes the configurations (not the input or the
    number of ranks and threads)."""
    other = {"run", "dir", "graph", "launchdate", "caliper-output", "triangles", "ranks", "threads", "num-threads",
             "cores", "git_commit", "cluster", "kagen_option_string"}
    return [c for c in runs.columns if c not in other and not c.startswith("kagen.") and runs[c].n_unique() > 1]


def main():
    ap = argparse.ArgumentParser(description="Mean of the per-iteration run time (maximum of the region 'total' "
                                 "across ranks) per input, configuration and core count.")
    ap.add_argument("dirs", nargs="+", help="kaval experiment directories (<suite>_<date>)")
    args = ap.parse_args()
    r = load(*args.dirs)
    machine = ["cluster"] if r.runs["cluster"].n_unique() > 1 else []
    keys = [*machine, "graph", *config_options(r.runs), "ranks", "threads"]
    summary = (
        r.timers.filter(region="total")
        .join(r.runs, on="run")
        .group_by(keys)
        .agg(pl.col("max").mean().alias("time"), pl.col("max").min().alias("time_min"),
             pl.len().alias("iterations"), pl.col("triangles").first())
        .sort(keys)
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(summary)


if __name__ == "__main__":
    sys.exit(main())
