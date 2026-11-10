#!/usr/bin/env python3
"""Merge the JSON written by katric's `katric-timers` and `katric-counters` Caliper configs (run with
`--caliper-config katric-timers,katric-counters --caliper-output <prefix>`) into one JSON document per run:

  {"config": ..., "triangles": N, "wall_times": [per iteration: maximum time of the "total" region],
   "timer":    [per iteration {region: {min, max, avg, exclusive_max, calls}}],
   "counters": [per iteration {region: {name: {min, max, sum}}}]}

Times are in seconds, min/max/avg are across ranks. Counters recorded outside a phase (e.g. `wedges`) are listed
under the region "iteration".

Given a directory (e.g. the output directory of a kaval suite), it prints one such document per run and line, with
the run's name (the prefix without directory) as "run"."""
import argparse
import json
import os
import sys


def unflatten(flat, prefix):
    """{'config.a.b': 1, 'config.l.0': 2} -> {'a': {'b': 1}, 'l': [2]} (numeric keys become lists)."""
    root = {}
    for key, value in flat.items():
        if not key.startswith(prefix + "."):
            continue
        node = root
        parts = key[len(prefix) + 1 :].split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def listify(node):
        if not isinstance(node, dict):
            return node
        node = {k: listify(v) for k, v in node.items()}
        if node and all(k.isdigit() for k in node):
            return [node[k] for k in sorted(node, key=int)]
        return node

    return listify(root)


def split_path(path):
    """'0/local_phase/x' -> (0, 'local_phase/x'); '0' -> (0, 'iteration'); no iteration -> None."""
    iteration, _, region = (path or "").partition("/")
    if not iteration.isdigit():
        return None
    return int(iteration), region or "iteration"


def load_run(prefix):
    """The merged document of the run written to `prefix` (see the module docstring)."""
    with open(f"{prefix}.timers.json") as f:
        timers = json.load(f)
    with open(f"{prefix}.counters.json") as f:
        counters = json.load(f)
    globals_ = timers["globals"]

    timer, counter = {}, {}
    for row in timers["records"]:
        key = split_path(row.get("path"))
        if key is None:
            continue  # time outside the iterations
        iteration, region = key
        timer.setdefault(iteration, {})[region] = {
            k: row.get(k) for k in ("min", "max", "avg", "exclusive_max", "calls")
        }
    for row in counters["records"]:
        key = split_path(row.get("path"))
        if key is None:
            continue
        iteration, region = key
        counter.setdefault(iteration, {}).setdefault(region, {})[row["counter.name"]] = {
            k: row[k] for k in ("min", "max", "sum")
        }

    return {
        "run": os.path.basename(prefix),
        "config": unflatten(globals_, "config"),
        "triangles": globals_.get("triangles"),
        "wall_times": [timer[i].get("total", {}).get("max") for i in sorted(timer)],
        "timer": [timer[i] for i in sorted(timer)],
        "counters": [counter.get(i, {}) for i in sorted(timer)],
    }


def prefixes_in(directory):
    """The --caliper-output prefixes of all runs in `directory` (e.g. a kaval output directory), sorted."""
    suffix = ".timers.json"
    return sorted(
        os.path.join(directory, name[: -len(suffix)]) for name in os.listdir(directory) if name.endswith(suffix)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "path",
        help="the --caliper-output prefix of one run (reads <prefix>.timers.json and <prefix>.counters.json), "
        "or a directory: then every run in it, one JSON document per line",
    )
    args = ap.parse_args()

    if os.path.isdir(args.path):
        for prefix in prefixes_in(args.path):
            try:
                print(json.dumps(load_run(prefix)))
            except (OSError, ValueError, KeyError) as e:  # e.g. a run that was killed before writing its output
                print(f"skipping {prefix}: {e!r}", file=sys.stderr)
    else:
        json.dump(load_run(args.path), sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
