#!/usr/bin/env python3
"""Merge the JSON written by katric's `katric-timers` and `katric-counters` Caliper configs (run with
`--caliper-config katric-timers,katric-counters --caliper-output <prefix>`) into one JSON document per run:

  {"config": ..., "triangles": N, "wall_times": [per iteration: maximum time of the "total" region],
   "timer":    [per iteration {region: {min, max, avg, exclusive_max, calls}}],
   "counters": [per iteration {region: {name: {min, max, sum}}}]}

Times are in seconds, min/max/avg are across ranks. Counters recorded outside a phase (e.g. `wedges`) are listed
under the region "iteration"."""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="the --caliper-output prefix (reads <prefix>.timers.json and <prefix>.counters.json)")
    args = ap.parse_args()

    with open(f"{args.prefix}.timers.json") as f:
        timers = json.load(f)
    with open(f"{args.prefix}.counters.json") as f:
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

    json.dump(
        {
            "config": unflatten(globals_, "config"),
            "triangles": globals_.get("triangles"),
            "wall_times": [timer[i].get("total", {}).get("max") for i in sorted(timer)],
            "timer": [timer[i] for i in sorted(timer)],
            "counters": [counter.get(i, {}) for i in sorted(timer)],
        },
        sys.stdout,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
