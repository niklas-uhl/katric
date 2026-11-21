"""Per-rank region times from Caliper's recorder (one rank-<rank>.cali per rank, see experiments/straggler.suite.yaml).

    uv run --group eval eval/perrank.py <dir with rank-*.cali> [--ranks-per-node 48] [--top 10] [region ...]

Regions are paths below the iteration (default: the ghost exchanges' `prepare` and `exchange`, and `total`). Prints,
per region, min / avg / max across ranks and the slowest ranks with their node (rank / ranks-per-node), per iteration
and summed over the iterations. Times are inclusive (the recorder's exclusive `sum#time.duration.ns`, summed over the
region's subtree).
"""

import argparse
import collections
import re
from pathlib import Path

import caliperreader

PLB = "total/primary_load_balancing/cost_function_communication"
DX = "total/preprocessing/degree_exchange"
DEFAULT_REGIONS = ["total", f"{PLB}/prepare", f"{PLB}/exchange", f"{DX}/prepare", f"{DX}/exchange"]


def read(directory, regions):
    """{rank: {(iteration, region): seconds}} for the iterations (paths starting with the iteration number)."""
    times = {}
    for f in Path(directory).glob("rank-*.cali"):
        rank = int(re.search(r"rank-(\d+)", f.name).group(1))
        records, _ = caliperreader.read_caliper_contents(str(f))
        t = collections.defaultdict(float)
        for r in records:
            path, ns = r.get("path"), r.get("sum#time.duration.ns")
            if path is None or ns is None:
                continue
            path = [path] if isinstance(path, str) else path
            if not path[0].isdigit():
                continue
            below = "/".join(path[1:])
            for region in regions:
                if below == region or below.startswith(region + "/"):
                    t[(int(path[0]), region)] += float(ns) * 1e-9
        times[rank] = t
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("regions", nargs="*", default=DEFAULT_REGIONS)
    ap.add_argument("--ranks-per-node", type=int, default=48)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    times = read(args.dir, args.regions)
    iterations = sorted({i for t in times.values() for i, _ in t})
    print(f"{len(times)} ranks, iterations {iterations}")
    for region in args.regions:
        print(f"\n{region}")
        for it in [*iterations, None]:
            values = {rank: (sum(t[(i, region)] for i in iterations) if it is None else t[(it, region)])
                      for rank, t in times.items()}
            if not any(values.values()):
                continue
            avg = sum(values.values()) / len(values)
            slowest = sorted(values, key=values.get, reverse=True)[: args.top]
            top = ", ".join(f"r{r}/n{r // args.ranks_per_node} {values[r]:.3f}" for r in slowest)
            label = "sum" if it is None else f"it {it}"
            print(f"  {label:>5}: min {min(values.values()):.3f}  avg {avg:.3f}  max {max(values.values()):.3f}"
                  f"  max/avg {max(values.values()) / avg if avg else 0:.2f}  slowest: {top}")


if __name__ == "__main__":
    main()
