"""Per-rank region times from Caliper's recorder (one rank-<rank>.cali per rank, see experiments/straggler.suite.yaml).

    uv run --group eval eval/perrank.py <dir with rank-*.cali> [--ranks-per-node 48] [--top 10] [region ...]

Regions are paths below the iteration (default: the ghost exchanges' `prepare` and `exchange`, and `total`). Prints,
per region, min / avg / max across ranks and the slowest ranks with their node (rank / ranks-per-node), per iteration
and summed over the iterations. Times are inclusive (the recorder's exclusive `sum#time.duration.ns`, summed over the
region's subtree).

With --counters, also the counters reported directly in each region (e.g. the resource counters of `pattern`: page
faults, cpu, numa_node, anon_huge_pages_kb, cycles, instructions, dtlb_load_misses; needs CALI_AGGREGATE_KEY with
counter.name), as the mean per iteration: the slowest ranks' values (by the region's time) against the median over
all ranks.
"""

import argparse
import collections
import re
import statistics
from pathlib import Path

import caliperreader

PLB = "total/primary_load_balancing/cost_function_communication"
DX = "total/preprocessing/degree_exchange"
DEFAULT_REGIONS = ["total", f"{PLB}/prepare", f"{PLB}/exchange", f"{DX}/prepare", f"{DX}/exchange"]


def read(directory, regions):
    """({rank: {(iteration, region): seconds}}, {rank: {(region, counter): [value per iteration]}}) for the
    iterations (paths starting with the iteration number)."""
    times, counters = {}, {}
    for f in Path(directory).glob("rank-*.cali"):
        rank = int(re.search(r"rank-(\d+)", f.name).group(1))
        records, _ = caliperreader.read_caliper_contents(str(f))
        t = collections.defaultdict(float)
        c = collections.defaultdict(list)
        for r in records:
            path = r.get("path")
            if path is None:
                continue
            path = [path] if isinstance(path, str) else path
            if not path[0].isdigit():
                continue
            below = "/".join(path[1:])
            name, value = r.get("counter.name"), r.get("sum#counter.value")
            if name is not None and value is not None and below in regions:
                c[(below, name)].append(float(value))
            ns = r.get("sum#time.duration.ns")  # counter records too carry time (since the previous snapshot)
            if ns is None:
                continue
            for region in regions:
                if below == region or below.startswith(region + "/"):
                    t[(int(path[0]), region)] += float(ns) * 1e-9
        times[rank] = t
        counters[rank] = c
    return times, counters


def print_counters(region, times, counters, iterations, top, ranks_per_node):
    """The counters of `region` on its slowest ranks (by the region's time summed over the iterations), as the mean
    per iteration, against the median over all ranks."""
    names = sorted({n for c in counters.values() for r, n in c if r == region})
    if not names:
        return
    total = {rank: sum(t[(i, region)] for i in iterations) for rank, t in times.items()}
    slowest = sorted(total, key=total.get, reverse=True)[:top]
    def mean(rank, name):
        values = counters[rank].get((region, name), [])
        return sum(values) / len(values) if values else 0

    print("  counters (mean per iteration; median over all ranks, then the slowest ranks):")
    print(f"    {'':>20} {'median':>14} " + " ".join(f"{f'r{r}/n{r // ranks_per_node}':>14}" for r in slowest))
    for name in names:
        median = statistics.median(mean(rank, name) for rank in counters)
        row = " ".join(f"{mean(r, name):>14.4g}" for r in slowest)
        print(f"    {name:>20} {median:>14.4g} {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("regions", nargs="*", default=DEFAULT_REGIONS)
    ap.add_argument("--ranks-per-node", type=int, default=48)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--counters", action="store_true", help="also the counters of each region (see above)")
    args = ap.parse_args()
    if args.counters and args.regions == DEFAULT_REGIONS:
        args.regions = [*DEFAULT_REGIONS, f"{PLB}/prepare/pattern", f"{DX}/prepare/pattern"]
    times, counters = read(args.dir, args.regions)
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
        if args.counters:
            print_counters(region, times, counters, iterations, args.top, args.ranks_per_node)


if __name__ == "__main__":
    main()
