"""usage: scripts/counters_compare.py <ref dir> <new dir> [ignored counter names...]
Compares triangle counts and counters (min/max/sum per name and region) of two counters_run.sh outputs."""
import json, pathlib, re, sys
ref, new = map(pathlib.Path, sys.argv[1:3])
ignore = set(sys.argv[3:])
def load(d):
    out = {}
    for f in sorted(d.glob("*.counters.json")):
        run = f.name.removesuffix(".counters.json")
        recs = {}
        for r in json.load(open(f))["records"]:
            name = r["counter.name"]
            if name.endswith("_ns") or name in ignore or ("grid" in run and name in GRID):
                continue
            recs[(name, r["path"])] = (r["min"], r["max"], r["sum"])
        tri = re.findall(r"(\d+) triangles", (d / f"{run}.log").read_text())
        out[run] = (tri[-1] if tri else None, recs)
    return out
GRID = {"sent_messages", "received_messages", "send_volume", "receive_volume"}
a, b = load(ref), load(new)
bad = 0
for run in sorted(set(a) | set(b)):
    if run not in a or run not in b:
        print("missing", run, "ref" if run not in a else "new"); bad += 1; continue
    (ta, ra), (tb, rb) = a[run], b[run]
    if ta != tb:
        print(f"{run}: triangles {ta} -> {tb}"); bad += 1
    for k in sorted(set(ra) | set(rb), key=str):
        if ra.get(k) != rb.get(k):
            print(f"{run}: {k[0]} @ {k[1]}: {ra.get(k)} -> {rb.get(k)}"); bad += 1
print(f"{len(a)} runs, {bad} differences")
