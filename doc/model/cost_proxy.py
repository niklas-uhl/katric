"""Cost functions for DiTriC's primary load balancing that need only the degrees (2026-11-21).

IDPD(u) = sum over in-neighbors v of d+(v) + d+(u) needs the ghosts' out-degrees: a second ghost exchange after the
degree exchange. Candidates from the degrees alone (d+(u) and indeg(u) follow from the degrees of u and its neighbors):
  IDD    d+(v) replaced by d(v) (an upper bound): indeg(u) d+(u) + sum over in-neighbors v of d(v)
  IDDc   d+(v) replaced by min(d(v), sqrt(2m)) (out-degrees are bounded by sqrt(2m))
  D2     d(u)^2, no neighbor degrees at all
Each partitions into contiguous ranges (as katric); measured: DiTriC's push work per rank (the receiver intersects),
send / receive volume (N+(v) once per rank of an out-neighbor), and the cost's max/avg on the equal-vertex input
partition (what a threshold would look at). Usage: python cost_proxy.py
"""
import math
import random

random.seed(7)


def rmat(scale, ef, a=.57, b=.19, c=.19):  # as model.py (which runs its experiment on import)
    n = 1 << scale; E = set()
    for _ in range(ef * n):
        u = v = 0
        for _ in range(scale):
            r = random.random(); u <<= 1; v <<= 1
            if r < a: pass
            elif r < a + b: v |= 1
            elif r < a + b + c: u |= 1
            else: u |= 1; v |= 1
        if u != v: E.add((min(u, v), max(u, v)))
    adj = [[] for _ in range(n)]
    for u, v in E: adj[u].append(v); adj[v].append(u)
    return adj


def ranges(weights, p):  # as model.py: contiguous ranges of about equal weight
    tot = sum(weights) or 1; acc = 0; own = [0] * len(weights)
    for v, w in enumerate(weights):
        own[v] = min(p - 1, acc * p // tot); acc += w
    return own


def permuted(adj):
    n = len(adj); perm = list(range(n)); random.shuffle(perm)
    out = [None] * n
    for v in range(n):
        out[perm[v]] = [perm[u] for u in adj[v]]
    return out


def rgg2d(n, avg_deg):
    r = math.sqrt(avg_deg / (math.pi * n)); cells = max(1, int(1 / r)); grid = {}
    pts = sorted((random.random(), random.random()) for _ in range(n))  # x-sorted ids: locality like KaGen
    for i, (x, y) in enumerate(pts):
        grid.setdefault((int(x * cells), int(y * cells)), []).append(i)
    adj = [[] for _ in range(n)]
    for (cx, cy), vs in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for u in vs:
                    for w in grid.get((cx + dx, cy + dy), ()):
                        if u < w and (pts[u][0] - pts[w][0]) ** 2 + (pts[u][1] - pts[w][1]) ** 2 < r * r:
                            adj[u].append(w); adj[w].append(u)
    return adj


def evaluate(adj, p, name):
    n = len(adj); deg = [len(a) for a in adj]; key = lambda v: (deg[v], v)
    out = [[u for u in adj[v] if key(u) > key(v)] for v in range(n)]
    inn = [[u for u in adj[v] if key(u) < key(v)] for v in range(n)]
    dp = [len(o) for o in out]; cap = math.isqrt(sum(deg))
    costs = {
        "none": [1] * n,
        "edges": deg,
        "IDPD": [sum(dp[v] + dp[u] for v in inn[u]) for u in range(n)],
        "IDD": [len(inn[u]) * dp[u] + sum(deg[v] for v in inn[u]) for u in range(n)],
        "IDDc": [len(inn[u]) * dp[u] + sum(min(deg[v], cap) for v in inn[u]) for u in range(n)],
        "D2": [d * d for d in deg],
    }
    equal = ranges([1] * n, p)

    def stats(own):
        work = [0] * p; send = [0] * p; recv = [0] * p
        for v in range(n):
            seen = set()
            for u in out[v]:
                work[own[u]] += dp[v] + dp[u]
                r = own[u]
                if r != own[v] and r not in seen:
                    seen.add(r); send[own[v]] += dp[v]; recv[r] += dp[v]
        ma = lambda x: max(x) / (sum(x) / p) if sum(x) else 0
        return ma(work), ma(send), ma(recv)

    print(f"{name} p={p}: n={n} m={sum(deg) // 2} max deg={max(deg)}")
    for label, c in costs.items():
        per_rank = [0] * p
        for v in range(n):
            per_rank[equal[v]] += c[v]
        gate = max(per_rank) / (sum(per_rank) / p)
        w, s, r = stats(ranges(c, p))
        print(f"   {label:6} work max/avg={w:5.2f}  send={s:5.2f}  recv={r:5.2f}   cost max/avg on input={gate:6.2f}")


if __name__ == "__main__":
    R = rmat(16, 16)
    graphs = {"rmat16": R, "rmat16 permuted": permuted(R), "rgg2d(2^16, 32)": rgg2d(1 << 16, 32)}
    graphs["rgg2d permuted"] = permuted(graphs["rgg2d(2^16, 32)"])
    for name, adj in graphs.items():
        for p in (16, 64, 256):
            evaluate(adj, p, name)
