# Share of global-phase message entries the receiver can use, on a graph without locality.
# DiTriC: v sends N+(v) once to every other rank r owning an out-neighbor u. Entry w is
#   necessary-relevant if w is owned by r or r holds a neighbor of w (r in R(w)),
#   useful if w in N+(u) for some u in N+(v) owned by r (it closes a triangle).
import numpy as np, sys
rng = np.random.default_rng(1)
n, m = 1 << 16, 1 << 20
src = rng.integers(0, n, m); dst = rng.integers(0, n, m)
keep = src != dst; src, dst = src[keep], dst[keep]
e = np.unique(np.sort(np.stack([src, dst], 1), 1), axis=0)
deg = np.bincount(e.ravel(), minlength=n)
key = deg.astype(np.int64) * n + np.arange(n)  # orientation (degree, id)
a, b = e[:, 0], e[:, 1]
fwd = key[a] < key[b]
tail = np.where(fwd, a, b); head = np.where(fwd, b, a)
order = np.argsort(tail, kind="stable"); tail, head = tail[order], head[order]
start = np.searchsorted(tail, np.arange(n + 1))
nbr = [set() for _ in range(n)]
for x, y in e: nbr[x].add(y); nbr[y].add(x)
outset = [set(head[start[v]:start[v + 1]]) for v in range(n)]
for p in (16, 64, 192, 768):
    owner = np.arange(n) * p // n          # contiguous ranges; ids are random, so no locality
    R = [set(owner[list(nbr[w])]) for w in range(n)]
    sent = rel = useful = 0
    for v in range(n):
        out = head[start[v]:start[v + 1]]
        if len(out) == 0: continue
        for r in set(owner[out]) - {owner[v]}:
            us = [u for u in out if owner[u] == r]
            for w in out:
                sent += 1
                if owner[w] == r or r in R[w]: rel += 1
                if any(w in outset[u] for u in us): useful += 1
    print(f"p={p:4d}: entries sent {sent:9d}, relevant (w on r or r in R(w)) {rel/sent:6.1%}, closes a triangle {useful/sent:6.2%}")
    sys.stdout.flush()
