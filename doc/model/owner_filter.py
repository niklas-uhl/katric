# Owner-side relevance: the owner of w sends the rank set R(w) (all ranks with a neighbor of w; piggybacks on the
# degree exchange) or P(w) (ranks with an in-neighbor u -> w; needs orientation) to every rank in that set.
# Sender drops entry w of a message to r if r is not in the set. Sets larger than `cap` ranks are not sent
# (entry always kept). Exchange cost: 16 bits per rank id per recipient. Messages: 64 bits per entry.
import numpy as np
rng = np.random.default_rng(1)
n, m = 1 << 16, 1 << 20
src = rng.integers(0, n, m); dst = rng.integers(0, n, m)
keep = src != dst
e = np.unique(np.sort(np.stack([src[keep], dst[keep]], 1), 1), axis=0)
deg = np.bincount(e.ravel(), minlength=n)
key = deg.astype(np.int64) * n + np.arange(n)
a, b_ = e[:, 0], e[:, 1]
fwd = key[a] < key[b_]
tail = np.where(fwd, a, b_); head = np.where(fwd, b_, a)
o = np.argsort(tail, kind="stable"); tail, head = tail[o], head[o]
start = np.searchsorted(tail, np.arange(n + 1))
und_x = np.concatenate([a, b_]); und_y = np.concatenate([b_, a])   # undirected pairs (w, neighbor)
print(f"{'p':>5} {'set':>4} {'cap':>5}  {'entries kept':>12}  {'exchange / messages':>19}  {'total volume':>12}")
for p in (16, 64, 192, 768):
    owner = np.arange(n) * p // n
    vr = np.unique(tail * p + owner[head]); v_ = vr // p; r_ = vr % p
    remote = r_ != owner[v_]; v_, r_ = v_[remote], r_[remote]
    lens = start[v_ + 1] - start[v_]
    idx = np.repeat(np.arange(len(v_)), lens)
    w = head[np.concatenate([np.arange(start[x], start[x + 1]) for x in v_])]
    r_e = r_[idx]
    sent = len(w)
    for name, (wx, ux) in {"R": (und_x, und_y), "P": (head, tail)}.items():
        pairs = np.unique(wx * p + owner[ux])           # (w, rank) membership
        size = np.bincount(pairs // p, minlength=n)     # |set(w)|
        for cap in (64, 1 << 30):
            capped = size > cap
            member = np.isin(w * p + r_e, pairs) | (owner[w] == r_e) | capped[w]
            kept = member.sum()
            # owner sends set(w) (16 bit per id) to each rank in it, except the owner itself; capped sets not sent
            s = size.astype(float)
            exch = np.where(capped, 0, s * np.maximum(s - 1, 0) * 16).sum()
            print(f"{p:5d} {name:>4} {('none' if cap > n else cap):>5}  {kept/sent:12.1%}  {exch/(sent*64):19.2f}  {(kept*64+exch)/(sent*64):11.2f}x")
