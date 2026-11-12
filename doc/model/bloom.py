# Bloom filter of W_r = union of N+(u), u on r: sender drops entries w of N+(v) not in the receiver's filter.
# Volume in bits: messages 64 per entry; filters |W_r| * b bits, sent to every rank that sends to r.
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
print(f"{'p':>5} {'relevant':>9} {'|W_r| avg':>10} {'senders/r':>10}  " + "  ".join(f"b={b}: kept / volume" for b in (2, 4, 8)))
for p in (16, 64, 192, 768, 3072):
    owner = np.arange(n) * p // n
    # W_r as a boolean matrix would be p x n; use per-rank sets of (r, w) pairs
    W = np.unique(owner[tail] * n + head)                   # pairs (r, w) with w in N+(u), u on r
    Wsize = np.bincount(W // n, minlength=p)
    # messages: (v, r) for r owning an out-neighbor of v, r != owner(v); entries = N+(v)
    vr = np.unique(tail * p + owner[head]); v_ = vr // p; r_ = vr % p
    remote = r_ != owner[v_]; v_, r_ = v_[remote], r_[remote]
    lens = start[v_ + 1] - start[v_]
    sent = lens.sum()
    # relevant entries: w in W_r
    idx = np.repeat(np.arange(len(v_)), lens)
    w = head[np.concatenate([np.arange(start[x], start[x + 1]) for x in v_])]
    rel = np.isin(r_[idx] * n + w, W).sum()
    senders = np.array([len(np.unique(owner[v_[r_ == r]])) for r in range(p)])
    out = []
    for b in (2, 4, 8):
        fp = 0.6185 ** b
        kept = rel + fp * (sent - rel)
        filt = (Wsize * b * senders).sum()
        out.append(f"{kept/sent:5.1%} / {(kept*64 + filt)/(sent*64):5.2f}x")
    print(f"{p:5d} {rel/sent:9.1%} {Wsize.mean():10.0f} {senders.mean():10.1f}  " + "   ".join(out))
