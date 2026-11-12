# Per-pair adaptive Bloom filter size vs. fixed bits per element (same random graph as bloom.py).
# Pair (s, r): traffic t = entries s sends to r, irr = irrelevant ones (w not in W_r).
# Adaptive: M = |W_r| * ln(30.8 t / |W_r|) / 0.48 bits (estimate irr by t), rounded down to a power of two,
# no filter if 30.8 t <= |W_r|; FP = 0.6185^(M/|W_r|). Volume in bits, messages 64 per entry.
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
print(f"{'p':>5}  {'b=4 fixed: kept / volume':>25}  {'adaptive: kept / volume':>24}  {'pairs w/ filter':>15}  {'bits/elem (median)':>18}")
for p in (16, 64, 192, 768):
    owner = np.arange(n) * p // n
    W = np.unique(owner[tail] * n + head)
    Wsize = np.bincount(W // n, minlength=p)
    vr = np.unique(tail * p + owner[head]); v_ = vr // p; r_ = vr % p
    remote = r_ != owner[v_]; v_, r_ = v_[remote], r_[remote]
    lens = start[v_ + 1] - start[v_]
    idx = np.repeat(np.arange(len(v_)), lens)
    w = head[np.concatenate([np.arange(start[x], start[x + 1]) for x in v_])]
    relevant = np.isin(r_[idx] * n + w, W)
    s_e, r_e = owner[v_[idx]], r_[idx]
    pair = s_e * p + r_e
    t = np.bincount(pair, minlength=p * p).astype(float)
    irr = np.bincount(pair, weights=~relevant, minlength=p * p)
    rel = t - irr
    Wr = np.tile(Wsize, p).astype(float)  # pair index s*p + r -> |W_r|
    active = t > 0
    sent = t.sum()
    # fixed b = 4 for every active pair
    fixed_kept = rel.sum() + 0.6185 ** 4 * irr.sum()
    fixed_vol = fixed_kept * 64 + (4 * Wr[active]).sum()
    # adaptive
    ratio = np.where(active, 30.8 * t / np.maximum(Wr, 1), 0)
    M = np.where(ratio > 1, Wr * np.log(np.maximum(ratio, 1)) / 0.48, 0)
    M = np.where(M >= 1, 2.0 ** np.floor(np.log2(np.maximum(M, 1))), 0)
    fp = np.where(M > 0, 0.6185 ** (M / np.maximum(Wr, 1)), 1.0)
    ad_kept = (rel + fp * irr).sum()
    ad_vol = ad_kept * 64 + M.sum()
    bpe = (M / np.maximum(Wr, 1))[M > 0]
    print(f"{p:5d}  {fixed_kept/sent:12.1%} / {fixed_vol/(sent*64):5.2f}x  {ad_kept/sent:12.1%} / {ad_vol/(sent*64):5.2f}x"
          f"  {(M > 0).sum() / active.sum():15.1%}  {np.median(bpe) if len(bpe) else 0:18.1f}")
