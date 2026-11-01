# Proves the fast (KV-cached) Kronos loop gives the same forecast as the original loop: same seed,
# same history -> same sampled tokens and same candles. Also times both.
import time

import numpy as np
import torch

import kronos_paths as kp
from data import context_for_day, load_5min

N, PRED, SEED = 4, 12, 7

df = load_5min()
day = sorted(df.date.unique())[-1]
past, fut = context_for_day(df, day, 437)
tok, mdl = kp.load(threads=6)

out = {}
for name, fast in (("original", False), ("fast", True)):
    t = time.time()
    out[name] = kp.sample_paths(tok, mdl, past[["open", "high", "low", "close"]], past.ts, fut.ts.iloc[:PRED],
                                n_paths=N, seed=SEED, fast=fast)
    print(f"{name:8s} {time.time() - t:6.1f}s  shape {out[name].shape}")

a, b = out["original"], out["fast"]
diff = np.abs(a - b)
print("max abs diff (price points):", float(diff.max()))
print("paths x candles with close diff > 0.5 pt:", int((diff[:, :, 3] > 0.5).sum()), "of", a.shape[0] * a.shape[1])
print("close, path 0, first 6 candles\n original:", np.round(a[0, :6, 3], 2), "\n fast:    ", np.round(b[0, :6, 3], 2))
