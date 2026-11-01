# Kronos vs a random estimate built the same way: for every forecast day, draw 30 random future
# paths by resampling the stock's own past hourly moves (and hourly volumes) from the SAME 360-candle
# history Kronos saw. No model, just chance. Repeated REPS times so we see the spread of luck.
#   python compare_random.py TCS results/<tag>
import os
import sys

import numpy as np
import pandas as pd

from backtest_hourly_data import load_hourly

SYM, RUN = sys.argv[1], sys.argv[2]
HIST, PRED, PATHS, REPS, H = 360, 24, 30, 200, 7

s = pd.read_csv(os.path.join(RUN, "summary.csv")).sort_values("origin").reset_index(drop=True)
df = load_hourly(SYM)
first = df.groupby("day").head(1)
idx_of = dict(zip(first.day, first.index))
rng = np.random.default_rng(2026)

n = len(s)
rw_up, rw_in90, rw_inmm, rw_vamp, rw_vol = (np.zeros((REPS, n)) for _ in range(5))
for j, day in enumerate(s.origin):
    i = idx_of[day]
    past, fut = df.iloc[i - HIST:i], df.iloc[i:i + PRED]
    r = np.diff(np.log(past.close.values))
    last = past.close.values[-1]
    draws = r[rng.integers(0, len(r), size=(REPS, PATHS, PRED))]
    paths = last * np.exp(np.cumsum(draws, axis=2))
    end = paths[:, :, H - 1]
    act = fut.close.values[H - 1]
    rw_up[:, j] = (end > last).mean(1)
    q05, q95 = np.percentile(end, [5, 95], axis=1)
    rw_in90[:, j] = (act >= q05) & (act <= q95)
    rw_inmm[:, j] = (act >= end.min(1)) & (act <= end.max(1))
    rw_vamp[:, j] = (draws.std(axis=2) > s.hist_vol[j]).mean(1)
    rw_vol[:, j] = past.volume.values[rng.integers(0, HIST, size=(REPS, PATHS, H))].sum(2).mean(1)

act = s[f"act{H}"].values
up = act > s["last"].values
vy = s.act_volamp.values


def spread(x):
    return f"{np.mean(x):.1%}  (luck range {np.percentile(x, 5):.1%} to {np.percentile(x, 95):.1%})"


def beat(k, x, higher=True):
    return f"random did better in {np.mean(x >= k if higher else x <= k):.0%} of 200 tries"


print(f"{SYM}: {n} daily forecasts, {s.origin.iloc[0]} .. {s.origin.iloc[-1]}, next-day = {H} candles\n")

k = ((s[f"up{H}"] > 0.5) == up).mean()
rw = ((rw_up > 0.5) == up).mean(1)
coin = rng.binomial(n, 0.5, 10000) / n
print("Q1 up or down tomorrow?")
print(f"  Kronos        {k:.1%}")
print(f"  random paths  {spread(rw)}   -> {beat(k, rw)}")
print(f"  coin flip     {spread(coin)}   -> {beat(k, coin)}")
print(f"  always up     {up.mean():.1%}")

k = ((s.volamp_prob > 0.5) == (vy == 1)).mean()
rw = ((rw_vamp > 0.5) == (vy == 1)).mean(1)
kb, rb = ((s.volamp_prob - vy) ** 2).mean(), ((rw_vamp - vy) ** 2).mean(1)
print("\nQ2 more volatile than the last 24 hours?")
print(f"  Kronos        {k:.1%}   (Brier {kb:.3f}, lower is better)")
print(f"  random paths  {spread(rw)}   -> {beat(k, rw)}   (Brier {rb.mean():.3f})")
print(f"  always yes    {vy.mean():.1%}")

av = s[f"actvol{H}"].values
kc = np.corrcoef(s[f"fcvol{H}"], av)[0, 1]
rc = np.array([np.corrcoef(rw_vol[t], av)[0, 1] for t in range(REPS)])
ke = np.median(np.abs(np.log(s[f"fcvol{H}"] / av)))
re = np.median(np.abs(np.log(rw_vol / av)), axis=1)
print("\nQ3 how much volume will trade tomorrow?")
print(f"  Kronos        corr {kc:+.2f}, typical miss {np.expm1(ke):.0%}")
print(f"  random draws  corr {rc.mean():+.2f} (luck range {np.percentile(rc, 5):+.2f} to {np.percentile(rc, 95):+.2f}), "
      f"typical miss {np.expm1(re.mean()):.0%}   -> {beat(kc, rc)}")

k90 = ((act >= s[f"q05_{H}"]) & (act <= s[f"q95_{H}"])).mean()
kmm = ((act >= s[f"min{H}"]) & (act <= s[f"max{H}"])).mean()
print("\nQ4 does the close stay inside the 90% range?  (target 90%)")
print(f"  Kronos        {k90:.1%}   | inside full 30-run range {kmm:.1%} (target ~94%)")
print(f"  random paths  {spread(rw_in90.mean(1))}   | inside full range {rw_inmm.mean():.1%}")
