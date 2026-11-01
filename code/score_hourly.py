# Scores a backtest_hourly.py run: direction, range coverage, return correlation (the paper's IC),
# the snap-back check, volume, and the demo's "volatility amplification" probability.
#   python score_hourly.py results/<tag>
import sys

import numpy as np
import pandas as pd

s = pd.read_csv(sys.argv[1] + "/summary.csv")
print(f"forecasts: {len(s)}  ({s.origin.iloc[0]} .. {s.origin.iloc[-1]})")
ctx_trend = s["last"] / s.ctx_first - 1

for h in sorted({int(c[3:]) for c in s.columns if c.startswith("act") and c[3:].isdigit()}):
    act = s[f"act{h}"] / s["last"] - 1
    fc = s[f"mean{h}"] / s["last"] - 1
    up, call = act > 0, s[f"up{h}"] > 0.5
    print(f"\n--- {h} candles ahead ---")
    print(f"direction hit (up-prob > 50%):   {(call == up).mean():.1%}")
    print(f"direction hit (mean path):       {((fc > 0) == up).mean():.1%}")
    print(f"always-up baseline:              {up.mean():.1%}")
    for thr in (0.7, 0.8, 0.9):
        m = (s[f"up{h}"] >= thr) | (s[f"up{h}"] <= 1 - thr)
        if m.sum():
            print(f"  confident >= {thr:.0%}:  n={m.sum():3d}  hit={(call[m] == up[m]).mean():.1%}")
    print(f"IC (corr forecast vs actual return): {np.corrcoef(fc, act)[0, 1]:+.3f}   RankIC: {fc.rank().corr(act.rank()):+.3f}")
    print(f"snap-back check: corr(forecast, past-360 trend) {np.corrcoef(fc, ctx_trend)[0, 1]:+.3f} | corr(actual, past trend) {np.corrcoef(act, ctx_trend)[0, 1]:+.3f}")
    print(f"close inside full min-max range: {((s[f'act{h}'] >= s[f'min{h}']) & (s[f'act{h}'] <= s[f'max{h}'])).mean():.1%}")
    print(f"close inside 5-95% band:         {((s[f'act{h}'] >= s[f'q05_{h}']) & (s[f'act{h}'] <= s[f'q95_{h}'])).mean():.1%}  (target 90%)")
    print(f"band width median {((s[f'q95_{h}'] - s[f'q05_{h}']) / s['last']).median():.2%} vs actual |move| median {act.abs().median():.2%}")
    print(f"volume: corr(forecast, actual) {np.corrcoef(s[f'fcvol{h}'], s[f'actvol{h}'])[0, 1]:+.3f}, forecast/actual median {(s[f'fcvol{h}'] / s[f'actvol{h}']).median():.2f}")

p, y = s.volamp_prob, s.act_volamp
print(f"\nvolatility amplification: avg forecast prob {p.mean():.1%}, actually happened {y.mean():.1%}, "
      f"hit (prob>50%) {((p > 0.5) == (y == 1)).mean():.1%}, Brier {((p - y) ** 2).mean():.3f} vs always-{y.mean():.2f} {((y.mean() - y) ** 2).mean():.3f}")
