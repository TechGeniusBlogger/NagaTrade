# Walk-forward test of Kronos-base on Nifty 50 5-minute candles. For each test day, the model sees
# only the 437 candles before that day's 9:15 open and forecasts all 75 candles of the day, many
# times. Nothing from the test day is visible to it. Resumable: finished days are skipped.
#   python backtest.py --days 40 --paths 100 --tag default
import argparse
import os
import time

import numpy as np
import pandas as pd

import kronos_paths as kp
from data import context_for_day, load_5min

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=40)
ap.add_argument("--paths", type=int, default=100)
ap.add_argument("--hist", type=int, default=437)
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--tag", default="default")
a = ap.parse_args()

out = os.path.join(kp.HERE, "results", a.tag)
os.makedirs(out, exist_ok=True)
summary = os.path.join(out, "summary.csv")
done = set(pd.read_csv(summary).date) if os.path.exists(summary) else set()

df = load_5min()
days = sorted(df.date.unique())[-a.days:]
tok, mdl = kp.load(threads=a.threads)

for day in days:
    if day in done:
        continue
    past, fut = context_for_day(df, day, a.hist)
    t = time.time()
    P = kp.sample_paths(tok, mdl, past[["open", "high", "low", "close"]], past.ts, fut.ts,
                        n_paths=a.paths, T=a.T, top_p=a.top_p, seed=int(day.replace("-", "")))
    np.savez_compressed(os.path.join(out, f"{day}.npz"), paths=P[:, :, :4].astype(np.float32))
    prev = float(past.close.iloc[-1])
    ends, hi, lo = P[:, -1, 3], P[:, :, 1].max(1), P[:, :, 2].min(1)
    row = dict(date=day, prev_close=prev, open=float(fut.open.iloc[0]), close=float(fut.close.iloc[-1]),
               high=float(fut.high.max()), low=float(fut.low.min()), up_share=float((ends > prev).mean()),
               **{f"c{q:02d}": float(np.percentile(ends, q)) for q in (5, 25, 50, 75, 95)},
               hi50=float(np.median(hi)), lo50=float(np.median(lo)),
               hi95=float(np.percentile(hi, 95)), lo05=float(np.percentile(lo, 5)),
               secs=round(time.time() - t, 1))
    pd.DataFrame([row]).to_csv(summary, mode="a", header=not os.path.exists(summary), index=False)
    print(day, "up", row["up_share"], "close", row["close"], "band", round(row["c05"]), round(row["c95"]), row["secs"], "s", flush=True)
