# Walk-forward test of Kronos on 1-hour stock candles, following the Kronos live demo's method:
# 360 past candles in, 24 candles out, 30 sampled paths -> mean path, full range, up-probability.
# One forecast per trading day, made before the 9:15 open from candles already closed.
#   python backtest_hourly.py --symbol TCS --days 250
import argparse
import os
import time

import numpy as np
import pandas as pd

import kronos_paths as kp

HOURLY = r"data_hourly"  # a folder of one-hour candle CSVs (research only)
SLOTS = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]

ap = argparse.ArgumentParser()
ap.add_argument("--symbol", default="TCS")
ap.add_argument("--days", type=int, default=250)
ap.add_argument("--hist", type=int, default=360)
ap.add_argument("--pred", type=int, default=24)
ap.add_argument("--paths", type=int, default=30)
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()

df = pd.read_csv(os.path.join(HOURLY, f"{a.symbol}.csv"), parse_dates=["date"])
df = df[df.date.dt.strftime("%H:%M").isin(SLOTS)].reset_index(drop=True)
df["day"] = df.date.dt.strftime("%Y-%m-%d")

tag = f"{a.symbol}_1h_h{a.hist}_p{a.pred}_n{a.paths}_T{a.T}_tp{a.top_p}"
out = os.path.join(kp.HERE, "results", tag)
os.makedirs(out, exist_ok=True)
summary = os.path.join(out, "summary.csv")
done = set(pd.read_csv(summary).origin) if os.path.exists(summary) else set()

first_idx = df.groupby("day").head(1).index.values
origins = [i for i in first_idx if i >= a.hist and i + a.pred <= len(df)][-a.days:]
tok, mdl = kp.load(threads=a.threads)

for i in origins:
    day = df.day[i]
    if day in done:
        continue
    past, fut = df.iloc[i - a.hist:i], df.iloc[i:i + a.pred]
    t = time.time()
    P = kp.sample_paths(tok, mdl, past[["open", "high", "low", "close", "volume"]], past.date, fut.date,
                        n_paths=a.paths, T=a.T, top_p=a.top_p, seed=int(day.replace("-", "")))
    np.savez_compressed(os.path.join(out, f"{day}.npz"), paths=P[:, :, :5].astype(np.float32))
    last = float(past.close.iloc[-1])
    row = dict(origin=day, last=last, ctx_first=float(past.close.iloc[0]), secs=round(time.time() - t, 1))
    hist_vol = np.std(np.diff(np.log(past.close.values[-a.pred - 1:])))
    for h in sorted({7, a.pred}):
        end = P[:, h - 1, 3]
        row[f"act{h}"] = float(fut.close.iloc[h - 1])
        row[f"mean{h}"] = float(end.mean())
        row[f"up{h}"] = float((end > last).mean())
        row[f"min{h}"], row[f"max{h}"] = float(end.min()), float(end.max())
        row[f"q05_{h}"], row[f"q95_{h}"] = float(np.percentile(end, 5)), float(np.percentile(end, 95))
        row[f"actvol{h}"] = float(fut.volume.iloc[:h].sum())
        row[f"fcvol{h}"] = float(P[:, :h, 4].sum(1).mean())
    fc_vol = np.std(np.diff(np.log(np.concatenate([np.full((a.paths, 1), last), P[:, :, 3]], 1)), axis=1), axis=1)
    act_vol = np.std(np.diff(np.log(np.concatenate([[last], fut.close.values]))))
    row.update(hist_vol=float(hist_vol), volamp_prob=float((fc_vol > hist_vol).mean()), act_volamp=int(act_vol > hist_vol))
    pd.DataFrame([row]).to_csv(summary, mode="a", header=not os.path.exists(summary), index=False)
    print(day, "up24", row[f"up{a.pred}"], "last", last, "act24", row[f"act{a.pred}"], row["secs"], "s", flush=True)
