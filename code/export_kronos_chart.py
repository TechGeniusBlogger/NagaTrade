# Builds the data behind the Kronos forecast page (/kronos/forecast), following the Kronos live
# demo's method: 360 past 1-hour candles in, the next 24 candles out, 30 Monte Carlo runs -> the
# mean path (solid line), the full min-max range of the runs (shaded), mean forecast volume, the
# up-probability and the "volatility amplification" probability. Same settings as backtest_hourly.py.
#
# One forecast from the very latest candle (the real "what's next"), plus one made before the open
# of each of the last few trading days, so the page can step back and reveal what really happened.
#   python export_kronos_chart.py --symbol CIPLA
import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

import kronos_paths as kp

MENTORSHIP = r"data_hourly"  # a folder of one-hour candle CSVs (research only)
SLOTS = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]

ap = argparse.ArgumentParser()
ap.add_argument("--symbol", default="CIPLA")
ap.add_argument("--past_days", type=int, default=10)
ap.add_argument("--hist", type=int, default=360)
ap.add_argument("--pred", type=int, default=24)
ap.add_argument("--paths", type=int, default=30)
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

dhan_file = os.path.join(kp.HERE, "data_hourly", f"{a.symbol}_1h.csv")
src_file, source = (dhan_file, "Dhan API (fetch_hourly.py)") if os.path.exists(dhan_file) else \
    (os.path.join(MENTORSHIP, f"{a.symbol}.csv"), "mentorship 1-hour file (Session 10)")
df = pd.read_csv(src_file, parse_dates=["date"])
df = df[df.date.dt.strftime("%H:%M").isin(SLOTS)].sort_values("date").reset_index(drop=True)
df["day"] = df.date.dt.strftime("%Y-%m-%d")


def next_slots(last_ts, n):
    # NSE hourly candle starts after last_ts, skipping weekends (exchange holidays are not known here).
    out, d, k = [], last_ts.normalize(), SLOTS.index(last_ts.strftime("%H:%M")) + 1
    while len(out) < n:
        if k == len(SLOTS):
            d, k = d + pd.Timedelta(days=1), 0
            while d.weekday() >= 5:
                d += pd.Timedelta(days=1)
            continue
        hh, mm = map(int, SLOTS[k].split(":"))
        out.append(d + pd.Timedelta(hours=hh, minutes=mm))
        k += 1
    return pd.Series(out)


fmt = lambda s: [t.strftime("%Y-%m-%d %H:%M") for t in s]
tok, mdl = kp.load(threads=a.threads)

first_idx = df.groupby("day").head(1).index.values
origins = [int(i) for i in first_idx if i >= a.hist][-a.past_days:] + [len(df)]
forecasts = []
for i in origins:
    past = df.iloc[i - a.hist:i]
    fut = df.iloc[i:i + a.pred]
    fut_ts = pd.concat([fut.date, next_slots(past.date.iloc[-1] if fut.empty else fut.date.iloc[-1], a.pred - len(fut))],
                       ignore_index=True) if len(fut) < a.pred else fut.date.reset_index(drop=True)
    seed_day = fut_ts.iloc[0].strftime("%Y%m%d")
    P = kp.sample_paths(tok, mdl, past[["open", "high", "low", "close", "volume"]], past.date, fut_ts,
                        n_paths=a.paths, T=a.T, top_p=a.top_p, seed=int(seed_day))
    last = float(past.close.iloc[-1])
    end = P[:, -1, 3]
    hist_vol = np.std(np.diff(np.log(past.close.values[-a.pred - 1:])))
    fc_vol = np.std(np.diff(np.log(np.concatenate([np.full((a.paths, 1), last), P[:, :, 3]], 1)), axis=1), axis=1)
    mean = P.mean(axis=0)
    forecasts.append(dict(
        origin=i, latest=i == len(df), asof=past.date.iloc[-1].strftime("%Y-%m-%d %H:%M"), ts=fmt(fut_ts),
        mean=np.round(mean[:, :5], 2).tolist(),
        lo=np.round(P[:, :, 3].min(0), 2).tolist(), hi=np.round(P[:, :, 3].max(0), 2).tolist(),
        paths=np.round(P[:, :, 3], 2).tolist(),
        last=last, up_prob=float((end > last).mean()), volamp_prob=float((fc_vol > hist_vol).mean()),
        mean_end=float(end.mean()), min_end=float(end.min()), max_end=float(end.max()),
        actual_n=int(len(fut))))
    print(forecasts[-1]["asof"], "-> up", forecasts[-1]["up_prob"], "volamp", forecasts[-1]["volamp_prob"], flush=True)

start = int(max(0, origins[0] - a.hist - 140))
c = df.iloc[start:]
out = dict(symbol=a.symbol, tf="1H", source=source, data_from=c.date.iloc[0].strftime("%Y-%m-%d"),
           data_to=df.date.iloc[-1].strftime("%Y-%m-%d %H:%M"), offset=start,
           settings=dict(model="Kronos-base", hist=a.hist, pred=a.pred, paths=a.paths, T=a.T, top_p=a.top_p),
           candles=[[t, *map(float, r)] for t, r in zip(fmt(c.date), c[["open", "high", "low", "close", "volume"]].values)],
           forecasts=[{**f, "origin": f["origin"] - start} for f in forecasts],
           made=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
os.makedirs(os.path.join(kp.HERE, "chart_data"), exist_ok=True)
path = os.path.join(kp.HERE, "chart_data", f"{a.symbol}.json")
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("wrote", path, len(out["candles"]), "candles,", len(forecasts), "forecasts")
