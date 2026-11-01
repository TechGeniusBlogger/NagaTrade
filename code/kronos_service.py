# On-demand Kronos forecasts for the /kronos/forecast page. The Kronos live demo's method: 360 past
# 1-hour candles in, the next 24 out, 30 runs -> mean path, min-max range, up-probability,
# "volatility amplification". The model loads once; each forecast (~5 s on CPU) is cached on disk.
#
# Candles: six full one-hour candles a day, 9:15 to 15:15. Since 3 Aug 2026 Dhan's intraday feed for
# stocks stops at 15:15 (the last 15 minutes are missing at every interval), so that short stub is
# left out on every day, for every symbol - every candle is then a true 60 minutes.
import json
import os
import threading
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_hourly")
CACHE = os.path.join(HERE, "chart_cache")
SLOTS = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15"]
HIST, PRED, PATHS, T, TOP_P, N_PAST, SHOW = 360, 24, 30, 1.0, 0.9, 10, 140
SETTINGS = dict(model="Kronos-base", hist=HIST, pred=PRED, paths=PATHS, T=T, top_p=TOP_P)
TAG = f"b_h{HIST}_p{PRED}_n{PATHS}_T{T}_tp{TOP_P}"

_lock = threading.Lock()
_model = None
_uni = None


def universe():
    global _uni
    if _uni is None:
        _uni = {u["symbol"]: u for u in json.load(open(os.path.join(HERE, "universe.json")))}
    return _uni


def _load_model():
    global _model
    if _model is None:
        import kronos_paths as kp
        _model = (kp,) + kp.load(threads=4)
    return _model


def candles(sym):
    path = os.path.join(DATA, sym.replace(" ", "_") + "_1h.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no candles for {sym} yet - run fetch_hourly.py")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df.date.dt.strftime("%H:%M").isin(SLOTS)].sort_values("date").reset_index(drop=True)
    df["day"] = df.date.dt.strftime("%Y-%m-%d")
    return df


def origins(df):
    first = df.groupby("day").head(1).index.values
    return [int(i) for i in first if i >= HIST][-N_PAST:] + [len(df)]


def _next_slots(last_ts, n):
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


def forecast(sym, df, i):
    """The forecast made from the HIST candles before index i (i == len(df): the latest one)."""
    past, fut = df.iloc[i - HIST:i], df.iloc[i:i + PRED]
    asof = past.date.iloc[-1].strftime("%Y-%m-%d %H:%M")
    cpath = os.path.join(CACHE, TAG, sym.replace(" ", "_"), asof.replace(":", "").replace(" ", "_") + ".json")
    if os.path.exists(cpath):
        f = json.load(open(cpath))
        if f["actual_n"] >= min(PRED, len(fut)):
            return f
    if len(fut) < PRED:
        fut_ts = pd.concat([fut.date, _next_slots((fut.date if len(fut) else past.date).iloc[-1], PRED - len(fut))], ignore_index=True)
    else:
        fut_ts = fut.date.reset_index(drop=True)
    seed = int(fut_ts.iloc[0].strftime("%Y%m%d")) + zlib.crc32(sym.encode()) % 100000
    with _lock:
        kp, tok, mdl = _load_model()
        P = kp.sample_paths(tok, mdl, past[["open", "high", "low", "close", "volume"]], past.date, fut_ts,
                            n_paths=PATHS, T=T, top_p=TOP_P, seed=seed)
    last = float(past.close.iloc[-1])
    end = P[:, -1, 3]
    hist_vol = np.std(np.diff(np.log(past.close.values[-PRED - 1:])))
    fc_vol = np.std(np.diff(np.log(np.concatenate([np.full((PATHS, 1), last), P[:, :, 3]], 1)), axis=1), axis=1)
    f = dict(asof=asof, ts=[t.strftime("%Y-%m-%d %H:%M") for t in fut_ts],
             mean=np.round(P.mean(0)[:, :5], 2).tolist(),
             lo=np.round(P[:, :, 3].min(0), 2).tolist(), hi=np.round(P[:, :, 3].max(0), 2).tolist(),
             paths=np.round(P[:, :, 3], 2).tolist(), last=last,
             up_prob=float((end > last).mean()), volamp_prob=float((fc_vol > hist_vol).mean()),
             mean_end=float(end.mean()), min_end=float(end.min()), max_end=float(end.max()),
             actual_n=int(len(fut)))
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    with open(cpath, "w") as fh:
        json.dump(f, fh, separators=(",", ":"))
    return f


def payload(sym):
    """Everything the page needs for one symbol: candles to draw, the forecast slots, and the latest
    forecast computed. Earlier forecasts are filled in one at a time by forecast_at()."""
    u = universe()[sym]
    df = candles(sym)
    if len(df) < HIST + 1:
        raise ValueError(f"only {len(df)} candles for {sym}; Kronos needs {HIST}")
    org = origins(df)
    start = max(0, org[0] - HIST - SHOW)
    c = df.iloc[start:]
    fcs = [None] * len(org)
    fcs[-1] = dict(forecast(sym, df, org[-1]), origin=org[-1] - start, latest=True)
    for k, i in enumerate(org[:-1]):  # already cached earlier forecasts come along for free
        f = _cached(sym, df, i)
        if f:
            fcs[k] = dict(f, origin=i - start, latest=False)
    return dict(symbol=sym, name=u["name"], tf="1H", source="Dhan API", settings=SETTINGS,
                data_from=c.date.iloc[0].strftime("%Y-%m-%d"), data_to=df.date.iloc[-1].strftime("%Y-%m-%d %H:%M"),
                candles=[[t.strftime("%Y-%m-%d %H:%M"), *map(float, r)] for t, r in
                         zip(c.date, c[["open", "high", "low", "close", "volume"]].values)],
                origins=[i - start for i in org], forecasts=fcs, offset=int(start))


def _cached(sym, df, i):
    past, fut = df.iloc[i - HIST:i], df.iloc[i:i + PRED]
    asof = past.date.iloc[-1].strftime("%Y-%m-%d %H:%M")
    cpath = os.path.join(CACHE, TAG, sym.replace(" ", "_"), asof.replace(":", "").replace(" ", "_") + ".json")
    if os.path.exists(cpath):
        f = json.load(open(cpath))
        if f["actual_n"] >= min(PRED, len(fut)):
            return f
    return None


def forecast_at(sym, k):
    """The k-th forecast slot for sym (0 = oldest of the last N_PAST days, last = latest)."""
    df = candles(sym)
    org = origins(df)
    k = max(0, min(len(org) - 1, int(k)))
    start = max(0, org[0] - HIST - SHOW)
    return dict(forecast(sym, df, org[k]), origin=org[k] - start, latest=k == len(org) - 1)
