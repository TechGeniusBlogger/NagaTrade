# Kronos Forecast (desktop app) - the forecasting service. Same method as the Kronos team's live
# demo: 360 past 1-hour candles in, the next 24 out, 30 runs -> average path, lowest-to-highest
# range, how the runs split up/down, and "volatility amplification".
#
# Candles come live from the user's OWN Dhan account (their client id + access token, stored only on
# this computer). Six full one-hour candles a day, 9:15 to 15:15: since Aug 2026 Dhan's intraday
# feed for stocks stops at 15:15, so the last short stub is left out on every day for every symbol.
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

import numpy as np
import pandas as pd

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def _home():
    if sys.platform == "win32":
        root = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        root = os.path.expanduser("~/Library/Application Support")
    else:
        root = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    d = os.path.join(root, "KronosForecast")
    os.makedirs(d, exist_ok=True)
    return d


HOME = _home()
DATA, CACHE, MODELS = (os.path.join(HOME, x) for x in ("candles", "forecasts", "models"))
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SLOTS = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15"]
HIST, PRED, PATHS, T, TOP_P, N_PAST, SHOW, DAYS = 360, 24, 30, 1.0, 0.9, 10, 140, 180
SETTINGS = dict(model="Kronos-base", hist=HIST, pred=PRED, paths=PATHS, T=T, top_p=TOP_P)
TAG = f"b_h{HIST}_p{PRED}_n{PATHS}_T{T}_tp{TOP_P}"
FRESH_SECONDS = 600

# The official Kronos release on Hugging Face, pinned by size and SHA-256 so a broken or swapped
# download is refused.
HF = "https://huggingface.co/NeoQuasar/{repo}/resolve/main/{file}"
MODEL_FILES = [
    ("Kronos-Tokenizer-base", "config.json", 301, "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2"),
    ("Kronos-Tokenizer-base", "model.safetensors", 15842368, "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"),
    ("Kronos-base", "config.json", 228, "77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2"),
    ("Kronos-base", "model.safetensors", 409264008, "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83"),
]


class DhanAuthError(Exception):
    pass


_keys = {"client_id": "", "token": ""}
_lock = threading.Lock()
_model = None
_uni = None
model_state = {"status": "missing", "done": 0, "total": sum(f[2] for f in MODEL_FILES), "error": ""}


def set_keys(client_id, token):
    _keys["client_id"], _keys["token"] = client_id, token


def universe():
    global _uni
    if _uni is None:
        _uni = {u["symbol"]: u for u in json.load(open(os.path.join(BASE, "assets", "universe.json")))}
    return _uni


# ---- the model: download once, verify, load once ---------------------------------------------------
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_ready():
    for repo, name, size, _ in MODEL_FILES:
        p = os.path.join(MODELS, repo, name)
        if not os.path.exists(p) or os.path.getsize(p) != size:
            return False
    return True


def refresh_model_state():
    if model_state["status"] != "downloading":
        model_state["status"] = "ready" if model_ready() else "missing"


def download_model():
    if model_state["status"] == "downloading":
        return
    model_state.update(status="downloading", done=0, error="")
    try:
        done = 0
        for repo, name, size, sha in MODEL_FILES:
            dest = os.path.join(MODELS, repo, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and os.path.getsize(dest) == size and _sha(dest) == sha:
                done += size
                model_state["done"] = done
                continue
            part = dest + ".part"
            with urllib.request.urlopen(HF.format(repo=repo, file=name), timeout=60) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    model_state["done"] = done
            if os.path.getsize(part) != size or _sha(part) != sha:
                os.remove(part)
                raise RuntimeError(f"{repo}/{name} did not match the official Kronos release - download refused")
            os.replace(part, dest)
        model_state["status"] = "ready"
    except Exception as e:
        model_state.update(status="error", error=str(e)[:300])


def _load_model():
    global _model
    if _model is None:
        import kronos_paths as kp
        threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        _model = (kp,) + kp.load(threads=threads, model_dir=MODELS)
    return _model


# ---- candles from the user's own Dhan account ---------------------------------------------------
def _dhan(body):
    req = urllib.request.Request("https://api.dhan.co/v2/charts/intraday", data=json.dumps(body).encode(),
                                 headers={"access-token": _keys["token"], "client-id": _keys["client_id"],
                                          "Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise DhanAuthError("Dhan says your access token is invalid or expired. Open Settings and paste a fresh one.")
            if e.code == 429 or e.code >= 500:
                time.sleep(2 + 3 * attempt)
                continue
            raise RuntimeError(f"Dhan returned HTTP {e.code}")
    raise RuntimeError("Dhan kept refusing the request - try again in a minute")


def _fetch(u, days=DAYS):
    end = dt.datetime.now(IST)
    cur = end - dt.timedelta(days=days)
    rows = {}
    while cur < end:
        to = min(cur + dt.timedelta(days=80), end)
        j = _dhan({"securityId": u["secid"], "exchangeSegment": u["segment"], "instrument": u["instrument"],
                   "interval": "60", "oi": False, "fromDate": cur.strftime("%Y-%m-%d %H:%M:%S"),
                   "toDate": to.strftime("%Y-%m-%d %H:%M:%S")})
        ts = j.get("timestamp") or []
        vols = j.get("volume") or [0] * len(ts)
        for t, o, h, l, c, v in zip(ts, j["open"], j["high"], j["low"], j["close"], vols):
            rows[dt.datetime.fromtimestamp(t, IST).strftime("%Y-%m-%d %H:%M:%S")] = (o, h, l, c, v)
        cur = to
    return rows


def check_keys(client_id, token):
    """A tiny real request (Nifty 50, last few days) to confirm the keys work before saving them."""
    old = dict(_keys)
    set_keys(client_id, token)
    try:
        _fetch(universe()["NIFTY 50"], days=5)
    finally:
        set_keys(old["client_id"], old["token"])


def candles(sym):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, sym.replace(" ", "_") + "_1h.csv")
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > FRESH_SECONDS:
        rows = _fetch(universe()[sym])
        pd.DataFrame([(k, *v) for k, v in sorted(rows.items())],
                     columns=["date", "open", "high", "low", "close", "volume"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df.date.dt.strftime("%H:%M").isin(SLOTS)].sort_values("date").reset_index(drop=True)
    df["volume"] = df.volume.fillna(0)
    df["day"] = df.date.dt.strftime("%Y-%m-%d")
    return df


# ---- forecasts --------------------------------------------------------------------------------
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


def _cpath(sym, asof):
    return os.path.join(CACHE, TAG, sym.replace(" ", "_"), asof.replace(":", "").replace(" ", "_") + ".json")


def _cached(sym, df, i):
    past, fut = df.iloc[i - HIST:i], df.iloc[i:i + PRED]
    p = _cpath(sym, past.date.iloc[-1].strftime("%Y-%m-%d %H:%M"))
    if os.path.exists(p):
        f = json.load(open(p))
        if f["actual_n"] >= min(PRED, len(fut)):
            return f
    return None


def forecast(sym, df, i):
    f = _cached(sym, df, i)
    if f:
        return f
    past, fut = df.iloc[i - HIST:i], df.iloc[i:i + PRED]
    asof = past.date.iloc[-1].strftime("%Y-%m-%d %H:%M")
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
    p = _cpath(sym, asof)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(f, fh, separators=(",", ":"))
    return f


def payload(sym):
    u = universe()[sym]
    df = candles(sym)
    if len(df) < HIST + 1:
        raise ValueError(f"only {len(df)} one-hour candles for {sym} so far; Kronos needs {HIST}")
    org = origins(df)
    start = max(0, org[0] - HIST - SHOW)
    c = df.iloc[start:]
    fcs = [None] * len(org)
    fcs[-1] = dict(forecast(sym, df, org[-1]), origin=org[-1] - start, latest=True)
    for k, i in enumerate(org[:-1]):
        f = _cached(sym, df, i)
        if f:
            fcs[k] = dict(f, origin=i - start, latest=False)
    return dict(symbol=sym, name=u["name"], tf="1H", source="Dhan API (your account)", settings=SETTINGS,
                data_from=c.date.iloc[0].strftime("%Y-%m-%d"), data_to=df.date.iloc[-1].strftime("%Y-%m-%d %H:%M"),
                candles=[[t.strftime("%Y-%m-%d %H:%M"), *map(float, r)] for t, r in
                         zip(c.date, c[["open", "high", "low", "close", "volume"]].values)],
                origins=[i - start for i in org], forecasts=fcs, offset=int(start))


def forecast_at(sym, k):
    df = candles(sym)
    org = origins(df)
    k = max(0, min(len(org) - 1, int(k)))
    start = max(0, org[0] - HIST - SHOW)
    return dict(forecast(sym, df, org[k]), origin=org[k] - start, latest=k == len(org) - 1)
