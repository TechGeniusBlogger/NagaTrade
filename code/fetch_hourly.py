"""
Pull 1-hour candles (with volume) from Dhan for the Kronos page's symbols.
Dhan's intraday endpoint serves at most 90 days per call, so we page in 80-day windows.

Source: DhanHQ API. The access token lives in the dhan_token.txt file next to this script (your own token; never share it) and
is never printed.

    python fetch_hourly.py --all            # every symbol in universe.json (build_universe.py)
    python fetch_hourly.py --symbol CIPLA   # one symbol from universe.json
Output: data_hourly/<SYMBOL>_1h.csv   date,open,high,low,close,volume   (date = IST candle start)
"""
import argparse
import csv
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dhan_token.txt")
CLIENT_CODE = "YOUR_DHAN_CLIENT_ID"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OUT = os.path.join(HERE, "data_hourly")


def fname(sym):
    return os.path.join(OUT, sym.replace(" ", "_") + "_1h.csv")


def fetch_chunk(tok, u, frm, to):
    body = json.dumps({"securityId": u["secid"], "exchangeSegment": u["segment"], "instrument": u["instrument"],
                       "interval": "60", "oi": False,
                       "fromDate": frm.strftime("%Y-%m-%d %H:%M:%S"),
                       "toDate": to.strftime("%Y-%m-%d %H:%M:%S")}).encode()
    req = urllib.request.Request("https://api.dhan.co/v2/charts/intraday", data=body,
                                 headers={"access-token": tok, "client-id": CLIENT_CODE,
                                          "Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise SystemExit("Dhan says the token is invalid or expired (401). Update dhan_token.txt.")
            if e.code == 429 or e.code >= 500:
                time.sleep(2 + 3 * attempt)
                continue
            raise
    raise RuntimeError("Dhan kept refusing the request")


def fetch(tok, u, days):
    end = dt.datetime.now(IST)
    cur = end - dt.timedelta(days=days)
    rows = {}
    while cur < end:
        to = min(cur + dt.timedelta(days=80), end)
        j = fetch_chunk(tok, u, cur, to)
        vols = j.get("volume") or [0] * len(j.get("timestamp", []))
        for t, o, h, l, c, v in zip(j.get("timestamp", []), j["open"], j["high"], j["low"], j["close"], vols):
            rows[dt.datetime.fromtimestamp(t, IST).strftime("%Y-%m-%d %H:%M:%S")] = (o, h, l, c, v)
        cur = to
        time.sleep(0.25)
    os.makedirs(OUT, exist_ok=True)
    with open(fname(u["symbol"]), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for k in sorted(rows):
            w.writerow([k, *rows[k]])
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    a = ap.parse_args()
    uni = json.load(open(os.path.join(HERE, "universe.json")))
    todo = uni if a.all else [u for u in uni if u["symbol"] == (a.symbol or "").upper()]
    if not todo:
        raise SystemExit("symbol not in universe.json")
    tok = open(TOKEN_FILE).read().strip()
    t0, fails = time.time(), []
    for k, u in enumerate(todo, 1):
        try:
            n = fetch(tok, u, a.days)
            print(f"[{k}/{len(todo)}] {u['symbol']}: {n} candles", flush=True)
        except SystemExit:
            raise
        except Exception as e:
            fails.append(u["symbol"])
            print(f"[{k}/{len(todo)}] {u['symbol']}: FAILED {str(e)[:120]}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; failed: {fails or 'none'}")


if __name__ == "__main__":
    main()
