# Fills the forecast cache for every symbol, so searches on the page are instant. Run it after
# fetch_hourly.py --all (e.g. every evening after the close).
#   python precompute_all.py            # the latest forecast for every symbol (~5 s each)
#   python precompute_all.py --past     # plus the last 10 days' forecasts (for "Reveal")
import argparse
import json
import os
import time

import kronos_service as ks

ap = argparse.ArgumentParser()
ap.add_argument("--past", action="store_true")
a = ap.parse_args()

t0, done, fails = time.time(), 0, []
syms = list(ks.universe())
for n, sym in enumerate(syms, 1):
    try:
        df = ks.candles(sym)
        org = ks.origins(df)
        for i in (org if a.past else org[-1:]):
            ks.forecast(sym, df, i)
            done += 1
        print(f"[{n}/{len(syms)}] {sym} ok  ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        fails.append(sym)
        print(f"[{n}/{len(syms)}] {sym} FAILED: {str(e)[:120]}", flush=True)
print(f"{done} forecasts in {time.time() - t0:.0f}s; failed: {fails or 'none'}")
