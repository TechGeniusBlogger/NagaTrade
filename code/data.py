# Nifty 50 5-minute candles from Dhan (fetched for Video 11 by fetch_intraday_5min.py).
# One trading day = 75 candles, 9:15 to 15:25. The 15:30 stub candle some days carry is dropped,
# and so is any special short session (e.g. a 12-candle Muhurat day).
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "video11_garch", "nifty_5min.csv")
PER_DAY = 75


def load_5min(path=SRC):
    df = pd.read_csv(path)
    df = df[df.time_min <= 370]
    n = df.groupby("date").size()
    df = df[df.date.isin(n[n == PER_DAY].index)].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df.date) + pd.Timedelta(hours=9, minutes=15) + pd.to_timedelta(df.time_min, unit="m")
    return df


def context_for_day(df, day, n_hist):
    """The n_hist candles before `day` (history the model sees) and that day's 75 candles (the truth)."""
    return df[df.date < day].tail(n_hist), df[df.date == day]
