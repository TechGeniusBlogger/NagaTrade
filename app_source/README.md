# Kronos Forecast (code file 2 of 5)

Kronos is an AI model that reads candles and predicts the next candle, the way ChatGPT reads words
and predicts the next word. It was built at Tsinghua University, accepted at AAAI 2026, and released
free by its authors (MIT licence). This app runs it on **Nifty 50, Sensex and all 210 NSE F&O stocks**,
using one-hour candles from **your own Dhan account**.

For the stock you pick, Kronos reads the last 360 one-hour candles (about 60 trading days) and
draws the next 24, **30 times over**. The chart shows:

- the **orange line**: the average of the 30 runs
- the **shaded band**: from the lowest run to the highest
- **direction of the runs**: how many of the 30 ended higher or lower (e.g. 11 up / 19 down = 37% / 63%)
- **volatility amplification**: how many runs swing harder than the last 24 real candles did
- the **volume** Kronos expects
- **Reveal** (key `R`, after stepping back with `←`): what really happened after an earlier forecast

## Run it

**Windows:** unzip `KronosForecast-windows.zip`, open the folder, double-click `KronosForecast.exe`.
Your browser opens at `http://localhost:8767`. Close the black window to stop the app.

**Mac:** open the `source` folder and double-click `run_mac.command` (first time: right-click →
Open, since it isn't from the App Store). **Linux:** `bash run_linux.sh` in the `source` folder.
The first run sets up Python + PyTorch once (a few minutes). Python 3.10+ is needed.

### First run (one time)
1. **Agreement.** Read and accept it. Nothing works until you do.
2. **Your Dhan keys.** Your client id and access token, from Dhan's web app (Profile →
   DhanHQ Trading APIs). The app checks them with Dhan before saving. They are stored only on this
   computer and sent only to `api.dhan.co`. The app never places an order. Dhan tokens expire:
   paste a fresh one any time from **Settings** at the top right.
3. **The AI model.** About 407 MB, downloaded once from its authors' official release on Hugging
   Face (NeoQuasar/Kronos-base). Every file is checked against the official release (size and
   SHA-256) before it is used.

Your settings, candles and saved forecasts live in `%APPDATA%\KronosForecast` (Windows),
`~/Library/Application Support/KronosForecast` (Mac) or `~/.config/KronosForecast` (Linux).

## Good to know
- A new stock takes about **5 seconds** on a normal laptop CPU (no graphics card needed); a stock
  you've already opened is instant.
- **Six full one-hour candles a day, 9:15 to 15:15.** Since August 2026, Dhan's intraday feed for
  stocks does not include the last 15 minutes of the day (15:15-15:30) at any interval, so that
  short stub is left out on every day for every symbol.
- **Read the direction honestly.** In our own tests, Kronos's up/down split was close to a coin
  flip, and the real price often moved outside its range. Use it as a first look at the range, and
  press Reveal on past forecasts to judge it yourself.

## Source
`source/` has everything: `kronos_app.py` (the app), `service.py` (Dhan candles, model download,
forecasts), `kronos_paths.py` (Kronos with all 30 runs kept, the stock Kronos code averages them
away, plus an exact faster loop: same seed, same answer, about 20x faster), `model/` (the Kronos
model code, unchanged, MIT licence), `assets/` (the chart page, setup page, agreement, symbol list).

Build the app yourself on any OS (PyInstaller builds for the machine it runs on):
`pip install -r requirements.txt pyinstaller` then `python build.py` → `dist/KronosForecast/`.

Checked: the Windows `.exe` gives exactly the same 30 runs as the Python version for the same stock
and time (0.0 price points difference).

## Not investment advice
Educational tool built on a published research model. Not investment advice, not a recommendation,
not a forecast you should trade on. The up/down split is only the share of the model's own runs.
Past results do not guarantee future results. Investments in securities markets are subject to
market risks; read all related documents carefully before investing.
