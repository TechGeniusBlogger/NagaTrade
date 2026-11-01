# The symbols the Kronos page can search: Nifty 50 and Sensex (indices), plus every NSE stock that
# has futures/options (F&O), taken from Dhan's own scrip master.
#   python build_universe.py  ->  universe.json
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "..", "video11_morning_brief", "data", "dhan_scrip_master.csv")

m = pd.read_csv(MASTER, low_memory=False)
fo = set(m[(m.EXCH_ID == "NSE") & m.INSTRUMENT.isin(["FUTSTK", "OPTSTK"])].UNDERLYING_SYMBOL.dropna())
fo = {s for s in fo if "NSETEST" not in s}  # exchange test instruments, not real stocks
eq = m[(m.EXCH_ID == "NSE") & (m.INSTRUMENT == "EQUITY") & (m.SERIES == "EQ") & m.UNDERLYING_SYMBOL.isin(fo)]
eq = eq.drop_duplicates("UNDERLYING_SYMBOL").sort_values("UNDERLYING_SYMBOL")

uni = [dict(symbol="NIFTY 50", name="Nifty 50 index", secid="13", segment="IDX_I", instrument="INDEX"),
       dict(symbol="SENSEX", name="BSE Sensex index", secid="51", segment="IDX_I", instrument="INDEX")]
uni += [dict(symbol=r.UNDERLYING_SYMBOL, name=r.DISPLAY_NAME, secid=str(int(r.SECURITY_ID)), segment="NSE_EQ",
             instrument="EQUITY") for r in eq.itertuples()]
with open(os.path.join(HERE, "universe.json"), "w") as f:
    json.dump(uni, f, indent=0)
print(len(uni), "symbols ->", os.path.join(HERE, "universe.json"))
