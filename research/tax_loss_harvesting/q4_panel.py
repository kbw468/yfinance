"""Build per-stock x per-year panel: Sep-30 signals + forward window returns."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for y in range(1999, 2026):
    t0 = D.me(y, 9)
    oct_, nov, dec = D.me(y, 10), D.me(y, 11), D.me(y, 12)
    dec15 = D.td_on_or_before(f"{y}-12-15")
    dec20 = D.td_on_or_before(f"{y}-12-20")
    jan = D.me(y + 1, 1)
    feb = D.me(y + 1, 2)
    h = {d: D.td_on_or_before(d) for d in (f"{y}-10-15", f"{y}-11-15", f"{y + 1}-01-15")}
    sig = D.signals(t0)
    cols = sig.index
    w = {
        "Oct": (t0, oct_), "Nov": (oct_, nov), "Dec": (nov, dec),
        "Dec1_15": (nov, dec15), "Dec16_31": (dec15, dec), "Dec20_31": (dec20, dec),
        "Q4": (t0, dec), "Jan": (dec, jan), "Dec20_Jan": (dec20, jan),
        "Q4_Jan": (t0, jan), "Feb": (jan, feb),
        "Oct1_15": (t0, h[f"{y}-10-15"]), "Oct16_31": (h[f"{y}-10-15"], oct_),
        "Nov1_15": (oct_, h[f"{y}-11-15"]), "Nov16_30": (h[f"{y}-11-15"], nov),
        "Jan1_15": (dec, h[f"{y + 1}-01-15"]), "Jan16_31": (h[f"{y + 1}-01-15"], jan),
    }
    df = sig.copy()
    for k, (a, b) in w.items():
        df[k] = D.window_ret(a, b, cols)
    # benchmark windows
    for k, (a, b) in w.items():
        df["SPY_" + k] = D.bench["SPY"].loc[b] / D.bench["SPY"].loc[a] - 1
    df["year"] = y
    df.index.name = "ticker"
    rows.append(df.reset_index())
    print(y, len(df), f"neg12={int((df.R12 < 0).sum())}", flush=True)

panel = pd.concat(rows, ignore_index=True)
# cross-sectional ranks within year
panel["R12_q"] = panel.groupby("year")["R12"].transform(lambda s: pd.qcut(s, 5, labels=False) + 1)
panel["DV_t"] = panel.groupby("year")["DV"].transform(lambda s: pd.qcut(s, 3, labels=False) + 1)
panel.to_parquet(OUT / "q4_panel.parquet")
print(panel.shape)
