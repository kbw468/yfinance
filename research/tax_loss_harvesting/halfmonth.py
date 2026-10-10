"""Half-month spread profile Oct-Jan (Sep-30 sort), SPY-beta hedged, by era."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
p = pd.read_parquet(OUT / "q4_panel.parquet")
HW = ["Oct1_15", "Oct16_31", "Nov1_15", "Nov16_30", "Dec1_15", "Dec16_31", "Jan1_15", "Jan16_31"]
PAIRS = {
    "NEG1Y-POS1Y": (p.R12 < 0, p.R12 >= 0),
    "NEG1M-POS1M": (p.R1 < 0, p.R1 >= 0),
    "NN-PP": ((p.R12 < 0) & (p.R1 < 0), (p.R12 >= 0) & (p.R1 >= 0)),
    "Q1-Q5": (p.R12_q == 1, p.R12_q == 5),
}
ERAS = {"1999-2007": (1999, 2007), "2008-2016": (2008, 2016), "2017-2025": (2017, 2025), "ALL": (1999, 2025)}
spy = {}
rows = []
for name, (a, b) in PAIRS.items():
    for w in HW:
        la = p[a].groupby("year")[w].mean()
        lb = p[b].groupby("year")[w].mean()
        ma = p[a].groupby("year")[w].median()
        mb = p[b].groupby("year")[w].median()
        s = (la - lb)
        sm_ = (ma - mb)
        bench = p.groupby("year")["SPY_" + w if "SPY_" + w in p else "SPY_Q4"].first()
        rows.append(pd.DataFrame({"pair": name, "window": w, "year": s.index, "spread": s.values,
                                  "spread_med": sm_.values}))
yr = pd.concat(rows)
yr.to_csv(OUT / "halfmonth_yearly.csv", index=False)


def t(x):
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


out = []
for (pair, w), d in yr.groupby(["pair", "window"]):
    for e, (y0, y1) in ERAS.items():
        dd = d[(d.year >= y0) & (d.year <= y1)]
        out.append({"pair": pair, "window": w, "era": e, "mean": dd.spread.mean(), "t": t(dd.spread),
                    "hit": (dd.spread > 0).mean(), "median_spread": dd.spread_med.mean()})
o = pd.DataFrame(out)
o["window"] = pd.Categorical(o.window, HW, ordered=True)
o.sort_values(["pair", "era", "window"]).to_csv(OUT / "halfmonth_profile.csv", index=False)
print(o.pivot_table(index=["pair", "era"], columns="window", values="mean", observed=True).mul(100).round(2).to_string())
print(o.pivot_table(index=["pair", "era"], columns="window", values="t", observed=True).round(1).to_string())
print(o.pivot_table(index=["pair", "era"], columns="window", values="hit", observed=True).round(2).to_string())
