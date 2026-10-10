"""Every month-end 1999-2026: sort on R12 / R1, measure next-month spread.
Gives the calendar-month seasonality control for the Q4 tax-loss test."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, MIN_DV  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
me = pd.DatetimeIndex(D.month_ends)
G = D.growth.ffill().loc[me]
M = G / G.shift(1) - 1                  # monthly stock return
R12 = G / G.shift(12) - 1
F1 = M.shift(-1)                        # next-month return
DV = D.dv63.loc[me]
SPY = D.bench["SPY"].loc[me]
spy_f1 = (SPY.shift(-1) / SPY - 1)

recs = []
for i, t in enumerate(me):
    if i < 12 or i >= len(me) - 1:
        continue
    ok = R12.loc[t].notna() & M.loc[t].notna() & F1.loc[t].notna() & (DV.loc[t] >= MIN_DV)
    r12, r1, f = R12.loc[t][ok], M.loc[t][ok], F1.loc[t][ok]
    q = pd.qcut(r12, 5, labels=False)
    nxt = me[i + 1]
    rec = {
        "form": t, "fwd_month": nxt.month, "fwd_year": nxt.year, "n": int(ok.sum()),
        "neg12": f[r12 < 0].mean(), "pos12": f[r12 >= 0].mean(),
        "neg12_med": f[r12 < 0].median(), "pos12_med": f[r12 >= 0].median(),
        "neg1": f[r1 < 0].mean(), "pos1": f[r1 >= 0].mean(),
        "nn": f[(r12 < 0) & (r1 < 0)].mean(), "pp": f[(r12 >= 0) & (r1 >= 0)].mean(),
        "np": f[(r12 < 0) & (r1 >= 0)].mean(), "pn": f[(r12 >= 0) & (r1 < 0)].mean(),
        "q1": f[q == 0].mean(), "q5": f[q == 4].mean(), "all": f.mean(),
        "spy": spy_f1.loc[t],
        "n_neg12": int((r12 < 0).sum()),
    }
    recs.append(rec)
m = pd.DataFrame(recs)
m["s12"] = m.neg12 - m.pos12
m["s12_med"] = m.neg12_med - m.pos12_med
m["s1"] = m.neg1 - m.pos1
m["s_nn_pp"] = m.nn - m.pp
m["s_q1q5"] = m.q1 - m.q5
m.to_csv(OUT / "monthly_spreads.csv", index=False)
print(m.shape, m.fwd_year.min(), m.fwd_year.max())
