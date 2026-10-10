"""Robustness: liquidity floor, point-in-time re-sort at Oct 15, ex-2020, medians."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
rows = []
for floor in (1e6, 5e6, 20e6):
    for y in range(1999, 2026):
        t0 = D.me(y, 9)
        o15 = D.td_on_or_before(f"{y}-10-15")
        d15 = D.td_on_or_before(f"{y}-12-15")
        dec, jan = D.me(y, 12), D.me(y + 1, 1)
        sig = D.signals(t0)
        sig = sig[sig.DV >= floor]
        # point-in-time re-sort at Oct 15 (R12 Oct15->Oct15, R1 Sep15->Oct15)
        r12o = D.window_ret(D.td_on_or_before(f"{y - 1}-10-15"), o15)
        r1o = D.window_ret(D.td_on_or_before(f"{y}-09-15"), o15)
        ok = r12o.notna() & r1o.notna() & (D.dv63.loc[o15] >= floor)
        r12o, r1o = r12o[ok], r1o[ok]
        sets = {
            "sep30": (sig.index[(sig.R12 < 0) & (sig.R1 < 0)], sig.index[(sig.R12 >= 0) & (sig.R1 >= 0)],
                      sig.index[sig.R12 < 0], sig.index[sig.R12 >= 0]),
            "oct15": (r12o.index[(r12o < 0) & (r1o < 0)], r12o.index[(r12o >= 0) & (r1o >= 0)],
                      r12o.index[r12o < 0], r12o.index[r12o >= 0]),
        }
        for sort, (nn, pp, n, p) in sets.items():
            for wn, (a, b) in {"Oct1_15": (t0, o15), "Oct15_Dec15": (o15, d15), "Oct15_Dec31": (o15, dec),
                               "Jan": (dec, jan)}.items():
                if sort == "oct15" and wn == "Oct1_15":
                    continue
                r = D.window_ret(a, b)
                rows.append({"floor": floor, "sort": sort, "year": y, "window": wn,
                             "nn_pp": r[nn].mean() - r[pp].mean(), "n_p": r[n].mean() - r[p].mean(),
                             "nn_pp_med": r[nn].median() - r[pp].median(), "n_p_med": r[n].median() - r[p].median(),
                             "n_nn": len(nn), "n_pp": len(pp)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "robustness_yearly.csv", index=False)
ERAS = {"1999-2007": (1999, 2007), "2008-2016": (2008, 2016), "2017-2025": (2017, 2025),
        "2017-2025 ex2020": (2017, 2025)}
out = []
for (fl, so, wn), d in df.groupby(["floor", "sort", "window"]):
    for e, (a, b) in ERAS.items():
        dd = d[(d.year >= a) & (d.year <= b)]
        if "ex2020" in e:
            dd = dd[dd.year != 2020]
        for col in ["nn_pp", "n_p", "nn_pp_med", "n_p_med"]:
            out.append({"floor": fl, "sort": so, "window": wn, "era": e, "metric": col,
                        "mean": dd[col].mean(), "t": tstat(dd[col]), "hit": (dd[col] > 0).mean(),
                        "avg_n_nn": dd.n_nn.mean()})
o = pd.DataFrame(out)
o.to_csv(OUT / "robustness_summary.csv", index=False)
pd.set_option("display.width", 250)
v = o[o.era.str.startswith("2017")].copy()
v["cell"] = v.apply(lambda r: f"{r['mean'] * 100:+.2f} t{r.t:+.1f} h{r.hit:.0%}", axis=1)
print(v.pivot_table(index=["window", "sort", "floor"], columns=["era", "metric"], values="cell", aggfunc="first").to_string())
