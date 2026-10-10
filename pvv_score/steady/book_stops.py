"""Same books as book.py, but each position is closed when the intraday low reaches 8% below the entry close (filled at that
level, or at the open if the stock gaps through it) and the slot sits in cash for the rest of the sleeve."""
import numpy as np, pandas as pd
from ..config import CACHE_DIR
from .portfolio import metrics
from .outcomes import STOP
from .book import BANDS
D = CACHE_DIR / "steady"


def stopped_paths(C, O_, L, d, tk, days):
    e = C.loc[d, tk]; lvl = e * (1 - STOP)
    cl = C.loc[days, tk]; lo = L.loc[days, tk]; op = O_.loc[days, tk]
    hit = lo.le(lvl); first = hit.idxmax().where(hit.any())
    prev = pd.concat([e.to_frame().T, cl.iloc[:-1]]).set_axis(days)
    r = cl / prev - 1
    for t in tk:
        if pd.notna(first[t]):
            k = days.get_loc(first[t]); fill = min(op[t].iloc[k], lvl[t])
            r.iloc[k, r.columns.get_loc(t)] = fill / prev[t].iloc[k] - 1
            r.iloc[k + 1:, r.columns.get_loc(t)] = 0.0
    return r.mean(axis=1), first.notna().mean()


def main():
    O = pd.read_parquet(D / "pathrisk_oos.parquet"); O = O[O.date >= "2018-01-01"]
    C = pd.read_parquet(CACHE_DIR / "Close.parquet").sort_index(); L = pd.read_parquet(CACHE_DIR / "Low.parquet").sort_index(); Op = pd.read_parquet(CACHE_DIR / "Open.parquet").sort_index()
    spy = C.SPY.pct_change(fill_method=None); idx = C.index
    pd.set_option("display.width", 240)
    for h in (21, 42, 63):
        for band, (lo, hi) in BANDS.items():
            dates = np.sort(O.dropna(subset=[f"ps_{h}"]).date.unique()); starts = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).min().tolist()
            res = {k: [] for k in ("low risk", "all in band", "high risk")}; hits = {k: [] for k in res}
            for d in starts:
                s = O[(O.date == d) & O.beta_l1_252.between(lo, hi)].dropna(subset=[f"ps_{h}"]); days = idx[idx > d][:h]
                if len(days) == 0: continue
                for k, sel in (("low risk", s.nlargest(20, f"ps_{h}")), ("all in band", s), ("high risk", s.nsmallest(20, f"ps_{h}"))):
                    tk = [t for t in sel.ticker if t in C.columns and pd.notna(C.loc[d, t])]
                    if tk:
                        r, f = stopped_paths(C, Op, L, d, tk, days); res[k].append(r); hits[k].append(f)
            print(f"\n======== {h}-session sleeves with an {STOP:.0%} stop executed, {band}")
            for start, end, lab in [("2018-01-01", "2023-12-31", "2018-2023"), ("2024-01-01", "2026-12-31", "2024 to date")]:
                M = pd.DataFrame({k: metrics(pd.concat(v, axis=1).mean(axis=1), spy, start, end) for k, v in res.items()} | {"SPY": metrics(spy, spy, start, end)}).T
                print(f"-- {lab}"); print(M.round(3).to_string())
            print("   share of positions stopped:", {k: f"{np.mean(v):.1%}" for k, v in hits.items()})


if __name__ == "__main__":
    main()
