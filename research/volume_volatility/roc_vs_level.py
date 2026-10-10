"""Is VV_ROC just VV level in disguise? Rank correlation, plus ROC Q5-Q1 averaged within VV terciles.

Usage: python roc_vs_level.py <bars.parquet>
"""
import sys
import numpy as np
import pandas as pd
import backtest as bt
import sweep as sw

sw.build_base(pd.read_parquet(sys.argv[1]))
G = sw.G
rk = lambda x: x.replace([np.inf, -np.inf], np.nan).rolling(252, min_periods=126).rank(pct=True).values
for w, lag in [(20, 10), (60, 10), (60, 20)]:
    vv = G["dlv"].rolling(w, min_periods=w - 2).std()
    pv, pr = rk(vv), rk(vv / vv.shift(lag) - 1)
    ok = ~np.isnan(pv) & ~np.isnan(pr)
    c = pd.Series(pv[ok][::7]).corr(pd.Series(pr[ok][::7]), method="spearman")
    print(f"\nVV{w} / ROC lag {lag}: rank corr(VV level, ROC) = {c:+.2f}")
    tv, q = np.ceil(pv * 3), bt.quintile(pr)
    for dt, day in G["day"].items():
        cells = []
        for h in bt.HORIZONS:
            parts = []
            for a in (1, 2, 3):
                d5, _ = bt.daily_bucket_mean(G["rax"][h], day & (tv == a) & (q == 5))
                d1, _ = bt.daily_bucket_mean(G["rax"][h], day & (tv == a) & (q == 1))
                parts.append(d5 - d1)
            avg = np.nanmean(np.vstack(parts), axis=0)  # ROC spread averaged across VV terciles
            t, _ = bt.newey_west_t(avg, h)
            cells.append(f"{h}d {np.nanmean(avg):+.3f} ({t:+.1f})")
        print(f"  ROC Q5-Q1 holding VV tercile fixed | {dt}: " + "  ".join(cells))
