"""Stage 2: conditional factor efficacy.

(a) IC within trailing-beta terciles (beta_252, re-cut every date)
(b) IC within realised-vol terciles (rv20)
(c) IC within sector groups (defensive / cyclical / growth) and per sector
(d) IC of SECTOR-NEUTRAL factor ranks (factor demeaned within sector-date) vs sector-excess Sharpe
(e) factor correlation clusters (for de-duplication in the composite)
Primary target for this stage: 42d excess Sharpe vs SPY (middle of the 21-63 window); 21/63 reported too.
"""
import sys
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR, HORIZONS, RECENT_START
from .evaluate import daily_ic, summarize_ic
from .run_eval import load_research, load_registry

FACTORS = load_registry()
SECTOR_GROUP = {
    "Utilities": "defensive", "Consumer Defensive": "defensive", "Healthcare": "defensive", "Real Estate": "defensive",
    "Industrials": "cyclical", "Financial": "cyclical", "Basic Materials": "cyclical", "Energy": "cyclical", "Consumer Cyclical": "cyclical",
    "Technology": "growth", "Communication Services": "growth",
}


def tercile(df, col):
    return df.groupby("date")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]) if s.notna().sum() >= 60 else pd.Series(index=s.index, dtype=object))


def ic_table(df, factors, target, h, label, min_n=50):
    ic = daily_ic(df, factors, target)
    full = summarize_ic(ic, h)[["ic", "t_nw"]].add_prefix(f"{label}|full|")
    rec = summarize_ic(ic, h, RECENT_START)[["ic", "t_nw"]].add_prefix(f"{label}|recent|")
    return pd.concat([full, rec], axis=1)


def main():
    df = load_research()
    factors = list(FACTORS)
    df["beta_bucket"] = tercile(df, "beta_252")
    df["vol_bucket"] = tercile(df, "rv20")
    df["sector_group"] = df["sector"].map(SECTOR_GROUP)

    out = {}
    for h in (21, 42, 63):
        tgt = f"xs_spy_sharpe_{h}"
        tabs = []
        for b in ["low", "mid", "high"]:
            tabs.append(ic_table(df[df.beta_bucket == b], factors, tgt, h, f"beta_{b}"))
        for b in ["low", "mid", "high"]:
            tabs.append(ic_table(df[df.vol_bucket == b], factors, tgt, h, f"vol_{b}"))
        for g in ["defensive", "cyclical", "growth"]:
            tabs.append(ic_table(df[df.sector_group == g], factors, tgt, h, f"grp_{g}"))
        out[h] = pd.concat(tabs, axis=1)
        out[h].to_csv(RESULTS_DIR / f"conditional_ic_{h}.csv")
        print(f"h={h} done", file=sys.stderr)

    # per-sector (42d) -- small cross-sections, treat as indicative
    tabs = []
    for s in sorted(df.sector.unique()):
        sub = df[df.sector == s]
        if sub.groupby("date").size().median() < 15:
            continue
        ic = daily_ic(sub, factors, "xs_spy_sharpe_42", min_n=15)
        ic = ic.dropna(how="all")
        rec = summarize_ic(ic, 42, RECENT_START)[["ic", "t_nw"]].add_prefix(f"{s}|recent|")
        tabs.append(rec)
    per_sector = pd.concat(tabs, axis=1)
    per_sector.to_csv(RESULTS_DIR / "conditional_ic_42_per_sector.csv")

    # sector-neutral factors vs sector-excess Sharpe
    neut = df.copy()
    for f in factors:
        neut[f] = df.groupby(["date", "sector"])[f].rank(pct=True)
        neut[f] = neut[f] - neut.groupby(["date", "sector"])[f].transform("mean")
    tabs = []
    for h in (21, 42, 63):
        tabs.append(ic_table(neut, factors, f"xs_sec_sharpe_{h}", h, f"secneut_sec{h}"))
        tabs.append(ic_table(neut, factors, f"xs_spy_sharpe_{h}", h, f"secneut_spy{h}"))
    sn = pd.concat(tabs, axis=1)
    sn.to_csv(RESULTS_DIR / "sector_neutral_ic.csv")

    # factor correlation clusters (Spearman, recent window, on a 1-in-5 day sample for speed)
    rec = df[df.date >= RECENT_START]
    rec = rec[rec.date.isin(sorted(rec.date.unique())[::5])]
    rk = rec.groupby("date")[factors].rank(pct=True)
    corr = rk.corr()
    corr.to_csv(RESULTS_DIR / "factor_rank_corr_recent.csv")

    # ---------- console summary: where does each factor work? (recent window, 42d)
    t42 = out[42]
    cols = [c for c in t42.columns if c.endswith("|recent|t_nw")]
    view = t42[cols].copy()
    view.columns = [c.split("|")[0] for c in cols]
    view["secneut_t42"] = sn["secneut_sec42|recent|t_nw"]
    view["max_abs_t"] = view.abs().max(axis=1)
    view = view.sort_values("max_abs_t", ascending=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print("\nRecent-window (2022-10+) NW t-stats of 42d IC by bucket:")
    print(view.round(2).head(45).to_string())
    print("\nPer-sector recent 42d t-stats, top |t| per sector:")
    for c in [c for c in per_sector.columns if c.endswith("t_nw")]:
        s = per_sector[c].dropna()
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(5)
        print(f"  {c.split('|')[0]:24s}", ", ".join(f"{k}={v:+.2f}" for k, v in top.items()))


if __name__ == "__main__":
    main()
