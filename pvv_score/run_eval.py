"""Stage 1: single-factor evaluation across horizons, benchmarks, full vs recent windows."""
import sys
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR, HORIZONS, RECENT_START, SAMPLE_START
from .evaluate import evaluate_factors, decile_spread, yearly_ic, factor_autocorr, bh_fdr

RESULTS_DIR.mkdir(exist_ok=True)


def load_registry() -> dict:
    """FACTORS registry is populated at compute time; build.py persists it."""
    reg = pd.read_csv(CACHE_DIR / "factor_registry.csv", index_col=0)
    return {f: (r.group, int(r.prior_sign), r.desc) for f, r in reg.iterrows()}


FACTORS = load_registry()


def load_research() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / "research_long.parquet")
    df = df[df.eligible & (df.date >= SAMPLE_START)].copy()
    return df


def main():
    df = load_research()
    factors = [f for f in FACTORS if f in df.columns]
    print(f"rows={len(df):,} dates={df.date.nunique()} tickers={df.ticker.nunique()} factors={len(factors)}", file=sys.stderr)

    targets = {}
    for h in HORIZONS:
        targets[f"spy_sh{h}"] = (f"xs_spy_sharpe_{h}", h)
        targets[f"sec_sh{h}"] = (f"xs_sec_sharpe_{h}", h)
        targets[f"spy_so{h}"] = (f"xs_spy_sortino_{h}", h)
    summary, ics = evaluate_factors(df, factors, targets)

    # consolidated view: average IC across horizons for SPY-excess Sharpe, full & recent
    cons = pd.DataFrame(index=factors)
    cons["group"] = [FACTORS[f][0] for f in factors]
    cons["prior"] = [FACTORS[f][1] for f in factors]
    for win in ["full", "recent"]:
        cons[f"ic_spy_{win}"] = summary[[f"spy_sh{h}|{win}|ic" for h in HORIZONS]].mean(axis=1)
        cons[f"t_spy_{win}"] = summary[[f"spy_sh{h}|{win}|t_nw" for h in HORIZONS]].mean(axis=1)
        cons[f"ic_sec_{win}"] = summary[[f"sec_sh{h}|{win}|ic" for h in HORIZONS]].mean(axis=1)
        cons[f"t_sec_{win}"] = summary[[f"sec_sh{h}|{win}|t_nw" for h in HORIZONS]].mean(axis=1)
        cons[f"ic_sortino_{win}"] = summary[[f"spy_so{h}|{win}|ic" for h in HORIZONS]].mean(axis=1)
    for h in HORIZONS:
        cons[f"ic_spy_recent_{h}"] = summary[f"spy_sh{h}|recent|ic"]
        cons[f"t_spy_recent_{h}"] = summary[f"spy_sh{h}|recent|t_nw"]
        cons[f"p_spy_recent_{h}"] = summary[f"spy_sh{h}|recent|p"]
    # sign agreement full vs recent, and FDR on the recent 42d test
    cons["sign_stable"] = np.sign(cons.ic_spy_full) == np.sign(cons.ic_spy_recent)
    cons["fdr_recent_42"] = bh_fdr(cons["p_spy_recent_42"]).reindex(cons.index)
    cons["fdr_full_42"] = bh_fdr(summary["spy_sh42|full|p"]).reindex(cons.index)
    cons["autocorr_21"] = factor_autocorr(df, factors, 21)

    # yearly IC (42d SPY-excess Sharpe) for regime stability
    yr = yearly_ic(ics["spy_sh42"])
    cons["ic42_pos_years"] = (np.sign(yr) == np.sign(cons.ic_spy_full.reindex(yr.columns))).mean().reindex(cons.index)

    # decile spreads (42d excess Sharpe) recent window
    rec = df[df.date >= RECENT_START]
    spreads = {}
    for f in factors:
        sp, _ = decile_spread(rec, f, "xs_spy_sharpe_42")
        spreads[f] = sp.mean()
    cons["d10_d1_spread_recent_42"] = pd.Series(spreads)

    cons = cons.sort_values("t_spy_recent", key=np.abs, ascending=False)
    cons.to_csv(RESULTS_DIR / "factor_ic_summary.csv")
    summary.to_csv(RESULTS_DIR / "factor_ic_detail.csv")
    yr.T.to_csv(RESULTS_DIR / "factor_ic42_by_year.csv")
    for k, v in ics.items():
        v.to_parquet(CACHE_DIR / f"ic_{k}.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(cons[["group", "prior", "ic_spy_full", "t_spy_full", "ic_spy_recent", "t_spy_recent", "ic_sec_recent",
                "t_sec_recent", "sign_stable", "fdr_recent_42", "ic42_pos_years", "d10_d1_spread_recent_42", "autocorr_21"]].round(3).to_string())


if __name__ == "__main__":
    main()
