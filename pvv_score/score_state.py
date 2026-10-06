"""Stage 6: identity-neutral live scoring of the FULL universe (every ticker in the Finviz file).

state_score      within-beta-bucket percentile of the walk-forward-validated STATE composite (z-factors only:
                 each factor is the name's deviation from its own 3y norm). Weights fitted on all labelled history.
state_score_pooled   same composite standardised within bucket, then percentile across the whole universe (single sortable number)
level_score      the earlier level-based composite percentile, for comparison (identity-heavy)
ml_score         walk-forward LightGBM percentile (reported, not used)
top_states       the three state readings most responsible for the name's score today (factor: z)
Names without enough history for a 3y norm get NaN with the reason stated; nothing is silently dropped.
"""
import sys
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR
from .data_io import load_universe
from .run_eval import load_research
from .composite import tercile_series, fit_weights, score_rows

EXCLUDE_GROUPS = {"roc"}  # tested, did not improve out-of-sample IC; kept out of the live score


def main():
    import os
    sfx = "_smooth" if os.environ.get("PVV_TARGET", "").startswith("smooth") else ""
    import pyarrow.parquet as pq
    from .config import SAMPLE_START
    uni = load_universe()
    schema_cols = pq.read_schema(CACHE_DIR / "research_long.parquet").names
    reg0 = pd.read_csv(CACHE_DIR / "factor_registry.csv", index_col=0)
    roc0 = set(reg0.index[reg0.group.isin(EXCLUDE_GROUPS)])
    zneed = [c for c in schema_cols if c.startswith("z_") and c[2:] not in roc0]
    keep_level = ["beta_252", "rv20", "mom_12_1", "dist_52w_high", "dist_52w_low", "rs_lead_126", "rs_spy_63", "sharpe_126", "rv20_pct_252",
                  "vol_dry_20_250", "updown_vol_ratio_50", "obv_price_div_63", "dn_up_vol_asym_63", "corr_spy_63", "idio_vol_63", "days_since_20pct_dd",
                  "ceiling_2x_low", "bbw_pct_252", "log_dvol_63", "amihud_21"]
    need = ["date", "ticker", "sector", "eligible", "y_blend_spy", "xs_spy_sharpe_42", "xs_sec_sharpe_42", "smooth_42", "y_smooth"] + keep_level + zneed
    need = [c for c in dict.fromkeys(need) if c in schema_cols]
    full = pq.read_table(CACHE_DIR / "research_long.parquet", columns=need).to_pandas()
    asof = full.date.max()
    last = full[full.date == asof]
    df = full[full.eligible & (full.date >= SAMPLE_START)].reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    df["beta_bucket"] = tercile_series(df)
    # Live weights exclude factor groups that failed the walk-forward test (the ROC family lowered OOS IC; see report 3b).
    reg = pd.read_csv(CACHE_DIR / "factor_registry.csv", index_col=0)
    excluded = set(reg.index[reg.group.isin(EXCLUDE_GROUPS)])
    zcols = [c for c in df.columns if c.startswith("z_") and c[2:] not in excluded]
    lab = df["y_blend_spy"].notna()
    tr = df[lab]

    samp = tr[tr.date >= tr.date.max() - pd.Timedelta(days=3 * 365)]
    samp = samp[samp.date.isin(sorted(samp.date.unique())[::10])]
    corr = samp.groupby("date")[zcols].rank(pct=True).corr()

    today = df[df.date == asof].copy()
    weights = {}
    today["state_raw"] = np.nan
    contrib = {}
    for b in ["low", "mid", "high"]:
        wb = fit_weights(tr[tr.beta_bucket == b], zcols, corr)
        weights[b] = wb
        m = today.beta_bucket == b
        today.loc[m, "state_raw"] = score_rows(today[m], wb).values
        # per-name contributions: weight x (rank - 0.5)
        rk = today.loc[m].groupby("date")[list(wb.index)].rank(pct=True) - 0.5
        contrib[b] = rk * wb.values
    import os
    sfx = "_smooth" if os.environ.get("PVV_TARGET", "").startswith("smooth") else ""
    pd.DataFrame(weights).to_csv(RESULTS_DIR / f"composite_state{sfx}_final_weights.csv")
    today["state_score"] = today.groupby("beta_bucket")["state_raw"].rank(pct=True)
    # raw composites use different weight vectors per bucket, so standardise within bucket before pooling
    zraw = (today["state_raw"] - today.groupby("beta_bucket")["state_raw"].transform("mean")) / today.groupby("beta_bucket")["state_raw"].transform("std")
    today["state_score_pooled"] = zraw.rank(pct=True)

    tops = {}
    for b, c in contrib.items():
        for i, row in c.iterrows():
            top3 = row.abs().sort_values(ascending=False).head(3).index
            tops[i] = "; ".join(f"{f[2:]} z={today.at[i, f]:+.1f}" for f in top3)
    today["top_states"] = pd.Series(tops)

    # ---------------- sector-relative state composite: target = Sharpe vs own sector ETF, ranks within (date, sector),
    # weights per sector group. Answers "beats its sector ETF?", which is a different question from the main score.
    from .composite import SECTOR_GROUP
    import pvv_score.composite as _c
    grp_map = {"defensive": "low", "cyclical": "mid", "growth": "high"}
    trs = tr[["date", "ticker", "sector", "beta_bucket", _c.TARGET if _c.TARGET in tr.columns else "xs_spy_sharpe_42", "xs_sec_sharpe_42"] + zcols].copy()
    trs[zcols] = trs.groupby(["date", "sector"])[zcols].rank(pct=True).astype("float32")
    trs["sgrp"] = trs["sector"].map(SECTOR_GROUP)
    tds = today[["date", "ticker", "sector", "beta_bucket"] + zcols].copy()
    tds[zcols] = tds.groupby(["date", "sector"])[zcols].rank(pct=True).astype("float32")
    tds["sgrp"] = tds["sector"].map(SECTOR_GROUP)
    del full
    import gc; gc.collect()
    old_target = _c.TARGET
    _c.TARGET = "xs_sec_sharpe_42"
    sec_w = {}
    today["vs_sector_raw"] = np.nan
    for g in ["defensive", "cyclical", "growth"]:
        wg = fit_weights(trs[trs.sgrp == g], zcols, corr)
        sec_w[g] = wg
        m = tds.sgrp == g
        today.loc[m[m].index, "vs_sector_raw"] = score_rows(tds[m], wg).values
    _c.TARGET = old_target
    pd.DataFrame(sec_w).to_csv(RESULTS_DIR / f"composite_state_sector{('_smooth' if os.environ.get('PVV_TARGET', '').startswith('smooth') else '')}_final_weights.csv")
    today["vs_sector_score"] = today.groupby("sector")["vs_sector_raw"].rank(pct=True)

    prev = pd.read_csv(RESULTS_DIR / f"current_rankings{sfx}.csv").set_index("ticker")
    today["level_score"] = today.ticker.map(prev["final_score"])
    today["ml_score"] = today.ticker.map(prev["ml_score"])

    # whole universe, including ineligible / short-history names, with reason
    out = uni[["Company", "Sector", "SectorETF"]].copy()
    out.index.name = "ticker"
    hist = full.groupby("ticker").size()
    last = last.set_index("ticker")
    out["sessions_of_history"] = hist.reindex(out.index).fillna(0).astype(int)
    out["eligible_today"] = last["eligible"].reindex(out.index).fillna(False).astype(bool)
    t = today.set_index("ticker")
    for c in ["beta_bucket", "state_score", "state_score_pooled", "level_score", "ml_score", "top_states", "vs_sector_score",
              "z_mom_12_1", "z_rs_lead_126", "z_dist_52w_high", "z_rv20_pct_252", "z_vol_dry_20_250", "z_obv_price_div_63",
              "z_updown_vol_ratio_50", "z_bbw_pct_252", "z_dn_up_vol_asym_63", "z_corr_spy_63", "z_idio_vol_63",
              "beta_252", "rv20", "mom_12_1", "dist_52w_high"]:
        out[c] = t[c].reindex(out.index)
    def reason(r):
        if pd.notna(r.state_score):
            return ""
        if r.sessions_of_history < 252:
            return f"insufficient history ({r.sessions_of_history} sessions; 252 needed)"
        if not r.eligible_today:
            return "fails liquidity / price eligibility today or no completed session (e.g. CTVA unadjusted spin-off)"
        if pd.isna(r.beta_bucket):
            return "no 252d beta (recent re-listing)"
        return "state norm needs 252 sessions of factor history"
    # recommended sort keys: identity-neutral state first, persistence layer as confirmation
    out["avg_score"] = (out["state_score_pooled"] + out["level_score"]) / 2
    out["both_agree"] = (out["state_score_pooled"] >= 0.8) & (out["level_score"] >= 0.8)
    def read(r):
        if pd.isna(r.avg_score):
            return ""
        spy = r.avg_score >= 0.8
        sec = pd.notna(r.vs_sector_score) and r.vs_sector_score >= 0.67
        return "SPY & sector" if (spy and sec) else "SPY only" if spy else "sector only" if sec else ""
    out["read"] = out.apply(read, axis=1)
    out["note"] = out.apply(reason, axis=1)
    out = out.sort_values("state_score_pooled", ascending=False)
    out.insert(0, "asof", asof.date())
    out.to_csv(RESULTS_DIR / f"universe_scores{sfx}.csv")

    pd.set_option("display.width", 300, "display.max_columns", 30, "display.max_rows", 60)
    show = ["Sector", "beta_bucket", "state_score", "state_score_pooled", "level_score", "ml_score", "z_mom_12_1", "z_rs_lead_126",
            "z_vol_dry_20_250", "z_rv20_pct_252", "z_obv_price_div_63", "top_states"]
    print(f"as of {asof.date()}: {out.state_score.notna().sum()} scored, {out.state_score.isna().sum()} unscored (see note)")
    print(out[show].head(40).round(2).to_string())
    print("\nDiagnostic: identity vs state for the names raised")
    print(out.loc[[x for x in ["DELL", "NVDA", "LLY", "PGR", "BRK-B", "AMD", "PANW"] if x in out.index], ["beta_bucket", "level_score", "state_score", "state_score_pooled", "top_states"]].round(2).to_string())
    print("\nrank corr level vs state:", round(out[["level_score", "state_score"]].corr(method="spearman").iloc[0, 1], 2))
    print("\nunscored:\n", out[out.state_score.isna()][["sessions_of_history", "note"]].to_string())


if __name__ == "__main__":
    main()
