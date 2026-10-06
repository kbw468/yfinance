"""Stage 6: identity-neutral live scoring of the FULL universe (every ticker in the Finviz file).

state_score      within-beta-bucket percentile of the walk-forward-validated STATE composite (z-factors only:
                 each factor is the name's deviation from its own 3y norm). Weights fitted on all labelled history.
state_score_pooled   same composite, percentile across the whole universe (for a single sortable number)
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


def main():
    full = pd.read_parquet(CACHE_DIR / "research_long.parquet")
    uni = load_universe()
    asof = full.date.max()
    df = full[full.eligible].reset_index(drop=True)
    df["beta_bucket"] = tercile_series(df)
    zcols = [c for c in df.columns if c.startswith("z_")]
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
    pd.DataFrame(weights).to_csv(RESULTS_DIR / "composite_state_final_weights.csv")
    today["state_score"] = today.groupby("beta_bucket")["state_raw"].rank(pct=True)
    today["state_score_pooled"] = today["state_raw"].rank(pct=True)

    tops = {}
    for b, c in contrib.items():
        for i, row in c.iterrows():
            top3 = row.abs().sort_values(ascending=False).head(3).index
            tops[i] = "; ".join(f"{f[2:]} z={today.at[i, f]:+.1f}" for f in top3)
    today["top_states"] = pd.Series(tops)

    prev = pd.read_csv(RESULTS_DIR / "current_rankings.csv").set_index("ticker")
    today["level_score"] = today.ticker.map(prev["final_score"])
    today["ml_score"] = today.ticker.map(prev["ml_score"])

    # whole universe, including ineligible / short-history names, with reason
    out = uni[["Company", "Sector", "SectorETF"]].copy()
    out.index.name = "ticker"
    hist = full.groupby("ticker").size()
    last = full[full.date == asof].set_index("ticker")
    out["sessions_of_history"] = hist.reindex(out.index).fillna(0).astype(int)
    out["eligible_today"] = last["eligible"].reindex(out.index).fillna(False).astype(bool)
    t = today.set_index("ticker")
    for c in ["beta_bucket", "state_score", "state_score_pooled", "level_score", "ml_score", "top_states",
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
    out["note"] = out.apply(reason, axis=1)
    out = out.sort_values("state_score_pooled", ascending=False)
    out.insert(0, "asof", asof.date())
    out.to_csv(RESULTS_DIR / "universe_scores.csv")

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
