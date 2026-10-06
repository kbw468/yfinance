"""Stage 4: score the universe as of the latest completed session.

Produces results/current_rankings.csv with, per eligible ticker:
  ml_score        - case-study model trained on ALL history to date (time-decay weighted), predicted
                    percentile of forward 21-63d excess Sharpe vs SPY
  comp_score      - transparent beta-bucket composite (weights fitted on all history to date)
  final_score     - within-beta-bucket composite percentile (the only layer with OOS support); ML reported alongside
  key factor readings, sector, beta bucket
  analogs         - the historical (date, ticker) cases nearest in factor space and what their
                    realised 42d excess Sharpe vs SPY was (median, hit rate), i.e. the case-study evidence
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors

from .config import CACHE_DIR, RESULTS_DIR, ML_HALFLIFE_DAYS
from .run_eval import load_research, load_registry
from .model import make_design, decay_weights, PARAMS, N_ROUNDS, EMBARGO, TRAIN_STRIDE
from .composite import tercile_series, fit_weights, score_rows

FACTORS = load_registry()
ANALOG_FACTORS = ["mom_12_1", "dist_52w_high", "dist_52w_low", "rs_lead_126", "rs_spy_126", "sharpe_126", "corr_spy_63",
                  "beta_252", "rv20_pct_252", "vol_dry_20_250", "obv_price_div_63", "dn_up_vol_asym_63", "skew_63",
                  "days_since_20pct_dd", "base_depth_126", "gap_share_21", "up_capture_63", "idio_vol_63"]
K_ANALOGS = 50


def main():
    full = pd.read_parquet(CACHE_DIR / "research_long.parquet")
    df = full[full.eligible].reset_index(drop=True)
    df["beta_bucket"] = tercile_series(df)
    asof = df.date.max()
    today = df[df.date == asof].copy()
    print(f"as of {asof.date()}: {len(today)} eligible names", file=sys.stderr)

    # ---------------- ML: train on everything with a label (labels need h days of future -> last labelled date is asof-63)
    X, cols = make_design(df)
    bdays = pd.DatetimeIndex(sorted(df.date.unique()))
    lab = df["y_blend_spy"].notna()
    train_end = df.loc[lab, "date"].max()
    keep_dates = bdays[bdays <= train_end][::TRAIN_STRIDE]
    tr_idx = df.index[lab & df.date.isin(keep_dates)]
    w = decay_weights(df.loc[tr_idx, "date"], train_end, bdays)
    model = lgb.train(PARAMS, lgb.Dataset(X.loc[tr_idx], label=df.loc[tr_idx, "y_blend_spy"], weight=w), num_boost_round=N_ROUNDS)
    today["ml_score"] = model.predict(X.loc[today.index])
    imp = pd.Series(model.feature_importance("gain"), index=cols)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(RESULTS_DIR / "ml_final_feature_importance.csv")

    # ---------------- composite: weights per beta bucket on all labelled history
    factors = list(FACTORS)
    tr = df[lab]
    samp = tr[tr.date >= tr.date.max() - pd.Timedelta(days=3 * 365)]
    samp = samp[samp.date.isin(sorted(samp.date.unique())[::10])]
    corr = samp.groupby("date")[factors].rank(pct=True).corr()
    weights = {}
    today["comp_score"] = np.nan
    for b in ["low", "mid", "high"]:
        wb = fit_weights(tr[tr.beta_bucket == b], factors, corr)
        weights[b] = wb
        m = today.beta_bucket == b
        today.loc[m, "comp_score"] = score_rows(today[m], wb).values
    pd.DataFrame({b: w for b, w in weights.items()}).to_csv(RESULTS_DIR / "composite_final_weights.csv")

    today["ml_rank"] = today.ml_score.rank(pct=True)
    today["comp_rank"] = today.groupby("beta_bucket").comp_score.rank(pct=True)
    # final score = within-beta-bucket composite percentile. The ML model showed ~zero OOS IC in the recent
    # window (see ml_oos_evaluation.csv) so it is reported as a column, not blended into the ranking.
    today["final_score"] = today.comp_rank.fillna(today.ml_rank)

    # ---------------- analogs: nearest historical cases in rank-factor space, with realised outcomes
    hist = tr[tr.date <= train_end]
    hist = hist[hist.date.isin(bdays[bdays <= train_end][::5])]  # thin to reduce same-ticker-adjacent-day clones
    rk_hist = hist.groupby("date")[ANALOG_FACTORS].rank(pct=True).fillna(0.5).values
    rk_now = today.groupby("date")[ANALOG_FACTORS].rank(pct=True).fillna(0.5).values
    nn = NearestNeighbors(n_neighbors=K_ANALOGS * 3).fit(rk_hist)
    dist, idx = nn.kneighbors(rk_now)
    a_med, a_hit, a_n, a_list = [], [], [], []
    for i in range(len(today)):
        cand = hist.iloc[idx[i]]
        # at most 3 cases per ticker so one name's history cannot dominate
        cand = cand.groupby("ticker", group_keys=False).head(3).head(K_ANALOGS)
        y = cand["xs_spy_sharpe_42"]
        a_med.append(y.median()); a_hit.append((y > 0).mean()); a_n.append(len(y))
        a_list.append("; ".join(f"{t}@{d.date()}:{v:+.1f}" for t, d, v in zip(cand.ticker.head(5), cand.date.head(5), y.head(5))))
    today["analog_med_xsSharpe42"] = a_med
    today["analog_hit_rate"] = a_hit
    today["analog_n"] = a_n
    # analog outcomes in absolute excess-Sharpe units sit near -1 in the recent era (SPY's own Sharpe was high);
    # the percentile across today's names is the comparable reading
    today["analog_pct"] = pd.Series(a_med, index=today.index).rank(pct=True)
    today["analog_examples"] = a_list

    show = ["ticker", "sector", "beta_bucket", "final_score", "ml_score", "comp_score", "analog_med_xsSharpe42", "analog_pct", "analog_hit_rate",
            "mom_12_1", "dist_52w_high", "dist_52w_low", "rs_lead_126", "rs_spy_63", "sharpe_126", "beta_252", "rv20", "rv20_pct_252",
            "vol_dry_20_250", "updown_vol_ratio_50", "obv_price_div_63", "dn_up_vol_asym_63", "corr_spy_63", "days_since_20pct_dd",
            "ceiling_2x_low", "analog_examples"]
    out = today.sort_values("final_score", ascending=False)[show]
    out.insert(0, "asof", asof.date())
    out.to_csv(RESULTS_DIR / "current_rankings.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(out.drop(columns=["analog_examples"]).head(40).round(3).to_string(index=False))
    print("\nbottom 10:")
    print(out.drop(columns=["analog_examples"]).tail(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
