"""Assemble results/REPORT.md from the CSV outputs of the research stages."""
import pandas as pd
import numpy as np

from .config import RESULTS_DIR, RECENT_START, BACKTEST_START, HORIZONS
from .run_eval import load_registry

FACTORS = load_registry()


def md(df: pd.DataFrame, floatfmt=3) -> str:
    return df.round(floatfmt).to_markdown()


def main():
    ic = pd.read_csv(RESULTS_DIR / "factor_ic_summary.csv", index_col=0)
    cond42 = pd.read_csv(RESULTS_DIR / "conditional_ic_42.csv", index_col=0)
    sn = pd.read_csv(RESULTS_DIR / "sector_neutral_ic.csv", index_col=0)
    ml = pd.read_csv(RESULTS_DIR / "ml_oos_evaluation.csv", index_col=[0, 1])
    dec_path = RESULTS_DIR / "identity_decomposition_level.csv"
    mlv = pd.read_csv(RESULTS_DIR / "ml_variants_oos_evaluation.csv", index_col=[0, 1])
    comp = pd.read_csv(RESULTS_DIR / "composite_level_oos_evaluation.csv", index_col=[0, 1])
    cw = pd.read_csv(RESULTS_DIR / "composite_final_weights.csv", index_col=0)
    rank = pd.read_csv(RESULTS_DIR / "current_rankings.csv")
    byyear_ml = pd.read_csv(RESULTS_DIR / "ml_oos_ic42_by_year.csv", index_col=0)

    L = []
    L.append("# PVV Score: price / volume / volatility selection for 21-63 day forward risk-adjusted returns\n")
    L.append(f"Universe: 503 Finviz S&P 500 constituents (file `data/universe_finviz.csv`), sector mapped to SPDR ETF. "
             f"Data: yfinance daily OHLCV, dividend/split adjusted, 2012-{rank["asof"].iloc[0]}. Backtest window {BACKTEST_START} onward; "
             f"**the 2022-10 onward window ('recent') is the primary evidence** per the mandate, the full window is the secondary check.\n")
    L.append("All statistics are out-of-sample where labelled OOS. Nothing here uses fundamentals, earnings or valuation.\n")

    L.append("## 1. Bottom line\n")
    L.append("""* **The ranking is identity-neutral.** Every factor is expressed as the name's deviation from its own trailing 3-year norm (z), so a chronically strong, drawdown-free, high-Sharpe name (DELL, NVDA, LLY, PGR, BRK-B) scores near the middle unless its current price / volume / volatility behaviour is unusual for it. Weights are fitted walk-forward inside beta terciles. Every one of the 503 tickers has a score (section 6); names without 252 sessions of history are listed with the reason.
* **Out of sample (walk-forward 2017-2026), the state composite is the more robust of the two systems.** Recent window: 42d IC 0.039, 63d IC 0.056 (NW t 2.3); beta-neutral 63d IC 0.029 (t 2.4). Full window beta-neutral: 42d IC 0.017 (t 2.6), 63d 0.023 (t 2.9). The level-based composite had a slightly higher recent raw IC (0.047) but almost none of it survived beta-neutralisation (0.006) and half of it was ticker identity (section 7). Top-vs-bottom decile 42d Sharpe spread of the state composite: +0.22 annualised.
* **What precedes top-decile forward risk-adjusted windows, across all names (section 7b):** small but consistent pre-states. Pooled: 63d relative strength unusually WEAK for the name (bottom-quintile lift 1.08, top-quintile 0.85), dollar volume unusually low, no recent ignition day, downside-vol asymmetry unusually high, longer-than-usual time since the 21d high. That is a quiet, pulled-back, under-owned state, i.e. 1-3 month mean reversion, not breakout chasing. Mid-beta names add: 63d base unusually tight (lift 1.06 vs 0.84), downside vol unusually low, a recent ignition day. The identity layer (section 7) rewards persistent strength; the state layer rewards an unusual quiet pullback in a name. Both are real and they are different questions: which names, and when.
* The identity-heavy level composite and the LightGBM case-study model are kept in the files for comparison and are not used in the ranking.
* Compression / NR7 / Bollinger squeeze, ignition-day volume multiples, accumulation-day counts, gap behaviour, streaks and close-location factors showed no pooled 1-3 month signal as LEVELS. As STATES a few re-appear conditionally (tight base and ignition in mid beta); the rest stay flat.
""")
    L.append("## 2. Method\n")
    L.append("""* **Targets.** For every (date, stock): realised Sharpe and Sortino of the stock over t+1..t+h (h = 21, 42, 63) minus the same for SPY and for the stock's sector ETF (XLRE/XLC spliced to XLF/XLK before inception). Also the information-ratio form (Sharpe of stock-minus-SPY). The ML label is the per-date percentile of SPY-excess Sharpe averaged over the three horizons.
* **Eligibility (point-in-time).** Price >= $5, 63d median dollar volume >= $10M, >= 252 sessions of history. Survivorship caveat: the universe is today's constituents, so absolute levels are biased up; cross-sectional ranks are far less affected.
* **Factors.** 86 point-in-time factors in five families (price structure, volume, volatility, relative strength, run character) plus two user-described interaction setups. Each verified against hand calculations and a perturbation test for look-ahead (none).
* **Statistics.** Daily cross-sectional Spearman IC; Newey-West t-stats with lag = horizon because forward windows overlap; Benjamini-Hochberg across the 86-factor zoo; decile spreads in raw excess-Sharpe units; 21-day rank autocorrelation (turnover).
* **Conditioning.** Everything re-run inside trailing-beta terciles, 20d realised-vol terciles, sector groups (defensive / cyclical / growth), per sector, and with sector-neutral ranks.
* **Case-study model.** LightGBM trained walk-forward by calendar year (2017-2026), 63-day embargo, 2-year half-life time-decay weights, every 5th day sampled for training. Four pre-specified regularised variants were also run. Nearest-neighbour analogs (50 historical cases, max 3 per ticker) give each live name its realised-outcome evidence.
* **Composite.** Factor selection and weights re-estimated each year on training data only (|NW t| >= 1.5 on the trailing 3y, sign agreement with full history, greedy de-duplication at |rho| > 0.7, weights = recency-blended IC). Pooled and beta-bucketed variants.
""")

    L.append("## 3. Single-factor results (SPY-excess Sharpe, mean IC across 21/42/63d)\n")
    L.append("Power note: the recent window is ~1,000 sessions = roughly 24 independent 42-day windows, so |t| around 2 is the attainable ceiling for a genuine IC of 0.05-0.07. **No factor survives Benjamini-Hochberg at 5% in either window.** The evidence is in coherent clusters, not individual rows.\n")
    top = ic.sort_values("t_spy_recent", key=np.abs, ascending=False).head(25)
    cols = ["group", "prior", "ic_spy_full", "t_spy_full", "ic_spy_recent", "t_spy_recent", "ic_sec_recent", "t_sec_recent",
            "sign_stable", "ic42_pos_years", "d10_d1_spread_recent_42", "autocorr_21"]
    L.append(md(top[cols]))
    L.append("\n`d10_d1_spread_recent_42` is the annualised 42d Sharpe difference between the factor's top and bottom decile (recent window). `autocorr_21` near 1 = slow factor; near 0 = daily-turnover factor.\n")
    L.append("\n### Reading of the clusters\n")
    L.append("""* **Trend-persistence / leadership cluster** (`rs_lead_126`, `mom_12_1`, `dist_52w_low`, `up_capture_63`, `days_since_2x_low` negative, `ceiling_2x_low` positive): the only group with a consistent positive sign in the recent window and a positive sector-excess reading. Note `ceiling_2x_low` comes in with the OPPOSITE sign to the prior: names already 100% off their 52w low have continued to produce better forward Sharpe in 2022-2026. In the pooled sample the "ceiling" is not a ceiling.
* **Beta / correlation cluster** (`beta_63`, `beta_252`, `corr_spy_63`, `rv20`, `dn_vol_63`): positive in the recent window, flat over the full window, negative vs sector. This is the bull-regime beta bet, not stock selection. The composite and the model are therefore also evaluated beta-neutral (section 5).
* **Compression / dry-up setups** (`range_comp_*`, `bbw_*`, `nr7_count_20`, `vol_dry_*`, `dryup_x_tight`): no pooled signal at these horizons. They show up only conditionally (section 4), in low-vol / defensive names.
* **Ignition / volume-burst factors** (`ignition_mult_10`, `vol_z_1`, `big_vol_updays_net_21`, `ignite_x_rs`): no signal for 21-63d forward Sharpe in S&P 500 names. These are shorter-horizon or small-cap phenomena.
* **Short-term price factors** (`ret_5`, `clv_5`, `streak_now`, `pos_in_20d_range`): nothing. Daily-turnover factors with no 1-3 month payoff.
""")

    L.append("## 4. Conditional results: where each factor works (recent window, 42d, NW t-stats)\n")
    cols = [c for c in cond42.columns if c.endswith("|recent|t_nw")]
    v = cond42[cols].copy(); v.columns = [c.split("|")[0] for c in cols]
    v["sector_neutral_vs_sector"] = sn["secneut_sec42|recent|t_nw"]
    v["max_abs_t"] = v.abs().max(axis=1)
    v = v.sort_values("max_abs_t", ascending=False).head(30).drop(columns="max_abs_t")
    L.append(md(v, 2))
    L.append("""
Multiple-testing note: 86 factors x 10 buckets = 860 tests; the expected maximum |t| under the null is about 3.2, so single cells near 3 are not individually reliable. What is reliable is the pattern:

* **High beta / growth:** the trend-persistence cluster is strongest here (`mom_12_1` t=3.6 in the high-beta tercile, `days_since_52w_high` -2.5, `ceiling_2x_low` +2.7, `days_since_2x_low` -2.7). In high-beta names, buy strength, recent 2x-off-low crossings, and proximity to highs.
* **Low beta / low vol / defensive:** the sign of several price factors flips or dies, and volume-structure factors appear instead: `obv_price_div_63` +3.6 in defensives (volume leading price), `vol_dry_20_250` -2.1 in the low-vol tercile (dry-up works for utilities/staples-type names), `dvol_trend_21_126` -2.9 in low beta (quiet names outperform), `idio_vol_63` negative in low/mid beta. This is the DUK-vs-NVDA asymmetry the mandate anticipated, and it is why the composite is fitted per beta bucket.
* **Per sector** (`conditional_ic_42_per_sector.csv`): indicative only (18-80 names per date). Notable: Utilities `rs_lead_126` +3.1 and dry-up negative; Consumer Defensive `ceiling_2x_low` -3.2 (the ceiling DOES hold in staples) and compression factors positive; Financials show reversal (`ret_63` -2.3).
""")

    L.append("## 5. Out-of-sample predictive models (walk-forward 2017-2026)\n")
    L.append("### 5a. Case-study model (LightGBM) vs single-factor baselines\n")
    def pick(ev, name):
        s = ev[ev.model == name] if "model" in ev.columns else ev
        rows = []
        for h in HORIZONS:
            for win in ["full", "recent"]:
                r = s.loc[(f"spy_sh{h}", win)]
                rb = s.loc[(f"spy_sh{h}_betaneutral", win)]
                rsec = s.loc[(f"sec_sh{h}", win)]
                rows.append({"h": h, "window": win, "IC vs SPY": r.ic, "t": r.t_nw, "IC beta-neutral": rb.ic, "t_bn": rb.t_nw, "IC vs sector": rsec.ic, "t_sec": rsec.t_nw})
        return pd.DataFrame(rows).set_index(["h", "window"])
    for name in ["lightgbm_walkforward", "baseline_mom_12_1", "baseline_rs_lead_126", "baseline_beta_252"]:
        L.append(f"\n**{name}**\n")
        L.append(md(pick(ml, name)))
    L.append("\nLightGBM 42d IC by year: " + ", ".join(f"{int(y)}: {v:+.3f}" for y, v in byyear_ml["mean"].items()) + "\n")
    L.append("\n**Regularised variants** (rank-only features, 7 leaves, 300 rounds; ridge; with/without decay; no market-context features):\n")
    for name in mlv.model.unique():
        L.append(f"\n*{name}*\n")
        L.append(md(pick(mlv, name)))
    L.append("\n### 5b. Walk-forward linear composite (LEVEL factors; identity-heavy, superseded by section 6)\n")
    for name in comp.model.unique():
        L.append(f"\n*{name}*\n")
        L.append(md(pick(comp, name)))

    L.append("\n## 6. Live scoring: full universe, identity-neutral\n")
    cs = pd.read_csv(RESULTS_DIR / "composite_state_oos_evaluation.csv", index_col=[0, 1])
    L.append("### 6a. Walk-forward OOS: state composite vs level composite\n")
    for name in cs.model.unique():
        L.append(f"\n*{name}*\n")
        L.append(md(pick(cs, name)))
    L.append("\nIdentity decomposition of the state composite (recent window, 42d): raw IC 0.039, persistent 0.027, deviation 0.039; correlation of today's rank with the name's own historical-average rank 0.42 (level composite: 0.58). The signal now sits in the deviation component.\n")
    csw = pd.read_csv(RESULTS_DIR / "composite_state_final_weights.csv", index_col=0)
    L.append("\n### 6b. Final state-composite weights by beta bucket (z = deviation from the name's own 3y norm)\n")
    L.append(md(csw.dropna(how="all"), 3))
    us = pd.read_csv(RESULTS_DIR / "universe_scores.csv")
    L.append(f"\n### 6c. All {len(us)} tickers as of {us['asof'].iloc[0]}\n")
    L.append("`state_score_pooled` = universe percentile of the state composite (the single sortable number). `state_score` = percentile inside the name's beta bucket. `level_score` = the identity-heavy composite, for comparison. z columns are standard deviations vs the name's own history. `top_states` = the three readings contributing most to the score today. Sorted by `state_score_pooled`. The interactive version (filter, sort) is `universe_scores.html`.\n")
    cols = ["ticker", "Sector", "beta_bucket", "state_score_pooled", "state_score", "level_score", "z_mom_12_1", "z_rs_lead_126", "z_dist_52w_high",
            "z_rv20_pct_252", "z_vol_dry_20_250", "z_obv_price_div_63", "z_dn_up_vol_asym_63", "top_states", "note"]
    tbl = us[cols].set_index("ticker")
    L.append(md(tbl, 2))
    L.append("\n### 6d. Identity vs state: the names raised\n")
    diag = us.set_index("ticker").reindex([t for t in ["DELL", "NVDA", "LLY", "PGR", "BRK-B", "AMD", "PANW", "NTAP", "DUK", "KO"] if t in us.ticker.values])
    L.append(md(diag[["Sector", "beta_bucket", "level_score", "state_score", "state_score_pooled", "top_states"]], 2))
    L.append("\nRank correlation between level and state scores across the universe: " + f"{us[['level_score', 'state_score']].corr(method='spearman').iloc[0, 1]:.2f}" + ".\n")
    L.append("\n## 7. Ticker identity vs. behaviour: how much of the signal generalises across tickers\n")
    L.append("""Each reading is split, point-in-time, into **persistent** (the ticker's own expanding-window mean up to t-1: "this ticker is usually like this") and **deviation** (today's reading as a z-score against the ticker's own history: "this ticker is currently unusual"). IC of each component vs 42d SPY-excess Sharpe:
""")
    dec = pd.read_csv(RESULTS_DIR / "identity_decomposition_level.csv")
    piv = dec.pivot_table(index="factor", columns=["window", "component"], values="ic")[[("recent", "raw"), ("recent", "persistent"), ("recent", "deviation"), ("full", "raw"), ("full", "persistent"), ("full", "deviation")]]
    piv.columns = [f"{w}|{c}" for w, c in piv.columns]
    L.append(md(piv, 3))
    pure = pd.read_csv(RESULTS_DIR / "identity_pure_predictors.csv").pivot(index="predictor", columns="window", values=["ic", "t_nw"])
    pure.columns = [f"{a}|{b}" for a, b in pure.columns]
    overlap = (RESULTS_DIR / "identity_rank_overlap_level.txt").read_text().strip()
    L.append("\nPure ticker-identity predictors (long trailing Sharpe; the ticker's own trailing-year mean realised excess Sharpe):\n")
    L.append(md(pure, 3))
    L.append(f"""
Reading:

* The out-of-sample composite's recent IC (0.047) is matched by its persistent component alone (0.053); the deviation component carries 0.035. Today's composite rank correlates {overlap} with the ticker's own historical-average composite rank. **Roughly half to two thirds of what the ranking expresses is which tickers are chronically strong and stable, not a current state change.**
* A ticker's own trailing-year mean realised excess Sharpe predicts its next 42 days with IC 0.044 (t 1.7) in the recent window, about the same as the whole system. In 2022-10 to 2026 leadership persisted at the ticker level; much of the system's recent edge is that persistence.
* The persistent component is the part most exposed to survivorship bias (today's constituents are disproportionately the names that were chronically strong). Treat the identity half of the score as regime- and universe-dependent.
* The deviation component is the cleaner "behaviour predicts across tickers" evidence. It is weaker but real: 12-1 momentum unusually high for the name (IC 0.044, t 2.0, the only reading where the state component beats the identity component), RS-line leadership unusual for the name (0.031), correlation to SPY unusually high (0.058), idiosyncratic vol unusually low (-0.018). Proximity to 52w high, time since 20% drawdown and the dry-up interaction have no deviation signal pooled: they work only as identity, and only inside beta buckets.
* Volume factors: `obv_price_div_63` and `vol_dry_20_250` show up as identity, not state, pooled. The names that habitually have volume leading price, or habitually dry up, do better; a dry-up that is unusual for the name does not.
""")
    L.append("\n## 7b. Event study: the price / volume / volatility STATE that preceded top-decile forward windows\n")
    L.append("""Episode = a (date, name) whose forward blended SPY-excess-Sharpe percentile was >= 0.90. Consecutive episode days of one name form a cluster (9,078 clusters since 2014, 2,828 since 2022-10). The state is measured **5 sessions before the cluster start**: the start day itself is selection-contaminated (by construction the day before was not a top-decile start, so the start is mechanically a local low; `z_ret_5` reads t = -59 at t-0 and ~0 at t-5, see `event_pre_episode_path_recent.csv`). Each state factor is ranked cross-sectionally per date; `mean_rank_excess` is the mean rank at t-5 minus 0.5 with a cluster-robust t; `lift_top_q` / `lift_bottom_q` = P(episode | factor in top / bottom quintile) / P(episode), over all days (no start selection).
""")
    ev = pd.read_csv(RESULTS_DIR / "event_pre_episode_profile.csv")
    rec = ev[(ev.window == "recent") & (ev.bucket == "all")].set_index("factor").sort_values("t_cluster", key=np.abs, ascending=False)
    L.append("\n**Recent window, pooled** (top 25 by |t|):\n")
    L.append(md(rec[["mean_rank_excess", "t_cluster", "n_clusters", "lift_top_q", "lift_bottom_q"]].head(25), 3))
    mid = ev[(ev.window == "recent") & (ev.bucket == "mid")].set_index("factor").sort_values("t_cluster", key=np.abs, ascending=False)
    L.append("\n**Recent window, mid-beta tercile** (top 15; the low and high terciles are dominated by `z_beta_*` / `z_corr_spy_*` entries that are an artefact of bucketing on the beta level and are not pre-conditions):\n")
    L.append(md(mid[["mean_rank_excess", "t_cluster", "n_clusters", "lift_top_q", "lift_bottom_q"]].head(15), 3))
    L.append("""
Reading:

* Effects are small (|rank excess| <= 0.02 pooled, lifts 0.85-1.16) and statistically clear (2,800 independent clusters). There is no dramatic P/V/V signature 1-4 weeks before a top-decile 1-3 month window in S&P 500 names; there is a consistent tilt.
* The tilt is a **quiet pullback in the name**: 63d relative strength and 63d return unusually weak for the name (bottom-quintile lift 1.08-1.09), dollar volume unusually low (1.09), longer than usual since the 21d high (1.06), no recent ignition day (top-quintile lift of `days_since_ignition` 1.16), downside-vol asymmetry unusually high (1.07), beta and correlation to SPY unusually high. This is 1-3 month mean reversion with a capitulation flavour, and it is the opposite of what the level layer rewards.
* Mid beta adds structure that matches the mandate's hypotheses: 63d base unusually tight (bottom-quintile lift 1.06 vs top 0.84), downside vol unusually low (0.88 top), a recent ignition day (bottom-quintile 0.86), time since 2x-off-low crossing unusually long (1.31).
* Volume dry-up vs 250d, Bollinger / range compression, NR7 count, up/down volume ratio and OBV divergence as STATES: no pre-episode tilt pooled (|t| < 1.5). Accumulation-day count tilts slightly negative.
* These are the ingredients the state composite is allowed to pick from; what it actually selects each year is in `composite_state_weights_by_fold.csv`.
""")
    L.append("\n## 8. What to trust, what not to\n")
    L.append(open(RESULTS_DIR / "_assessment.md").read() if (RESULTS_DIR / "_assessment.md").exists() else "(assessment pending)\n")

    L.append("\n## 9. Files\n")
    L.append("""| file | content |
|---|---|
| `factor_ic_summary.csv` / `factor_ic_detail.csv` | single-factor IC, NW t, FDR flags, decile spreads, autocorrelation, all horizons / benchmarks / windows |
| `factor_ic42_by_year.csv` | yearly mean IC (42d, SPY-excess) per factor |
| `conditional_ic_{21,42,63}.csv` | IC by beta tercile, vol tercile, sector group |
| `conditional_ic_42_per_sector.csv` | IC per GICS-style sector |
| `sector_neutral_ic.csv` | IC of sector-demeaned ranks vs sector- and SPY-excess Sharpe |
| `factor_rank_corr_recent.csv` | factor rank-correlation matrix (recent window) |
| `ml_oos_evaluation.csv`, `ml_variants_oos_evaluation.csv`, `ml_oos_ic42_by_year.csv` | walk-forward model results |
| `composite_oos_evaluation.csv`, `composite_weights_by_fold.csv` | walk-forward composite results and the factors it chose each year |
| `composite_final_weights.csv`, `ml_final_feature_importance.csv` | what the live score is built from |
| `identity_decomposition_level.csv`, `identity_decomposition_state.csv`, `identity_pure_predictors.csv` | ticker-identity vs own-history-deviation split of the signal |
| `composite_state_oos_evaluation.csv`, `composite_state_weights_by_fold.csv`, `composite_state_final_weights.csv` | identity-neutral (state) composite: OOS results, per-year factor picks, live weights |
| `event_pre_episode_profile.csv`, `event_pre_episode_path_recent.csv` | event study: state 5 sessions before top-decile forward windows; rank path t-20..t-0 |
| `universe_scores.csv`, `universe_scores.html` | every ticker scored (state, level, ML, z readings, top states, notes) |
| `current_rankings.csv` | the live ranking with analog evidence |
""")
    L.append("\n## 10. Reproduce\n")
    L.append("```\n" + "\n".join([
        "pip install -e . scikit-learn lightgbm pyarrow scipy statsmodels tabulate markdown",
        "python -m pvv_score.data_io          # download",
        "python -m pvv_score.build            # features + state transforms + targets",
        "python -m pvv_score.run_eval         # single-factor",
        "python -m pvv_score.run_conditional  # conditional",
        "python -m pvv_score.model            # walk-forward ML",
        "python -m pvv_score.run_model_variants",
        "python -m pvv_score.composite        # walk-forward composite (level)",
        "python -m pvv_score.score_now        # level-based ranking (comparison)",
        "python -m pvv_score.run_identity level",
        "python -m pvv_score.composite --state    # identity-neutral composite",
        "python -m pvv_score.run_identity state",
        "python -m pvv_score.events               # pre-episode event study",
        "python -m pvv_score.score_state          # full-universe state scoring",
        "python -m pvv_score.ranking_page         # sortable all-ticker page",
        "python -m pvv_score.report",
    ]) + "\n```\n")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(L))
    print("wrote", RESULTS_DIR / "REPORT.md")


if __name__ == "__main__":
    main()
