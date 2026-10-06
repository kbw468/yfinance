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
    mlv = pd.read_csv(RESULTS_DIR / "ml_variants_oos_evaluation.csv", index_col=[0, 1])
    comp = pd.read_csv(RESULTS_DIR / "composite_oos_evaluation.csv", index_col=[0, 1])
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
    L.append("""* Two readings carry out-of-sample signal for 21-63d forward Sharpe in this universe in the current regime: relative-strength leadership (`rs_lead_126`: RS line nearer its 6-month high than price is to its own) and 12-1 momentum. Both survive beta-neutralisation (recent-window 42d IC 0.03-0.05, NW t 2.1-2.4). Both were dead 2014-2021.
* The setup that works depends on beta. High-beta names: buy persistence (momentum, trailing Sharpe, proximity to highs, recent 2x-off-low crossings; the 100%-off-low ceiling is the wrong sign there). Low-beta names: price-structure factors die and quiet volume-led accumulation takes over (volume dry-up vs 250d, falling dollar volume, OBV leading price, shallow drawdowns, low idiosyncratic vol, and the dry-up x tight-range x near-high interaction).
* The live ranking (`current_rankings.csv`) is the within-beta-bucket walk-forward composite built from those weights: recent-window OOS 42d IC 0.047, 63d 0.065, top-vs-bottom decile 42d Sharpe spread +0.29 annualised, 57% of windows positive. A pooled (beta-blind) composite has no edge.
* Compression / NR7 / Bollinger squeeze, ignition-day volume multiples, accumulation-day counts, gap behaviour, streaks and close-location factors showed no 1-3 month signal in S&P 500 names, pooled. The LightGBM case-study model, in five configurations, did not beat the two single factors out of sample (recent 42d IC 0.00-0.02) and is reported but not used in the ranking.
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
    L.append("\n### 5b. Walk-forward linear composite\n")
    for name in comp.model.unique():
        L.append(f"\n*{name}*\n")
        L.append(md(pick(comp, name)))

    L.append("\n## 6. Live scoring\n")
    L.append("`current_rankings.csv` columns: `final_score` = within-beta-bucket composite percentile (the ML percentile is reported alongside, not blended, because of its OOS record); `analog_med_xsSharpe42` / `analog_pct` / `analog_hit_rate` = realised 42d SPY-excess Sharpe of the 50 nearest historical cases (median, its percentile across today's names, share > 0); key factor readings follow.\n")
    L.append("\n### Final composite weights by beta bucket\n")
    L.append(md(cw.dropna(how="all"), 3))
    L.append("\n### Top 30 as of " + str(rank["asof"].iloc[0]) + "\n")
    show = ["ticker", "sector", "beta_bucket", "final_score", "ml_score", "comp_score", "analog_pct", "analog_hit_rate",
            "mom_12_1", "dist_52w_high", "dist_52w_low", "rs_lead_126", "beta_252", "rv20_pct_252", "vol_dry_20_250", "obv_price_div_63"]
    L.append(md(rank[show].head(30).set_index("ticker"), 2))

    L.append("\n## 7. What to trust, what not to\n")
    L.append(open(RESULTS_DIR / "_assessment.md").read() if (RESULTS_DIR / "_assessment.md").exists() else "(assessment pending)\n")

    L.append("\n## 8. Files\n")
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
| `current_rankings.csv` | the live ranking with analog evidence |
""")
    L.append("\n## 9. Reproduce\n")
    L.append("```\npip install -e . scikit-learn lightgbm pyarrow scipy statsmodels tabulate\npython -m pvv_score.data_io          # download\npython -m pvv_score.build            # features + targets\npython -m pvv_score.run_eval         # single-factor\npython -m pvv_score.run_conditional  # conditional\npython -m pvv_score.model            # walk-forward ML\npython -m pvv_score.run_model_variants\npython -m pvv_score.composite        # walk-forward composite\npython -m pvv_score.score_now        # live ranking\npython -m pvv_score.report\n```\n")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(L))
    print("wrote", RESULTS_DIR / "REPORT.md")


if __name__ == "__main__":
    main()
