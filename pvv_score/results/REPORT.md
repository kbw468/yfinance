# PVV Score: price / volume / volatility selection for 21-63 day forward risk-adjusted returns

Universe: 503 Finviz S&P 500 constituents (file `data/universe_finviz.csv`), sector mapped to SPDR ETF. Data: yfinance daily OHLCV, dividend/split adjusted, 2012-2026-10-05. Backtest window 2014-01-01 onward; **the 2022-10 onward window ('recent') is the primary evidence** per the mandate, the full window is the secondary check.

All statistics are out-of-sample where labelled OOS. Nothing here uses fundamentals, earnings or valuation.

## 1. Bottom line

* Two readings carry out-of-sample signal for 21-63d forward Sharpe in this universe in the current regime: relative-strength leadership (`rs_lead_126`: RS line nearer its 6-month high than price is to its own) and 12-1 momentum. Both survive beta-neutralisation (recent-window 42d IC 0.03-0.05, NW t 2.1-2.4). Both were dead 2014-2021.
* The setup that works depends on beta. High-beta names: buy persistence (momentum, trailing Sharpe, proximity to highs, recent 2x-off-low crossings; the 100%-off-low ceiling is the wrong sign there). Low-beta names: price-structure factors die and quiet volume-led accumulation takes over (volume dry-up vs 250d, falling dollar volume, OBV leading price, shallow drawdowns, low idiosyncratic vol, and the dry-up x tight-range x near-high interaction).
* The live ranking (`current_rankings.csv`) is the within-beta-bucket walk-forward composite built from those weights: recent-window OOS 42d IC 0.047, 63d 0.065, top-vs-bottom decile 42d Sharpe spread +0.29 annualised, 57% of windows positive. A pooled (beta-blind) composite has no edge.
* Roughly half to two thirds of the ranking is ticker identity (names that are chronically near highs, drawdown-free and high-Sharpe) rather than a current state change; the state-change half is weaker (IC 0.035) and is led by momentum and RS leadership that are unusual for the name (section 7).
* Compression / NR7 / Bollinger squeeze, ignition-day volume multiples, accumulation-day counts, gap behaviour, streaks and close-location factors showed no 1-3 month signal in S&P 500 names, pooled. The LightGBM case-study model, in five configurations, did not beat the two single factors out of sample (recent 42d IC 0.00-0.02) and is reported but not used in the ranking.

## 2. Method

* **Targets.** For every (date, stock): realised Sharpe and Sortino of the stock over t+1..t+h (h = 21, 42, 63) minus the same for SPY and for the stock's sector ETF (XLRE/XLC spliced to XLF/XLK before inception). Also the information-ratio form (Sharpe of stock-minus-SPY). The ML label is the per-date percentile of SPY-excess Sharpe averaged over the three horizons.
* **Eligibility (point-in-time).** Price >= $5, 63d median dollar volume >= $10M, >= 252 sessions of history. Survivorship caveat: the universe is today's constituents, so absolute levels are biased up; cross-sectional ranks are far less affected.
* **Factors.** 86 point-in-time factors in five families (price structure, volume, volatility, relative strength, run character) plus two user-described interaction setups. Each verified against hand calculations and a perturbation test for look-ahead (none).
* **Statistics.** Daily cross-sectional Spearman IC; Newey-West t-stats with lag = horizon because forward windows overlap; Benjamini-Hochberg across the 86-factor zoo; decile spreads in raw excess-Sharpe units; 21-day rank autocorrelation (turnover).
* **Conditioning.** Everything re-run inside trailing-beta terciles, 20d realised-vol terciles, sector groups (defensive / cyclical / growth), per sector, and with sector-neutral ranks.
* **Case-study model.** LightGBM trained walk-forward by calendar year (2017-2026), 63-day embargo, 2-year half-life time-decay weights, every 5th day sampled for training. Four pre-specified regularised variants were also run. Nearest-neighbour analogs (50 historical cases, max 3 per ticker) give each live name its realised-outcome evidence.
* **Composite.** Factor selection and weights re-estimated each year on training data only (|NW t| >= 1.5 on the trailing 3y, sign agreement with full history, greedy de-duplication at |rho| > 0.7, weights = recency-blended IC). Pooled and beta-bucketed variants.

## 3. Single-factor results (SPY-excess Sharpe, mean IC across 21/42/63d)

Power note: the recent window is ~1,000 sessions = roughly 24 independent 42-day windows, so |t| around 2 is the attainable ceiling for a genuine IC of 0.05-0.07. **No factor survives Benjamini-Hochberg at 5% in either window.** The evidence is in coherent clusters, not individual rows.

|                           | group    |   prior |   ic_spy_full |   t_spy_full |   ic_spy_recent |   t_spy_recent |   ic_sec_recent |   t_sec_recent | sign_stable   |   ic42_pos_years |   d10_d1_spread_recent_42 |   autocorr_21 |
|:--------------------------|:---------|--------:|--------------:|-------------:|----------------:|---------------:|----------------:|---------------:|:--------------|-----------------:|--------------------------:|--------------:|
| rs_lead_126               | relative |       1 |        -0.002 |       -0.127 |           0.066 |          2.436 |           0.013 |          0.705 | False         |            0.538 |                     0.515 |         0.72  |
| ceiling_2x_low            | run      |      -1 |         0.01  |        1.151 |           0.038 |          2.275 |           0.019 |          1.662 | True          |            0.615 |                     0.28  |         0.743 |
| corr_spy_63               | relative |      -1 |         0.02  |        1.176 |           0.072 |          2.119 |          -0.001 |         -0.1   | True          |            0.615 |                     0.437 |         0.848 |
| parkinson_cc_20           | vol      |       0 |        -0.004 |       -0.483 |          -0.029 |         -1.977 |           0.008 |          1.081 | True          |            0.462 |                    -0.182 |         0.081 |
| dist_52w_low              | run      |       0 |         0.002 |        0.154 |           0.047 |          1.915 |           0.027 |          1.569 | True          |            0.538 |                     0.417 |         0.869 |
| up_capture_63             | relative |       1 |         0.005 |        0.219 |           0.069 |          1.911 |           0.001 |          0.024 | True          |            0.538 |                     0.55  |         0.817 |
| overnight_intraday_vol_20 | vol      |       0 |         0.011 |        1.099 |           0.034 |          1.908 |          -0.001 |         -0.083 | True          |            0.692 |                     0.211 |         0.203 |
| beta_63                   | relative |       0 |        -0.004 |       -0.23  |           0.072 |          1.817 |          -0.007 |         -0.403 | False         |            0.538 |                     0.539 |         0.88  |
| days_since_2x_low         | run      |       0 |        -0     |       -0.013 |          -0.037 |         -1.791 |          -0.028 |         -2.47  | True          |            0.385 |                    -0.262 |         0.923 |
| gap_share_21              | price    |       0 |         0.014 |        1.335 |           0.037 |          1.765 |           0.007 |          0.623 | True          |            0.692 |                     0.32  |         0.26  |
| beta_252                  | relative |       0 |         0.002 |        0.06  |           0.072 |          1.727 |          -0.013 |         -0.661 | True          |            0.462 |                     0.398 |         0.985 |
| mom_12_1                  | price    |       1 |         0.005 |        0.374 |           0.04  |          1.66  |           0.043 |          2.288 | True          |            0.615 |                     0.277 |         0.892 |
| up_gap_freq_21            | price    |       1 |        -0.009 |       -0.532 |           0.041 |          1.583 |          -0.003 |         -0.188 | False         |            0.538 |                     0.318 |         0.452 |
| down_capture_63           | relative |      -1 |         0.003 |        0.112 |           0.051 |          1.461 |          -0.02  |         -1.236 | True          |            0.538 |                     0.308 |         0.792 |
| dn_vol_63                 | vol      |      -1 |        -0.011 |       -0.681 |           0.043 |          1.457 |          -0.014 |         -0.846 | False         |            0.615 |                     0.239 |         0.845 |
| rv20                      | vol      |       0 |        -0.015 |       -0.961 |           0.04  |          1.37  |          -0.01  |         -0.67  | False         |            0.538 |                     0.274 |         0.545 |
| rs_newhigh_price_not_20   | relative |       1 |        -0.006 |       -0.694 |          -0.017 |         -1.362 |           0     |          0.046 | True          |            0.692 |                     0.011 |         0.196 |
| dvol_trend_21_126         | volume   |       0 |        -0.002 |       -0.333 |          -0.015 |         -1.343 |           0.011 |          1.395 | True          |            0.462 |                    -0.087 |         0.242 |
| accum_dist_50             | volume   |       1 |        -0.009 |       -1.034 |          -0.019 |         -1.255 |           0.006 |          0.506 | True          |            0.769 |                    -0.173 |         0.563 |
| rv_ratio_5_20             | vol      |       0 |         0.001 |        0.384 |           0.006 |          1.188 |           0.002 |          0.671 | True          |            0.538 |                     0.026 |        -0.006 |
| base_tightness_63         | price    |      -1 |        -0.011 |       -0.734 |           0.033 |          1.136 |          -0.004 |         -0.283 | False         |            0.615 |                     0.21  |         0.752 |
| bbw_squeeze_126           | price    |      -1 |        -0.004 |       -0.64  |           0.015 |          1.134 |           0.011 |          1.316 | False         |            0.538 |                     0.121 |         0.108 |
| dn_up_vol_asym_63         | vol      |      -1 |         0.011 |        1.059 |           0.016 |          1.087 |          -0.001 |         -0.058 | True          |            0.538 |                     0.059 |         0.614 |
| rs_sec_21                 | relative |       1 |         0.002 |        0.23  |           0.014 |          1.087 |           0.008 |          0.603 | True          |            0.462 |                     0.237 |        -0.01  |
| obv_price_div_63          | volume   |       1 |         0.003 |        0.454 |           0.011 |          1.08  |           0.007 |          0.814 | True          |            0.615 |                     0.077 |         0.575 |

`d10_d1_spread_recent_42` is the annualised 42d Sharpe difference between the factor's top and bottom decile (recent window). `autocorr_21` near 1 = slow factor; near 0 = daily-turnover factor.


### Reading of the clusters

* **Trend-persistence / leadership cluster** (`rs_lead_126`, `mom_12_1`, `dist_52w_low`, `up_capture_63`, `days_since_2x_low` negative, `ceiling_2x_low` positive): the only group with a consistent positive sign in the recent window and a positive sector-excess reading. Note `ceiling_2x_low` comes in with the OPPOSITE sign to the prior: names already 100% off their 52w low have continued to produce better forward Sharpe in 2022-2026. In the pooled sample the "ceiling" is not a ceiling.
* **Beta / correlation cluster** (`beta_63`, `beta_252`, `corr_spy_63`, `rv20`, `dn_vol_63`): positive in the recent window, flat over the full window, negative vs sector. This is the bull-regime beta bet, not stock selection. The composite and the model are therefore also evaluated beta-neutral (section 5).
* **Compression / dry-up setups** (`range_comp_*`, `bbw_*`, `nr7_count_20`, `vol_dry_*`, `dryup_x_tight`): no pooled signal at these horizons. They show up only conditionally (section 4), in low-vol / defensive names.
* **Ignition / volume-burst factors** (`ignition_mult_10`, `vol_z_1`, `big_vol_updays_net_21`, `ignite_x_rs`): no signal for 21-63d forward Sharpe in S&P 500 names. These are shorter-horizon or small-cap phenomena.
* **Short-term price factors** (`ret_5`, `clv_5`, `streak_now`, `pos_in_20d_range`): nothing. Daily-turnover factors with no 1-3 month payoff.

## 4. Conditional results: where each factor works (recent window, 42d, NW t-stats)

|                         |   beta_low |   beta_mid |   beta_high |   vol_low |   vol_mid |   vol_high |   grp_defensive |   grp_cyclical |   grp_growth |   sector_neutral_vs_sector |
|:------------------------|-----------:|-----------:|------------:|----------:|----------:|-----------:|----------------:|---------------:|-------------:|---------------------------:|
| mom_12_1                |       1.41 |       1.09 |        3.59 |      2.06 |      2.54 |       2.02 |            0.99 |           1.72 |         0.82 |                       1.79 |
| obv_price_div_63        |       0.5  |      -0.34 |        0.91 |      0.36 |      0.91 |      -0.72 |            3.56 |          -0.67 |        -0.8  |                       0.32 |
| dvol_trend_21_126       |      -2.93 |      -0.3  |        1.47 |     -1.76 |     -1.23 |      -0.49 |           -1.3  |          -1.36 |         1.16 |                       0.2  |
| skew_63                 |       0.44 |       0.33 |       -2.1  |     -0.16 |     -0.92 |       1.02 |           -2.24 |           0.56 |         2.88 |                       0.45 |
| corr_spy_63             |       0.83 |       2.72 |        1.17 |      2.09 |      1.83 |       1.85 |            0.72 |           1.83 |         0.43 |                       1.72 |
| ceiling_2x_low          |       1.61 |      -1.1  |        2.71 |      0.76 |      2.17 |       2.22 |           -0.02 |           1.02 |         2.58 |                       2.2  |
| idio_vol_63             |      -1.1  |      -2.39 |        0.47 |     -2.12 |     -2.69 |       1.03 |           -0.04 |          -1.57 |         1.75 |                      -0.06 |
| days_since_2x_low       |      -0.1  |       0.79 |       -2.68 |     -1.08 |     -1.62 |      -1.51 |            0.13 |          -0.82 |        -2.25 |                      -2.54 |
| rs_lead_126             |       1.55 |       1.97 |        1.77 |      2.59 |      1.68 |       2.35 |            1.31 |           2.43 |         1.18 |                       2.11 |
| capture_spread_63       |      -0.93 |       1.63 |        1.7  |     -0.41 |      0.22 |       1.39 |           -1.64 |          -0.07 |         2.54 |                       0.36 |
| days_since_52w_high     |      -1.11 |      -0.59 |       -2.52 |     -0.53 |     -1.43 |      -1.55 |           -0.42 |          -0.63 |        -0.6  |                      -0.75 |
| dist_52w_low            |       0.85 |       1.59 |        2.5  |      1.9  |      2.11 |       2.01 |            0.31 |           1.48 |         1.82 |                       1.79 |
| up_day_frac_63          |      -0.94 |       0.87 |        2.44 |      0.23 |      0.14 |       0.27 |           -0.11 |          -1.09 |         0.76 |                      -0.1  |
| dn_up_vol_asym_63       |       0.72 |      -1.12 |        0.5  |      0.5  |      0.98 |      -0.91 |            2.33 |           1.07 |        -2.42 |                       0.57 |
| base_depth_126          |      -0.14 |       2.36 |        1.1  |      0.42 |      0.85 |       0.36 |           -0.86 |           0.34 |         0.85 |                       0.28 |
| rs_sec_21               |       1.19 |       2.35 |        1.74 |      1.77 |      1.85 |       2    |            0.43 |           0.71 |         1.71 |                       0.91 |
| rs_sec_63               |      -0.33 |       2.34 |        1.33 |      0.75 |      0.24 |       1.28 |           -0.84 |          -0.76 |         1.47 |                      -0.4  |
| rv_ratio_20_60          |       1.48 |      -0.72 |       -0.86 |      0.33 |     -0.16 |      -2.3  |            0.94 |           0.63 |        -1.08 |                      -0.07 |
| rs_dist_high_126        |      -0.17 |       2.3  |        1.18 |      1.57 |      1.1  |       0.65 |           -0.88 |           1.16 |         0.98 |                       0.75 |
| base_depth_63           |      -0.55 |       2.14 |        0.91 |      0.04 |      0.44 |       0.29 |           -1.14 |          -0.13 |         0.85 |                      -0.17 |
| vol_dry_20_250          |      -1.41 |      -0.84 |        0.81 |     -2.12 |     -1.23 |      -0.52 |           -0.32 |           0.07 |         0.17 |                       0.21 |
| days_since_20pct_dd     |       0.83 |       1.64 |        2.09 |      0.31 |      1.31 |       0.6  |           -0.17 |           0.47 |        -0.18 |                       0.32 |
| dd_from_252_high        |       0.32 |       1.71 |        2.06 |      0.67 |      1.49 |       0.9  |           -0.57 |           0.74 |         1.02 |                       0.62 |
| ignition_mult_10        |       1.39 |      -1.25 |       -0.16 |      0.49 |     -0.27 |       1.21 |            1.18 |          -0.35 |         2.01 |                       1.62 |
| rs_newhigh_price_not_20 |      -0.07 |      -0.61 |        0.07 |      0.22 |     -0.67 |      -0.65 |           -2.01 |          -0.28 |        -0.3  |                      -0.23 |
| gap_share_21            |       1.48 |       0.92 |        0.83 |      2    |      1.45 |       1.07 |            0.51 |           1.01 |         1.29 |                       1.46 |
| dist_52w_high           |       0.34 |       1.71 |        2    |      0.64 |      1.46 |       0.86 |           -0.54 |           0.63 |         0.96 |                       0.57 |
| beta_63                 |       0.32 |       1.5  |        1.24 |      1.77 |      1.77 |       1.99 |            0.62 |           1.24 |         1.3  |                       1.59 |
| rv_ratio_5_20           |       1.66 |       0.24 |        0.64 |      1.08 |      1.87 |       1.98 |            0.87 |           1.15 |        -0.08 |                       0.42 |
| up_day_frac_21          |      -0.76 |       1.48 |        1.98 |      0.71 |      0.99 |       0.51 |           -0.08 |           0.34 |         0.97 |                       0.81 |

Multiple-testing note: 86 factors x 10 buckets = 860 tests; the expected maximum |t| under the null is about 3.2, so single cells near 3 are not individually reliable. What is reliable is the pattern:

* **High beta / growth:** the trend-persistence cluster is strongest here (`mom_12_1` t=3.6 in the high-beta tercile, `days_since_52w_high` -2.5, `ceiling_2x_low` +2.7, `days_since_2x_low` -2.7). In high-beta names, buy strength, recent 2x-off-low crossings, and proximity to highs.
* **Low beta / low vol / defensive:** the sign of several price factors flips or dies, and volume-structure factors appear instead: `obv_price_div_63` +3.6 in defensives (volume leading price), `vol_dry_20_250` -2.1 in the low-vol tercile (dry-up works for utilities/staples-type names), `dvol_trend_21_126` -2.9 in low beta (quiet names outperform), `idio_vol_63` negative in low/mid beta. This is the DUK-vs-NVDA asymmetry the mandate anticipated, and it is why the composite is fitted per beta bucket.
* **Per sector** (`conditional_ic_42_per_sector.csv`): indicative only (18-80 names per date). Notable: Utilities `rs_lead_126` +3.1 and dry-up negative; Consumer Defensive `ceiling_2x_low` -3.2 (the ceiling DOES hold in staples) and compression factors positive; Financials show reversal (`ret_63` -2.3).

## 5. Out-of-sample predictive models (walk-forward 2017-2026)

### 5a. Case-study model (LightGBM) vs single-factor baselines


**lightgbm_walkforward**

|                |   IC vs SPY |      t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|-------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.01  |  1.094 |             0.012 |  1.624 |          0.01  |   1.38  |
| (21, 'recent') |      -0     | -0.015 |             0.004 |  0.349 |          0.017 |   1.72  |
| (42, 'full')   |       0.013 |  1.066 |             0.014 |  1.56  |          0.014 |   1.624 |
| (42, 'recent') |       0.003 |  0.198 |             0.003 |  0.204 |          0.013 |   1.071 |
| (63, 'full')   |       0.018 |  1.21  |             0.017 |  1.844 |          0.016 |   1.641 |
| (63, 'recent') |       0.018 |  0.884 |             0.009 |  0.788 |          0.009 |   0.613 |

**baseline_mom_12_1**

|                |   IC vs SPY |     t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.018 | 1.347 |             0.026 |  2.286 |          0.013 |   1.325 |
| (21, 'recent') |       0.038 | 1.918 |             0.044 |  2.534 |          0.037 |   2.337 |
| (42, 'full')   |       0.015 | 0.847 |             0.027 |  1.879 |          0.008 |   0.639 |
| (42, 'recent') |       0.041 | 1.598 |             0.048 |  2.361 |          0.043 |   2.097 |
| (63, 'full')   |       0.014 | 0.659 |             0.03  |  1.93  |          0.006 |   0.414 |
| (63, 'recent') |       0.041 | 1.464 |             0.054 |  2.609 |          0.051 |   2.429 |

**baseline_rs_lead_126**

|                |   IC vs SPY |     t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.009 | 0.718 |             0.006 |  0.64  |         -0.001 |  -0.085 |
| (21, 'recent') |       0.048 | 2.261 |             0.026 |  2.058 |          0.005 |   0.448 |
| (42, 'full')   |       0.015 | 0.897 |             0.006 |  0.524 |         -0     |  -0.007 |
| (42, 'recent') |       0.068 | 2.393 |             0.034 |  2.057 |          0.015 |   0.853 |
| (63, 'full')   |       0.017 | 0.92  |             0.007 |  0.548 |         -0.001 |  -0.079 |
| (63, 'recent') |       0.082 | 2.654 |             0.041 |  2.237 |          0.017 |   0.814 |

**baseline_beta_252**

|                |   IC vs SPY |     t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.008 | 0.415 |             0.001 |  0.508 |         -0.019 |  -2.127 |
| (21, 'recent') |       0.05  | 1.538 |            -0.004 | -1.427 |         -0.013 |  -0.837 |
| (42, 'full')   |       0.02  | 0.775 |             0.004 |  1.248 |         -0.015 |  -1.164 |
| (42, 'recent') |       0.076 | 1.734 |            -0.003 | -0.859 |         -0.011 |  -0.537 |
| (63, 'full')   |       0.027 | 0.868 |             0.003 |  0.978 |         -0.014 |  -0.896 |
| (63, 'recent') |       0.089 | 1.909 |            -0.006 | -1.307 |         -0.014 |  -0.608 |

LightGBM 42d IC by year: 2017: -0.011, 2018: +0.037, 2019: +0.141, 2020: +0.006, 2021: -0.005, 2022: -0.042, 2023: -0.037, 2024: +0.006, 2025: +0.047, 2026: -0.021


**Regularised variants** (rank-only features, 7 leaves, 300 rounds; ridge; with/without decay; no market-context features):


*gbm_reg_rankonly_decay*

|                |   IC vs SPY |     t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.007 | 0.661 |             0.005 |  0.662 |          0.011 |   1.688 |
| (21, 'recent') |       0.006 | 0.385 |             0     |  0.023 |          0.007 |   0.938 |
| (42, 'full')   |       0.011 | 0.823 |             0.009 |  0.915 |          0.016 |   1.824 |
| (42, 'recent') |       0.016 | 0.752 |             0.001 |  0.095 |          0.01  |   1.139 |
| (63, 'full')   |       0.013 | 0.817 |             0.012 |  1.226 |          0.02  |   1.874 |
| (63, 'recent') |       0.023 | 0.935 |             0.007 |  0.504 |          0.009 |   0.93  |

*gbm_reg_rankonly_nodecay*

|                |   IC vs SPY |      t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|-------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.003 |  0.276 |             0.003 |  0.409 |          0.01  |   1.489 |
| (21, 'recent') |      -0.013 | -0.905 |            -0.011 | -1.047 |         -0.002 |  -0.314 |
| (42, 'full')   |       0.008 |  0.611 |             0.008 |  0.821 |          0.016 |   1.805 |
| (42, 'recent') |      -0.002 | -0.105 |            -0.009 | -0.722 |         -0.001 |  -0.094 |
| (63, 'full')   |       0.015 |  0.929 |             0.014 |  1.335 |          0.024 |   2.122 |
| (63, 'recent') |       0.013 |  0.669 |             0.001 |  0.083 |          0.003 |   0.358 |

*ridge_rank_decay*

|                |   IC vs SPY |     t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.011 | 0.95  |             0.009 |  0.945 |          0.016 |   2.414 |
| (21, 'recent') |       0.011 | 0.564 |             0.002 |  0.152 |          0.012 |   1.539 |
| (42, 'full')   |       0.014 | 0.911 |             0.007 |  0.65  |          0.014 |   1.479 |
| (42, 'recent') |       0.023 | 0.927 |             0     |  0.025 |          0.005 |   0.533 |
| (63, 'full')   |       0.016 | 0.922 |             0.009 |  0.789 |          0.016 |   1.283 |
| (63, 'recent') |       0.032 | 1.149 |             0.004 |  0.247 |         -0.003 |  -0.264 |

*gbm_reg_noctx_decay*

|                |   IC vs SPY |      t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|-------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.007 |  0.667 |             0.008 |  0.95  |          0.004 |   0.665 |
| (21, 'recent') |      -0     | -0.001 |            -0     | -0.011 |          0.009 |   1.233 |
| (42, 'full')   |       0.01  |  0.732 |             0.009 |  0.901 |          0.009 |   1.003 |
| (42, 'recent') |       0.01  |  0.468 |             0     |  0.008 |          0.012 |   1.168 |
| (63, 'full')   |       0.014 |  0.846 |             0.012 |  1.12  |          0.012 |   1.149 |
| (63, 'recent') |       0.022 |  0.941 |             0.006 |  0.41  |          0.013 |   1.124 |

### 5b. Walk-forward linear composite


*composite_pooled_wf*

|                |   IC vs SPY |      t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|-------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.005 |  0.37  |            -0.001 | -0.058 |          0.008 |   1.115 |
| (21, 'recent') |      -0.014 | -0.79  |            -0.027 | -2.276 |         -0.013 |  -1.288 |
| (42, 'full')   |       0.006 |  0.376 |            -0.002 | -0.124 |          0.006 |   0.561 |
| (42, 'recent') |      -0.005 | -0.231 |            -0.035 | -2.844 |         -0.019 |  -1.65  |
| (63, 'full')   |       0.01  |  0.526 |             0.002 |  0.138 |          0.007 |   0.49  |
| (63, 'recent') |       0.015 |  0.648 |            -0.024 | -1.685 |         -0.02  |  -1.837 |

*composite_betabucket_wf*

|                |   IC vs SPY |     t |   IC beta-neutral |   t_bn |   IC vs sector |   t_sec |
|:---------------|------------:|------:|------------------:|-------:|---------------:|--------:|
| (21, 'full')   |       0.009 | 0.722 |             0.006 |  0.812 |         -0.005 |  -0.716 |
| (21, 'recent') |       0.027 | 1.109 |             0.002 |  0.205 |          0.002 |   0.163 |
| (42, 'full')   |       0.018 | 1.086 |             0.006 |  0.699 |         -0.003 |  -0.361 |
| (42, 'recent') |       0.047 | 1.463 |             0.006 |  0.471 |          0.005 |   0.268 |
| (63, 'full')   |       0.025 | 1.346 |             0.008 |  0.827 |         -0.002 |  -0.141 |
| (63, 'recent') |       0.065 | 1.905 |             0.015 |  1.097 |          0.006 |   0.294 |

## 6. Live scoring

`current_rankings.csv` columns: `final_score` = within-beta-bucket composite percentile (the ML percentile is reported alongside, not blended, because of its OOS record); `analog_med_xsSharpe42` / `analog_pct` / `analog_hit_rate` = realised 42d SPY-excess Sharpe of the 50 nearest historical cases (median, its percentile across today's names, share > 0); key factor readings follow.


### Final composite weights by beta bucket

|                      |     low |     mid |    high |
|:---------------------|--------:|--------:|--------:|
| accum_dist_50        | nan     |   0.033 | nan     |
| base_depth_126       | nan     |   0.091 | nan     |
| capture_spread_63    | nan     | nan     |   0.075 |
| ceiling_2x_low       | nan     | nan     |   0.081 |
| clv_21               | nan     |   0.042 | nan     |
| clv_5                | nan     |   0.02  | nan     |
| comp_then_exp        |  -0.036 | nan     | nan     |
| corr_spy_63          | nan     |   0.087 | nan     |
| days_since_20pct_dd  |   0.248 |   0.075 |   0.082 |
| days_since_2x_low    | nan     | nan     |  -0.079 |
| days_since_52w_high  | nan     | nan     |  -0.093 |
| dist_52w_low         | nan     |   0.061 | nan     |
| dryup_x_tight        |   0.16  |   0.055 | nan     |
| dvol_trend_21_126    |  -0.101 | nan     |   0.041 |
| higher_low_cadence   | nan     |   0.046 | nan     |
| idio_vol_63          |  -0.242 |  -0.076 | nan     |
| max_up_streak_21     | nan     | nan     |   0.032 |
| mdd_63               |   0.213 | nan     | nan     |
| mom_12_1             | nan     | nan     |   0.136 |
| mom_6_1              | nan     |   0.039 | nan     |
| pullback_21          | nan     |   0.053 |   0.029 |
| pullback_duration_21 | nan     |  -0.037 | nan     |
| rs_lead_126          | nan     |   0.052 |   0.056 |
| rs_sec_21            | nan     |   0.054 | nan     |
| rs_sec_63            | nan     |   0.065 | nan     |
| sharpe_126           | nan     | nan     |   0.095 |
| skew_63              | nan     | nan     |  -0.039 |
| up_day_frac_21       | nan     |   0.056 |   0.052 |
| up_day_frac_63       | nan     |   0.059 |   0.075 |
| vol_cv_20            | nan     | nan     |   0.036 |

### Top 30 as of 2026-10-05

| ticker   | sector                 | beta_bucket   |   final_score |   ml_score |   comp_score |   analog_pct |   analog_hit_rate |   mom_12_1 |   dist_52w_high |   dist_52w_low |   rs_lead_126 |   beta_252 |   rv20_pct_252 |   vol_dry_20_250 |   obv_price_div_63 |
|:---------|:-----------------------|:--------------|--------------:|-----------:|-------------:|-------------:|------------------:|-----------:|----------------:|---------------:|--------------:|-----------:|---------------:|-----------------:|-------------------:|
| BRK-B    | Financial              | low           |          1    |       0.53 |         0.51 |         0.87 |              0.48 |       0.02 |           -0.06 |           0.09 |         -0.06 |       0.01 |           0.86 |             0.99 |               1.36 |
| NTAP     | Technology             | high          |          1    |       0.56 |         0.66 |         0.32 |              0.36 |       0.57 |           -0.04 |           1.41 |         -0.01 |       1.31 |           1.3  |             1.02 |               1.3  |
| EXPD     | Industrials            | mid           |          1    |       0.5  |         0.77 |         0.58 |              0.34 |       0.54 |           -0.01 |           0.74 |         -0.02 |       0.53 |           1.05 |             0.87 |               1.13 |
| ATO      | Utilities              | low           |          0.99 |       0.58 |         0.49 |         0.32 |              0.44 |       0.02 |           -0.17 |           0.02 |         -0.11 |      -0.16 |           0.68 |             0.9  |              -0.09 |
| NDSN     | Industrials            | mid           |          0.99 |       0.54 |         0.75 |         0.96 |              0.54 |       0.38 |           -0.01 |           0.52 |         -0.02 |       0.84 |           0.87 |             0.92 |              -0.59 |
| P        | Technology             | high          |          0.99 |       0.63 |         0.64 |         0.42 |              0.36 |       0.11 |           -0    |           1.53 |          0    |       2.44 |           0.98 |             2.09 |               0.7  |
| ED       | Utilities              | low           |          0.99 |       0.63 |         0.47 |         0.47 |              0.38 |       0.15 |           -0.1  |           0.12 |         -0.14 |      -0.46 |           0.79 |             1    |               1.01 |
| IEX      | Industrials            | mid           |          0.99 |       0.52 |         0.71 |         0.13 |              0.3  |       0.37 |           -0.03 |           0.5  |         -0.02 |       0.65 |           0.71 |             0.88 |               1.12 |
| CRWD     | Technology             | high          |          0.99 |       0.56 |         0.61 |         0.4  |              0.38 |       0.73 |           -0.01 |           2.18 |          0    |       1.49 |           1.11 |             0.87 |              -1    |
| REG      | Real Estate            | low           |          0.98 |       0.57 |         0.45 |         0.36 |              0.4  |       0.09 |           -0.14 |           0.11 |         -0.08 |       0.05 |           0.61 |             1.05 |              -0.41 |
| WAT      | Healthcare             | mid           |          0.98 |       0.45 |         0.7  |         0.51 |              0.38 |       0.28 |           -0.02 |           0.56 |         -0.01 |       0.84 |           0.88 |             0.84 |               1.12 |
| KEYS     | Technology             | high          |          0.98 |       0.57 |         0.61 |         0.43 |              0.38 |       0.84 |           -0.01 |           1.42 |         -0.01 |       1.85 |           0.97 |             0.78 |              -0.62 |
| WELL     | Real Estate            | low           |          0.98 |       0.61 |         0.44 |         0.85 |              0.5  |       0.39 |           -0.12 |           0.39 |         -0.04 |      -0.02 |           0.68 |             0.83 |              -0.71 |
| MET      | Financial              | mid           |          0.98 |       0.51 |         0.68 |         0.26 |              0.36 |       0.25 |           -0.04 |           0.46 |         -0.03 |       0.66 |           0.76 |             0.89 |               0.77 |
| HPE      | Technology             | high          |          0.98 |       0.63 |         0.6  |         0.72 |              0.46 |       1.26 |           -0.03 |           2.49 |         -0.01 |       2.05 |           1.6  |             0.96 |              -0.23 |
| ABBV     | Healthcare             | low           |          0.97 |       0.53 |         0.43 |         0.61 |              0.42 |       0.13 |           -0.01 |           0.4  |         -0.05 |      -0.11 |           0.66 |             0.71 |               0.32 |
| ILMN     | Healthcare             | mid           |          0.97 |       0.52 |         0.68 |         0.1  |              0.32 |       1.23 |           -0.01 |           2.34 |          0    |       0.8  |           1.32 |             2.09 |              -1.82 |
| PANW     | Technology             | high          |          0.97 |       0.53 |         0.59 |         0.8  |              0.48 |       0.59 |           -0.01 |           1.91 |          0    |       1.18 |           1.26 |             0.81 |              -0.07 |
| D        | Utilities              | low           |          0.96 |       0.63 |         0.43 |         0.62 |              0.46 |       0.13 |           -0.16 |           0.13 |         -0.04 |      -0.07 |           0.81 |             0.89 |               0.3  |
| WBD      | Communication Services | mid           |          0.96 |       0.47 |         0.68 |         0.91 |              0.52 |       0.47 |           -0    |           0.81 |         -0.04 |       0.5  |           1.88 |             2.23 |               1.92 |
| AMD      | Technology             | high          |          0.96 |       0.53 |         0.57 |         0.74 |              0.4  |       1.69 |           -0.02 |           2.87 |         -0.01 |       3.2  |           0.79 |             0.61 |               0.48 |
| KO       | Consumer Defensive     | low           |          0.96 |       0.52 |         0.42 |         0.01 |              0.22 |       0.37 |           -0.06 |           0.35 |         -0.03 |      -0.25 |           0.72 |             0.96 |               0.91 |
| FTNT     | Technology             | high          |          0.96 |       0.46 |         0.57 |         0.14 |              0.36 |       0.81 |           -0    |           1.5  |          0    |       0.92 |           1    |             0.69 |              -0.6  |
| PH       | Industrials            | mid           |          0.96 |       0.54 |         0.65 |         0.45 |              0.44 |       0.28 |           -0.11 |           0.38 |         -0.04 |       0.83 |           0.73 |             0.82 |               0.68 |
| PPL      | Utilities              | low           |          0.95 |       0.58 |         0.42 |         0.86 |              0.5  |      -0.01 |           -0.17 |           0.04 |         -0.11 |      -0.03 |           0.78 |             0.89 |              -1.32 |
| A        | Healthcare             | mid           |          0.95 |       0.47 |         0.64 |         0.84 |              0.48 |       0.09 |           -0.03 |           0.58 |         -0.01 |       0.85 |           1.2  |             1.14 |              -1.1  |
| DELL     | Technology             | high          |          0.95 |       0.62 |         0.56 |         0.53 |              0.4  |       2.55 |           -0.07 |           4.03 |         -0.02 |       2.23 |           1.1  |             1.02 |               0.55 |
| PG       | Consumer Defensive     | low           |          0.95 |       0.5  |         0.41 |         1    |              0.66 |      -0.01 |           -0.11 |           0.08 |         -0.08 |      -0.03 |           0.9  |             0.84 |               1.41 |
| CRL      | Healthcare             | high          |          0.95 |       0.54 |         0.55 |         0.57 |              0.46 |       0.67 |           -0    |           1.13 |          0    |       1.25 |           0.95 |             1.07 |              -1.08 |
| VEEV     | Healthcare             | mid           |          0.95 |       0.45 |         0.63 |         0.83 |              0.46 |      -0.05 |           -0.09 |           0.91 |         -0.01 |       0.56 |           0.74 |             0.71 |               0.13 |

## 7. Ticker identity vs. behaviour: how much of the signal generalises across tickers

Each reading is split, point-in-time, into **persistent** (the ticker's own expanding-window mean up to t-1: "this ticker is usually like this") and **deviation** (today's reading as a z-score against the ticker's own history: "this ticker is currently unusual"). IC of each component vs 42d SPY-excess Sharpe:

| factor              |   recent|raw |   recent|persistent |   recent|deviation |   full|raw |   full|persistent |   full|deviation |
|:--------------------|-------------:|--------------------:|-------------------:|-----------:|------------------:|-----------------:|
| comp_bucketed       |        0.047 |               0.053 |              0.035 |      0.015 |             0.017 |            0.016 |
| corr_spy_63         |        0.077 |               0.055 |              0.058 |      0.028 |             0.033 |            0.007 |
| days_since_20pct_dd |       -0.008 |              -0.043 |              0.021 |      0.009 |             0.006 |            0.008 |
| dist_52w_high       |       -0.004 |              -0.032 |              0.005 |      0.009 |             0.015 |           -0     |
| dryup_x_tight       |       -0.003 |              -0.015 |              0.001 |      0.006 |             0.018 |            0.001 |
| idio_vol_63         |        0.006 |               0.028 |             -0.018 |     -0.024 |            -0.008 |           -0.017 |
| mom_12_1            |        0.039 |               0.019 |              0.044 |      0.007 |             0.022 |            0.003 |
| obv_price_div_63    |        0.01  |               0.023 |              0.007 |      0.006 |             0.01  |            0.004 |
| rs_lead_126         |        0.068 |               0.067 |              0.031 |      0.004 |             0.032 |           -0.013 |
| sharpe_126          |        0.016 |               0.009 |              0.012 |      0.003 |             0.021 |           -0.006 |
| up_capture_63       |        0.073 |               0.064 |              0.028 |      0.014 |             0.021 |           -0.007 |
| vol_dry_20_250      |       -0.01  |              -0.021 |             -0.009 |      0.001 |            -0.018 |           -0     |

Pure ticker-identity predictors (long trailing Sharpe; the ticker's own trailing-year mean realised excess Sharpe):

| predictor       |   ic|full |   ic|recent |   t_nw|full |   t_nw|recent |
|:----------------|----------:|------------:|------------:|--------------:|
| past_xs_1y_mean |     0.015 |       0.044 |       0.927 |         1.699 |
| sharpe_252      |     0.013 |       0.042 |       0.758 |         1.615 |
| sharpe_504      |     0.013 |       0.026 |       0.651 |         0.865 |
| sharpe_756      |     0.024 |       0.034 |       1.266 |         1.303 |

Reading:

* The out-of-sample composite's recent IC (0.047) is matched by its persistent component alone (0.053); the deviation component carries 0.035. Today's composite rank correlates 0.580 with the ticker's own historical-average composite rank. **Roughly half to two thirds of what the ranking expresses is which tickers are chronically strong and stable, not a current state change.**
* A ticker's own trailing-year mean realised excess Sharpe predicts its next 42 days with IC 0.044 (t 1.7) in the recent window, about the same as the whole system. In 2022-10 to 2026 leadership persisted at the ticker level; much of the system's recent edge is that persistence.
* The persistent component is the part most exposed to survivorship bias (today's constituents are disproportionately the names that were chronically strong). Treat the identity half of the score as regime- and universe-dependent.
* The deviation component is the cleaner "behaviour predicts across tickers" evidence. It is weaker but real: 12-1 momentum unusually high for the name (IC 0.044, t 2.0, the only reading where the state component beats the identity component), RS-line leadership unusual for the name (0.031), correlation to SPY unusually high (0.058), idiosyncratic vol unusually low (-0.018). Proximity to 52w high, time since 20% drawdown and the dry-up interaction have no deviation signal pooled: they work only as identity, and only inside beta buckets.
* Volume factors: `obv_price_div_63` and `vol_dry_20_250` show up as identity, not state, pooled. The names that habitually have volume leading price, or habitually dry up, do better; a dry-up that is unusual for the name does not.


## 8. What to trust, what not to

Direct read of the evidence, in order of reliability.

**1. What has out-of-sample support for 21-63d forward Sharpe in this universe**

* Relative-strength leadership: `rs_lead_126` (RS line closer to its 6-month high than price is to its own). Walk-forward baseline, recent window: 42d IC 0.068 (NW t 2.4), 63d IC 0.082 (t 2.7); still 0.034-0.041 (t 2.1-2.2) after removing beta. The single most useful reading in the library.
* 12-1 momentum: recent beta-neutral 42d IC 0.048 (t 2.4), sector-excess IC 0.043 (t 2.1). Dead over 2014-2021 (full-window IC 0.005), alive since 2022-10. Consistent with the regime shift the mandate describes.
* Beta-bucketed walk-forward composite: recent 42d IC 0.047 (t 1.5), 63d 0.065 (t 1.9), top-minus-bottom decile 42d Sharpe spread +0.29 annualised with near-monotone deciles. The pooled composite has nothing (recent IC negative). Conditioning on beta is what makes the composite work. Its beta-neutral IC is small (0.006-0.015): inside each bucket it still leans on correlation / up-capture type factors, so part of its recent edge is bull-regime beta.
* Low-beta / defensive names respond to volume structure, not price structure: OBV leading price (`obv_price_div_63`, t 3.6 in defensives), volume dry-up vs 250d (`vol_dry_20_250`, t -2.1 in the low-vol tercile), falling dollar volume (`dvol_trend_21_126`, t -2.9 in low beta). Treat as a cluster, not three independent facts.

**2. What did not show signal at this horizon in S&P 500 names (tested, not assumed)**

* Compression setups (daily-range percentile, Bollinger width, NR7 clusters, base tightness), pooled: IC within +/-0.015 in every window and horizon. They only appear inside the low-vol / defensive bucket. The "compression then expansion" sequence factor is flat everywhere.
* Ignition-day volume multiple, volume z-score, big-volume up-day counts, the ignition x RS interaction: flat. Nothing in a 1-10 day volume burst predicts the next 1-3 months of risk-adjusted return here.
* Accumulation/distribution day counts, up/down volume ratio: flat pooled, mildly negative in places.
* Higher-low cadence, close location value, gap frequency, streaks, 20d range position: flat.
* The 100%-off-low "ceiling": wrong sign pooled and in high beta (names already 2x off the low kept producing better Sharpe, t +2.3 to +2.7); right sign only in Consumer Defensive (t -3.2). Do not use it as a universal exclusion.
* The LightGBM case-study model, in all five configurations: recent-window 42d IC between -0.001 and +0.023, none significant, every variant below the two single-factor baselines. A great 2019 (IC 0.11-0.14) and nothing stable since. Non-linear interaction mining over 86 factors does not beat two robust factors at this horizon with ~10 years of 500-name data. The analog columns in the ranking are therefore descriptive evidence per name, not a forecast.

**3. Statistical honesty**

* Identity vs behaviour (section 7): about half of the composite's recent IC is reproduced by the ticker's own historical average of the score. That half is regime- and survivorship-exposed. The generalising half is led by momentum / RS leadership unusual for the name.

* No single factor passes a 5% false-discovery-rate test across the 86-factor zoo in either window. The recent window has roughly 24 independent 42-day periods; |t| of 2-2.7 is the ceiling a true IC of 0.05-0.08 can reach there. The claims above rest on (a) coherent clusters pointing the same way, (b) survival after beta-neutralisation, (c) walk-forward reproduction, not on any one t-stat.
* Multiple-testing in the conditional tables: 860 bucket tests (expected max |t| under the null about 3.2) and 946 per-sector tests. Cells near 3 are suggestive only.
* Survivorship: today's constituents. Ranks are robust to it; absolute forward-Sharpe levels (the deciles all sit near -1.1 in the recent window because SPY's own realised Sharpe was high) are not the point.
* Variant selection (5 ML configurations, 2 composite variants) is a mild in-sample choice and is disclosed; the picked composite is the beta-bucketed one.

**4. How to use the ranking**

* `final_score` is the within-beta-bucket composite percentile, so the top of the list mixes low-, mid- and high-beta leaders by design. Read it with `beta_bucket`: high-beta names are there for trend persistence (RS leading, 12-1 momentum, proximity to highs, recent 2x-off-low crossings); low-beta names are there for quiet volume-led accumulation.
* Expected magnitude: a top-decile vs bottom-decile difference of roughly 0.3 annualised Sharpe over the next 42 days, with the top decile beating the bottom in about 57% of windows. That is a real but modest edge; it is a screen that concentrates probability, not a signal that resolves individual names.
* Refit cadence: the factors turn over slowly (21d rank autocorrelation 0.7-0.9 for the ones that matter), so re-scoring weekly is sufficient; the walk-forward re-estimates weights annually and that was enough.


## 9. Files

| file | content |
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
| `identity_decomposition.csv`, `identity_pure_predictors.csv` | ticker-identity vs own-history-deviation split of the signal |
| `current_rankings.csv` | the live ranking with analog evidence |


## 10. Reproduce

```
pip install -e . scikit-learn lightgbm pyarrow scipy statsmodels tabulate
python -m pvv_score.data_io          # download
python -m pvv_score.build            # features + targets
python -m pvv_score.run_eval         # single-factor
python -m pvv_score.run_conditional  # conditional
python -m pvv_score.model            # walk-forward ML
python -m pvv_score.run_model_variants
python -m pvv_score.composite        # walk-forward composite
python -m pvv_score.score_now        # live ranking
python -m pvv_score.run_identity     # identity vs behaviour decomposition
python -m pvv_score.report
```
