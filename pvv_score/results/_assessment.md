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
