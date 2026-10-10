# Steady-alpha track (MSI-2024 type): rebuild and findings

Separate from THE LIST. Nothing here changes the production model.

## Rules
- No Gaussian statistics anywhere: no standard deviation, variance, z-scores, Sharpe, OLS, correlation, skew, Bollinger.
- Features (features.py, 75): returns and excess returns over windows, distance to highs and lows, sessions since highs,
  drawdown paths (deepest and mean, inside 21-252 sessions), efficiency, shares of up days / up pairs / new highs, gain-to-pain,
  higher lows, close location, gap share, median absolute move and median range and their rates of change, own-history
  percentiles, median-volume rates of change, up-volume share, pullback volume, accumulation minus distribution, dollar volume,
  up/down capture and the L1 beta  sum(r * sign(r_SPY)) / sum(|r_SPY|)  (rank correlation 0.985 with OLS beta).
- Model input: same-day cross-sectional percentile ranks only. Raw levels carry the market regime: on them a shuffled-label
  control scored 0.39-0.46 within-date AUC; on ranks it sits at 0.48-0.51.
- Universe: every eligible name (price >= $5, $10M median dollar volume, 252 sessions), 2014-06 to date, COVID Feb-Jun 2020 out of training.
- Walk-forward: test years 2018-2026, training ends 100 calendar days before each test year, every 3rd session, fixed LightGBM
  hyperparameters, isotonic calibration on earlier years only. Skill reported as the mean of per-date AUCs (name selection only).

## Outcomes (outcomes.py), 21 / 42 / 63 sessions
excess return vs SPY, deepest close-to-close drawdown on the path, lowest intraday low vs entry (8% stop).

## Findings
1. First target (beat SPY, drawdown cap, no stop): learnable, but the model met it with low-beta names that rarely draw down and lag
   SPY. Top-20 book 2018+: 7-8% a year vs SPY 14.7%. Retired.
2. Alpha-per-pain target (excess / max(|drawdown|, floor), top 20% of the date, positive, no stop): within-date AUC 0.50-0.56,
   deciles flat. MSI sat in deciles 1-4 during its 2024 run.
3. Decomposition, 42 sessions, 2019-2025 (decomposition.txt): beat SPY 0.50; excess top 20% 0.615 and bottom 20% 0.594 (it is
   predicting size of move, not direction); shallow, unstopped path 0.73; beat SPY inside the low-risk names 0.51. Every single
   directional signal (12m / 6m / 63d momentum, efficiency, up-day share, gain-to-pain, volume) scores 0.48-0.53.
   Direction of 21-63 session excess return is not forecastable from price, volume and volatility in this universe.
4. Path risk is forecastable: within-date AUC 0.715 / 0.732 / 0.737 at 21 / 42 / 63 sessions, every year 0.67-0.78.
   Holding beta equal, the low-risk third has the same median excess and 10-17 points fewer stop-outs (pathrisk_beta_matched.txt).
5. Book level (book_no_stops.txt, book_with_stops.txt): the low-path-risk books trail the high-risk books in most bands and
   periods, with or without an executed 8% stop. The filter halves stop-outs but excludes the names that carried the return.

## Files
features.py, outcomes.py, build.py (table), model.py (alpha-per-pain walk-forward), control.py, decompose.py, pathrisk.py,
book.py, book_stops.py, evaluate.py, portfolio.py, today.py (path-risk scores for every name on the latest session:
results/steady/pathrisk_latest.csv).
