# Volume / volatility microstructure screens — backtest on 431-name finviz list

As-of: 2026-10-02 close (Monday 10/05 partial bar dropped). Data: yfinance daily OHLCV, 7y, auto-adjusted.
Universe: `universe_finviz.csv` (435 tickers; FDXF, HONA, Q, VYLR had no history).

## Pipeline
1. `dl.py` – download panel to `px.pkl`
2. `feats.py` – every screen in the catalog computed daily (see column list in `screens_2026-10-02.csv`), stacked weekly from 2020-10 with forward 21d/63d targets
3. `bt2.py` – rank IC per feature vs forward horizon-Sharpe (fwd return / fwd realised vol), raw and sector-neutral, HAC t-stats
4. `bt3.py` – threshold state screens, sector-demeaned forward excess, HAC t-stats
5. `bt4.py` – walk-forward composite (expanding-window feature selection |t|>1.5, OOS from 2022-02)
6. `bt5.py` – subperiod robustness of the screens that cleared t>2
7. `bt6.py` – corrected accumulation×compression definition re-test
8. `snap.py` – current snapshot, quadrant/flow labels, levels, CSV

## Result
No feature in the catalog has a robust cross-sectional forward 21d/63d risk-adjusted signal on this universe
(302 weekly dates, ~430 names, 2020-10 → 2026-07). Max |HAC t| of any single-feature rank IC: 2.1 (clv, 63d, negative = reversal).
Realised-vol ratios (5/21, 10/21, 21/63, 21/126, 21/252, 63/252, own-history percentile): all |t| < 1.5.
Walk-forward composite OOS IC: −0.017 (t −1.9) at 21d, −0.012 at 63d. In-sample winners do not hold.
The one in-sample "winner" (accumulation × vol-compression, t 2.9) was a sign bug: negative × negative. Corrected: t ≈ 0.

Only consistently-signed effects:
- One down day ≥ 50% of 21d variance → negative forward excess, all three subperiods, t −2.9 at 63d in 2024-10→2026-07. Avoid list.
- Mild 1-month reversal at 63d (ret21, clv, up-down share all negative), not stable across subperiods.

`screens_2026-10-02.csv` holds every measure for every name as of 10/02 for descriptive use.
