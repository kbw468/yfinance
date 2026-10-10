# Volume volatility factor backtest

Universe: 2,367 tickers from the finviz export. Daily adjusted bars via yfinance, 2016-10-10 to 2026-10-09.
Observation filter: price >= $1, 20d median dollar volume >= $500k.

## Factors (known at close t)

- **VV**: 20-session stdev of ln(V_t / V_{t-1}), the realized volatility of daily volume.
- **VV_ROC**: VV_t / VV_{t-10} - 1, the 10-session rate of change in VV.
- Each factor is ranked against the ticker's own trailing 252 sessions and bucketed into quintiles (Q1 low, Q5 high).
- Robustness checks: a cross-sectional rank (vs. the universe that day) and an alternate VV definition, CV = stdev/mean of volume levels (`results/cv`).

## Conditioning

- UP day: close_t > close_{t-1}. DOWN day: close_t < close_{t-1}.

## Forward risk-adjusted return (entry at close t)

- RA_h = fwd_ret_h / (trailing 20d sigma x sqrt(h)), winsorized at 0.5%/99.5%.
- RAX_h = RA_h minus that day's universe average RA_h. Zero means in line with the universe.
- h = 7, 14, 21 and 42 sessions.
- Each bucket is averaged across names per date. t-stats are Newey-West with lag h.

## Run

```
python download.py finviz.csv data 10y
python backtest.py data/bars.parquet results/dlogv dlogv
python backtest.py data/bars.parquet results/cv cv
python summarize.py results/dlogv
```

## Outputs (`results/<variant>/`)

- `RESULTS.md`: all summary tables.
- `buckets_time_series.csv` / `buckets_cross_section.csv`: every factor x day x horizon x quintile, with RA, RAX and raw return, t-stat and hit rate.
- `per_ticker.csv`: per-ticker Q5 vs Q1 RAX and IC for each factor x day x horizon.
- `combo_vv_x_roc.csv`: VV tercile x VV_ROC tercile grid, by day type.
- `yearly_q5_minus_q1.csv`: Q5-Q1 spread by calendar year.

## Lookback sweep

`python sweep.py data/bars.parquet results/sweep` tests VV windows 5/10/20/40/60 and ROC lags 3/5/10/20 for each window.
`python roc_vs_level.py data/bars.parquet` checks whether ROC adds anything once VV level is held fixed.
Outputs are in `results/sweep/`.
