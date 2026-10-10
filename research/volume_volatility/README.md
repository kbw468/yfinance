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
python backtest.py data/bars.parquet results/dlogv                       # VV window 20, ROC lag 10
python backtest.py data/bars.parquet results/dlogv_w60 --vv-win 60       # other window
python backtest.py data/bars.parquet results/cv --def cv                 # CV definition
python summarize.py results/dlogv
python sweep.py data/bars.parquet results/sweep                          # window x lag sweep, IS/OOS
python compare.py out.md VV20=results/dlogv VV60=results/dlogv_w60       # side by side
```
Other flags on `backtest.py`: `--roc-lag`, `--rank-win`, and `--eval-start` (scores every setting on the same dates).

## Outputs (`results/<variant>/`)

- `RESULTS.md`: all summary tables.
- `buckets_time_series.csv` / `buckets_cross_section.csv`: every factor x day x horizon x quintile, with RA, RAX and raw return, t-stat and hit rate.
- `per_ticker.csv`: per-ticker Q5 vs Q1 RAX and IC for each factor x day x horizon.
- `combo_vv_x_roc.csv`: VV tercile x VV_ROC tercile grid, by day type.
- `yearly_q5_minus_q1.csv`: Q5-Q1 spread by calendar year.

## Lookback sweep and validation

`sweep.py` tests VV windows 5-120 and ROC lags 3-60. Every setting is scored on the same dates from 2018-07-01.
The sample is split in-sample (2018-07 to 2022-06) and out-of-sample (2022-07 onward).
ROC is also scored with VV level held fixed: its Q5-Q1 inside each VV tercile, averaged.
`compare.py` puts runs side by side: other VV definitions, rank lookbacks, cross-sectional rank, per-ticker breadth, yearly spreads and quintile grids.

- `results/sweep/`: `SWEEP.md` and `sweep.csv`.
- `results/compare/COMPARE_common_sample.md`: VV 20/40/60, CV 20/40/60 and rank 126/504, all from 2018-07.
- `results/compare/COMPARE_full_sample.md`: VV 20/40/60 over the full sample, by year.
- `results/dlogv_w40/`, `results/dlogv_w60/`: full outputs at those windows.

## Findings

- High VV (own-history Q5) underperforms on both UP and DOWN days. The sign holds for every window from 5 to 60 in both halves.
- Windows of 5-15 are weak. Windows of 40-60 are the strongest and most stable.
- In-sample window ranking does not predict out-of-sample ranking (rank corr -0.15). Treat 40-60 as a range, not a tuned point.
- VV60 Q5-Q1 is negative on DOWN days in 9 of 10 years. 2021-2022 is flat for every window.
- VV_ROC with VV level held fixed is negative in-sample and roughly zero out-of-sample at almost every setting. It has no edge beyond the VV level.
