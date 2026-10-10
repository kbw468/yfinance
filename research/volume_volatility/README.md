# Volume volatility factor backtest

Universe: 2,367 tickers from the finviz export. Daily adjusted bars via yfinance, 2005-01 to 2026-10 (first rounds used 2016-10 to 2026-10).
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

## Report

`Volume_Volatility_Backtest.pdf` holds the findings for 2009-2026. Rebuild it with:
```
python report.py <sweep_dir> <sweep_rawvolume_dir> <runs_dir> Volume_Volatility_Backtest.pdf
```

## Run

```
python download.py finviz.csv data20 2005-01-01                          # history back to 2005
python sweep.py data20/bars.parquet results/sweep_2009_2026              # VV windows 5-252 x ROC lags 3-252
python sweep.py data20/bars.parquet results/sweep_rawvol --raw-volume    # cleaning check
python backtest.py data20/bars.parquet runs/w20 --vv-win 20 --eval-start 2009-01-01 --min-history 756
python compare.py out.md VV20=runs/w20 VV60=runs/w60                     # side by side
```
Other `backtest.py` flags: `--def cv`, `--roc-lag`, `--rank-win`, `--raw-volume`, `--min-history`.
Volume cleaning is on by default: prints below 2% of the trailing 20d median are dropped, and daily log-volume changes are capped at 50x.

## Results layout

- `results/2009_2026/<run>/`: full backtest outputs behind the PDF. Runs: w20, w50, w60, w120, w252, w20_cv, w60_cv, w20_rank126, w20_rank504, w20_raw, w20_alluniverse. Per-ticker results are in w20 only.
- `results/sweep_2009_2026/`, `results/sweep_2009_2026_rawvolume/`: window x lag sweep with three periods: 2009-01..2016-09, 2016-10..2022-06 and 2022-07..2026-10.
- Earlier rounds on 2016-2026 data: `results/dlogv`, `results/cv`, `results/dlogv_w40`, `results/dlogv_w60`, `results/sweep_2018_2026`, `results/compare`.

## Findings (2009-2026)

- High VV (own-history Q5) underperforms on both UP and DOWN days. The 20-session window is strongest: DOWN Q5-Q1 t -3.7 to -4.7, negative in all 17 full years from 2009 to 2025.
- Windows of 10-50 work. 60-252 only worked in 2022-2026. 252 is the weakest. Window rankings do not carry across periods.
- Slow VV_ROC (window 50-120, lag 10-20, VV level held fixed) worked 2009-2022 and has been flat since. No setting holds in all three periods.
- Fast VV_ROC (window 5-15, lag 3-5, VV level held fixed) is positive in all three periods: a short-term jump in VV outperforms. Full-sample t +2.1 to +2.6.
- Volume cleaning, CV definition, rank lookback 126/504 and the no-history-minimum universe all keep the 20-session result's sign.
