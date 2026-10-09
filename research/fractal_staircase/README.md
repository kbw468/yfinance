# Staircase precursor study

Backtest of what precedes a smooth, low-drawdown 21–63 session advance across the
2,367-ticker universe in `universe.csv`, using Yahoo Finance daily OHLCV only.
Every statistic is distribution-free: medians, ranks, ranges, drawdowns, Beta
posteriors, label-shuffle nulls. No standard deviation, z-score, OLS, Sharpe,
t-test or moving average.

## Hit definitions

* `y63`: forward 63-session log return >= +10% and max peak-to-trough drawdown
  along the path <= 1/3 of that gain.
* `y21`: +5% within 21 sessions, same 1/3 rule.
* Lift is regime-neutral: hits / sum of each row's own-week universe hit rate.

## Pipeline

Set `FS_DATA` to a scratch directory (defaults to `./data`, which is git-ignored).

| step | script | output |
|---|---|---|
| 1 | `download.py` | `ohlcv.parquet` (Yahoo, adjusted, 2004 →) |
| 2 | `features.py` | weekly per-ticker features + forward outcomes |
| 3 | `xsec.py` | universe / sector / industry ranks, group breadth, regime |
| 4 | `analog.py` | shape match to the reference chart at 40/80/160 bars |
| 5 | `univariate.py`, `ic.py`, `conditional.py` | single-feature lifts, rank ICs, conditional states |
| 6 | `interactions.py`, `grids.py` | ROC triad cubes, 5×5 lift grids |
| 7 | `hypothesis.py`, `robust.py` | thesis ladder vs quiet-base ladder, robustness grid, confirmers, segments |
| 8 | `rules.py` | conjunction rule search, train 2006–2016 / test 2017–2026, shuffle null |
| 9 | `shapes.py` | k-medians shape alphabet at 63/126 bars |
| 10 | `model.py` | walk-forward gradient-boosted trees (`binary`, `binary_xs`, `rank`) |
| 11 | `regime.py`, `segments.py` | when staircases cluster; per-sector leaders |
| 12 | `scan.py` | latest-session scan → `results/scan_<date>.csv` |
| 13 | `build_report.py` | `report.html` |

`results/` holds every table the report is built from.
