# pvv_score

Price / volume / volatility factor research and live scoring for 21-63 trading-day forward
risk-adjusted returns (Sharpe / Sortino vs SPY and vs sector ETF) on the S&P 500 universe in
`data/universe_finviz.csv`. Data comes from this repository's `yfinance`.

Read `results/REPORT.md` first. It holds the method, the full factor table, the conditional
(beta / vol / sector) results, the out-of-sample model results and the live ranking with the
assessment of what is and is not supported by the evidence.

| module | role |
|---|---|
| `config.py` | universe, sector-ETF map, horizons, backtest / recent windows, recency weight |
| `data_io.py` | yfinance download (OHLCV, dividend- and split-adjusted), parquet cache |
| `clean.py` | data-quality rules motivated by the audit (pre-listing stubs, partial session, OHLC repair, unadjusted corporate actions) |
| `features.py` | 86 point-in-time factors + eligibility mask; unit-tested vs hand calculations and a look-ahead perturbation test |
| `targets.py` | forward Sharpe / Sortino vs SPY and sector ETF, 21/42/63d; blended percentile label |
| `build.py` | assembles the long research table |
| `evaluate.py` | daily rank IC, Newey-West t, decile spreads, FDR, factor autocorrelation |
| `run_eval.py` | stage 1: single-factor evaluation |
| `run_conditional.py` | stage 2: beta / vol / sector-group / per-sector / sector-neutral evaluation |
| `model.py`, `run_model_variants.py` | stage 3a: walk-forward LightGBM case-study model (+ regularised variants, ridge) |
| `composite.py` | stage 3b: walk-forward linear composite, pooled and beta-bucketed |
| `score_now.py` | stage 4: live ranking with nearest-neighbour historical analogs |
| `report.py` | writes `results/REPORT.md` from the result files |

Run order and dependencies are in the report's last section.
