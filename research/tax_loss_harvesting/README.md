# Q4 tax-loss-harvesting seasonality study

Backtest of how trailing 1Y / 1M losers vs winners behave in Q4 and around year end,
across the 2,367-ticker list in `universe_finviz.csv`, with the live Q4 2026 screen.

The report (`Q4_Tax_Loss_Seasonality_2018-2025.pdf`) covers Q4 2018 through Q4 2025; the full-history
1999-2025 tables stay in `results/`.

Run in order (`DATA` holds downloaded prices, `OUT` the results):

```
python download.py universe_finviz.csv DATA        # adjusted closes + volume via yfinance
python -c "import yfinance as yf; yf.download('^IRX', start='1998-06-01', auto_adjust=True)['Close'].to_parquet('DATA/irx.parquet')"
python q4_panel.py DATA OUT                         # per-stock Sep-30 signals + forward windows
python monthly_seasonality.py DATA OUT              # every month-end sort, calendar control
python analyze.py DATA OUT                          # quarter risk, long/short, trough, Dec-15 rebound, volume, live screen
python halfmonth.py OUT
python trade_window.py DATA OUT
python robustness.py DATA OUT
python export_dashboard.py OUT universe_finviz.csv DASH
python study_since.py DATA OUT 2018                # 2018+ tables for the report
python megacap.py DATA universe_finviz.csv OUT 2018 # top-100/top-50 point-in-time cut + per-name history
python cells.py DATA OUT 2018                      # finer baskets (1Y rank x September)
python candidates.py OUT universe_finviz.csv        # Q4 2026 names in baskets A/B/C
python rank_strategies.py DATA universe_finviz.csv OUT 2018   # every strategy by market-cap tier
python make_pdf.py OUT/since_2018.json OUT/megacap_2018.json OUT/cells_2018.csv \
    OUT/q4_2026_candidates.csv OUT/quintiles_2026.json OUT/strategy_rank_2018.json OUT/sep_sort_2026_tiers.csv \
    DASH Q4_Tax_Loss_Seasonality_2018-2025.pdf
```

Method: formation at the last September close; 1Y = trailing 12-month total return, 1M = September
return; eligibility = full 12M history and 63-day median dollar volume >= $1M. Buckets are
equal-weight buy-and-hold. Daily returns clipped to [-80%, +150%]. Risk stats pool daily returns
inside each window across years; alpha/beta vs SPY, Sharpe over 13-week T-bills. Eras: 1999-2007,
2008-2016, 2017-2025. Universe is current survivors only.

`results/` holds the output tables (the 7 MB stock-level panel and the 1 MB per-year path file are
not committed; rerun to regenerate).
