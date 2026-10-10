# Q4 tax-loss-harvesting seasonality study

Backtest of how trailing 1Y / 1M losers vs winners behave in Q4 and around year end,
across the 2,367-ticker list in `universe_finviz.csv`, 1999-2025, with the live Q4 2026 screen.

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
python make_pdf.py DASH results Q4_Tax_Loss_Seasonality.pdf
```

Method: formation at the last September close; 1Y = trailing 12-month total return, 1M = September
return; eligibility = full 12M history and 63-day median dollar volume >= $1M. Buckets are
equal-weight buy-and-hold. Daily returns clipped to [-80%, +150%]. Risk stats pool daily returns
inside each window across years; alpha/beta vs SPY, Sharpe over 13-week T-bills. Eras: 1999-2007,
2008-2016, 2017-2025. Universe is current survivors only.

`results/` holds the output tables (the 7 MB stock-level panel and the 1 MB per-year path file are
not committed; rerun to regenerate).
