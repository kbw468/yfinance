from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = Path("/tmp/claude-0/-home-user-yfinance/6b0db6a0-0646-53b0-950e-60d46b6034fa/scratchpad/pvv_cache")

UNIVERSE_CSV = DATA_DIR / "universe_combined.csv"   # S&P 500 + S&P 400 (Finviz exports), deduplicated

START = "2012-01-01"  # 10y backtest from 2014 + 2y warm-up

MARKET = "SPY"
# Finviz sector -> SPDR sector ETF
SECTOR_ETF = {
    "Technology": "XLK",
    "Industrials": "XLI",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
    "Energy": "XLE",
}
# XLRE (2015) and XLC (2018) are young; fall back to parent ETFs for history
SECTOR_ETF_FALLBACK = {"XLRE": "XLF", "XLC": "XLK"}

HORIZONS = (21, 42, 63)

# Evaluation windows. RECENT is the regime that matters most (CTA / vol-control / 0DTE flow era).
BACKTEST_START = "2014-01-01"
RECENT_START = "2018-01-01"   # probability / calibration window: the full sample (regime-matched), COVID window excluded
# nothing before this date enters any evaluation, fit or probability table (market structure regime cut)
SAMPLE_START = "2018-01-01"
# periods removed from every fit, discovery and probability table (2022: bear market for everything but energy/commodities;
# behaviour that precedes smooth climbs in a normal tape did not get to express itself)
EXCLUDE_PERIODS = [("2020-02-20", "2020-06-30")]   # COVID crash and V-recovery: no analogue elsewhere in the sample. 2022 stays in as risk-off context (regime.py)
# walk-forward test years (training starts at SAMPLE_START, so the first test year needs >= 1y of history + embargo)
WF_YEARS = range(2020, 2027)
# weight on recent-window IC when forming composite weights (rest on full window)
RECENT_WEIGHT = 0.67
# time-decay half-life (trading days) for ML sample weights
ML_HALFLIFE_DAYS = 504
