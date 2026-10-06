"""Build the long research table: one row per (date, ticker) with all factors, targets and metadata."""
import sys
import time
import numpy as np
import pandas as pd

from .config import CACHE_DIR, BACKTEST_START, MARKET
from .data_io import load_panel, load_universe
from .clean import clean_panel
from .features import compute_features, eligibility, FACTORS
from .state import to_state
from .targets import build_targets, blended_rank_target, sector_bench_returns


def sector_close(panel_close: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    # price-level benchmark via cumulated returns (handles XLRE/XLC fallback splice)
    r = panel_close.pct_change()
    rb = sector_bench_returns(r, universe)
    first = pd.DataFrame(np.zeros(rb.shape, dtype=bool), index=rb.index, columns=rb.columns)
    first.iloc[0] = True
    return (1 + rb.fillna(0)).cumprod().where(rb.notna() | first)


def main():
    t0 = time.time()
    panel, _ = clean_panel(load_panel())
    universe = load_universe()
    stocks = [t for t in universe.index if t in panel["Close"].columns]
    print(f"{len(stocks)} stocks", file=sys.stderr)

    sb = sector_close(panel["Close"], universe)
    F = compute_features(panel, universe, sb)
    print(f"features done {time.time()-t0:.0f}s", file=sys.stderr)
    # state versions: the 0/1 flag is excluded (z of a binary is not meaningful)
    Z = to_state(F, skip=("ceiling_2x_low",))
    print(f"state transforms done {time.time()-t0:.0f}s", file=sys.stderr)
    T = build_targets(panel["Close"], universe)
    T["y_blend_spy"] = blended_rank_target(T, "xs_spy_sharpe")
    T["y_blend_sec"] = blended_rank_target(T, "xs_sec_sharpe")
    print(f"targets done {time.time()-t0:.0f}s", file=sys.stderr)
    elig = eligibility(panel, stocks)

    # wide -> long, restricted to backtest window and eligible rows
    dates = panel["Close"].loc[BACKTEST_START:].index
    pieces = {}
    for k, v in {**F, **Z, **T}.items():
        pieces[k] = v.reindex(index=dates, columns=stocks).stack(future_stack=True)
    long = pd.DataFrame(pieces)
    long["eligible"] = elig.reindex(index=dates, columns=stocks).stack(future_stack=True).fillna(False).astype(bool)
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    long["sector"] = long["ticker"].map(universe["Sector"])
    long["sector_etf"] = long["ticker"].map(universe["SectorETF"])
    # keep rows that have a close (factor rows) - drop entirely-NaN rows
    long = long[long["rv20"].notna() | long["eligible"]]
    out = CACHE_DIR / "research_long.parquet"
    long.to_parquet(out, index=False)
    print(f"saved {out} shape={long.shape} eligible={long.eligible.sum()} {time.time()-t0:.0f}s", file=sys.stderr)
    pd.Series({k: v[2] for k, v in FACTORS.items()}).to_frame("desc").assign(
        group=[v[0] for v in FACTORS.values()], prior_sign=[v[1] for v in FACTORS.values()]
    ).to_csv(CACHE_DIR / "factor_registry.csv")


if __name__ == "__main__":
    main()
