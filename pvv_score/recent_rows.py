"""Nightly feature build over a recent window only: every factor and state z-score for the last few sessions.

Same per-ticker code as build.py (compute_features, to_state, eligibility) on a panel that starts WINDOW_YEARS before the
last session, so every rolling quantity at the last session has its full window inside the panel (longest: 504-session
drawdown clock inside a 252-session factor inside a 756-session state norm). No forward targets. Writes
CACHE_DIR/recent_rows.parquet with the last KEEP sessions; score_frozen reads the last one."""
import sys
import time
import gc
import numpy as np
import pandas as pd
from .config import CACHE_DIR
from .data_io import load_panel, load_universe
from .clean import clean_panel
from .features import compute_features, eligibility, FACTORS
from .state import to_state
from .build import sector_close, _stack, CHUNK

WINDOW_YEARS = 7
KEEP = 5


def main(window_years: int = WINDOW_YEARS, keep: int = KEEP):
    t0 = time.time()
    panel, _ = clean_panel(load_panel())
    asof = panel["Close"].index[-1]
    start = asof - pd.DateOffset(years=window_years)
    panel = {k: v.loc[start:] for k, v in panel.items()}
    universe = load_universe()
    stocks = [t for t in universe.index if t in panel["Close"].columns]
    etfs = [c for c in panel["Close"].columns if c not in stocks]
    sb_all = sector_close(panel["Close"], universe)
    dates = panel["Close"].index[-keep:]
    print(f"{len(stocks)} stocks, panel {panel['Close'].index[0].date()}..{asof.date()}, keeping {len(dates)} sessions", file=sys.stderr)
    parts = []
    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        sub = {k: v[chunk + etfs] for k, v in panel.items()}
        F = compute_features(sub, universe.loc[chunk], sb_all[chunk], cross_sectional=False)
        Z = to_state(F, skip=("ceiling_2x_low",) + tuple(k for k, v in FACTORS.items() if v[0] == "roc"))
        E = {"eligible": eligibility(sub, chunk).astype("float32")}
        parts.append(_stack({**F, **Z, **E}, dates, chunk))
        del F, Z, sub
        gc.collect()
        print(f"  chunk {i // CHUNK + 1}: {len(chunk)} names, {time.time() - t0:.0f}s", file=sys.stderr)
    long = pd.concat(parts).reset_index()
    long = long[long["rv20"].notna() | (long["eligible"] > 0)].reset_index(drop=True)
    long["eligible"] = long["eligible"] > 0
    long["sector"] = long["ticker"].map(universe["Sector"])
    long["sector_etf"] = long["ticker"].map(universe["SectorETF"])
    g = long.groupby("date")
    rk = lambda c: g[c].rank(pct=True).astype("float32")
    long["dryup_x_tight"] = (1 - rk("vol_dry_20_250")) + (1 - rk("range_comp_10_252")) + rk("dist_52w_high")
    long["ignite_x_rs"] = rk("ignition_mult_10") + rk("rs_spy_63") + rk("updown_vol_ratio_50")
    out = CACHE_DIR / "recent_rows.parquet"
    long.to_parquet(out, index=False)
    print(f"saved {out} shape={long.shape} asof={asof.date()} eligible_today={int(long[long.date == asof].eligible.sum())} {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
