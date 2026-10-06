"""Build the long research table: one row per (date, ticker) with all factors, targets and metadata.

Memory-safe for ~900 names: per-ticker computations run in ticker chunks and are stacked to the long table as
float32; the handful of cross-sectional columns (interaction factors, blended labels, smooth-path target) are then
computed on the long table with per-date ranks.
"""
import sys
import time
import gc
import numpy as np
import pandas as pd

from .config import CACHE_DIR, BACKTEST_START, MARKET, HORIZONS
from .data_io import load_panel, load_universe
from .clean import clean_panel
from .features import compute_features, eligibility, FACTORS
from .state import to_state
from .targets import build_targets, sector_bench_returns, smooth_components

CHUNK = 160
PANEL_START = "2010-01-01"   # 2 years of warm-up before the 2012 feature start is plenty; earlier rows only cost memory


def sector_close(panel_close: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    r = panel_close.pct_change()
    rb = sector_bench_returns(r, universe)
    first = pd.DataFrame(np.zeros(rb.shape, dtype=bool), index=rb.index, columns=rb.columns)
    first.iloc[0] = True
    return (1 + rb.fillna(0)).cumprod().where(rb.notna() | first)


def _stack(frames: dict, dates, cols) -> pd.DataFrame:
    pieces = {k: v.reindex(index=dates, columns=cols).astype("float32").stack(future_stack=True) for k, v in frames.items()}
    out = pd.DataFrame(pieces)
    out.index.names = ["date", "ticker"]
    return out


def main():
    t0 = time.time()
    panel, _ = clean_panel(load_panel())
    panel = {k: v.loc[PANEL_START:] for k, v in panel.items()}
    universe = load_universe()
    stocks = [t for t in universe.index if t in panel["Close"].columns]
    etfs = [c for c in panel["Close"].columns if c not in stocks]
    print(f"{len(stocks)} stocks, {len(etfs)} benchmarks", file=sys.stderr)
    sb_all = sector_close(panel["Close"], universe)
    dates = panel["Close"].loc[BACKTEST_START:].index

    parts = []
    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        sub = {k: v[chunk + etfs] for k, v in panel.items()}
        uni_c = universe.loc[chunk]
        F = compute_features(sub, uni_c, sb_all[chunk], cross_sectional=False)
        Z = to_state(F, skip=("ceiling_2x_low",) + tuple(k for k, v in FACTORS.items() if v[0] == "roc"))
        T = build_targets(sub["Close"], uni_c)
        T.update(smooth_components(sub["Close"][chunk]))
        E = {"eligible": eligibility(sub, chunk).astype("float32")}
        parts.append(_stack({**F, **Z, **T, **E}, dates, chunk))
        del F, Z, T, sub
        gc.collect()
        print(f"  chunk {i // CHUNK + 1}: {len(chunk)} names, {time.time() - t0:.0f}s", file=sys.stderr)
    long = pd.concat(parts)
    del parts
    gc.collect()
    long = long[long["rv20"].notna() | (long["eligible"] > 0)].reset_index()
    long["eligible"] = long["eligible"] > 0
    long["sector"] = long["ticker"].map(universe["Sector"])
    long["sector_etf"] = long["ticker"].map(universe["SectorETF"])
    print(f"stacked {long.shape} {time.time() - t0:.0f}s", file=sys.stderr)

    # ---- cross-sectional columns (per-date ranks on the long table)
    g = long.groupby("date")
    rk = lambda c: g[c].rank(pct=True).astype("float32")
    long["dryup_x_tight"] = (1 - rk("vol_dry_20_250")) + (1 - rk("range_comp_10_252")) + rk("dist_52w_high")
    long["ignite_x_rs"] = rk("ignition_mult_10") + rk("rs_spy_63") + rk("updown_vol_ratio_50")
    for key, lab in [("xs_spy_sharpe", "y_blend_spy"), ("xs_sec_sharpe", "y_blend_sec")]:
        long[lab] = sum(rk(f"{key}_{h}") for h in HORIZONS) / len(HORIZONS)
    # smooth-path target: components ranked within (date, beta tercile)
    brk = rk("beta_252")
    long["_bb"] = np.select([brk <= 1 / 3, brk <= 2 / 3], [0, 1], 2)
    long.loc[brk.isna(), "_bb"] = -1
    gb = long.groupby(["date", "_bb"])
    for h in HORIZONS:
        parts4 = [gb[f"fwd_sharpe_{h}"].rank(pct=True), -long[f"fwd_mdd_{h}"], long[f"fwd_r2_{h}"], long[f"fwd_up_{h}"]]
        parts4[1] = parts4[1].groupby([long.date, long._bb]).rank(pct=True)
        parts4[2] = gb[f"fwd_r2_{h}"].rank(pct=True)
        parts4[3] = gb[f"fwd_up_{h}"].rank(pct=True)
        long[f"smooth_{h}"] = (sum(parts4) / 4).astype("float32")
        long.loc[long._bb < 0, f"smooth_{h}"] = np.nan
    long["y_smooth"] = sum(long[f"smooth_{h}"] for h in HORIZONS) / len(HORIZONS)
    long = long.drop(columns=["_bb"])
    print(f"cross-sectional done {time.time() - t0:.0f}s", file=sys.stderr)

    out = CACHE_DIR / "research_long.parquet"
    long.to_parquet(out, index=False)
    print(f"saved {out} shape={long.shape} eligible={int(long.eligible.sum())} {time.time() - t0:.0f}s", file=sys.stderr)
    pd.Series({k: v[2] for k, v in FACTORS.items()}).to_frame("desc").assign(
        group=[v[0] for v in FACTORS.values()], prior_sign=[v[1] for v in FACTORS.values()]
    ).to_csv(CACHE_DIR / "factor_registry.csv")


if __name__ == "__main__":
    main()
