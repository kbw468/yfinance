"""Tonight's composite percentile, built by the SAME process as every row the calibration tables were measured on:
the walk-forward out-of-sample composite (the current test fold's weights, fitted on earlier years only) for the state
and level layers, each ranked across the universe on the date, then averaged.

score_state / score_now also produce 'final-weight' scores (weights refit on all data through today, percentiled
differently). Those never went through an out-of-sample test and are not on the calibration tables' scale; looking a
cell up with them mis-keyed roughly half the universe (rank correlation 0.65 to the calibration scale). They remain in
universe_scores as diagnostics only."""
import pandas as pd
from .config import CACHE_DIR

STATE = CACHE_DIR / "composite_state_smooth_oos_preds.parquet"
LEVEL = CACHE_DIR / "composite_level_smooth_oos_preds.parquet"


def live_composite_percentile(asof=None) -> pd.Series:
    st = pd.read_parquet(STATE, columns=["date", "ticker", "comp_bucketed"])
    lv = pd.read_parquet(LEVEL, columns=["date", "ticker", "comp_bucketed"])
    d = st.date.max()
    assert lv.date.max() == d, f"state preds end {d.date()}, level preds end {lv.date.max().date()}"
    if asof is not None:
        assert pd.Timestamp(asof) == d, f"composite preds end {d.date()} but the list is as of {pd.Timestamp(asof).date()}"
    s = st[st.date == d].set_index("ticker")["comp_bucketed"].rename("s")
    l = lv[lv.date == d].set_index("ticker")["comp_bucketed"].rename("l")
    both = pd.concat([s, l], axis=1).dropna()
    return ((both.s.rank(pct=True) + both.l.rank(pct=True)) / 2).round(8).rename("comp_p")   # 8 dp: tie order is then identical in every code path
