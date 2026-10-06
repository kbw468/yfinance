"""Stage 5: how much of the signal is ticker identity vs. behaviour that generalises across tickers.

For each factor (and the OOS composite) split the reading into
  persistent = the ticker's own expanding-window mean up to t-1 (point-in-time "this ticker is usually like this")
  deviation  = (today's reading - persistent) / own expanding std   ("this ticker is unusual vs itself")
and compute IC of raw / persistent / deviation vs 42d SPY-excess Sharpe. Also IC of pure identity predictors:
long trailing Sharpe and the ticker's own trailing-year mean realised excess Sharpe.
"""
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .evaluate import daily_ic, nw_tstat
from .run_eval import load_research
from .data_io import load_panel
from .clean import clean_panel

TGT, H = "xs_spy_sharpe_42", 42
KEY = ["comp_bucketed", "dist_52w_high", "days_since_20pct_dd", "sharpe_126", "mom_12_1", "rs_lead_126", "idio_vol_63",
       "vol_dry_20_250", "obv_price_div_63", "dryup_x_tight", "corr_spy_63", "up_capture_63"]


def main():
    df = load_research().reset_index(drop=True)
    oos = pd.read_parquet(CACHE_DIR / "composite_oos_preds.parquet")
    df = df.merge(oos, on=["date", "ticker"], how="left").sort_values(["ticker", "date"])
    p, _ = clean_panel(load_panel(), verbose=False)
    rr = p["Close"].pct_change()
    for w in (252, 504, 756):
        s = (rr.rolling(w).mean() / rr.rolling(w).std() * np.sqrt(252)).stack(future_stack=True).rename(f"sharpe_{w}")
        s.index.names = ["date", "ticker"]
        df = df.merge(s.reset_index(), on=["date", "ticker"], how="left")
    df["past_xs_1y_mean"] = df.groupby("ticker")[TGT].transform(lambda s: s.shift(H).rolling(252, min_periods=126).mean())

    rows = []
    for col in KEY:
        g = df.groupby("ticker")[col]
        mean = g.transform(lambda s: s.expanding(min_periods=252).mean().shift(1))
        std = g.transform(lambda s: s.expanding(min_periods=252).std().shift(1))
        tmp = pd.DataFrame({"date": df.date, "ticker": df.ticker, TGT: df[TGT], "raw": df[col], "persistent": mean,
                            "deviation": (df[col] - mean) / std}).dropna()
        ic = daily_ic(tmp, ["raw", "persistent", "deviation"], TGT)
        for win, start in [("full", None), ("recent", RECENT_START)]:
            for c in ["raw", "persistent", "deviation"]:
                m, t, _ = nw_tstat(ic.loc[start:, c], H)
                rows.append({"factor": col, "window": win, "component": c, "ic": m, "t_nw": t})
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS_DIR / "identity_decomposition.csv", index=False)

    idc = ["sharpe_252", "sharpe_504", "sharpe_756", "past_xs_1y_mean"]
    tmp = df[["date", "ticker", TGT] + idc].dropna()
    ic = daily_ic(tmp, idc, TGT)
    rows = []
    for win, start in [("full", None), ("recent", RECENT_START)]:
        for c in idc:
            m, t, _ = nw_tstat(ic.loc[start:, c], H)
            rows.append({"predictor": c, "window": win, "ic": m, "t_nw": t})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "identity_pure_predictors.csv", index=False)

    rec = df[(df.date >= RECENT_START) & df.comp_bucketed.notna()]
    own = rec.groupby("ticker")["comp_bucketed"].transform(lambda s: s.expanding(min_periods=252).mean().shift(1))
    rk_c = rec.groupby("date")["comp_bucketed"].rank(pct=True)
    rk_o = own.groupby(rec.date).rank(pct=True)
    ok = rk_o.notna()
    corr = float(np.corrcoef(rk_c[ok], rk_o[ok])[0, 1])
    (RESULTS_DIR / "identity_rank_overlap.txt").write_text(f"{corr:.3f}")
    print(dec.pivot_table(index="factor", columns=["window", "component"], values="ic").round(3).to_string())
    print("composite-rank vs own-history-rank correlation (recent):", round(corr, 2))


if __name__ == "__main__":
    main()
