"""Stage 3: historical case-study model.

A gradient-boosted regressor learns, from every (date, ticker) in the past, how a stock's full
price/volume/volatility profile mapped to its subsequent cross-sectional excess-Sharpe percentile.
That is the formalised version of "find historical case studies of tickers exhibiting these same
characteristics and see what they did next" - with the advantage that it uses ALL of them, weighted
to the current regime, instead of a hand-picked handful.

Protocol (no shortcuts):
  * features = per-date percentile ranks of all factors (regime-stationary) + raw beta/vol level
               + market context (SPY 63d return, SPY 20d RV, their 252d percentiles) so the model
               can learn factor x regime interactions
  * label    = per-date percentile of forward excess Sharpe vs SPY, averaged over 21/42/63d
  * walk-forward by calendar year, 2017..2026; train on all earlier data with a 63-trading-day
    embargo before the test year (labels that would overlap the test period are dropped)
  * time-decay sample weights, half-life 504 trading days (2y), so the recent flow regime dominates
  * training rows sampled every 5th day (overlapping labels add little information); testing uses every day
  * OOS evaluation: daily rank IC vs 21/42/63d excess Sharpe (SPY and sector), Newey-West t, decile
    spread, by year, and a beta-partialled IC to show what is NOT just a beta bet
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import WF_YEARS, CACHE_DIR, RESULTS_DIR, HORIZONS, RECENT_START, ML_HALFLIFE_DAYS, MARKET
from .evaluate import daily_ic, summarize_ic, nw_tstat, decile_spread
from .run_eval import load_research, load_registry
from .data_io import load_panel
from .clean import clean_panel

FACTORS = load_registry()
EMBARGO = 63
TRAIN_STRIDE = 5
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=400,
              feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              num_threads=3, verbose=-1, seed=7)
N_ROUNDS = 600


def market_context(dates: pd.DatetimeIndex) -> pd.DataFrame:
    panel, _ = clean_panel(load_panel(), verbose=False)
    c = panel["Close"][MARKET]
    lr = np.log(c).diff()
    ctx = pd.DataFrame({
        "mkt_ret_63": c / c.shift(63) - 1,
        "mkt_ret_21": c / c.shift(21) - 1,
        "mkt_rv20": lr.rolling(20).std() * np.sqrt(252),
        "mkt_dd_252": c / c.rolling(252).max() - 1,
    })
    ctx["mkt_rv20_pct"] = ctx["mkt_rv20"].rolling(252).rank(pct=True)
    ctx["mkt_ret63_pct"] = ctx["mkt_ret_63"].rolling(252).rank(pct=True)
    return ctx.reindex(dates)


def make_design(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    factors = list(FACTORS)
    X = df.groupby("date")[factors].rank(pct=True).astype("float32")
    X.columns = [f"rk_{f}" for f in factors]
    X["beta_252_raw"] = df["beta_252"].astype("float32")
    X["rv20_raw"] = df["rv20"].astype("float32")
    X["sector_code"] = df["sector"].astype("category").cat.codes.astype("int16")
    ctx = market_context(pd.DatetimeIndex(sorted(df.date.unique())))
    X = X.join(ctx.reindex(df.date.values).set_index(df.index).astype("float32"))
    return X, list(X.columns)


def decay_weights(dates: pd.Series, end: pd.Timestamp, bdays_index: pd.DatetimeIndex) -> np.ndarray:
    pos = pd.Series(np.arange(len(bdays_index)), index=bdays_index)
    age = pos[end] - pos.reindex(dates.values).values
    return np.power(0.5, age / ML_HALFLIFE_DAYS)


def walk_forward(df: pd.DataFrame, X: pd.DataFrame, label: str, years=WF_YEARS, params=None, n_rounds=None,
                 decay=True, tag="ml") -> pd.Series:
    params = params or PARAMS
    n_rounds = n_rounds or N_ROUNDS
    bdays = pd.DatetimeIndex(sorted(df.date.unique()))
    pos = pd.Series(np.arange(len(bdays)), index=bdays)
    preds = pd.Series(np.nan, index=df.index, dtype="float32")
    imp = []
    for y in years:
        test_start = pd.Timestamp(f"{y}-01-01")
        test_end = pd.Timestamp(f"{y}-12-31")
        first_test_day = bdays[bdays >= test_start]
        if len(first_test_day) == 0:
            continue
        cut_pos = pos[first_test_day[0]] - EMBARGO
        if cut_pos < 252:
            continue
        train_end = bdays[cut_pos]
        tr = (df.date <= train_end) & df[label].notna()
        tr_idx = df.index[tr]
        # stride sampling by date position
        keep_dates = bdays[:cut_pos + 1][::TRAIN_STRIDE]
        tr_idx = tr_idx[df.loc[tr_idx, "date"].isin(keep_dates)]
        te = (df.date >= test_start) & (df.date <= test_end)
        w = decay_weights(df.loc[tr_idx, "date"], train_end, bdays) if decay else None
        ds = lgb.Dataset(X.loc[tr_idx], label=df.loc[tr_idx, label], weight=w, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=n_rounds)
        preds[te] = model.predict(X[te]).astype("float32")
        gain = pd.Series(model.feature_importance("gain"), index=X.columns, name=y)
        imp.append(gain / gain.sum())
        print(f"  fold {y}: train rows={len(tr_idx):,} (to {train_end.date()}), test rows={int(te.sum()):,}", file=sys.stderr)
    pd.concat(imp, axis=1).to_csv(RESULTS_DIR / f"{tag}_feature_importance_by_fold.csv")
    return preds


def beta_partial_ic(df: pd.DataFrame, pred: str, target: str) -> pd.DataFrame:
    """IC of prediction after removing (within each date) the part linearly explained by beta rank."""
    sub = df[[ "date", pred, target, "beta_252"]].dropna().copy()
    rk = sub.groupby("date")[[pred, "beta_252"]].rank(pct=True)
    rk = rk - rk.groupby(sub.date).transform("mean")
    cov = (rk[pred] * rk["beta_252"]).groupby(sub.date).transform("sum")
    var = (rk["beta_252"] ** 2).groupby(sub.date).transform("sum")
    sub["pred_resid"] = rk[pred] - rk["beta_252"] * cov / var
    return daily_ic(sub, ["pred_resid"], target)


def evaluate_prediction(df: pd.DataFrame, pred: str, tag: str) -> pd.DataFrame:
    rows = {}
    for h in HORIZONS:
        col = f"smooth_{h}"
        if col in df.columns:
            ic = daily_ic(df, [pred], col)[pred]
            for win, start in [("full", None), ("recent", RECENT_START)]:
                m, t, p = nw_tstat(ic.loc[start:], h)
                rows[(f"smooth{h}", win)] = {"ic": m, "t_nw": t, "hit": (ic.loc[start:] > 0).mean()}
    for h in HORIZONS:
        for bench in ["spy", "sec"]:
            tgt = f"xs_{bench}_sharpe_{h}"
            ic = daily_ic(df, [pred], tgt)[pred]
            for win, start in [("full", None), ("recent", RECENT_START)]:
                m, t, p = nw_tstat(ic.loc[start:], h)
                rows[(f"{bench}_sh{h}", win)] = {"ic": m, "t_nw": t, "hit": (ic.loc[start:] > 0).mean()}
        bp = beta_partial_ic(df, pred, f"xs_spy_sharpe_{h}")["pred_resid"]
        m, t, p = nw_tstat(bp.loc[RECENT_START:], h)
        rows[(f"spy_sh{h}_betaneutral", "recent")] = {"ic": m, "t_nw": t, "hit": (bp.loc[RECENT_START:] > 0).mean()}
        m, t, p = nw_tstat(bp, h)
        rows[(f"spy_sh{h}_betaneutral", "full")] = {"ic": m, "t_nw": t, "hit": (bp > 0).mean()}
    out = pd.DataFrame(rows).T
    out.index.names = ["target", "window"]
    out["model"] = tag
    return out


def main():
    df = load_research()
    df = df.reset_index(drop=True)
    print(f"rows={len(df):,}", file=sys.stderr)
    X, cols = make_design(df)
    df["ml_pred"] = walk_forward(df, X, "y_blend_spy")
    oos = df[df.ml_pred.notna()].copy()
    oos[["date", "ticker", "ml_pred"]].to_parquet(CACHE_DIR / "ml_oos_preds.parquet", index=False)

    evals = [evaluate_prediction(oos, "ml_pred", "lightgbm_walkforward")]
    # baselines on the same OOS rows
    oos["bl_mom"] = oos["mom_12_1"]
    oos["bl_beta"] = oos["beta_252"]
    oos["bl_rslead"] = oos["rs_lead_126"]
    evals += [evaluate_prediction(oos, "bl_mom", "baseline_mom_12_1"),
              evaluate_prediction(oos, "bl_beta", "baseline_beta_252"),
              evaluate_prediction(oos, "bl_rslead", "baseline_rs_lead_126")]
    ev = pd.concat(evals)
    ev.to_csv(RESULTS_DIR / "ml_oos_evaluation.csv")

    # by-year IC of the ML prediction (42d, SPY)
    ic42 = daily_ic(oos, ["ml_pred"], "xs_spy_sharpe_42")["ml_pred"]
    byyear = ic42.groupby(ic42.index.year).agg(["mean", "std", "count"])
    byyear["t_naive"] = byyear["mean"] / byyear["std"] * np.sqrt(byyear["count"] / 42)
    byyear.to_csv(RESULTS_DIR / "ml_oos_ic42_by_year.csv")

    # decile analysis, recent window, 42d
    rec = oos[oos.date >= RECENT_START]
    sp, dec = decile_spread(rec, "ml_pred", "xs_spy_sharpe_42")
    dec_mean = dec.mean()
    dec_mean.to_csv(RESULTS_DIR / "ml_oos_decile_means_recent_42.csv")

    pd.set_option("display.width", 250, "display.max_rows", 200)
    print("\nOOS evaluation (walk-forward 2017-2026):")
    print(ev.round(3).to_string())
    print("\nML 42d IC by year:\n", byyear.round(3).to_string())
    print("\nML decile means of 42d excess Sharpe vs SPY (recent window, D1 low .. D10 high):\n", dec_mean.round(3).to_string())
    print("D10-D1 spread mean:", round(sp.mean(), 3), " hit:", round((sp > 0).mean(), 3))


if __name__ == "__main__":
    main()
