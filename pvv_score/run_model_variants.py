"""Model-variant sweep (disclosed in the report: 4 pre-specified variants, judged on OOS 42d IC in the recent window
AND the full window; the choice is a mild in-sample selection and is reported as such)."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .evaluate import daily_ic, nw_tstat
from .run_eval import load_research
from .model import make_design, walk_forward, evaluate_prediction, decay_weights, EMBARGO, TRAIN_STRIDE

REG = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=2000, feature_fraction=0.5,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, num_threads=3, verbose=-1, seed=7)


def ridge_walk_forward(df, X, label, years=range(2017, 2027), decay=True):
    bdays = pd.DatetimeIndex(sorted(df.date.unique()))
    pos = pd.Series(np.arange(len(bdays)), index=bdays)
    preds = pd.Series(np.nan, index=df.index, dtype="float32")
    Xf = X.fillna(0.5)
    for y in years:
        first = bdays[bdays >= pd.Timestamp(f"{y}-01-01")]
        cut = pos[first[0]] - EMBARGO
        train_end = bdays[cut]
        keep = bdays[:cut + 1][::TRAIN_STRIDE]
        tr = df.index[(df.date <= train_end) & df[label].notna() & df.date.isin(keep)]
        te = (df.date >= first[0]) & (df.date <= pd.Timestamp(f"{y}-12-31"))
        w = decay_weights(df.loc[tr, "date"], train_end, bdays) if decay else None
        m = Ridge(alpha=1000.0).fit(Xf.loc[tr], df.loc[tr, label], sample_weight=w)
        preds[te] = m.predict(Xf[te]).astype("float32")
        print(f"  ridge fold {y}", file=sys.stderr)
    return preds


def main():
    df = load_research().reset_index(drop=True)
    X, cols = make_design(df)
    rank_cols = [c for c in cols if c.startswith("rk_")]
    variants = {}
    print("variant A: regularised GBM, rank features only, decay", file=sys.stderr)
    variants["gbm_reg_rankonly_decay"] = walk_forward(df, X[rank_cols], "y_blend_spy", params=REG, n_rounds=300, decay=True, tag="varA")
    print("variant B: regularised GBM, rank features only, no decay", file=sys.stderr)
    variants["gbm_reg_rankonly_nodecay"] = walk_forward(df, X[rank_cols], "y_blend_spy", params=REG, n_rounds=300, decay=False, tag="varB")
    print("variant C: ridge on rank features, decay", file=sys.stderr)
    variants["ridge_rank_decay"] = ridge_walk_forward(df, X[rank_cols], "y_blend_spy", decay=True)
    print("variant D: regularised GBM, rank + beta/vol raw + sector, decay (no market context)", file=sys.stderr)
    variants["gbm_reg_noctx_decay"] = walk_forward(df, X[rank_cols + ["beta_252_raw", "rv20_raw", "sector_code"]], "y_blend_spy", params=REG, n_rounds=300, decay=True, tag="varD")
    for k, v in variants.items():
        df[k] = v
    oos = df[df[list(variants)[0]].notna()].copy()
    oos[["date", "ticker"] + list(variants)].to_parquet(CACHE_DIR / "ml_variants_oos_preds.parquet", index=False)
    ev = pd.concat([evaluate_prediction(oos, k, k) for k in variants])
    ev.to_csv(RESULTS_DIR / "ml_variants_oos_evaluation.csv")
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(ev.round(3).to_string())
    ic = daily_ic(oos, list(variants), "xs_spy_sharpe_42")
    print("\n42d IC by year:\n", ic.groupby(ic.index.year).mean().round(3).to_string())


if __name__ == "__main__":
    main()
