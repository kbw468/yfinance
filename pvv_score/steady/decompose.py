"""Which half of the target is forecastable? Walk-forward on rank features, within-date AUC, 42 sessions, test years 2019-2025.
  beat      : excess return vs SPY > 0
  alpha_top : excess return in the top 20% of the universe that date
  shallow   : path drawdown shallower than -7% and never 8% below entry intraday
  beat | low-risk : 'beat', trained and scored only on names in the top 30% of the shallow-path model that date"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from .model import PARAMS, feature_list, load, EMBARGO_DAYS, _daily
warnings.filterwarnings("ignore")


def wf(T, feats, y, years, mask=None):
    out = {}; pred = pd.Series(np.nan, index=T.index)
    for Y in years:
        te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31") & T[y].notna()
        tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & T[y].notna() & T.thin
        if mask is not None: te &= mask; tr &= mask
        m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], T.loc[tr, y])
        pred[te] = m.predict_proba(T.loc[te, feats])[:, 1]
        out[Y] = round(_daily(T.loc[te, ["date", y]].assign(s=pred[te]), y, "s"), 3)
    return out, pred


def main():
    feats = feature_list(); T = load(); T["thin"] = T.date.isin(set(np.sort(T.date.unique())[::3]))
    T["beat"] = (T.xs_42 > 0).astype(float).where(T.xs_42.notna())
    T["alpha_top"] = (T.xs_42.groupby(T.date).rank(pct=True) >= 0.8).astype(float).where(T.xs_42.notna())
    T["shallow"] = ((T.ptt_42 >= -0.07) & (T.stop_42 > -0.08)).astype(float).where(T.xs_42.notna())
    yrs = range(2019, 2026); res = {}
    for y in ("beat", "alpha_top", "shallow"):
        res[y], p = wf(T, feats, y, yrs); print(y, res[y], "mean", round(np.mean(list(res[y].values())), 3), flush=True)
        if y == "shallow": T["p_shallow"] = p
    T["lowrisk"] = T.p_shallow.groupby(T.date).rank(pct=True) >= 0.7
    # p_shallow only exists for test years; for training rows of later folds use it where available (2019+), so start low-risk test at 2021
    res["beat | low-risk"], _ = wf(T, feats, "beat", range(2021, 2026), mask=T.lowrisk & T.p_shallow.notna())
    print("beat | low-risk", res["beat | low-risk"], "mean", round(np.mean(list(res["beat | low-risk"].values())), 3), flush=True)
    # univariate checks: classic signals on 'beat', within-date AUC
    for f in ["cs_xs_spy_252", "cs_roc_126", "cs_xs_spy_63", "cs_off_high_252", "cs_eff_63", "cs_mdd_63", "cs_up_share_63", "cs_gain_pain_63", "cs_beta_l1_252"]:
        X = T[(T.date >= "2019-01-01") & T.beat.notna()]
        print(f"univariate {f:<20} within-date AUC for beat: {_daily(X, 'beat', f):.3f}   for shallow: {_daily(X, 'shallow', f):.3f}", flush=True)


if __name__ == "__main__":
    main()
