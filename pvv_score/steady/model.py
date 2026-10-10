"""Walk-forward models for hit_21 / hit_42 / hit_63 over the entire eligible universe.

Inputs: same-day cross-sectional percentile ranks of the non-Gaussian features (features.csv lists the raw names).
Learner: gradient-boosted trees (rank/threshold splits, no distributional assumption), binary log loss, fixed hyperparameters.
Walk-forward: test years 2018..2026; training = sessions ending 100 calendar days before the test year (covers the
63-session outcome window), every 3rd session (outcome windows overlap), COVID Feb-Jun 2020 excluded.
Calibration: isotonic map fitted only on earlier test years' out-of-sample predictions (none before two years exist).
Baselines scored on the same rows: the steady-state flag alone, and a transparent rank-sum steadiness score.
Control: the identical pipeline with labels shuffled within each date, one test year per horizon."""
import sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from ..config import CACHE_DIR
from .outcomes import HORIZONS, alpha_target
TARGET = "ah"
warnings.filterwarnings("ignore")
D = CACHE_DIR / "steady"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=1000, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=1.0, n_estimators=400, verbose=-1, n_jobs=4)
TEST_YEARS = range(2018, 2027)
EMBARGO_DAYS = 100
COVID = ("2020-02-20", "2020-06-30")
# transparent baseline: equal-weight rank-sum of steadiness features (higher is steadier)
BASE_UP = ["cs_xs_spy_63", "cs_mdd_63", "cs_off_high_63", "cs_eff_63", "cs_up_share_63", "cs_gain_pain_63", "cs_new_high63_share_63"]
BASE_DOWN = ["cs_mad_21", "cs_down_up_move_63"]


def _daily(df, y, s) -> float:
    """Mean of per-date AUCs: name-selection skill only, no market timing."""
    a = [roc_auc_score(g[y], g[s]) for _, g in df.groupby("date") if g[y].nunique() == 2 and g[s].nunique() > 1 and len(g) > 50]
    return float(np.mean(a)) if a else np.nan


def feature_list() -> list:
    """Same-day cross-sectional percentile ranks only. Raw levels carry the market regime: with them a shuffled-label control
    scored a within-date AUC of 0.39-0.46 (it learned regime, which runs against the cross-section); on ranks alone the control
    sits at 0.48-0.50 and the real model keeps its full within-date skill (2024: 0.62 / 0.65 / 0.69 at 21 / 42 / 63 sessions)."""
    raw = pd.read_csv(D / "features.csv").iloc[:, 0].tolist()
    return [f"cs_{c}" for c in raw]


def load(cols=None) -> pd.DataFrame:
    T = pd.read_parquet(D / "table.parquet", columns=cols)
    T = T[~((T.date >= COVID[0]) & (T.date <= COVID[1]))].reset_index(drop=True)
    return alpha_target(T) if cols is None else T


def main():
    t0 = time.time()
    feats = feature_list()
    T = load()
    T["baseline"] = T[BASE_UP].sum(axis=1) - T[BASE_DOWN].sum(axis=1)
    thin = set(np.sort(T.date.unique())[::3]); T["thin"] = T.date.isin(thin)
    print(f"rows {len(T):,} features {len(feats)} load {time.time()-t0:.0f}s", flush=True)
    keep = ["date", "ticker", "sector", "steady", "baseline", "beta_l1_252"] + [f"{k}_{h}" for h in HORIZONS for k in ("hit", "ah", "ratio", "xs", "ptt", "stop", "ret")]
    O = T[keep].copy(); log = []; imps = {}
    for h in HORIZONS:
        y = f"{TARGET}_{h}"; O[f"raw_{h}"] = np.nan; O[f"p_{h}"] = np.nan; gi = pd.Series(0.0, index=feats)
        for Y in TEST_YEARS:
            te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31")
            tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & T[y].notna() & T.thin
            m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], T.loc[tr, y])
            O.loc[te, f"raw_{h}"] = m.predict_proba(T.loc[te, feats])[:, 1]
            gi += pd.Series(m.booster_.feature_importance("gain"), index=feats)
            prior = (O.date < f"{Y}-01-01") & O[f"raw_{h}"].notna() & O[y].notna() & (O.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS))
            if O.loc[prior, "date"].dt.year.nunique() >= 2:
                iso = IsotonicRegression(out_of_bounds="clip").fit(O.loc[prior, f"raw_{h}"], O.loc[prior, y])
                O.loc[te, f"p_{h}"] = iso.predict(O.loc[te, f"raw_{h}"])
            ok = te & O[y].notna()
            rec = {"h": h, "year": Y, "train_rows": int(tr.sum()), "test_rows": int(ok.sum()), "base_rate": O.loc[ok, y].mean() if ok.any() else np.nan}
            if ok.sum() > 1000:
                rec.update(auc_model=_daily(O.loc[ok], y, f"raw_{h}"), auc_baseline=_daily(O.loc[ok], y, "baseline"), auc_steady_flag=_daily(O.loc[ok], y, "steady"))
            log.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, f"{time.time()-t0:.0f}s", flush=True)
        imps[h] = gi / gi.sum()
        # shuffled-label control on 2024
        Y = 2024; te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31") & T[y].notna()
        tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & T[y].notna() & T.thin
        rng = np.random.default_rng(h)
        ys = T.loc[tr].groupby("date")[y].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        ms = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], ys)
        a = _daily(T.loc[te, ['date', y]].assign(s=ms.predict_proba(T.loc[te, feats])[:, 1]), y, 's')
        log.append({"h": h, "year": "2024 shuffled-label control", "auc_model": a}); print(f"[{h}] shuffled control AUC {a:.3f}", flush=True)
    pd.DataFrame(log).to_csv(D / "walkforward_log.csv", index=False)
    pd.DataFrame(imps).to_csv(D / "importance.csv")
    O.to_parquet(D / "oos.parquet")
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
