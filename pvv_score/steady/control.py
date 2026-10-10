"""Name-selection skill measured within each date (mean of daily AUCs), so market timing cannot leak in.
Model and baseline from the walk-forward; shuffled-label control re-fitted for 2024 and scored the same way."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score
from ..config import CACHE_DIR
from .model import PARAMS, feature_list, load, EMBARGO_DAYS
from .outcomes import HORIZONS
warnings.filterwarnings("ignore"); D = CACHE_DIR / "steady"


def daily_auc(df, y, s):
    a = [roc_auc_score(g[y], g[s]) for _, g in df.groupby("date") if g[y].nunique() == 2 and len(g) > 50]
    return np.mean(a), np.percentile(a, 10), np.mean(np.array(a) > 0.5), len(a)


def main():
    O = pd.read_parquet(D / "oos.parquet"); rows = []
    for h in HORIZONS:
        X = O.dropna(subset=[f"raw_{h}", f"hit_{h}"])
        for Y, g in X.groupby(X.date.dt.year):
            for s, lab in ((f"raw_{h}", "model"), ("baseline", "baseline")):
                m, p10, share, n = daily_auc(g, f"hit_{h}", s); rows.append({"h": h, "year": Y, "score": lab, "mean_daily_auc": m, "p10": p10, "days_above_0.5": share, "days": n})
    feats = feature_list(); T = load(); thin = set(np.sort(T.date.unique())[::3])
    for h in HORIZONS:
        y = f"hit_{h}"; Y = 2024
        te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31") & T[y].notna()
        tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & T[y].notna() & T.date.isin(thin)
        rng = np.random.default_rng(h)
        ys = T.loc[tr].groupby("date")[y].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], ys)
        G = T.loc[te, ["date", y]].copy(); G["s"] = m.predict_proba(T.loc[te, feats])[:, 1]
        a, p10, share, n = daily_auc(G, y, "s"); rows.append({"h": h, "year": Y, "score": "shuffled-label control", "mean_daily_auc": a, "p10": p10, "days_above_0.5": share, "days": n})
        print(h, "control", round(a, 3), flush=True)
    R = pd.DataFrame(rows); R.to_csv(D / "daily_auc.csv", index=False)
    print(R.pivot_table(index="year", columns=["h", "score"], values="mean_daily_auc").round(3).to_string())
    print(R[R.score == "shuffled-label control"].round(3).to_string(index=False))
    print(R.groupby(["h", "score"])[["mean_daily_auc", "days_above_0.5"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
