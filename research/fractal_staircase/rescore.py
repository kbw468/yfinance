"""Score the latest session with the saved walk-forward models (no retraining)."""
import glob
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import DATA_DIR, family


def main(tag="binary_y63"):
    files = sorted(glob.glob(f"{DATA_DIR}/model_{tag}_*.txt"))
    boosters = [lgb.Booster(model_file=f) for f in files]
    feats = boosters[0].feature_name()
    dates = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=["date"])["date"]
    last_date = dates.max()
    cols = ["date", "ticker", "sector", "industry"] + [c for c in feats if c != "sector_code"]
    p = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=cols)
    sectors = sorted(p["sector"].dropna().unique())
    last = p[p["date"] == last_date].copy()
    last["sector_code"] = last["sector"].map({s: i for i, s in enumerate(sectors)}).fillna(-1).astype(np.float32)
    X = last[feats].to_numpy(np.float32)
    last["score"] = np.mean([b.predict(X) for b in boosters], axis=0)
    contrib = np.mean([b.predict(X, pred_contrib=True)[:, :-1] for b in boosters], axis=0)
    cdf = pd.DataFrame(contrib, columns=feats, index=last.index)
    out = last[["date", "ticker", "sector", "industry", "score"]].copy()
    out["top_drivers"] = cdf.apply(lambda r: ", ".join(r.sort_values(ascending=False).index[:4]), axis=1)
    famc = cdf.T.groupby([family(c) for c in feats]).sum().T
    famc.columns = ["why_" + c for c in famc.columns]
    out = pd.concat([out, famc], axis=1)
    out["pct"] = out["score"].rank(pct=True)
    out.to_parquet(f"{DATA_DIR}/latest_{tag}.parquet", index=False)
    print(tag, "scored", len(out), "names on", last_date.date(), "with", len(boosters), "models")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["binary_y63", "binary_xs_y63"]):
        main(t)
