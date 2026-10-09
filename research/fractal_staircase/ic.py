"""Rank information coefficients: per-week Spearman rank correlation between every
feature and every forward outcome component, then averaged across weeks.

Splitting the staircase into its parts shows which features push toward
  * forward gain           (fr21, fr63)
  * forward smoothness     (fer21, fer63: net move / total distance travelled)
  * shallow drawdown       (-fdd21, -fdd63)
  * gain-to-pain           (fr / (fdd + 1%))
  * the binary staircase   (y21, y63, y_hold)
Reported: mean weekly rank-IC, share of weeks with IC > 0, and mean IC per era.
"""
import numpy as np
import pandas as pd

from common import DATA_DIR, ERAS, RES_DIR, load_panel

OUTCOMES = ["fr21", "fer21", "neg_fdd21", "gp21f", "y21", "fr63", "fer63", "neg_fdd63", "gp63f", "y63", "y_hold"]


def main():
    import pyarrow.parquet as pq
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    cols = ["date", "ticker", "fr21", "fdd21", "fer21", "fr42", "fdd42", "fr63", "fdd63", "fer63"] + \
        [c for c in names if c.startswith(("q_", "sq_", "iq_"))]
    df = load_panel(columns=cols)
    df = df[(df["date"] >= "2006-01-01") & df["fr63"].notna()].reset_index(drop=True)
    df["neg_fdd21"], df["neg_fdd63"] = -df["fdd21"], -df["fdd63"]
    df["gp21f"] = df["fr21"] / (df["fdd21"] + 0.01)
    df["gp63f"] = df["fr63"] / (df["fdd63"] + 0.01)
    qcols = [c for c in df.columns if c.startswith(("q_", "sq_", "iq_"))]
    dates = df["date"].to_numpy()
    res = []
    for d, idx in pd.Series(np.arange(len(df))).groupby(dates).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 100:
            continue
        F = df.iloc[idx][qcols].to_numpy(np.float64)
        F = np.where(np.isnan(F), 0.5, F) - 0.5
        O = df.iloc[idx][OUTCOMES].rank(pct=True).to_numpy(np.float64) - 0.5
        O = np.where(np.isnan(O), 0.0, O)
        num = F.T @ O
        den = np.sqrt((F ** 2).sum(0))[:, None] * np.sqrt((O ** 2).sum(0))[None, :]
        res.append((d, num / np.where(den > 0, den, np.nan)))
    dts = pd.DatetimeIndex([r[0] for r in res])
    IC = np.stack([r[1] for r in res])                 # weeks x features x outcomes
    rows = []
    for j, f in enumerate(qcols):
        row = {"feature": f}
        for k, o in enumerate(OUTCOMES):
            s = IC[:, j, k]
            row[f"ic_{o}"] = np.nanmean(s)
            row[f"pos_{o}"] = np.nanmean(s > 0)
        for a, b in ERAS:
            m = np.asarray((dts >= pd.Timestamp(a)) & (dts <= pd.Timestamp(b)))
            row[f"ic_y63_{a[:4]}"] = np.nanmean(IC[m, j, OUTCOMES.index("y63")])
            row[f"ic_fer63_{a[:4]}"] = np.nanmean(IC[m, j, OUTCOMES.index("fer63")])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES_DIR}/ic_all.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["feature", "ic_fr63", "ic_fer63", "ic_neg_fdd63", "ic_gp63f", "ic_y63", "pos_y63", "ic_y21", "ic_y_hold"]
    for key in ("ic_y63", "ic_gp63f", "ic_fer63", "ic_y21"):
        print(f"\n== top 25 by {key}")
        print(out.sort_values(key, ascending=False)[show].head(25).round(3).to_string(index=False))
        print(f"== bottom 10 by {key}")
        print(out.sort_values(key)[show].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
