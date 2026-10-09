"""5x5 regime-neutral lift grids: volatility level against each rate of change.
Saved for the report heatmaps (results/grid_<row>__<col>.csv)."""
import numpy as np
import pandas as pd

from common import RES_DIR, add_week_base, load_panel

PAIRS = [("q_mar63", "q_roc63"), ("q_mar63", "q_roc21"), ("q_mar63", "q_vroc21"), ("q_mar63", "q_rv10_126"),
         ("q_mar63", "q_vlroc21"), ("q_mar63", "q_acc21"), ("q_rng63", "q_roc126"), ("q_mdd126", "q_roc126")]


def main():
    cols = ["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"] + sorted({c for p in PAIRS for c in p})
    df = load_panel(columns=cols)
    df = df[(df["date"] >= "2006-01-01") & df["y63"].notna()]
    df = add_week_base(df, "y63")
    q = lambda s: np.floor(np.minimum(s * 5, 4.999)).astype(int) + 1
    for a, b in PAIRS:
        d = df[[a, b, "y63", "_wb", "fr63", "fdd63"]].dropna()
        g = d.groupby([q(d[a]), q(d[b])])
        t = pd.DataFrame({"lift": g["y63"].sum() / g["_wb"].sum(), "hit": g["y63"].mean(), "n": g.size(),
                          "med_fr63": g["fr63"].median(), "med_fdd63": g["fdd63"].median()}).reset_index()
        t.columns = ["row_q", "col_q", "lift", "hit", "n", "med_fr63", "med_fdd63"]
        t.to_csv(f"{RES_DIR}/grid_{a}__{b}.csv", index=False)
        print(a, b, "max lift", t["lift"].max().round(3), "min", t["lift"].min().round(3))


if __name__ == "__main__":
    main()
