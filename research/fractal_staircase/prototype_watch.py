"""Prototype watch: the quiet profiles that precede MSI-grade runs, with the filters the
backtest supports. Writes results/prototype_watch.csv (segments) and
results/prototype_watch_years.csv (lift by year)."""
import numpy as np
import pandas as pd

import prototype as P
from common import DATA_DIR, RES_DIR, adj_lift


def profiles(d):
    qb = (d.q_mar63 <= 0.2) & (d.q_mdd126 <= 0.3) & (d.q_roc126 <= 0.4) & (d.ddh252 > -0.105)
    jm = (d.q_rng63 <= 0.10) & (d.q_mar63 <= 0.10) & (d.q_mdd126 <= 0.15) & (d.ddh252 > -0.10) & (d.q_roc126 >= 0.5)
    td = (d.q_rng63 <= 0.10) & (d.q_roc126 >= 0.8) & (d.ddh252 > -0.10) & (d.q_mar63 <= 0.2)
    pinned = (d.q_mar63 <= 0.01) & (d.q_rng63 <= 0.01)
    union = (qb | jm | td).fillna(False)
    watch = union & (d.mcap >= 2000) & (d.industry != "Banks - Regional") & ~pinned.fillna(False)
    return dict(union=union.to_numpy(), watch=watch.to_numpy(), pinned=pinned.fillna(False).to_numpy(),
                small=(d.mcap < 2000).to_numpy(), bank=(d.industry == "Banks - Regional").to_numpy())


def main():
    d = P.load()
    m = profiles(d)
    lab = d.proto.notna().to_numpy()
    tr = (d.date <= "2016-12-31").to_numpy()
    te = (d.date >= "2017-07-01").to_numpy()
    rows = []
    for name, mask in [("All three profiles, no filters", m["union"]), ("  of which below $2B", m["union"] & m["small"]),
                       ("  of which regional banks", m["union"] & m["bank"]), ("  of which pinned (vol and range bottom 1%)", m["union"] & m["pinned"]),
                       ("Prototype watch (filters applied)", m["watch"])]:
        r = dict(segment=name, per_week=(mask & lab).sum() / d.loc[lab, "date"].nunique())
        for nm, sel in (("all", np.ones(len(d), bool)), ("2006_16", tr), ("2017_26", te)):
            mm = mask & lab & sel
            r[f"lift_{nm}"] = adj_lift(d.proto[mm], d._wb_proto[mm])
            r[f"hit_{nm}"] = d.proto[mm].mean()
        mm = mask & lab
        r["up126"] = (d.fr126[mm] > 0).mean()
        r["med_fr126"] = np.exp(d.fr126[mm].median()) - 1
        r["med_dd126"] = d.fdd126[mm].median()
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{RES_DIR}/prototype_watch.csv", index=False)
    w = m["watch"] & lab
    g = d[w].groupby(d.loc[w, "date"].dt.year)
    yr = pd.DataFrame({"n": g.size(), "hits": g["proto"].sum(), "exp": g["_wb_proto"].sum()})
    yr["lift"] = yr["hits"] / yr["exp"]
    yr.index.name = "year"
    yr.reset_index().to_csv(f"{RES_DIR}/prototype_watch_years.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(yr.round(2).T.to_string())


if __name__ == "__main__":
    main()
