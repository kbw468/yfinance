"""Transparent characteristics score, chosen only on 2015-2020 and tested on 2021 onward.
A feature enters if its 2015-2020 per-date AUC for sup_21 is at least 0.008 away from 0.5 and on the same side in at least
5 of the 6 years. Score = sum over entrants of sign x (same-day universe percentile - 0.5), equal weights (no fitting).
Reported: per-date AUC by year, top-decile superior rate and trade statistics, 2021-2023 and 2024 onward."""
import numpy as np
import pandas as pd
from .common import load, date_auc, OUT
from .plain import name


def main():
    Y = pd.read_csv(OUT / "characteristics_by_year.csv", index_col=0)
    Y.columns = Y.columns.astype(int)
    disc = Y[[c for c in Y.columns if 2015 <= c <= 2020]]
    m = disc.mean(axis=1); side = np.sign(m - 0.5)
    same = disc.sub(0.5).apply(np.sign).eq(side, axis=0).sum(axis=1)
    pick = m[((m - 0.5).abs() >= 0.008) & (same >= 5)]
    sel = pd.DataFrame({"auc_2015_2020": pick, "sign": np.sign(pick - 0.5), "plain": [name(f) for f in pick.index]}).sort_values("auc_2015_2020")
    print("entrants chosen on 2015-2020:\n", sel.round(3).to_string())
    T = load(list(pick.index) + ["sup_21", "sup_42", "xs_21", "mar_21", "stopped_21", "beta_l1_252"], start="2021-01-01")
    R = T.groupby("date")[list(pick.index)].rank(pct=True) - 0.5
    T["score"] = (R * sel.sign.reindex(R.columns).values).sum(axis=1, min_count=1)
    dcode = T.date.factorize(sort=True)[0]
    a = date_auc(T.score.to_numpy(float), T.sup_21.to_numpy(float), dcode)
    yrs = T.groupby(dcode).date.first().dt.year
    by = a.groupby(yrs.reindex(a.index).values).mean()
    T["dec"] = (T.groupby("date").score.rank(pct=True) * 10).clip(upper=9.999).astype(int) + 1
    tab = T.groupby(["dec"]).agg(superior_21=("sup_21", "mean"), superior_42=("sup_42", "mean"), med_xs_21=("xs_21", "median"), mean_xs_21=("xs_21", "mean"),
                                  stopped=("stopped_21", "mean"), med_beta=("beta_l1_252", "median"))
    t2 = T[T.date >= "2024-01-01"].groupby("dec").agg(superior_21=("sup_21", "mean"), med_xs_21=("xs_21", "median"), stopped=("stopped_21", "mean"))
    txt = ("Transparent characteristics score (entrants chosen on 2015-2020 only), tested 2021 onward\n" + sel.round(3).to_string() +
           f"\n\nper-date AUC for sup_21 by year: {by.round(3).to_dict()}  mean {by.mean():.3f}\n\ndeciles 2021 onward (10 = highest):\n" + tab.round(3).to_string() +
           "\n\ndeciles 2024 onward:\n" + t2.round(3).to_string())
    (OUT / "transparent_score.txt").write_text(txt); print(txt)


if __name__ == "__main__":
    main()
