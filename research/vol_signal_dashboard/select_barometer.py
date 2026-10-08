"""Re-check which inputs the drawdown barometer should use.

Walk-forward search over every equal-weight combination of 3-6 inputs from build_dashboard.CANDIDATES.
Each year's pick is made from earlier years only. The live set in barometer.json is replaced only
when a different multifactor set (3+ inputs, no single input above 50% of the score's movement)
beats it out of sample since mid-2020 on both the 5% and 8% drawdown odds by at least 0.01 AUC.

Usage:
    python select_barometer.py            # report only
    python select_barometer.py --apply    # also rewrite barometer.json when the rule says switch
"""

import argparse
import datetime as dt
import itertools
import json

import numpy as np
import pandas as pd

import build_dashboard as B

MIN_GAIN = 0.01
MAX_SHARE = 50.0
POST = "2020-07-01"


def auc(y, s):
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate_fixed(names, P, d, ok):
    score = P[names].mean(axis=1)
    reading, _ = B.baro_readings(score, ok & P[names].notna().all(axis=1))
    sel = reading.notna() & d.f40.notna() & (reading.index >= pd.Timestamp(POST))
    s = score[sel]
    f = d.f40[sel]
    return auc((f <= -5).values.astype(int), s.values), auc((f <= -8).values.astype(int), s.values)


def shares(names, P, idx):
    c = P.loc[idx, names] / len(names)
    cov = c.cov()
    tot = c.sum(axis=1).var()
    return {k: round(float(cov.loc[k].sum() / tot * 100), 1) for k in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    raw = B.fetch_yahoo("2007-01-01")
    d = B.build_frame(raw, B.fetch_dspx())
    t = B.tiers(d)
    names_all = list(B.CANDIDATES)
    P = B.candidate_inputs(d, names_all, B.fetch_extras(names_all))
    ok = t.gate & P.notna().all(axis=1)
    near = d[ok & d.f40.notna()].loc[B.BARO_START:]
    X = P.loc[near.index].values
    y5 = (near.f40 <= -5).values.astype(int)
    y8 = (near.f40 <= -8).values.astype(int)
    dates = near.index
    subsets = [c for k in range(3, 7) for c in itertools.combinations(range(len(names_all)), k)]

    def objective(mask):
        return np.array([0.5 * (auc(y5[mask], X[mask][:, s].mean(axis=1)) + auc(y8[mask], X[mask][:, s].mean(axis=1)))
                         for s in subsets])

    picks = {}
    for yr in range(2018, dates[-1].year + 2):
        train = dates < pd.Timestamp(f"{yr}-01-01")
        if train.sum() <= 290:
            continue
        last_ok = np.where(train)[0][-41]            # outcomes must be realized before the year starts
        train = train & (np.arange(len(dates)) <= last_ok)
        best = subsets[int(np.nanargmax(objective(train)))]
        picks[yr] = [names_all[i] for i in best]
    obj_all = objective(np.ones(len(dates), bool))
    order = np.argsort(-obj_all)
    best_all = [names_all[i] for i in subsets[order[0]]]

    cfg = B.load_baro_config()
    live = cfg["inputs"]
    live_auc = evaluate_fixed(live, P, d, ok)
    cand_auc = evaluate_fixed(best_all, P, d, ok)
    cand_share = shares(best_all, P, near.loc[POST:].index)
    multifactor = len(best_all) >= 3 and max(cand_share.values()) <= MAX_SHARE
    better = (cand_auc[0] >= live_auc[0] + MIN_GAIN) and (cand_auc[1] >= live_auc[1] + MIN_GAIN)
    switch = sorted(best_all) != sorted(live) and multifactor and better

    report = {
        "run": dt.date.today().isoformat(),
        "live": live, "live_auc_post_covid": [round(live_auc[0], 3), round(live_auc[1], 3)],
        "best_full_history": best_all, "best_auc_post_covid": [round(cand_auc[0], 3), round(cand_auc[1], 3)],
        "best_shares_post_covid": cand_share, "walk_forward_picks": {str(k): v for k, v in picks.items()},
        "top5": [{"inputs": [names_all[j] for j in subsets[i]], "objective": round(float(obj_all[i]), 3)} for i in order[:5]],
        "decision": "switch" if switch else "keep",
    }
    if switch and args.apply:
        new_cfg = {"inputs": best_all, "selected": report["run"],
                   "evidence": (f"Walk-forward search over {len(subsets)} combinations of 3-6 of {len(names_all)} candidates. "
                                f"Post-COVID out-of-sample AUC {cand_auc[0]:.3f} (5% drop) / {cand_auc[1]:.3f} (8% drop), "
                                f"vs {live_auc[0]:.3f} / {live_auc[1]:.3f} for the previous set {', '.join(live)}.")}
        with open(B.BARO_CONFIG, "w", encoding="utf-8") as fh:
            json.dump(new_cfg, fh, indent=2)
            fh.write("\n")
        report["applied"] = True
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
