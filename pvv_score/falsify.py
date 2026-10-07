"""Falsification tests for the signature engine, run on the same rows the list is built from.

  reversed  : discover on the confirmation half (2021+), confirm on the discovery half (2015-2020 ex-COVID).
              A real signature family discovers in either half and confirms in the other.
  permuted  : forward outcome shuffled across names within each date (conditions intact, link to what followed destroyed),
              then the full discover-and-confirm pass. A clean engine confirms ~0 signatures; whatever confirms is what the
              thresholds alone can manufacture.
Writes results/falsification_summary.csv, results/falsification_reversed_signatures.csv, results/falsification_tonight.csv."""
import sys
import itertools
import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .run_eval import load_research
from .regime import tag, today_regime, same_regime
from .signatures import (QUINTILE_FACTORS, build_conditions, smooth_top_quartile, DISCOVER_END, CONFIRM_START,
                         MIN_N, MIN_LIFT, MIN_N_CONFIRM, CONFIRM_LIFT, MAX_PAIRS_TO_EXTEND)

N_PERM = 2


def search(conds: dict, y: np.ndarray, disc: np.ndarray, conf: np.ndarray) -> pd.DataFrame:
    """Same pass as signatures.main: pairs, then triples that add lift; confirmation = lift >= CONFIRM_LIFT, n >= MIN_N_CONFIRM, p_conf >= 0.8 p_disc."""
    names = list(conds)
    base_d, base_c = np.nanmean(y[disc]), np.nanmean(y[conf])
    yy = np.nan_to_num(y)

    def stats(m):
        md, mc = m & disc, m & conf
        nd, nc = md.sum(), mc.sum()
        return nd, (yy[md].mean() if nd else np.nan), nc, (yy[mc].mean() if nc else np.nan)

    rows = []
    for a, b in itertools.combinations(names, 2):
        if a.split(":")[0] == b.split(":")[0]:
            continue
        nd, pd_, nc, pc = stats(conds[a] & conds[b])
        if nd >= MIN_N and pd_ / base_d >= MIN_LIFT:
            rows.append({"signature": f"{a} & {b}", "k": 2, "n_disc": nd, "p_disc": pd_, "lift_disc": pd_ / base_d, "n_conf": nc, "p_conf": pc, "lift_conf": pc / base_c if nc else np.nan})
    if not rows:
        return pd.DataFrame(columns=["signature", "k", "n_disc", "p_disc", "lift_disc", "n_conf", "p_conf", "lift_conf", "confirmed"]), base_d, base_c
    pair_df = pd.DataFrame(rows).sort_values("lift_disc", ascending=False)
    pair_lift = dict(zip(pair_df.signature, pair_df.lift_disc))
    seen = set()
    for sig in pair_df.signature.head(MAX_PAIRS_TO_EXTEND):
        a, b = sig.split(" & ")
        for c in names:
            if c.split(":")[0] in (a.split(":")[0], b.split(":")[0]):
                continue
            key = tuple(sorted([a, b, c]))
            if key in seen:
                continue
            seen.add(key)
            nd, pd_, nc, pc = stats(conds[a] & conds[b] & conds[c])
            if nd >= MIN_N and pd_ / base_d >= max(MIN_LIFT + 0.1, pair_lift[sig] + 0.1):
                rows.append({"signature": " & ".join(key), "k": 3, "n_disc": nd, "p_disc": pd_, "lift_disc": pd_ / base_d, "n_conf": nc, "p_conf": pc, "lift_conf": pc / base_c if nc else np.nan})
    res = pd.DataFrame(rows)
    res["confirmed"] = (res.n_conf >= MIN_N_CONFIRM) & (res.lift_conf >= CONFIRM_LIFT) & (res.p_conf >= 0.8 * res.p_disc)
    return res.sort_values(["confirmed", "p_conf"], ascending=False), base_d, base_c


def fires_today(df, conds, confirmed):
    today = (df.date == df.date.max()).values
    tick = df.ticker.values[today]
    best = {t: (0.0, "") for t in tick}; cnt = {t: 0 for t in tick}
    for _, r in confirmed.sort_values("p_disc", ascending=False).iterrows():
        m = np.ones(len(df), dtype=bool)
        for p in r.signature.split(" & "):
            m &= conds[p]
        for t in df.ticker.values[today & m]:
            cnt[t] += 1
            if not best[t][1]:
                best[t] = (r.p_conf, r.signature)
    return pd.DataFrame({"ticker": tick, "n_sig": [cnt[t] for t in tick], "best_p": [best[t][0] for t in tick], "best_sig": [best[t][1] for t in tick]})


def main():
    need = ["date", "ticker", "sector", "beta_252", "vol_roc_21", "vol_roc_10", "rv20_roc_21", "rv20_roc_10", "ret_21", "ret_5", "dvol_roc_21", "vol_accel_5", "rv_accel_5",
            "fwd_sharpe_42", "fwd_mdd_42", "fwd_r2_42", "fwd_up_42"] + QUINTILE_FACTORS
    df = tag(load_research()[list(dict.fromkeys(need))].reset_index(drop=True))
    same = same_regime(df, today_regime())
    y = smooth_top_quartile(df, 42).astype(float).values
    ok = ~np.isnan(y) & same
    early = (df.date <= DISCOVER_END).values & ok
    late = (df.date >= CONFIRM_START).values & ok
    conds = build_conditions(df)
    print(f"rows early={early.sum():,} late={late.sum():,}; {len(conds)} conditions", file=sys.stderr)

    summary = []
    fwd, bd, bc = search(conds, y, early, late)
    fwd_set = set(fwd[fwd.confirmed].signature)
    summary.append({"test": "forward (as deployed)", "discovered": len(fwd), "confirmed": len(fwd_set), "base_disc": bd, "base_conf": bc,
                    "median_p_conf_confirmed": fwd[fwd.confirmed].p_conf.median()})
    print(f"forward: discovered {len(fwd)} confirmed {len(fwd_set)}", file=sys.stderr)

    rev, bd, bc = search(conds, y, late, early)
    rev_set = set(rev[rev.confirmed].signature)
    inter = fwd_set & rev_set
    summary.append({"test": "reversed halves", "discovered": len(rev), "confirmed": len(rev_set), "base_disc": bd, "base_conf": bc,
                    "median_p_conf_confirmed": rev[rev.confirmed].p_conf.median(),
                    "overlap_with_forward": len(inter), "jaccard": len(inter) / max(1, len(fwd_set | rev_set))})
    print(f"reversed: discovered {len(rev)} confirmed {len(rev_set)}; overlap {len(inter)}", file=sys.stderr)
    rev.to_csv(RESULTS_DIR / "falsification_reversed_signatures.csv", index=False)

    # how the deployed forward-confirmed set performs when measured only on the half it was NOT selected on is already p_conf;
    # the reversed set measured on 2015-2020 tells whether early years carried the same family
    tf = fires_today(df, conds, fwd[fwd.confirmed]).rename(columns={"n_sig": "n_sig_forward", "best_p": "best_p_forward", "best_sig": "best_sig_forward"})
    tr = fires_today(df, conds, rev[rev.confirmed]).rename(columns={"n_sig": "n_sig_reversed", "best_p": "best_p_reversed", "best_sig": "best_sig_reversed"})
    tonight = tf.merge(tr, on="ticker")
    try:
        L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv")[["rank", "tier", "ticker", "P_topq_42d"]]
        tonight = L.merge(tonight, on="ticker", how="left").sort_values("rank")
    except FileNotFoundError:
        pass
    tonight.to_csv(RESULTS_DIR / "falsification_tonight.csv", index=False)

    rng = np.random.default_rng(20151231)
    dates = df.date.values
    order = np.argsort(dates, kind="stable")
    bounds = np.flatnonzero(np.diff(dates[order])) + 1
    for s in range(N_PERM):
        yp = y.copy()
        for grp in np.split(order, bounds):
            yp[grp] = y[rng.permutation(grp)]
        per, bd, bc = search(conds, yp, early, late)
        n_conf = int(per.confirmed.sum()) if len(per) else 0
        summary.append({"test": f"permuted targets, seed {s}", "discovered": len(per), "confirmed": n_conf, "base_disc": bd, "base_conf": bc,
                        "median_p_conf_confirmed": per[per.confirmed].p_conf.median() if n_conf else np.nan,
                        "max_lift_disc": per.lift_disc.max() if len(per) else np.nan})
        print(f"permuted seed {s}: discovered {len(per)} confirmed {n_conf}", file=sys.stderr)

    out = pd.DataFrame(summary)
    out.to_csv(RESULTS_DIR / "falsification_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    t12 = tonight[tonight.tier <= 2] if "tier" in tonight else tonight.head(40)
    print("\nTier 1-2 tonight under forward vs reversed engines:")
    print(t12[["rank", "tier", "ticker", "P_topq_42d", "n_sig_forward", "best_p_forward", "n_sig_reversed", "best_p_reversed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
