"""Which characteristics precede a superior 21-session trade? For every feature: per-date AUC of the feature for sup_21
(0.5 = no information, above = higher values precede superior trades, below = lower values do), its mean by year and the
share of years on the same side, the same inside beta terciles and inside typical-move terciles (so a feature cannot score
by being a volatility proxy), the split by market state, and the mean at 42 and 63 sessions. 2015 onward, COVID window out."""
import time
import numpy as np
import pandas as pd
from .common import load, feature_sets, date_auc, OUT, MARKET

H = 21


def main(feats=None, tag=""):
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    fs = feature_sets(); feats = feats or fs["stock"]
    T = load(feats + MARKET + ["beta_l1_252", "mad_63", "sup_21", "sup_42", "sup_63", "xs_21", "mar_21"])
    dcode = T.date.factorize(sort=True)[0]; dates = np.sort(T.date.unique())
    years = pd.DatetimeIndex(dates).year
    bb = T.groupby("date").beta_l1_252.rank(pct=True); vb = T.groupby("date").mad_63.rank(pct=True)
    bcode = dcode * 3 + np.clip((bb.fillna(0.5).to_numpy() * 3).astype(int), 0, 2)
    vcode = dcode * 3 + np.clip((vb.fillna(0.5).to_numpy() * 3).astype(int), 0, 2)
    # market state per date (tercile cut points from the whole sample, fixed)
    M = T.groupby("date")[MARKET].first()
    reg = pd.DataFrame(index=M.index)
    reg["spy_near_high"] = np.where(M.mkt_off_high_252 >= -0.05, "SPY within 5% of high", "SPY >5% below high")
    reg["vix"] = pd.cut(M.mkt_vix_pctile, [-0.01, 0.33, 0.67, 1.0], labels=["VIX low vs own year", "VIX mid", "VIX high vs own year"]).astype(str)
    reg["breadth"] = pd.qcut(M.mkt_breadth_63h, 3, labels=["breadth low", "breadth mid", "breadth high"]).astype(str)
    reg["spy_21"] = np.where(M.mkt_roc_21 >= 0, "SPY 21d up", "SPY 21d down")
    print(f"rows {len(T):,} dates {len(dates)} features {len(feats)} {time.time()-t0:.0f}s", flush=True)
    base = T.groupby(T.date.dt.year)[["sup_21", "sup_42", "sup_63"]].mean()
    print("base rate by year:\n", base.round(3).to_string(), flush=True)
    rows = []; peryear = {}; perreg = {}
    y21 = T.sup_21.to_numpy(float)
    for i, f in enumerate(feats):
        x = T[f].to_numpy(float)
        a = date_auc(x, y21, dcode).reindex(range(len(dates))).to_numpy()
        ab = np.nanmean(date_auc(x, y21, bcode).reindex(range(3 * len(dates))).to_numpy().reshape(-1, 3), axis=1)
        av = np.nanmean(date_auc(x, y21, vcode).reindex(range(3 * len(dates))).to_numpy().reshape(-1, 3), axis=1)
        a42 = np.nanmean(date_auc(x, T.sup_42.to_numpy(float), dcode)); a63 = np.nanmean(date_auc(x, T.sup_63.to_numpy(float), dcode))
        s = pd.Series(a, index=dates); yr = s.groupby(years).mean(); peryear[f] = yr
        side = np.sign(np.nanmean(a) - 0.5)
        same = float((np.sign(yr - 0.5) == side).mean())
        rg = {f"{c}: {lab}": float(np.nanmean(s[reg[c].reindex(s.index).values == lab])) for c in reg for lab in sorted(reg[c].unique())}
        perreg[f] = rg
        rows.append({"feature": f, "auc_21": np.nanmean(a), "auc_21_in_beta_terciles": np.nanmean(ab), "auc_21_in_move_terciles": np.nanmean(av),
                     "years_same_side": same, "worst_year": float(yr.min() if side > 0 else yr.max()), "auc_42": a42, "auc_63": a63,
                     "auc_2015_2020": float(s[s.index < "2021-01-01"].mean()), "auc_2021_2023": float(s[(s.index >= "2021-01-01") & (s.index < "2024-01-01")].mean()),
                     "auc_2024_on": float(s[s.index >= "2024-01-01"].mean())})
        if i % 20 == 0: print(f"  {i}/{len(feats)} {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R["edge"] = (R.auc_21 - 0.5).abs()
    R["vol_neutral_edge"] = np.minimum((R.auc_21_in_beta_terciles - 0.5) * np.sign(R.auc_21 - 0.5), (R.auc_21_in_move_terciles - 0.5) * np.sign(R.auc_21 - 0.5))
    R = R.sort_values("edge", ascending=False)
    R.round(4).to_csv(OUT / f"characteristics_univariate{tag}.csv", index=False)
    pd.DataFrame(peryear).T.round(4).to_csv(OUT / f"characteristics_by_year{tag}.csv")
    pd.DataFrame(perreg).T.round(4).to_csv(OUT / f"characteristics_by_market_state{tag}.csv")
    # base rate structure of the target itself: by beta and typical-move quintile
    T["beta_q"] = pd.qcut(bb, 5, labels=[1, 2, 3, 4, 5]); T["move_q"] = pd.qcut(vb, 5, labels=[1, 2, 3, 4, 5])
    st = pd.concat({"by L1 beta quintile": T.groupby("beta_q", observed=True).sup_21.mean(), "by typical-move quintile": T.groupby("move_q", observed=True).sup_21.mean()}, axis=1)
    if not tag: st.round(3).to_csv(OUT / "target_by_risk_quintile.csv")
    print(st.round(3).to_string())
    print(R.head(40).round(3).to_string(index=False))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--feats":
        main(sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else "_subset")
    else:
        main()
