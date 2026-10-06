"""Precedent engine: identity-neutral nearest-neighbour analogs.

For a (date, ticker) in a given beta bucket, the analogs are the K historical (date', ticker') cases in the SAME
bucket, DIFFERENT ticker, whose price/volume/volatility state is closest in rank space. The precedent probability
is the share of those cases whose forward 42d Sharpe landed in the top half (and top quartile) of the universe on
their own date; the precedent expectation is their mean excess Sharpe vs the universe at 42 and 63 days.

Features = the factors the walk-forward composites selected (level and state), each as a per-date percentile rank
within the bucket, so a state is comparable across years.

Validation = walk-forward by calendar year with a 63-session embargo; neighbours come only from before the cutoff.
Live = neighbours from all labelled history (which necessarily ends 63 sessions before today).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .run_eval import load_research
from .composite import tercile_series
from .evaluate import daily_ic, nw_tstat

K = 100
RAW_K = 400          # pulled before the same-ticker / max-per-ticker filter
MAX_PER_TICKER = 3
EMBARGO = 63
TRAIN_STRIDE = 5
TEST_STRIDE = 5


def feature_list() -> list:
    lw = pd.read_csv(RESULTS_DIR / "composite_final_weights.csv", index_col=0)
    sw = pd.read_csv(RESULTS_DIR / "composite_state_final_weights.csv", index_col=0)
    feats = list(lw.dropna(how="all").index) + list(sw.dropna(how="all").index)
    core = ["mom_12_1", "rs_lead_126", "dist_52w_high", "days_since_20pct_dd", "sharpe_126", "idio_vol_63", "vol_dry_20_250",
            "obv_price_div_63", "rv20_pct_252", "dn_up_vol_asym_63", "corr_spy_63", "log_dvol_63", "base_depth_126", "rs_sec_63"]
    seen = []
    for f in core + feats:
        if f not in seen:
            seen.append(f)
    return seen


def prepare(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    df = df.copy()
    df["beta_bucket"] = tercile_series(df)
    df = df[df.beta_bucket.notna()]
    R = df.groupby(["date", "beta_bucket"], observed=True)[feats].rank(pct=True).fillna(0.5).astype("float32")
    R.columns = [f"r_{f}" for f in feats]
    out = pd.concat([df[["date", "ticker", "beta_bucket", "xs_spy_sharpe_42", "xs_spy_sharpe_63"]], R], axis=1)
    for h in (42, 63):
        y = out[f"xs_spy_sharpe_{h}"]
        med = y.groupby(out.date).transform("median")
        q75 = y.groupby(out.date).transform(lambda s: s.quantile(0.75))
        mean = y.groupby(out.date).transform("mean")
        out[f"top_half_{h}"] = (y > med).astype("float32").where(y.notna())
        out[f"top_q_{h}"] = (y > q75).astype("float32").where(y.notna())
        out[f"xs_u_{h}"] = (y - mean).astype("float32")
    return out


def analog_outcomes(train: pd.DataFrame, query: pd.DataFrame, rcols: list, want_cases=False):
    """Return DataFrame indexed like query with precedent stats (and named cases if want_cases)."""
    res = pd.DataFrame(index=query.index, columns=["an_p_top_half_42", "an_p_top_q_42", "an_xs_42", "an_p_top_half_63", "an_xs_63", "an_n", "an_dist"], dtype=float)
    cases = pd.Series("", index=query.index, dtype=object) if want_cases else None
    for b in ["low", "mid", "high"]:
        tr = train[train.beta_bucket == b]
        q = query[query.beta_bucket == b]
        if len(tr) < 1000 or q.empty:
            continue
        nn = NearestNeighbors(n_neighbors=min(RAW_K, len(tr)), algorithm="brute", n_jobs=3).fit(tr[rcols].values)
        dist, idx = nn.kneighbors(q[rcols].values)
        tr_t = tr.ticker.values; tr_th42 = tr.top_half_42.values; tr_tq42 = tr.top_q_42.values; tr_xs42 = tr.xs_u_42.values
        tr_th63 = tr.top_half_63.values; tr_xs63 = tr.xs_u_63.values; tr_d = tr.date.values; tr_y = tr.xs_spy_sharpe_42.values
        for i, (qi, qt) in enumerate(zip(q.index, q.ticker.values)):
            cand = idx[i]; tick = tr_t[cand]
            keep = tick != qt
            cand, tick, dd = cand[keep], tick[keep], dist[i][keep]
            # max MAX_PER_TICKER per ticker, preserving distance order
            counts = {}; sel = []
            for j, t in enumerate(tick):
                c = counts.get(t, 0)
                if c < MAX_PER_TICKER:
                    sel.append(j); counts[t] = c + 1
                if len(sel) >= K:
                    break
            sel = np.array(sel, dtype=int); c = cand[sel]
            res.loc[qi] = [np.nanmean(tr_th42[c]), np.nanmean(tr_tq42[c]), np.nanmean(tr_xs42[c]), np.nanmean(tr_th63[c]), np.nanmean(tr_xs63[c]), len(c), dd[sel].mean()]
            if want_cases:
                top = c[:6]
                cases[qi] = "; ".join(f"{tr_t[k]}@{pd.Timestamp(tr_d[k]).date()}:{tr_y[k]:+.1f}" for k in top)
    return (res, cases) if want_cases else res


def walk_forward(P: pd.DataFrame, rcols: list, years=range(2017, 2027)) -> pd.DataFrame:
    bdays = pd.DatetimeIndex(sorted(P.date.unique())); pos = pd.Series(np.arange(len(bdays)), index=bdays)
    outs = []
    for y in years:
        first = bdays[bdays >= pd.Timestamp(f"{y}-01-01")]
        if len(first) == 0:
            continue
        cut = bdays[pos[first[0]] - EMBARGO]
        tr = P[(P.date <= cut) & P.top_half_42.notna() & P.date.isin(bdays[:pos[cut] + 1][::TRAIN_STRIDE])]
        te_dates = bdays[(bdays >= first[0]) & (bdays <= pd.Timestamp(f"{y}-12-31"))][::TEST_STRIDE]
        te = P[P.date.isin(te_dates)]
        r = analog_outcomes(tr, te, rcols)
        outs.append(pd.concat([te[["date", "ticker", "beta_bucket", "xs_spy_sharpe_42", "xs_spy_sharpe_63", "top_half_42", "top_q_42"]], r], axis=1))
        print(f"  analog fold {y}: train {len(tr):,} test {len(te):,}", file=sys.stderr)
    return pd.concat(outs)


def main():
    df = load_research()[["date", "ticker", "sector", "beta_252", "xs_spy_sharpe_42", "xs_spy_sharpe_63"] + feature_list()].reset_index(drop=True)
    feats = feature_list()
    P = prepare(df, feats)
    rcols = [f"r_{f}" for f in feats]
    print(f"{len(feats)} features, {len(P):,} rows", file=sys.stderr)
    oos = walk_forward(P, rcols)
    oos.to_parquet(CACHE_DIR / "analog_oos.parquet", index=False)

    # evaluation: IC of precedent probability vs realised 42d/63d excess Sharpe; calibration of P(top half)
    rows = {}
    for col in ["an_p_top_half_42", "an_xs_42", "an_p_top_q_42"]:
        for tgt, h in [("xs_spy_sharpe_42", 42), ("xs_spy_sharpe_63", 63)]:
            ic = daily_ic(oos.dropna(subset=[col]), [col], tgt)[col]
            for win, start in [("full", None), ("recent", RECENT_START)]:
                m, t, _ = nw_tstat(ic.loc[start:], h)
                rows[(col, tgt, win)] = {"ic": m, "t_nw": t, "hit": (ic.loc[start:] > 0).mean()}
    ev = pd.DataFrame(rows).T; ev.index.names = ["predictor", "target", "window"]
    ev.to_csv(RESULTS_DIR / "analog_oos_evaluation.csv")
    rec = oos[oos.date >= RECENT_START].dropna(subset=["an_p_top_half_42"])
    rec["bin"] = pd.cut(rec.an_p_top_half_42, [0, .44, .47, .50, .53, .56, .60, 1.0])
    cal = rec.groupby("bin", observed=True).agg(n=("top_half_42", "size"), realised_top_half=("top_half_42", "mean"), realised_top_q=("top_q_42", "mean"),
                                                 realised_xs42=("xs_spy_sharpe_42", lambda s: (s - rec.loc[s.index].groupby(rec.loc[s.index, "date"]).xs_spy_sharpe_42.transform("mean")).mean()))
    cal.to_csv(RESULTS_DIR / "analog_calibration_recent.csv")
    pd.set_option("display.width", 200)
    print(ev.round(3).to_string()); print("\ncalibration (recent): predicted P(top half) bin -> realised\n", cal.round(3).to_string())


if __name__ == "__main__":
    main()
