"""Rank names by signal strength and MSI likeness, after testing each ranking key.

Keys (all on the MSI-grade target: +25% over 126 sessions, no pullback over 6%):
  analog   analog odds: among eligible past stock-weeks whose state sits in a small box around
           the name's state (volatility, 63-day range, 6-month drawdown, 6-month return
           percentiles; distance from the 52-week high), hits / expected hits, shrunk toward 1
           with 6 pseudo expected hits. Counting in a neighbourhood, nothing fitted.
  msi_dist distance to MSI's own pre-run state (median weekly percentiles Jun 1 - Oct 27 2023),
           mean absolute percentile gap over the four percentiles and the 52-week-high percentile.
  gap126 / gap260  shape gaps to MSI's 2024 path (msi_like.py).
  model    walk-forward model score (63-session target).
Test: analog odds built from 2006-2016 rows only, every key applied to watch rows from
2017-07 on, ranked within each week, realized regime-neutral lift by third. MSI excluded.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common import DATA_DIR, RES_DIR
from msi_like import T126, T260, template
from prototype import forward126

DIMS = ["q_mar63", "q_rng63", "q_mdd126", "q_roc126", "ddh252"]
HALF = np.array([0.05, 0.05, 0.05, 0.10, 0.03])
K = 6.0
MSI_DIMS = ["q_mar63", "q_rng63", "q_mdd126", "q_roc126", "q_ddh252"]
BAND = 0.03


def load():
    cols = ["date", "ticker", "sector", "industry", "mcap", "q_mar63", "q_rng63", "q_mdd126", "q_roc126", "q_ddh252", "ddh252"]
    d = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=cols)
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    d = d.merge(forward126(), on=["date", "ticker"], how="left")
    d = d[d["date"] >= "2006-01-01"].reset_index(drop=True)
    d["proto"] = ((d["fr126"] >= np.log(1.25)) & (d["fdd126"] <= 0.06)).astype("float32").where(d["fr126"].notna())
    d["_wb"] = d.groupby("date")["proto"].transform("mean")
    d["pinned"] = (d["q_mar63"] <= 0.01) & (d["q_rng63"] <= 0.01)
    d["elig"] = (d["mcap"] >= 2000) & (d["industry"] != "Banks - Regional") & ~d["pinned"]
    d["p_quiet"] = (d["q_mar63"] <= 0.2) & (d["q_mdd126"] <= 0.3) & (d["q_roc126"] <= 0.4) & (d["ddh252"] > -0.105)
    d["p_jnj_msi"] = ((d["q_rng63"] <= 0.10) & (d["q_mar63"] <= 0.10) & (d["q_mdd126"] <= 0.15) & (d["ddh252"] > -0.10)
                      & (d["q_roc126"] >= 0.5))
    d["p_td"] = (d["q_rng63"] <= 0.10) & (d["q_roc126"] >= 0.8) & (d["ddh252"] > -0.10) & (d["q_mar63"] <= 0.2)
    d["watch"] = (d["p_quiet"] | d["p_jnj_msi"] | d["p_td"]) & d["elig"]
    return d


def shape_gaps(dates):
    px = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet", columns=["date", "ticker", "close"])
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    tm = {"gap126": template(px, *T126), "gap260": template(px, *T260)}
    keep = pd.DatetimeIndex(dates)
    parts = []
    for tkr, g in px.groupby("ticker", sort=False):
        g = g.sort_values("date")
        x = np.log(g["close"].to_numpy(float))
        dd = pd.DatetimeIndex(g["date"])
        idx = np.flatnonzero(dd.isin(keep))
        out = {"date": dd[idx].to_numpy(), "ticker": tkr}
        for name, t in tm.items():
            n = len(t)
            gap = np.full(len(idx), np.nan)
            ok = idx >= n - 1
            if ok.any():
                W = sliding_window_view(x, n)[idx[ok] - (n - 1)]
                lo, hi = np.nanmin(W, axis=1, keepdims=True), np.nanmax(W, axis=1, keepdims=True)
                S = (W - lo) / np.where(hi > lo, hi - lo, np.nan)
                gap[ok] = np.nanmean(np.abs(S - t), axis=1)
            out[name] = gap
        parts.append(pd.DataFrame(out))
    return pd.concat(parts, ignore_index=True)


class Pool:
    """Labelled eligible rows sorted on volatility for fast box queries."""

    def __init__(self, s):
        s = s.sort_values("q_mar63")
        self.X = s[DIMS].to_numpy(np.float64)
        self.y = s["proto"].to_numpy(np.float64)
        self.w = s["_wb"].to_numpy(np.float64)
        self.first = (s["date"] <= "2016-12-31").to_numpy()
        self.tk = s["ticker"].to_numpy()

    def query(self, v, min_e=8.0, detail=False):
        if np.isnan(v).any():
            return dict(analog=np.nan)
        for mult in (1.0, 1.5, 2.25, 3.4):
            h = HALF * mult
            lo = np.searchsorted(self.X[:, 0], v[0] - h[0], "left")
            hi = np.searchsorted(self.X[:, 0], v[0] + h[0], "right")
            m = np.all(np.abs(self.X[lo:hi, 1:] - v[1:]) <= h[1:], axis=1)
            e = self.w[lo:hi][m].sum()
            if e >= min_e:
                break
        y, w = self.y[lo:hi][m], self.w[lo:hi][m]
        out = dict(analog=(y.sum() + K) / (e + K))
        if detail:
            f = self.first[lo:hi][m]
            out.update(box_weeks=int(m.sum()), box_hits=int(y.sum()), box_hit_names=len(np.unique(self.tk[lo:hi][m][y > 0])),
                       box_hit_rate=y.mean() if len(y) else np.nan, box_mult=mult,
                       analog_2006_16=y[f].sum() / w[f].sum() if w[f].sum() > 0 else np.nan,
                       analog_2017_26=y[~f].sum() / w[~f].sum() if w[~f].sum() > 0 else np.nan)
        return out


def lifts(s):
    f = s["date"] <= "2021-12-31"
    return dict(weeks=len(s), hit=s["proto"].mean(), lift=s["proto"].sum() / s["_wb"].sum(),
                lift_2017_21=s.loc[f, "proto"].sum() / s.loc[f, "_wb"].sum() if f.any() else np.nan,
                lift_2022_26=s.loc[~f, "proto"].sum() / s.loc[~f, "_wb"].sum() if (~f).any() else np.nan)


def thirds(t, key, ascending):
    """Rank key within each week, realized lift by third (3 = best: analog/score high, distances low)."""
    r = t.groupby("date")[key].rank(pct=True, ascending=not ascending)
    b = np.ceil(r * 3).clip(1, 3)
    return [dict(key=key, third=k, **lifts(t[b == k])) for k in (1, 2, 3)]


def test_all(lab, train, step=4):
    """Analog odds as a stand-alone screen: every 4th week from 2017-07, all eligible names."""
    weeks = np.sort(lab.loc[lab["date"] >= "2017-07-01", "date"].unique())[::step]
    t = lab[lab["date"].isin(weeks) & (lab["ticker"] != "MSI")].copy()
    t["analog"] = [train.query(v)["analog"] for v in t[DIMS].to_numpy(np.float64)]
    t["bin"] = pd.cut(t["analog"], [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf], right=False)
    rows = [dict(analog=str(b), group=g, **lifts(s)) for b, s0 in t.groupby("bin", observed=True)
            for g, s in (("all", s0), ("watch", s0[s0["watch"]]), ("not watch", s0[~s0["watch"]])) if len(s)]
    for thr in (2.5, 3.0):
        hi = t["analog"] >= thr
        rows += [dict(analog=f">= {thr}", group=g, **lifts(t[m])) for g, m in
                 (("all", hi), ("watch", hi & t["watch"]), ("not watch", hi & ~t["watch"]), ("watch or analog", hi | t["watch"]))]
    rows.append(dict(analog="any", group="watch", **lifts(t[t["watch"]])))
    for name, m in tiers(t).items():
        rows.append(dict(analog="tier", group=name, **lifts(t[m])))
    t[["date", "ticker", "analog", "watch", "proto", "_wb"]].to_parquet(f"{DATA_DIR}/rank_test_rows.parquet", index=False)
    return pd.DataFrame(rows)


TIER_NAMES = {1: "1: watch, analog >= 2.5", 2: "2: watch, analog < 2.5", 3: "3: off watch, analog >= 3"}


def tiers(t):
    """Cut points read off the 2017-26 bins: inside the watch the odds step up at 2.5, outside it at 3."""
    w = t["watch"].astype(bool)
    return {TIER_NAMES[1]: w & (t["analog"] >= 2.5), TIER_NAMES[2]: w & (t["analog"] < 2.5),
            TIER_NAMES[3]: ~w & (t["analog"] >= 3.0), "rest: off watch, analog < 3": ~w & (t["analog"] < 3.0)}


def main_band():
    """MSI-grade odds by 20-session high-low band width, eligible names and the watch."""
    d = load()
    px = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet", columns=["date", "ticker", "high", "low", "close"])
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    px = px.sort_values(["ticker", "date"])
    g = px.groupby("ticker", sort=False)
    px["rng20"] = (g["high"].transform(lambda s: s.rolling(20).max()) - g["low"].transform(lambda s: s.rolling(20).min())) / px["close"]
    d = d.merge(px[["date", "ticker", "rng20"]], on=["date", "ticker"], how="left")
    lab = d[d["proto"].notna() & d["elig"]].copy()
    lab["band"] = pd.cut(lab["rng20"], [0, 0.02, 0.025, 0.03, 0.04, 0.06, np.inf], right=False).astype(str)
    lab["side"] = np.where(lab["rng20"] < BAND, f"under {BAND:.0%}", f"{BAND:.0%} or wider")
    rows = []
    for grp, m in (("eligible", np.ones(len(lab), bool)), ("watch", lab["watch"].to_numpy())):
        for b, s in list(lab[m].groupby("band")) + list(lab[m].groupby("side")):
            f = s["date"] <= "2016-12-31"
            rows.append(dict(group=grp, band=b, weeks=len(s), names=s["ticker"].nunique(), hit=s["proto"].mean(),
                             lift=s["proto"].sum() / s["_wb"].sum(), lift_2006_16=s.loc[f, "proto"].sum() / s.loc[f, "_wb"].sum(),
                             lift_2017_26=s.loc[~f, "proto"].sum() / s.loc[~f, "_wb"].sum()))
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES_DIR}/rank_band_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


def main_test_all():
    d = load()
    lab = d[d["proto"].notna() & d["elig"]]
    out = test_all(lab, Pool(lab[lab["date"] <= "2016-12-31"]))
    out.to_csv(f"{RES_DIR}/rank_tests_all.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


def main():
    d = load()
    sg = shape_gaps(d["date"].unique())
    d = d.merge(sg, on=["date", "ticker"], how="left")
    msi = d[(d["ticker"] == "MSI") & (d["date"] >= "2023-06-01") & (d["date"] <= "2023-10-27")]
    ref = msi[MSI_DIMS].median()
    print("MSI pre-run state (median Jun 1 - Oct 27 2023):", ref.round(3).to_dict(), "ddh252", round(msi["ddh252"].median(), 3))
    d["msi_dist"] = (d[MSI_DIMS] - ref).abs().mean(axis=1)

    # the walk-forward scores sit on the earlier weekly grid: take each name's latest score from the prior 6 days
    oos = pd.read_parquet(f"{DATA_DIR}/oos_binary_y63.parquet", columns=["date", "ticker", "score"])
    oos["date"] = pd.to_datetime(oos["date"]).astype("datetime64[ns]")
    d = pd.merge_asof(d.sort_values("date"), oos.sort_values("date"), on="date", by="ticker",
                      tolerance=pd.Timedelta(days=6), direction="backward").reset_index(drop=True)

    lab = d[d["proto"].notna() & d["elig"]]
    train = Pool(lab[lab["date"] <= "2016-12-31"])
    test = lab[lab["watch"] & (lab["date"] >= "2017-07-01") & (lab["ticker"] != "MSI")].copy()
    test["analog"] = [train.query(v)["analog"] for v in test[DIMS].to_numpy(np.float64)]
    rows = []
    for key, asc in (("analog", False), ("msi_dist", True), ("gap126", True), ("gap260", True), ("score", False)):
        rows += thirds(test.dropna(subset=[key]), key, asc)
    tr = pd.DataFrame(rows)
    tr.to_csv(f"{RES_DIR}/rank_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\nwatch rows 2017-07 on: {len(test):,}, hit {test['proto'].mean():.4f}, lift {test['proto'].sum() / test['_wb'].sum():.2f}")
    print("third 3 = best on the key within each week (analog high, distances low, score high)")
    print(tr.round(3).to_string(index=False))

    full = Pool(lab)
    now = d[d["date"] == d["date"].max()].copy()
    now = now[now["elig"]]
    q = pd.DataFrame([full.query(v, detail=True) for v in now[DIMS].to_numpy(np.float64)], index=now.index)
    now = pd.concat([now, q], axis=1)
    ml = pd.read_csv(f"{RES_DIR}/msi_like_{now['date'].iloc[0]:%Y-%m-%d}.csv",
                     usecols=["ticker", "gain126", "worst_pullback126", "pre_run", "in_run"])
    sc = pd.read_csv(f"{RES_DIR}/scan_{now['date'].iloc[0]:%Y-%m-%d}.csv", usecols=["ticker", "price", "model_pct", "proto_profile", "rules_list"])
    now = now.drop(columns=["score"]).merge(ml, on="ticker", how="left").merge(sc, on="ticker", how="left")
    # 20-session high-low band under 3%: held at a deal price or stalled; inside the watch these ran at ~0.4x (main_band)
    px = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet", columns=["date", "ticker", "high", "low", "close"])
    px = px[pd.to_datetime(px["date"]) >= pd.to_datetime(px["date"]).max() - pd.Timedelta(days=40)].sort_values("date")
    g = px.groupby("ticker").tail(20).groupby("ticker")
    now["rng20"] = now["ticker"].map((g["high"].max() - g["low"].min()) / g["close"].last())
    now["deal_pinned"] = now["rng20"] < BAND
    now["tier"] = np.nan
    for k, name in TIER_NAMES.items():
        now.loc[tiers(now)[name] & ~now["deal_pinned"].fillna(False), "tier"] = k
    # within a tier the odds are flat, so order by closeness to MSI's 2023 base (neutral on odds)
    now = now.sort_values(["tier", "msi_dist", "analog"], ascending=[True, True, False]).reset_index(drop=True)
    now["order"] = np.where(now["tier"].notna(), now["tier"].notna().cumsum(), np.nan)
    now.to_csv(f"{RES_DIR}/rank_{now['date'].iloc[0]:%Y-%m-%d}.csv", index=False)
    show = ["ticker", "industry", "mcap", "watch", "proto_profile", "analog", "analog_2006_16", "analog_2017_26", "box_weeks", "box_hits",
            "box_hit_names", "msi_dist", "gap126", "gap260", "gain126", "worst_pullback126", "model_pct"]
    print("\ntop 40 by analog odds:")
    print(now.sort_values("analog", ascending=False)[show].head(40).round(3).to_string(index=False))
    print("\nranked (tier, then closeness to MSI's 2023 base):")
    print(now[now["tier"].notna()][["order", "tier"] + show + ["rng20"]].round(3).to_string(index=False))
    print("\nheld at a deal price:", now.loc[now["deal_pinned"].fillna(False) & now["watch"], "ticker"].tolist())


if __name__ == "__main__":
    import sys
    {"test_all": main_test_all, "band": main_band}.get((sys.argv[1:] or ["main"])[0], main)()
