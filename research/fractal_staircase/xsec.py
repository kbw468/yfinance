"""Cross-sectional, group (sector/industry) and market-regime layer.

Adds to the weekly panel:
  * per-date percentile ranks of every feature across the whole universe (prefix q_)
  * within-sector / within-industry percentile ranks of key features (prefix sq_, iq_)
  * sector / industry participation: medians and breadth of key features (prefix sec_, ind_)
  * market regime from the universe itself and from Yahoo index series (prefix mkt_)
All of it is rank / median / fraction based.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))
MIN_DOLLAR_VOL = 2e6      # median daily dollar volume (63 bars) needed to be in the sample

ID = ["date", "ticker"]
LABEL_PREFIX = ("fr", "fdd", "fae", "fer", "fmar")
GROUP_KEYS = ["roc21", "roc63", "roc126", "er21", "er63", "kt63", "vc21_126", "vroc21",
              "rc21_252", "rv10_126", "vlroc21", "mdd63", "udv63", "hexp126", "acc21"]


def is_label(c):
    return any(c.startswith(p) and c[len(p):].isdigit() for p in LABEL_PREFIX)


def aux_series():
    tick = {"SPY": "spy", "RSP": "rsp", "IWM": "iwm", "^VIX": "vix", "^VIX3M": "vix3m", "^VVIX": "vvix"}
    raw = yf.download(list(tick), start="2004-01-01", auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns=tick)
    out = pd.DataFrame(index=raw.index)
    lx = np.log(raw)
    for k in ("spy", "rsp", "iwm"):
        out[f"mkt_{k}_roc21"] = lx[k] - lx[k].shift(21)
        out[f"mkt_{k}_roc63"] = lx[k] - lx[k].shift(63)
        out[f"mkt_{k}_ddh252"] = lx[k] - lx[k].rolling(252, min_periods=200).max()
    out["mkt_rsp_spy_roc63"] = out["mkt_rsp_roc63"] - out["mkt_spy_roc63"]
    out["mkt_iwm_spy_roc63"] = out["mkt_iwm_roc63"] - out["mkt_spy_roc63"]
    out["mkt_vix"] = raw["vix"]
    out["mkt_vix_roc21"] = lx["vix"] - lx["vix"].shift(21)
    out["mkt_vix_pct252"] = raw["vix"].rolling(252, min_periods=200).rank(pct=True)
    out["mkt_vix_term"] = np.log(raw["vix"] / raw["vix3m"])
    out["mkt_vvix_vix"] = np.log(raw["vvix"] / raw["vix"])
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.ffill()
    out.index.name = "date"
    return out.reset_index()


def consecutive(flag, key):
    """Length of the current run of True values, per ticker (weekly rows)."""
    f = flag.astype(int)
    grp = (f != f.groupby(key).shift()).cumsum()
    return f.groupby([key, grp]).cumsum().where(f == 1, 0)


def derived(panel):
    """Cross-family ratios, multi-scale agreement, staircase memory, setup age.

    Weekly rows: shift(4) ~ 20 bars, shift(13) ~ 65 bars.
    """
    p = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    tk = p["ticker"]
    lg = lambda c, k: p.groupby("ticker")[c].shift(k)
    # multi-scale agreement (self-similarity)
    p["kt_min"] = p[["kt21", "kt63", "kt126"]].min(axis=1)
    p["er_min3"] = p[["er21", "er63", "er126"]].min(axis=1)
    p["roc_align"] = sum((p[c] > 0).astype(int) for c in ("roc21", "roc63", "roc126", "roc252"))
    # cross-family rate-of-change spreads (all log changes, so differences are ratios)
    p["pv_spread21"] = p["roc21"] - p["vroc21"]           # price up while volatility falls
    p["pv_spread63"] = p["roc63"] - p["vroc63"]
    p["pr_spread21"] = p["roc21"] - p["rroc21"]           # same with daily range
    p["vv_spread21"] = p["vlroc21"] - p["vroc21"]         # volume up while volatility falls
    p["vv_spread63"] = p["vlroc63"] - p["vroc63"]
    p["pvv21"] = p["roc21"] - p["vroc21"] + p["vlroc21"]  # the full triad
    p["pvv63"] = p["roc63"] - p["vroc63"] + p["vlroc63"]
    p["pv_ratio63"] = p["roc63"] / (p["mar63"] * 63) / np.exp(p["vroc63"])
    # rate of change of the ratios themselves (~20 bars)
    for c in ("vc21_126", "rc21_252", "rv10_126", "udv63", "er63", "kt63", "mdd63_rel", "gp63", "dn63"):
        p[f"d_{c}"] = p[c] - lg(c, 4)
    # backward-looking staircase state at three scales (fractal recurrence)
    p["stair_b21"] = ((p["roc21"] >= np.log(1.05)) & (p["mdd21"] <= p["roc21"] / 3)).astype(float)
    p["stair_b63"] = ((p["roc63"] >= np.log(1.10)) & (p["mdd63"] <= p["roc63"] / 3)).astype(float)
    p["stair_b126"] = ((p["roc126"] >= np.log(1.20)) & (p["mdd126"] <= p["roc126"] / 3)).astype(float)
    p["stair_scales"] = p["stair_b21"] + p["stair_b63"] + p["stair_b126"]
    # setup age: how many consecutive weeks the condition has held
    p["age_trend"] = consecutive((p["roc63"] > 0) & (p["er63"] > 0.1), tk).astype(float)
    p["age_comp"] = consecutive(p["vc21_126"] < 0, tk).astype(float)
    p["age_stair"] = consecutive(p["stair_b63"] == 1, tk).astype(float)
    # memory: share of fully-realised past windows that were staircases
    y63 = ((p["fr63"] >= np.log(1.10)) & (p["fdd63"] <= p["fr63"] / 3)).astype(float).where(p["fr63"].notna())
    y21 = ((p["fr21"] >= np.log(1.05)) & (p["fdd21"] <= p["fr21"] / 3)).astype(float).where(p["fr21"].notna())
    p["_y63"], p["_y21"] = y63, y21
    g = p.groupby("ticker")
    p["mem_y63"] = g["_y63"].transform(lambda s: s.shift(13).rolling(52, min_periods=20).mean())
    p["mem_y21"] = g["_y21"].transform(lambda s: s.shift(5).rolling(52, min_periods=20).mean())
    p["mem_y63_long"] = g["_y63"].transform(lambda s: s.shift(13).rolling(260, min_periods=52).mean())
    return p.drop(columns=["_y63", "_y21"])


def main():
    panel = derived(pd.read_parquet(os.path.join(DATA_DIR, "panel_raw.parquet")))
    uni = pd.read_csv(os.path.join(HERE, "universe.csv")).rename(
        columns={"Ticker": "ticker", "Sector": "sector", "Industry": "industry", "Market Cap": "mcap"})
    panel = panel.merge(uni[["ticker", "sector", "industry", "mcap"]], on="ticker", how="left")
    panel = panel[panel["ldv63"] >= np.log(MIN_DOLLAR_VOL)].reset_index(drop=True)

    feats = [c for c in panel.columns if c not in ID + ["sector", "industry", "mcap", "bars"] and not is_label(c)]
    gb = panel.groupby("date", sort=False)

    # universe-wide percentile ranks (chunked to keep peak memory down)
    qs = []
    for i in range(0, len(feats), 20):
        part = gb[feats[i:i + 20]].rank(pct=True).astype("float32")
        part.columns = ["q_" + c for c in feats[i:i + 20]]
        qs.append(part)
    q = pd.concat(qs, axis=1)
    del qs

    # within sector / within industry ranks of key features
    sq = panel.groupby(["date", "sector"])[GROUP_KEYS].rank(pct=True).astype("float32")
    sq.columns = ["sq_" + c for c in GROUP_KEYS]
    icount = panel.groupby(["date", "industry"])["ticker"].transform("count")
    iq = panel.groupby(["date", "industry"])[GROUP_KEYS].rank(pct=True).astype("float32")
    iq = iq.where(icount >= 5)
    iq.columns = ["iq_" + c for c in GROUP_KEYS]

    # group participation: medians and breadth
    def group_stats(key, prefix, min_n):
        g = panel.groupby(["date", key])
        med = g[GROUP_KEYS].transform("median").astype("float32")
        med.columns = [f"{prefix}_med_{c}" for c in GROUP_KEYS]
        n = g["ticker"].transform("count")
        brd = pd.DataFrame({
            f"{prefix}_brd_up63": (panel["roc63"] > 0).groupby([panel["date"], panel[key]]).transform("mean"),
            f"{prefix}_brd_eff": ((panel["er63"] > 0.15) & (panel["roc63"] > 0)).groupby([panel["date"], panel[key]]).transform("mean"),
            f"{prefix}_brd_hi": (panel["ddh63"] > -0.02).groupby([panel["date"], panel[key]]).transform("mean"),
            f"{prefix}_brd_vcomp": (panel["vc21_126"] < 0).groupby([panel["date"], panel[key]]).transform("mean"),
        }).astype("float32")
        out = pd.concat([med, brd], axis=1)
        return out.where(n >= min_n)

    sec = group_stats("sector", "sec", 5)
    ind = group_stats("industry", "ind", 5)
    # the stock relative to its group (rate-of-change spread)
    rel = pd.DataFrame({
        "rel_ind_roc63": panel["roc63"] - ind["ind_med_roc63"],
        "rel_ind_roc21": panel["roc21"] - ind["ind_med_roc21"],
        "rel_sec_roc63": panel["roc63"] - sec["sec_med_roc63"],
        "rel_ind_er63": panel["er63"] - ind["ind_med_er63"],
        "rel_ind_vroc21": panel["vroc21"] - ind["ind_med_vroc21"],
        "rel_ind_mdd63": panel["mdd63"] - ind["ind_med_mdd63"],    # drawdown vs peers
        "rel_sec_mdd63": panel["mdd63"] - sec["sec_med_mdd63"],
        "rel_ind_vlroc21": panel["vlroc21"] - ind["ind_med_vlroc21"],
    }).astype("float32")

    # market regime from the universe itself
    m = pd.DataFrame({
        "mkt_brd_up63": (panel["roc63"] > 0).groupby(panel["date"]).transform("mean"),
        "mkt_brd_hi": (panel["ddh63"] > -0.02).groupby(panel["date"]).transform("mean"),
        "mkt_brd_eff": ((panel["er63"] > 0.15) & (panel["roc63"] > 0)).groupby(panel["date"]).transform("mean"),
        "mkt_med_roc21": gb["roc21"].transform("median"),
        "mkt_med_roc63": gb["roc63"].transform("median"),
        "mkt_med_vc21_126": gb["vc21_126"].transform("median"),
        "mkt_med_vroc21": gb["vroc21"].transform("median"),
        "mkt_med_rv10_126": gb["rv10_126"].transform("median"),
        "mkt_med_mar21": gb["mar21"].transform("median"),
    }).astype("float32")

    out = pd.concat([panel, q, sq, iq, sec, ind, rel, m], axis=1)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["age_hot"] = consecutive(out["q_roc63"] >= 0.8, out["ticker"]).astype(float)
    out["age_quiet"] = consecutive(out["q_mar21"] <= 0.3, out["ticker"]).astype(float)
    # industry momentum and its rate of change (~20 bars)
    out["ind_d_roc63"] = out["ind_med_roc63"] - out.groupby("ticker")["ind_med_roc63"].shift(4)
    out["sec_d_roc63"] = out["sec_med_roc63"] - out.groupby("ticker")["sec_med_roc63"].shift(4)
    # rank dynamics: rate of change of the stock's position inside the universe (~20 bars)
    for c in ("roc63", "mar21", "er63", "kt63", "rs63", "vlroc21", "mdd63"):
        if "q_" + c in out.columns:
            out[f"dq_{c}"] = out["q_" + c] - out.groupby("ticker")["q_" + c].shift(4)
    for c in ("age_hot", "age_quiet", "ind_d_roc63", "sec_d_roc63") + tuple(c for c in out.columns if c.startswith("dq_")):
        out["q_" + c] = out.groupby("date")[c].rank(pct=True).astype("float32")
    aux = aux_series()
    out["date"] = out["date"].astype("datetime64[ns]")
    aux["date"] = aux["date"].astype("datetime64[ns]")
    out = pd.merge_asof(out.sort_values("date"), aux.sort_values("date"), on="date")
    out["mcap_bucket"] = pd.cut(out["mcap"], [0, 2e3, 10e3, 50e3, 200e3, 1e9],
                                labels=["<2B", "2-10B", "10-50B", "50-200B", ">200B"])
    out = out.sort_values(ID).reset_index(drop=True)
    out.to_parquet(os.path.join(DATA_DIR, "panel.parquet"), index=False)
    print("panel", out.shape, "dates", out["date"].nunique(), "tickers", out["ticker"].nunique())


if __name__ == "__main__":
    main()
