"""Build the SPY drawdown signal dashboard (single self-contained HTML page).

Data: Yahoo Finance via yfinance (SPY, QQQ, HYG, IEF, ^VIX, ^VVIX, ^VXN, DX-Y.NYB) and the
Cboe S&P 500 Dispersion Index (DSPX) history CSV.

Tiers (evaluated only while SPY closes within 1.5% of its 52-week closing high):
  Watch            VIX or VXN 10-session change > +10% with VVIX 10-session change
                   below half of that move
  Working          Watch + (duration-hedged HYG 10-session change < 0
                            or DSPX >= 80th percentile of its trailing year)
  High conviction  Watch + duration-hedged HYG 10-session change < 0
                         + DSPX >= 60th percentile of its trailing year
  Low-risk         VXN/VIX 10-session change < -5%
  Size flag        DXY in the top 20% of its trailing-year range: when a tier fires with
                   this on, the odds of an 8%+ drop have run about 3x higher (not a tier)
Barometer (1-10): equal-weight average of the input percentiles listed in barometer.json,
  read as a decile of past near-high scores. select_barometer.py picks the inputs (walk-forward
  search over every 3-6 input combination of ten candidates) and is re-run monthly.
  History is walk-forward (each year scored by a model fit on earlier years); the odds
  shown for each band are what happened out of sample.
In a drawdown, the VVIX/VIX ratio's compression from its close on the SPY
52-week-high day is the severity read (25% by the first -5% close, 40% = 10%+).

Usage:
    python build_dashboard.py --out vol_signal_dashboard.html
"""

import argparse
import datetime as dt
import io
import json
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DSPX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/DSPX_History.csv"
ET = ZoneInfo("America/New_York")

GATE_PCT = -1.5        # SPY within 1.5% of its 52-week closing high
ROC_WIN = 10           # sessions for every rate-of-change input
VOL_TRIG = 10.0        # VIX / VXN 10-session change trigger, %
HEDGE_BETA = 0.45      # HYG minus 0.45 x IEF (duration hedge)
FWD = 40               # outcome window, sessions
DEDUPE = 20            # one print per 20 sessions
CHART_START = "2020-07-01"
RECORD_STARTS = {"since_2015": "2015-06-01", "post_covid": "2020-07-01"}

# NYSE full-day holidays used to lay out the next sessions
NYSE_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def fetch_yahoo(start):
    tickers = ["SPY", "QQQ", "HYG", "IEF", "^VIX", "^VVIX", "^VXN", "DX-Y.NYB"]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return raw


def fetch_dspx():
    resp = requests.get(DSPX_URL, timeout=30)
    resp.raise_for_status()
    x = pd.read_csv(io.StringIO(resp.text))
    if not {"DATE", "DSPX"}.issubset(x.columns):
        raise ValueError("unexpected DSPX file layout")
    x["DATE"] = pd.to_datetime(x["DATE"], format="%m/%d/%Y")
    return pd.to_numeric(x.set_index("DATE")["DSPX"], errors="coerce").dropna()


def pct_rank_1y(s):
    def rank(v):
        if np.isnan(v[-1]):
            return np.nan
        w = v[~np.isnan(v)]
        return (w <= v[-1]).mean() * 100
    return s.rolling(252, min_periods=200).apply(rank, raw=True)


def build_frame(raw, dspx):
    d = pd.DataFrame(index=raw["SPY"].dropna().index)
    for col, tk in (("spy", "SPY"), ("qqq", "QQQ"), ("hyg", "HYG"), ("ief", "IEF"),
                    ("vix", "^VIX"), ("vvix", "^VVIX"), ("vxn", "^VXN"), ("dxy", "DX-Y.NYB")):
        d[col] = raw[tk].reindex(d.index).ffill(limit=3)
    d["dspx"] = dspx.reindex(d.index).ffill(limit=3)
    d["dspx_date"] = pd.Series(dspx.index, index=dspx.index).reindex(d.index).ffill(limit=3)
    d["hi52"] = d.spy.rolling(252, min_periods=1).max()
    d["dd"] = (d.spy / d.hi52 - 1) * 100
    d["ratio"] = d.vvix / d.vix
    d["nv"] = d.vxn / d.vix
    excess = (d.hyg.pct_change() - HEDGE_BETA * d.ief.pct_change()).fillna(0)
    d["cred"] = (1 + excess).cumprod()
    for k in ("vix", "vvix", "vxn", "nv", "cred", "spy", "dxy"):
        d[f"{k}10"] = d[k].pct_change(ROC_WIN) * 100
    d["dspx_p1y"] = pct_rank_1y(d.dspx)
    d["dxy_p1y"] = pct_rank_1y(d.dxy)
    fmin = d.spy[::-1].rolling(FWD, min_periods=FWD).min()[::-1].shift(-1)
    d["f40"] = (fmin / d.spy - 1) * 100
    return d


def tiers(d):
    gate = d.dd >= GATE_PCT
    v = (d.vix10 > VOL_TRIG) & (d.vvix10 < d.vix10 / 2)
    vn = (d.vxn10 > VOL_TRIG) & (d.vvix10 < d.vxn10 / 2)
    watch = gate & (v | vn)
    credit = d.cred10 < 0
    d80 = d.dspx_p1y >= 80
    d60 = d.dspx_p1y >= 60
    return pd.DataFrame({
        "gate": gate, "v": v, "vn": vn, "watch": watch, "credit": credit, "d80": d80, "d60": d60,
        "size": d.dxy_p1y >= 80,
        "working": watch & (credit | d80),
        "high": watch & credit & d60,
        "lowrisk": gate & (d.nv10 < -5),
    }, index=d.index)


def dedupe(mask):
    keep, last = [], -10**9
    for i in np.where(mask.fillna(False).values)[0]:
        if i - last >= DEDUPE:
            keep.append(i)
            last = i
    return mask.index[keep]


def episodes(d, th=5.0):
    spy, dd, idx, n = d.spy.values, d.dd.values, d.index, len(d)
    out, i = [], 0
    while i < n:
        if dd[i] <= -th:
            p = i
            while p > 0 and dd[p] < 0:
                p -= 1
            e = i
            while e < n and dd[e] < 0:
                e += 1
            t = p + int(np.argmin(spy[p:e]))
            path = (spy[p:t + 1] / spy[p] - 1) * 100
            c5 = p + int(np.where(path <= -th)[0][0])
            out.append({"peak": idx[p], "trough": idx[t], "c5": idx[c5], "c5_i": c5,
                        "depth": (spy[t] / spy[p] - 1) * 100, "open": e >= n})
            i = e + 1
        else:
            i += 1
    return out


def rate(series, th):
    return None if len(series) == 0 else round(float((series <= th).mean() * 100), 1)


def record(d, t, eps):
    rows = []
    idx = d.index
    for key, start in RECORD_STARTS.items():
        sub = d.loc[start:]
        sub = sub[sub.f40.notna()]
        ts = t.loc[sub.index]
        years = (sub.index[-1] - sub.index[0]).days / 365.25
        E = [e for e in eps if e["peak"] >= pd.Timestamp(start) and e["c5"] <= sub.index[-1]]
        for tier, mask in (("base", ts.gate), ("watch", ts.watch), ("working", ts.working), ("high", ts.high),
                           ("working_size", ts.working & ts["size"]), ("working_nosize", ts.working & ~ts["size"])):
            prints = dedupe(mask)
            f = sub.loc[prints, "f40"]
            c5 = c8 = t8 = 0
            for e in E:
                lo = idx[max(0, e["c5_i"] - FWD)]
                has = any(lo <= p <= e["c5"] for p in prints)
                c5 += has
                if e["depth"] <= -8:
                    t8 += 1
                    c8 += has
            rows.append({"period": key, "tier": tier, "prints": len(prints), "per_year": round(len(prints) / years, 1),
                         "hit5": rate(f, -5), "hit8": rate(f, -8), "caught5": c5, "n5": len(E), "caught8": c8, "n8": t8,
                         "sessions": int(mask.sum())})
    return rows


# ---------- barometer ----------
# Equal-weight average of the inputs listed in barometer.json, each as its percentile within its own
# trailing year, signed so that higher = more drawdown risk. The inputs are chosen by
# select_barometer.py: a walk-forward search over every 3-6 input combination of the candidates below,
# re-run monthly. It only switches when a different multifactor set beats the live one out of sample.
BARO_START = "2015-06-01"      # DSPX history
BARO_BANDS = [(1, 2, "Low"), (3, 4, "Below average"), (5, 6, "Average"), (7, 8, "Elevated"), (9, 10, "High")]
BARO_CONFIG = os.path.join(HERE, "barometer.json")
COR1M_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv"
CANDIDATES = {
    "dspx": "DSPX percentile",
    "dxy": "DXY percentile",
    "lag": "VVIX lag behind the VIX/VXN move",
    "credit": "Credit weakness (hedged HYG)",
    "vxnvix": "VXN/VIX 10-session",
    "volmove": "VIX/VXN 10-session move",
    "xlu": "XLU vs SPY 10-session",
    "cor1m": "Low implied correlation (COR1M)",
    "vrp": "VIX cheap vs SPY realized vol",
    "move": "MOVE 10-session",
}
EXTRA_NEEDS = {"xlu": "XLU", "move": "^MOVE", "cor1m": "COR1M"}


def load_baro_config():
    with open(BARO_CONFIG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    bad = [k for k in cfg["inputs"] if k not in CANDIDATES]
    if bad or len(cfg["inputs"]) < 3:
        raise ValueError(f"barometer.json needs 3+ known inputs, got {cfg['inputs']}")
    return cfg


def fetch_extras(names, start="2007-01-01"):
    out = {}
    tick = [EXTRA_NEEDS[n] for n in names if n in EXTRA_NEEDS and EXTRA_NEEDS[n] != "COR1M"]
    if tick:
        px = yf.download(tick, start=start, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(tick[0])
        for tk in tick:
            out[tk] = px[tk]
    if "cor1m" in names:
        resp = requests.get(COR1M_URL, timeout=30)
        resp.raise_for_status()
        x = pd.read_csv(io.StringIO(resp.text))
        x["DATE"] = pd.to_datetime(x["DATE"], format="%m/%d/%Y")
        out["COR1M"] = pd.to_numeric(x.set_index("DATE")["CLOSE"], errors="coerce").dropna()
    return out


def candidate_inputs(d, names, extras):
    """Percentile (trailing year) of each named candidate, signed so higher = more drawdown risk."""
    cols = {}
    vol10 = np.maximum(d.vix10, d.vxn10)
    for n in names:
        if n == "dspx":
            cols[n] = d.dspx_p1y
        elif n == "dxy":
            cols[n] = d.dxy_p1y
        elif n == "lag":
            cols[n] = pct_rank_1y(vol10 - d.vvix10)
        elif n == "credit":
            cols[n] = 100 - pct_rank_1y(d.cred10)
        elif n == "vxnvix":
            cols[n] = pct_rank_1y(d.nv10)
        elif n == "volmove":
            cols[n] = pct_rank_1y(vol10)
        elif n == "xlu":
            cols[n] = pct_rank_1y((extras["XLU"].reindex(d.index).ffill(limit=3) / d.spy).pct_change(ROC_WIN) * 100)
        elif n == "cor1m":
            cols[n] = 100 - pct_rank_1y(extras["COR1M"].reindex(d.index).ffill(limit=3))
        elif n == "vrp":
            rv10 = np.log(d.spy).diff().rolling(10).std() * np.sqrt(252) * 100
            cols[n] = 100 - pct_rank_1y(d.vix / rv10)
        elif n == "move":
            cols[n] = pct_rank_1y(extras["^MOVE"].reindex(d.index).ffill(limit=3).pct_change(ROC_WIN) * 100)
    return pd.DataFrame(cols, index=d.index)


def baro_readings(score, ok):
    """Walk-forward deciles: each year is read against the near-high scores of earlier years only."""
    near = score[ok].loc[BARO_START:]
    reading = pd.Series(np.nan, index=score.index)
    for yr in range(2018, score.index[-1].year + 1):
        past = near[near.index < pd.Timestamp(f"{yr}-01-01")]
        cur_yr = near[(near.index >= pd.Timestamp(f"{yr}-01-01")) & (near.index < pd.Timestamp(f"{yr + 1}-01-01"))]
        if len(cur_yr) == 0 or len(past) < 250:
            continue
        cuts = np.quantile(past.values, np.linspace(0.1, 0.9, 9))
        reading.loc[cur_yr.index] = 1 + np.searchsorted(cuts, cur_yr.values, side="right")
    return reading, near


def barometer(d, t, cfg, extras):
    names = cfg["inputs"]
    P = candidate_inputs(d, names, extras)
    score = P.mean(axis=1)
    ok = t.gate & P.notna().all(axis=1)
    reading, near = baro_readings(score, ok)
    live = None
    if bool(ok.iloc[-1]):
        cuts = np.quantile(near.iloc[:-1].values, np.linspace(0.1, 0.9, 9))
        live = int(1 + np.searchsorted(cuts, score.iloc[-1], side="right"))
        reading.iloc[-1] = live
    f40 = d.f40
    oos = reading.dropna().index.intersection(f40.dropna().index)
    cal = []
    for lo, hi, lab in BARO_BANDS:
        idx = reading.loc[oos][(reading.loc[oos] >= lo) & (reading.loc[oos] <= hi)].index
        f = f40.loc[idx]
        cal.append({"lo": lo, "hi": hi, "label": lab, "sessions": len(idx), "p5": rate(f, -5), "p8": rate(f, -8),
                    "med40": None if len(f) == 0 else round(float(f.median()), 2)})
    cur = P.iloc[-1]
    drivers = [{"name": CANDIDATES[k], "pct": None if np.isnan(cur[k]) else round(float(cur[k]), 0),
                "push": None if np.isnan(cur[k]) else round(float((cur[k] - 50) / len(names)), 1)} for k in names]
    band = next((x for x in cal if live is not None and x["lo"] <= live <= x["hi"]), None)
    return {"reading": live, "score": None if np.isnan(score.iloc[-1]) else round(float(score.iloc[-1]), 1), "band": band,
            "cal": cal, "drivers": drivers, "base5": rate(f40.loc[oos], -5), "base8": rate(f40.loc[oos], -8),
            "oos_from": str(reading.dropna().index[0].date()) if reading.notna().any() else None, "series": reading,
            "inputs": [CANDIDATES[k] for k in names], "selected": cfg.get("selected"), "evidence": cfg.get("evidence")}


def next_sessions(after, n):
    out, day = [], after
    while len(out) < n:
        day = day + dt.timedelta(days=1)
        if day.weekday() < 5 and day.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS:
            out.append(day)
    return out


def dspx_level_for_pct(window, pct):
    """Smallest next print whose percentile in (window + itself) reaches pct."""
    w = np.sort(np.asarray(window, dtype=float))
    need = math.ceil(pct / 100 * (len(w) + 1)) - 1   # count of prior values <= x
    if need <= 0:
        return float(w[0])
    return float(w[min(need, len(w)) - 1])


def build(out_path, start):
    now_et = dt.datetime.now(ET)
    raw = fetch_yahoo(start)
    dspx = fetch_dspx()
    d = build_frame(raw, dspx)
    t = tiers(d)
    eps = episodes(d)
    cfg = load_baro_config()
    baro = barometer(d, t, cfg, fetch_extras(cfg["inputs"]))
    idx = d.index
    last_day = idx[-1].date()
    live = last_day == now_et.date() and dt.time(9, 30) <= now_et.time() < dt.time(16, 15)
    b = len(d) - 2 if live else len(d) - 1      # last completed close
    cur = len(d) - 1

    # ---------- current component readings (live bar if session is open) ----------
    r = d.iloc[cur]
    hi_day = d.spy.iloc[max(0, cur - 251):cur + 1].idxmax()
    ratio_hi = float(d.loc[hi_day, "ratio"])
    comp = (float(r.ratio) / ratio_hi - 1) * 100
    tnow = t.iloc[cur]

    # ---------- trigger ladder ----------
    sessions = ([last_day] if live else []) + next_sessions(last_day, 5 if live else 6)
    ladder = []
    for j, day in enumerate(sessions[:6], start=1):
        a = b + j - ROC_WIN
        row = d.iloc[a]
        item = {"date": day.strftime("%Y-%m-%d"), "anchor": idx[a].strftime("%Y-%m-%d"),
                "vix_anchor": round(float(row.vix), 2), "vix_trig": round(float(row.vix) * 1.1, 2),
                "vxn_anchor": round(float(row.vxn), 2), "vxn_trig": round(float(row.vxn) * 1.1, 2),
                "vvix_anchor": round(float(row.vvix), 2), "vvix_ceiling": round(float(row.vvix) * 1.05, 2),
                "today": live and j == 1}
        if j == 1:
            item["credit_cutoff"] = round((float(d.cred.iloc[a]) / float(d.cred.iloc[b]) - 1) * 100, 3)
            window = d.dspx.dropna().loc[:d.dspx_date.dropna().iloc[-1]].iloc[-251:]
            item["dspx60"] = round(dspx_level_for_pct(window, 60), 2)
            item["dspx80"] = round(dspx_level_for_pct(window, 80), 2)
            item["dxy80"] = round(dspx_level_for_pct(d.dxy.iloc[b - 250:b + 1].dropna(), 80), 2)
        ladder.append(item)

    # ---------- record + history ----------
    rec = record(d, t, eps)
    hist = []
    post = d.loc[CHART_START:]
    tp = t.loc[post.index]
    wp, hp = set(dedupe(tp.working)), set(dedupe(tp.high))
    for day in sorted(wp | hp):
        rr = d.loc[day]
        tier = "Working + High" if (day in wp and day in hp) else ("High" if day in hp else "Working")
        hist.append({"date": day.strftime("%Y-%m-%d"), "tier": tier, "dd": round(float(rr.dd), 2),
                     "vix10": round(float(rr.vix10), 1), "vxn10": round(float(rr.vxn10), 1), "vvix10": round(float(rr.vvix10), 1),
                     "cred10": round(float(rr.cred10), 2), "dspx_p1y": None if np.isnan(rr.dspx_p1y) else round(float(rr.dspx_p1y), 0),
                     "dxy_p1y": None if np.isnan(rr.dxy_p1y) else round(float(rr.dxy_p1y), 0),
                     "f40": None if np.isnan(rr.f40) else round(float(rr.f40), 2)})

    # ---------- chart series ----------
    cs = d.loc[CHART_START:]
    ct = t.loc[cs.index]
    marks_w = {p.strftime("%Y-%m-%d") for p in dedupe(ct.working)}
    marks_h = {p.strftime("%Y-%m-%d") for p in dedupe(ct.high)}

    def col(s, nd=2):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), nd) for v in s]
    series = {
        "dates": [x.strftime("%Y-%m-%d") for x in cs.index],
        "spy": col(cs.spy), "dd": col(cs.dd), "vix10": col(cs.vix10, 1), "vxn10": col(cs.vxn10, 1), "vvix10": col(cs.vvix10, 1),
        "cred10": col(cs.cred10, 2), "dspx_p1y": col(cs.dspx_p1y, 0), "ratio": col(cs.ratio, 2),
        "dxy_p1y": col(cs.dxy_p1y, 0), "baro": col(baro["series"].loc[cs.index], 0),
        "watch": [int(x) for x in ct.watch.fillna(False)], "gate": [int(x) for x in ct.gate.fillna(False)],
        "wk": [int(x) for x in ct.working.fillna(False)], "hi": [int(x) for x in ct.high.fillna(False)],
        "mw": [int(x in marks_w) for x in (i.strftime("%Y-%m-%d") for i in cs.index)],
        "mh": [int(x in marks_h) for x in (i.strftime("%Y-%m-%d") for i in cs.index)],
    }
    dd_eps = [{"peak": e["peak"].strftime("%Y-%m-%d"), "trough": e["trough"].strftime("%Y-%m-%d"), "depth": round(e["depth"], 1)}
              for e in eps if e["peak"] >= pd.Timestamp(CHART_START)]

    def fnum(v, nd=2):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)
    anchor_today = d.iloc[cur - ROC_WIN]
    payload = {
        "built_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "live": bool(live),
        "last_bar": last_day.strftime("%Y-%m-%d"),
        "dspx_date": d.dspx_date.dropna().iloc[-1].strftime("%Y-%m-%d"),
        "anchor_date": idx[cur - ROC_WIN].strftime("%Y-%m-%d"),
        "now": {
            "spy": fnum(r.spy), "hi52": fnum(r.hi52), "dd": fnum(r.dd), "hi_day": hi_day.strftime("%Y-%m-%d"),
            "vix": fnum(r.vix), "vix10": fnum(r.vix10, 1), "vix_anchor": fnum(anchor_today.vix),
            "vxn": fnum(r.vxn), "vxn10": fnum(r.vxn10, 1), "vxn_anchor": fnum(anchor_today.vxn),
            "vvix": fnum(r.vvix), "vvix10": fnum(r.vvix10, 1), "vvix_anchor": fnum(anchor_today.vvix),
            "cred10": fnum(r.cred10, 2), "dspx": fnum(r.dspx), "dspx_p1y": fnum(r.dspx_p1y, 1),
            "dxy": fnum(r.dxy), "dxy10": fnum(r.dxy10, 2), "dxy_anchor": fnum(anchor_today.dxy), "dxy_p1y": fnum(r.dxy_p1y, 1),
            "dxy_hi": fnum(d.dxy.iloc[max(0, cur - 251):cur + 1].max()), "dxy_lo": fnum(d.dxy.iloc[max(0, cur - 251):cur + 1].min()),
            "nv": fnum(r.nv, 3), "nv10": fnum(r.nv10, 1), "ratio": fnum(r.ratio, 3), "ratio_hi": round(ratio_hi, 3),
            "comp": round(comp, 1), "ratio_25": round(ratio_hi * 0.75, 2), "ratio_40": round(ratio_hi * 0.60, 2),
        },
        "flags": {k: bool(tnow[k]) for k in t.columns},
        "roc_short": {f"{w}d": {k: fnum((d[k].iloc[cur] / d[k].iloc[cur - w] - 1) * 100, 1) for k in ("spy", "vix", "vxn", "vvix", "nv")}
                      for w in (1, 2, 3)},
        "baro": {k: v for k, v in baro.items() if k != "series"},
        "ladder": ladder, "record": rec, "history": hist, "series": series, "dd_eps": dd_eps,
    }
    with open(os.path.join(HERE, "template.html.in"), encoding="utf-8") as fh:
        html = fh.read()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__DATA__", blob)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="vol_signal_dashboard.html")
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--json", help="also write the computed payload to this path")
    args = ap.parse_args()
    payload = build(args.out, args.start)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
    f = payload["flags"]
    print(f"wrote {args.out}  bar={payload['last_bar']} live={payload['live']}  "
          f"gate={f['gate']} watch={f['watch']} working={f['working']} high={f['high']} lowrisk={f['lowrisk']}")


if __name__ == "__main__":
    main()
