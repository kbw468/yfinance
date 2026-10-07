"""Intraday projection of tonight's list.

Takes every name's bar so far today (price now as the close, today's high and low so far, volume so far scaled to a full-day
estimate by time of day), scores it through the same frozen model as the nightly, and compares with the last published list.
Per name: P at the last close, projected P now, change in signatures, the day's move, and a status. Outputs go to
results/intraday/ and do not touch the nightly history, except a projection log that the nightly uses to measure how close
the projection came to the actual close.

Sources: Yahoo partial bars for the whole universe (default), or --csv PATH with columns Ticker, Price (or Last/Close),
Volume and optionally Open, High, Low. --whatif TICKER=-2.8 moves one or more names to a hypothetical close.
--vol-pace 0.77 overrides the time-of-day volume share."""
import argparse
import datetime as dt
import html
import subprocess
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
from .config import CACHE_DIR, RESULTS_DIR
from .data_io import load_panel, all_tickers, load_universe
from .clean import clean_panel
from .recent_rows import build_rows
from .score_frozen import main as score_frozen

OUT = RESULTS_DIR / "intraday"
NY = ZoneInfo("America/New_York")
# cumulative share of a typical full session's volume by New York time (U-shaped profile, closing auction heavy)
VOL_CURVE = [(9.5, 0.0), (10.0, 0.15), (10.5, 0.24), (11.0, 0.31), (11.5, 0.37), (12.0, 0.42), (12.5, 0.46), (13.0, 0.50), (13.5, 0.54),
             (14.0, 0.59), (14.5, 0.64), (15.0, 0.70), (15.25, 0.73), (15.5, 0.77), (15.75, 0.83), (16.0, 1.0)]
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def vol_share(now_ny: dt.datetime) -> float:
    h = now_ny.hour + now_ny.minute / 60
    return float(np.interp(h, [x for x, _ in VOL_CURVE], [y for _, y in VOL_CURVE]))


def fetch_live(tickers: list) -> tuple[pd.DataFrame, pd.Timestamp]:
    d = yf.download(tickers, period="5d", interval="1d", auto_adjust=False, progress=False, group_by="column", threads=True)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    last = d.index[-1]
    live = pd.DataFrame({f: d[f].iloc[-1] for f in ["Open", "High", "Low", "Close", "Volume"]})
    return live, last


def read_csv_bars(path: str, prev_close: pd.Series) -> pd.DataFrame:
    c = pd.read_csv(path)
    cols = {k.lower(): k for k in c.columns}
    tk = cols.get("ticker") or cols.get("symbol")
    px = cols.get("price") or cols.get("last") or cols.get("close")
    vol = cols.get("volume")
    assert tk and px and vol, f"csv needs Ticker, Price/Last/Close and Volume columns; has {list(c.columns)}"
    c[tk] = c[tk].astype(str).str.replace(".", "-", regex=False)
    c = c.set_index(tk)
    live = pd.DataFrame(index=c.index)
    live["Close"] = pd.to_numeric(c[px].astype(str).str.replace(",", ""), errors="coerce")
    live["Volume"] = pd.to_numeric(c[vol].astype(str).str.replace(",", ""), errors="coerce")
    pc = prev_close.reindex(live.index)
    live["Open"] = pd.to_numeric(c[cols["open"]], errors="coerce") if "open" in cols else pc
    live["High"] = pd.to_numeric(c[cols["high"]], errors="coerce") if "high" in cols else np.maximum(pc, live["Close"])
    live["Low"] = pd.to_numeric(c[cols["low"]], errors="coerce") if "low" in cols else np.minimum(pc, live["Close"])
    return live


def provisional_panel(csv: str | None, whatif: dict, vol_pace: float | None) -> tuple[dict, dict]:
    panel = load_panel()
    now_ny = dt.datetime.now(NY)
    tickers = [t for t in all_tickers() if t in panel["Close"].columns]
    if csv:
        live = read_csv_bars(csv, panel["RawClose"].iloc[-1]); live_date = pd.Timestamp(now_ny.date())
        source = f"csv {csv}"
    else:
        live, live_date = fetch_live(tickers); source = "yahoo partial bars"
    last_final = panel["Close"].index[-1]
    market_open = now_ny.weekday() < 5 and dt.time(9, 30) <= now_ny.time() < dt.time(16, 0)
    is_partial = live_date.date() == now_ny.date() and market_open
    share = vol_pace if vol_pace else (vol_share(now_ny) if is_partial else 1.0)
    live = live.reindex(tickers)
    missing = live.index[live.Close.isna()].tolist()
    live = live.dropna(subset=["Close"])
    for f in ["Open", "High", "Low", "Close"]:
        panel[f].loc[live_date, live.index] = live[f].values
    panel["RawClose"].loc[live_date, live.index] = live["Close"].values
    panel["Volume"].loc[live_date, live.index] = (live["Volume"] / share).values
    for t, pct in whatif.items():
        if t in panel["Close"].columns:
            c = float(panel["Close"].loc[live_date, t]) * (1 + pct / 100)
            panel["Close"].loc[live_date, t] = c; panel["RawClose"].loc[live_date, t] = c
            panel["High"].loc[live_date, t] = max(float(panel["High"].loc[live_date, t]), c)
            panel["Low"].loc[live_date, t] = min(float(panel["Low"].loc[live_date, t]), c)
    panel = {k: v.sort_index() for k, v in panel.items()}
    cleaned, rep = clean_panel(panel, keep_partial=True, verbose=False)
    meta = {"now_ny": now_ny.strftime("%Y-%m-%d %H:%M ET"), "bar_date": str(live_date.date()), "last_final": str(last_final.date()), "partial": bool(is_partial),
            "vol_share": round(share, 3), "source": source, "missing": missing, "whatif": whatif}
    return cleaned, meta


def status_of(r) -> str:
    if pd.isna(r.tier_close):
        return "new"
    if r.tier_now < r.tier_close:
        return "UP A TIER"
    if r.tier_now > r.tier_close:
        return "down a tier"
    if r.dP >= 0.01:
        return "strengthening"
    if r.dP <= -0.01:
        return "weakening"
    return "holding"


def main(csv=None, whatif=None, vol_pace=None, send_label=None):
    whatif = whatif or {}
    panel, meta = provisional_panel(csv, whatif, vol_pace)
    rows = build_rows(panel, CACHE_DIR / "intraday_rows.parquet", keep=2)
    OUT.mkdir(exist_ok=True)
    ranked = score_frozen(str(rows), OUT)
    base = pd.read_csv(RESULTS_DIR / "THE_LIST.csv").set_index("ticker")
    now = ranked.set_index("ticker")
    prev_close = panel["RawClose"].iloc[-2] if meta["partial"] or meta["bar_date"] != meta["last_final"] else panel["RawClose"].iloc[-2]
    close_now = panel["RawClose"].iloc[-1]
    uni = load_universe()
    rd = pd.DataFrame({"Sector": now.Sector, "tier_close": base.tier.reindex(now.index), "P_close": base.P_topq_42d.reindex(now.index), "sigs_close": base.n_signatures.reindex(now.index),
                       "tier_now": now.tier, "P_now": now.P_topq_42d, "sigs_now": now.n_signatures, "rank_now": now["rank"], "rank_close": base["rank"].reindex(now.index),
                       "day_ret": (close_now.reindex(now.index) / prev_close.reindex(now.index) - 1), "basis_now": now.basis})
    rd["dP"] = rd.P_now - rd.P_close; rd["dsig"] = rd.sigs_now - rd.sigs_close
    rd["status"] = rd.apply(status_of, axis=1)
    rd = rd.sort_values(["tier_now", "P_now"], ascending=[True, False])
    stamp = pd.Timestamp(meta["bar_date"]).strftime("%Y-%m-%d") + "_" + dt.datetime.now(NY).strftime("%H%M")
    rd.reset_index().to_csv(OUT / f"INTRADAY_{stamp}.csv", index=False)
    rd.reset_index().to_csv(OUT / "INTRADAY_latest.csv", index=False)
    # projection log (the nightly measures the projection against the actual close)
    if meta["partial"] and not whatif and not csv:
        lg = OUT / "projection_log.csv"
        rec = rd.reset_index()[["ticker", "tier_now", "P_now", "sigs_now"]].rename(columns={"tier_now": "tier_proj", "P_now": "P_proj", "sigs_now": "sigs_proj"})
        rec.insert(0, "time_ny", dt.datetime.now(NY).strftime("%H:%M")); rec.insert(0, "asof", meta["bar_date"])
        old = pd.read_csv(lg) if lg.exists() else pd.DataFrame()
        pd.concat([old[~((old.asof == meta["bar_date"]) & (old.time_ny == rec.time_ny.iloc[0]))] if len(old) else old, rec]).to_csv(lg, index=False)
    # text
    L = [f"INTRADAY READ, {meta['now_ny']}. Bar: {meta['bar_date']} ({'partial, volume so far scaled by 1/' + str(meta['vol_share']) if meta['partial'] else 'final session bar'}); source {meta['source']}."
         + (f"  WHAT-IF: {', '.join(f'{k} {v:+.1f}%' for k, v in whatif.items())}" if whatif else ""),
         f"Projected tonight's list vs the published {meta['last_final']} list, same frozen model. Status: UP A TIER / strengthening (>= +1 pt) / holding / weakening (<= -1 pt) / down a tier."]
    if meta["missing"]:
        L.append(f"no live bar for {len(meta['missing'])} names (previous close carried): {' '.join(meta['missing'][:12])}")
    L.append("")
    def line(t, r):
        return (f"  {t:<6} {str(r.Sector)[:14]:<14} day {r.day_ret*100:+5.1f}%  tier {int(r.tier_close) if pd.notna(r.tier_close) else '-'}->{int(r.tier_now)}  "
                f"P {r.P_close*100 if pd.notna(r.P_close) else float('nan'):4.1f}% -> {r.P_now*100:4.1f}% ({r.dP*100:+4.1f})  sigs {int(r.sigs_close) if pd.notna(r.sigs_close) else 0}->{int(r.sigs_now)}  {r.status}")
    t12 = rd[(rd.tier_now <= 2) | (rd.tier_close <= 2)]
    L.append(f"TIERS 1-2 (projected or at last close), {len(t12)} names:")
    L += [line(t, r) for t, r in t12.iterrows()]
    f1 = rd[(rd.day_ret < 0) & (rd.tier_now <= 2) & rd.status.isin(["holding", "strengthening", "UP A TIER"])].sort_values("day_ret")
    L += ["", f"DOWN ON THE DAY, PROJECTED TIER 1-2 AND HOLDING OR STRONGER ({len(f1)}):"] + [line(t, r) for t, r in f1.iterrows()]
    f2 = rd[(rd.day_ret > 0) & (rd.tier_close <= 2) & rd.status.isin(["weakening", "down a tier"])].sort_values("day_ret", ascending=False)
    L += ["", f"UP ON THE DAY BUT WEAKENING FROM TIER 1-2 ({len(f2)}):"] + [line(t, r) for t, r in f2.iterrows()]
    f3 = rd[(rd.tier_now <= 2) & (rd.tier_close > 2)].sort_values("P_now", ascending=False)
    L += ["", f"PROJECTED TO ENTER TIER 1-2 ({len(f3)}):"] + [line(t, r) for t, r in f3.iterrows()]
    big = rd.reindex(rd.dP.abs().sort_values(ascending=False).index).head(15)
    L += ["", "LARGEST PROJECTED PROBABILITY CHANGES, ANY TIER:"] + [line(t, r) for t, r in big.iterrows()]
    txt = "\n".join(L)
    (OUT / f"INTRADAY_{stamp}.txt").write_text(txt); (OUT / "INTRADAY.txt").write_text(txt)
    h = OUT / "INTRADAY.html"
    style = "@page{size:A4 portrait;margin:11mm} body{font-family:Menlo,Consolas,monospace;font-size:8.3px;color:#1a1f2b;white-space:pre-wrap;line-height:1.38}"
    h.write_text('<!doctype html><html><head><meta charset="utf-8"><style>' + style + '</style></head><body>' + html.escape(txt) + '</body></html>')
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT / 'INTRADAY.pdf'}", f"file://{h}"], capture_output=True)
    print(txt[:6000])
    return rd, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None); ap.add_argument("--whatif", nargs="*", default=[]); ap.add_argument("--vol-pace", type=float, default=None)
    a = ap.parse_args()
    wi = {kv.split("=")[0].upper(): float(kv.split("=")[1].rstrip("%")) for kv in a.whatif}
    main(a.csv, wi, a.vol_pace)
