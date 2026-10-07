"""Track the list session over session.

  results/history/snapshots/THE_LIST_<asof>.csv   the list as published that session (plus model id)
  results/history/list_history.csv                long table: one row per (asof, ticker)
  results/history/list_metrics.csv                one row per session: tier counts, breadth, vol readings, model id
  results/history/scorecard.csv                   realised hit rates of past snapshots once 42 / 63 sessions have elapsed
  results/movers_today.csv                        per-name 1-session and 5-session change in rank and probability (page / PDF columns)
  results/MOVERS.txt                              entries and exits of Tiers 1-2, biggest movers, list-level change, scorecard
Run after score_frozen (or final_list): python -m pvv_score.history"""
import sys
import numpy as np
import pandas as pd
from .config import CACHE_DIR, RESULTS_DIR
from .volindex import readings_at

HIST = RESULTS_DIR / "history"
SNAP = HIST / "snapshots"
KEEP_COLS = ["rank", "tier", "ticker", "Sector", "beta_bucket", "P_topq_42d", "P_topq_63d", "basis", "n_signatures", "avg_score"]


def _sessions() -> pd.DatetimeIndex:
    return pd.read_parquet(CACHE_DIR / "Close.parquet", columns=["SPY"]).dropna().index


def _snapshots() -> list:
    return sorted(pd.Timestamp(p.stem.split("_")[-1]) for p in SNAP.glob("THE_LIST_*.csv"))


def load_snapshot(asof) -> pd.DataFrame:
    return pd.read_csv(SNAP / f"THE_LIST_{pd.Timestamp(asof).date()}.csv")


def record(asof=None) -> pd.Timestamp:
    HIST.mkdir(exist_ok=True); SNAP.mkdir(exist_ok=True)
    L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv"); U = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")
    asof = pd.Timestamp(asof or U["asof"].iloc[0])
    model = str(U["model"].iloc[0]) if "model" in U.columns else ""
    snap = L[KEEP_COLS].copy(); snap.insert(0, "asof", asof.date()); snap["model"] = model
    snap.to_csv(SNAP / f"THE_LIST_{asof.date()}.csv", index=False)
    hp = HIST / "list_history.csv"
    hist = pd.read_csv(hp, parse_dates=["asof"]) if hp.exists() else pd.DataFrame()
    hist = pd.concat([hist[hist["asof"] != asof] if len(hist) else hist, snap.assign(asof=asof)]).sort_values(["asof", "rank"])
    hist.to_csv(hp, index=False)
    v = readings_at(asof)
    C = pd.read_parquet(CACHE_DIR / "Close.parquet", columns=["SPY"])
    row = {"asof": asof.date(), "n_ranked": len(L), **{f"tier{t}": int((L.tier == t).sum()) for t in range(1, 7)}, "n_ge40": int((L.P_topq_42d >= .40).sum()),
           "mean_p_top50": round(L.P_topq_42d.head(50).mean(), 4), "mean_p_tier1": round(L[L.tier == 1].P_topq_42d.mean(), 4) if (L.tier == 1).any() else np.nan,
           "n_firing": int((L.n_signatures > 0).sum()), "n_50plus": int((L.n_signatures >= 50).sum()),
           "spy": round(float(C.loc[asof, "SPY"]), 2) if asof in C.index else np.nan,
           "vix": v["vix"], "vxn": v["vxn"], "move": v["move"], "iwm_rv20": v["iwm_rv20"], "move_band": v["move_band"], "model": model}
    mp = HIST / "list_metrics.csv"
    met = pd.read_csv(mp, parse_dates=["asof"]) if mp.exists() else pd.DataFrame()
    met = pd.concat([met[met["asof"] != asof] if len(met) else met, pd.DataFrame([row]).assign(asof=asof)]).sort_values("asof")
    met.to_csv(mp, index=False)
    return asof


def movers(asof) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    snaps = [d for d in _snapshots() if d < asof]
    cur = load_snapshot(asof).set_index("ticker")
    out = cur[["rank", "tier", "P_topq_42d", "n_signatures", "Sector", "beta_bucket"]].copy()
    for lag, suf in [(1, "1"), (5, "5")]:
        if len(snaps) >= lag:
            prev = load_snapshot(snaps[-lag]).set_index("ticker")
            out[f"prev_asof_{suf}"] = snaps[-lag].date()
            out[f"d_rank_{suf}"] = prev["rank"].reindex(out.index) - out["rank"]          # positive = moved up the list
            out[f"d_p42_{suf}"] = out["P_topq_42d"] - prev["P_topq_42d"].reindex(out.index)
            out[f"d_tier_{suf}"] = prev["tier"].reindex(out.index) - out["tier"]           # positive = better tier
            out[f"d_sig_{suf}"] = out["n_signatures"] - prev["n_signatures"].reindex(out.index)
            out[f"prev_tier_{suf}"] = prev["tier"].reindex(out.index)
            out[f"model_changed_{suf}"] = str(prev["model"].iloc[0]) != str(cur["model"].iloc[0])
        else:
            for c in [f"d_rank_{suf}", f"d_p42_{suf}", f"d_tier_{suf}", f"d_sig_{suf}", f"prev_tier_{suf}"]:
                out[c] = np.nan
            out[f"prev_asof_{suf}"] = None; out[f"model_changed_{suf}"] = False
    if len(snaps) >= 1:
        prev = load_snapshot(snaps[-1]).set_index("ticker")
        in12 = cur.index[cur.tier <= 2]; was12 = prev.index[prev.tier <= 2]
        out["status"] = ""
        out.loc[[t for t in in12 if t not in was12], "status"] = "entered T1-2"
        exits = [t for t in was12 if t not in in12]
        ex = pd.DataFrame({"rank": cur["rank"].reindex(exits), "tier": cur["tier"].reindex(exits), "P_topq_42d": cur["P_topq_42d"].reindex(exits),
                           "prev_rank": prev["rank"].reindex(exits), "prev_tier": prev["tier"].reindex(exits), "prev_P": prev["P_topq_42d"].reindex(exits)}, index=exits)
    else:
        out["status"] = ""; ex = pd.DataFrame()
    out = out.reset_index()
    out.to_csv(RESULTS_DIR / "movers_today.csv", index=False)
    return out, ex, snaps


def realised_top_quartile(tickers: list, start, h: int) -> pd.Series | None:
    """Did each name's path over the h sessions after `start` land in the universe top quartile on the smooth-climb score?
    Same construction as the research target (Sharpe, -max drawdown, straightness, up-day share; percentiles across the
    snapshot's names on that date, re-ranked so exactly 25% qualify). None until h sessions have completed."""
    from .targets import sharpe_sortino, forward_path_metrics
    C = pd.read_parquet(CACHE_DIR / "Close.parquet")
    idx = C.index
    i0 = idx.get_loc(pd.Timestamp(start))
    if i0 + h >= len(idx):
        return None
    tk = [t for t in tickers if t in C.columns]
    close = C.iloc[i0: i0 + h + 1][tk]
    r = close.pct_change()
    sh, _ = sharpe_sortino(r, h)
    pm = forward_path_metrics(close, h)
    parts = [sh.iloc[0], -pm["mdd"].iloc[0], pm["r2"].iloc[0], pm["up_frac"].iloc[0]]
    u = sum(p.rank(pct=True) for p in parts) / 4
    ur = u.rank(pct=True)
    return (ur >= 0.75).astype(float).where(u.notna())


def scorecard(asof) -> pd.DataFrame:
    """For every snapshot old enough, realised hit rate by printed tier and by P band, vs what was printed."""
    asof = pd.Timestamp(asof)
    rows = []
    for d in _snapshots():
        snap = load_snapshot(d)
        for h, pcol in [(42, "P_topq_42d"), (63, "P_topq_63d")]:
            y = realised_top_quartile(snap.ticker.tolist(), d, h)
            if y is None:
                continue
            s = snap.set_index("ticker").join(y.rename("y"))
            s = s[s.y.notna()]
            for grp, g in [("all", s)] + [(f"tier{t}", s[s.tier == t]) for t in range(1, 7)] + [("P>=45", s[s[pcol] >= .45]), ("P 40-45", s[(s[pcol] >= .40) & (s[pcol] < .45)]), ("P<25", s[s[pcol] < .25])]:
                if len(g) == 0:
                    continue
                rows.append({"snapshot": d.date(), "horizon": h, "group": grp, "n": len(g), "printed": round(g[pcol].mean(), 4), "realised": round(g.y.mean(), 4), "model": snap.model.iloc[0]})
    sc = pd.DataFrame(rows)
    if len(sc):
        sc.to_csv(HIST / "scorecard.csv", index=False)
    return sc


def write_text(asof, mv: pd.DataFrame, ex: pd.DataFrame, snaps: list, sc: pd.DataFrame):
    asof = pd.Timestamp(asof)
    met = pd.read_csv(HIST / "list_metrics.csv", parse_dates=["asof"]).set_index("asof")
    cur = met.loc[asof]
    L = [f"MOVERS, {asof.date()} close.  model {cur.model}" + ("  (first session on record: no prior snapshot)" if not snaps else f"  vs {snaps[-1].date()}" + (f" and {snaps[-5].date()}" if len(snaps) >= 5 else ""))]
    if snaps and bool(mv["model_changed_1"].iloc[0]):
        L.append("NOTE: the model was refit between these sessions; part of every move below is refit, not tape.")
    L.append("")
    L.append("list level" + ("" if not snaps else f" (now | {snaps[-1].date()}" + (f" | {snaps[-5].date()})" if len(snaps) >= 5 else ")")))
    def fmt(k):
        vals = [cur[k]] + [met.loc[d, k] for d in ([snaps[-1]] if snaps else []) + ([snaps[-5]] if len(snaps) >= 5 else [])]
        dec = 3 if k.startswith("mean_p") else 1 if k in ("vix", "vxn", "iwm_rv20", "move", "spy") else 0
        return " | ".join(f"{float(v):.{dec}f}" if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v) else str(v) for v in vals)
    for k, lab in [("tier1", "Tier 1"), ("tier2", "Tier 2"), ("tier3", "Tier 3"), ("n_ge40", "names >= 40%"), ("mean_p_top50", "mean P, top 50"), ("n_firing", "names firing"), ("n_50plus", "names with 50+ signatures"),
                   ("spy", "SPY"), ("vix", "VIX"), ("vxn", "VXN"), ("move", "MOVE"), ("iwm_rv20", "IWM 20d rv")]:
        L.append(f"  {lab:<26} {fmt(k)}")
    if snaps:
        ent = mv[mv.status == "entered T1-2"].sort_values("rank")
        L += ["", f"entered Tiers 1-2 ({len(ent)}):"] + [f"  {r.ticker:<6} rank {int(r['rank']):>3}  tier {int(r.tier)}  P42 {r.P_topq_42d:.1%}  (was tier {int(r.prev_tier_1) if pd.notna(r.prev_tier_1) else '-'}, {'+' if r.d_p42_1 >= 0 else ''}{r.d_p42_1*100:.1f} pts, {int(r.d_sig_1) if pd.notna(r.d_sig_1) else 0:+d} sigs)" if pd.notna(r.prev_tier_1) else f"  {r.ticker:<6} rank {int(r['rank']):>3}  tier {int(r.tier)}  P42 {r.P_topq_42d:.1%}  (not ranked previously)" for _, r in ent.iterrows()]
        L += ["", f"left Tiers 1-2 ({len(ex)}):"] + [f"  {t:<6} now rank {int(r['rank']) if pd.notna(r['rank']) else '-':>3}  tier {int(r.tier) if pd.notna(r.tier) else '-'}  P42 {r.P_topq_42d:.1%}  (was rank {int(r.prev_rank)}, tier {int(r.prev_tier)}, {r.prev_P:.1%})" if pd.notna(r["rank"]) else f"  {t:<6} not ranked today (was rank {int(r.prev_rank)}, tier {int(r.prev_tier)})" for t, r in ex.iterrows()]
        for suf, lab in [("1", "1 session"), ("5", "5 sessions")]:
            if mv[f"d_p42_{suf}"].notna().any():
                up = mv.sort_values(f"d_p42_{suf}", ascending=False).head(15); dn = mv.sort_values(f"d_p42_{suf}").head(15)
                L += ["", f"biggest probability gains, {lab}:"] + [f"  {r.ticker:<6} {r.d_p42_1 if suf == '1' else r.d_p42_5:+.1%}  rank {int(r['rank']):>3} ({int(r[f'd_rank_{suf}']):+d})  tier {int(r.tier)}  P42 {r.P_topq_42d:.1%}  sigs {int(r.n_signatures)} ({int(r[f'd_sig_{suf}']):+d})" for _, r in up.iterrows() if r[f"d_p42_{suf}"] > 0]
                L += ["", f"biggest probability drops, {lab}:"] + [f"  {r.ticker:<6} {r[f'd_p42_{suf}']:+.1%}  rank {int(r['rank']):>3} ({int(r[f'd_rank_{suf}']):+d})  tier {int(r.tier)}  P42 {r.P_topq_42d:.1%}  sigs {int(r.n_signatures)} ({int(r[f'd_sig_{suf}']):+d})" for _, r in dn.iterrows() if r[f"d_p42_{suf}"] < 0]
    L += ["", "realised scorecard (snapshots with 42 / 63 sessions elapsed):"]
    if len(sc):
        for (d, h), g in sc.groupby(["snapshot", "horizon"]):
            g = g.set_index("group")
            L.append(f"  {d} {h}d: all n={int(g.loc['all','n'])} printed {g.loc['all','printed']:.1%} realised {g.loc['all','realised']:.1%}" +
                     "".join(f" | {k} n={int(g.loc[k,'n'])} {g.loc[k,'printed']:.0%}->{g.loc[k,'realised']:.0%}" for k in ["tier1", "tier2", "P>=45", "P<25"] if k in g.index))
    else:
        first = _snapshots()[0] if _snapshots() else asof
        L.append(f"  none matured yet; the first snapshot ({first.date()}) reads out after 42 sessions.")
    (RESULTS_DIR / "MOVERS.txt").write_text("\n".join(L))
    print("\n".join(L[:60]))


def projection_accuracy(asof) -> pd.DataFrame | None:
    """How close the latest intraday projection for this session came to the actual close list."""
    lg = RESULTS_DIR / "intraday" / "projection_log.csv"
    if not lg.exists():
        return None
    log = pd.read_csv(lg); log = log[log.asof == str(pd.Timestamp(asof).date())]
    if log.empty:
        return None
    last_t = sorted(log.time_ny.unique())[-1]
    proj = log[log.time_ny == last_t].set_index("ticker")
    act = load_snapshot(asof).set_index("ticker")
    j = proj.join(act[["tier", "P_topq_42d", "n_signatures"]], how="inner")
    t12p, t12a = set(j.index[j.tier_proj <= 2]), set(j.index[j.tier <= 2])
    row = {"asof": pd.Timestamp(asof).date(), "time_ny": last_t, "names": len(j), "tier_match": round((j.tier_proj == j.tier).mean(), 3),
           "mean_abs_dP_pts": round((j.P_proj - j.P_topq_42d).abs().mean() * 100, 2), "mean_abs_dP_t12_pts": round((j.loc[list(t12p | t12a)].P_proj - j.loc[list(t12p | t12a)].P_topq_42d).abs().mean() * 100, 2) if (t12p | t12a) else np.nan,
           "t12_projected": len(t12p), "t12_actual": len(t12a), "t12_overlap": len(t12p & t12a)}
    ap = HIST / "projection_accuracy.csv"
    acc = pd.read_csv(ap) if ap.exists() else pd.DataFrame()
    acc = pd.concat([acc[acc.asof.astype(str) != str(row["asof"])] if len(acc) else acc, pd.DataFrame([row])])
    acc.to_csv(ap, index=False)
    return pd.DataFrame([row])


def main(asof=None):
    asof = record(asof)
    mv, ex, snaps = movers(asof)
    sc = scorecard(asof)
    write_text(asof, mv, ex, snaps, sc)
    pa = projection_accuracy(asof)
    if pa is not None:
        r = pa.iloc[0]
        line = (f"\nintraday projection at {r.time_ny} ET vs the close: tier matched for {r.tier_match:.0%} of names; mean |ΔP| {r.mean_abs_dP_pts:.2f} pts "
                f"(Tier 1-2 names {r.mean_abs_dP_t12_pts:.2f}); Tier 1-2 projected {int(r.t12_projected)}, actual {int(r.t12_actual)}, overlap {int(r.t12_overlap)}.")
        with open(RESULTS_DIR / "MOVERS.txt", "a") as f:
            f.write(line + "\n")
        print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
