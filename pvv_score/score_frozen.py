"""Score the universe for one session from the frozen model in results/model/. Nothing is fitted here.

Rows in: every name's row on the session (factor levels, state z-scores, eligibility, beta, sector), from the research
table (weekly refit / verification) or from the nightly recent-window build (--rows PATH).
Out: THE_LIST.csv / .txt, signatures_today.csv, universe_scores_smooth.csv, identical in form to the full chain's."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .data_io import load_universe
from .calib import BinnedIsotonic
from .composite import tercile_series, score_rows
from .signatures import QUINTILE_FACTORS, FIXED, build_conditions
from .signature_rule import plain, COMP_Q

MODEL = RESULTS_DIR / "model"
TIER_EDGES = [-1, .25, .30, .35, .40, .45, 2]


def load_model() -> dict:
    W = json.load(open(MODEL / "weights.json"))
    cc = {b: {c: BinnedIsotonic.from_dict(d) for c, d in cs.items()} for b, cs in json.load(open(MODEL / "calib_composite.json")).items()}
    cs = json.load(open(MODEL / "calib_signature.json"))
    sc = {c: {q: BinnedIsotonic.from_dict(d) for q, d in qs.items()} for c, qs in cs["curves"].items()}
    conf = pd.read_csv(MODEL / "signatures.csv")
    return {"weights": W, "comp_curves": cc, "sig_curves": sc, "comp_q_edges": cs["comp_q_edges"], "signatures": conf, "meta": json.load(open(MODEL / "model.json"))}


def needed_columns(W: dict) -> list:
    fac = sorted({f for layer in ("state", "level") for b in W[layer].values() for f in b})
    zc = [f for f in fac if f.startswith("z_")]
    return list(dict.fromkeys(["date", "ticker", "sector", "eligible", "beta_252"] + fac + QUINTILE_FACTORS + sorted({v[0] for v in FIXED.values()}) + zc))


def load_rows(source: str, W: dict, asof=None) -> pd.DataFrame:
    cols = needed_columns(W)
    if source == "research":
        path = CACHE_DIR / "research_long.parquet"
    else:
        path = source
    have = pq.read_schema(path).names
    miss = [c for c in cols if c not in have]
    assert not miss, f"rows at {path} lack columns {miss}"
    dates = pq.read_table(path, columns=["date"]).column("date").to_pandas()
    asof = pd.Timestamp(asof) if asof is not None else dates.max()
    assert asof in set(dates), f"{asof.date()} not in rows at {path}"
    rows = pq.read_table(path, columns=cols, filters=[("date", "==", asof)]).to_pandas().reset_index(drop=True)
    rows["eligible"] = rows["eligible"].astype(bool)
    return rows


def main(source: str = "research", out_dir=RESULTS_DIR, asof=None, vol_header=True):
    M = load_model(); W = M["weights"]
    rows = load_rows(source, W, asof)
    asof = pd.Timestamp(rows.date.iloc[0])
    uni = load_universe()
    el = rows[rows.eligible].copy().reset_index(drop=True)
    el["beta_bucket"] = tercile_series(el)
    # composite layers: within-bucket percentile ranks x frozen weights, then universe percentile of each, averaged
    contrib = {}
    for layer in ("state", "level"):
        raw = pd.Series(np.nan, index=el.index)
        for b, wd in W[layer].items():
            w = pd.Series(wd, dtype=float)
            m = (el.beta_bucket == b).values
            if m.any() and len(w):
                raw[m] = score_rows(el[m], w).values
                if layer == "state":
                    rk = el.loc[m].groupby("date")[list(w.index)].rank(pct=True) - 0.5
                    contrib[b] = rk * w.values
        el[f"comp_{layer}"] = raw
    both = el.comp_state.notna() & el.comp_level.notna()
    el["avg_score"] = ((el.comp_state.rank(pct=True) + el.comp_level.rank(pct=True)) / 2).round(8).where(both)   # 8 dp, as livecomp
    el["state_score"] = el.comp_state.rank(pct=True)
    tops = {}
    for b, c in contrib.items():
        for i, row in c.iterrows():
            top = row.sort_values(ascending=False).head(3)
            tops[i] = "; ".join(f"{f[2:]} z={el.loc[i, f]:+.1f}" for f in top.index)
    el["top_states"] = pd.Series(tops)
    # signatures
    conds = build_conditions(el)
    conf = M["signatures"]
    n = len(el); cnt = np.zeros(n, int); liftsum = np.zeros(n); bestp = np.zeros(n); best_idx = np.full(n, -1)
    top5 = [[] for _ in range(n)]
    for i, r in conf.iterrows():
        m = np.ones(n, bool)
        for p in r.signature.split(" & "):
            m &= conds[p]
        cnt += m; liftsum += m * (r.lift_conf - 1.0); bestp = np.maximum(bestp, m * r.p_conf)
        upd = m & (best_idx < 0); best_idx[upd] = i
        for j in np.flatnonzero(m):
            if len(top5[j]) < 5:
                top5[j].append(f"{r.signature}|{r.p_conf:.2f}|{int(r.n_conf)}")
    el["n_fire"] = cnt; el["liftsum"] = liftsum; el["bestp"] = bestp
    el["best_signature"] = [conf.signature.iloc[i] if i >= 0 else "" for i in best_idx]
    el["best_signature_plain"] = el.best_signature.map(lambda s: " AND ".join(plain(c) for c in s.split(" & ")) if s else "")
    el["best_p_conf"] = [conf.p_conf.iloc[i] if i >= 0 else np.nan for i in best_idx]
    el["best_n_conf"] = [conf.n_conf.iloc[i] if i >= 0 else np.nan for i in best_idx]
    el["top5_signatures"] = [" || ".join(t) for t in top5]
    # calibration lookups
    for c in ["iso_q42", "iso_h42", "iso_q63", "iso_h63"]:
        el[c] = [M["comp_curves"][b][c].predict([x])[0] if (isinstance(b, str) and pd.notna(x)) else np.nan for b, x in zip(el.beta_bucket, el.avg_score)]
    el["comp_q"] = pd.cut(el.avg_score, M["comp_q_edges"], labels=range(len(M["comp_q_edges"]) - 1), include_lowest=True)
    for c in ["ls_iso_q42", "ls_iso_q63", "bp_iso_q42", "bp_iso_q63"]:
        x = np.log1p(el.liftsum.values) if c.startswith("ls_") else el.bestp.values
        pred = np.array([M["sig_curves"][c][str(int(q)) if pd.notna(q) else "all"].predict([xi])[0] for xi, q in zip(x, el.comp_q)])
        el[c] = np.where(el.n_fire.values > 0, pred, np.nan)
    # assemble P exactly as final_list does
    p42, p63, basis = [], [], []
    for _, r in el.iterrows():
        cands = []
        if pd.notna(r.iso_q42):
            cands.append((float(r.iso_q42), float(r.iso_q63), f"composite {r.avg_score:.2f}"))
        if r.n_fire > 0 and pd.notna(r.ls_iso_q42):
            cands.append((float(r.ls_iso_q42), float(r.ls_iso_q63), f"signatures x{int(r.n_fire)}"))
        if r.n_fire > 0 and pd.notna(r.bp_iso_q42):
            cands.append((float(r.bp_iso_q42), float(r.bp_iso_q63), f"signatures x{int(r.n_fire)}"))
        if not cands:
            p42.append(np.nan); p63.append(np.nan); basis.append("no cell"); continue
        best = max(cands, key=lambda x: x[0])
        p42.append(best[0]); p63.append(best[1])
        others = sorted({x[2].split(" ")[0] for x in cands if x[2] != best[2]})
        basis.append(best[2] + (f" (+{others[0]})" if others else ""))
    el["P_topq_42d"] = np.round(p42, 4); el["P_topq_63d"] = np.round(p63, 4); el["basis"] = basis
    el["n_signatures"] = el.n_fire
    el["Company"] = el.ticker.map(uni["Company"]); el["Sector"] = el.ticker.map(uni["Sector"])
    ranked = el[el.P_topq_42d.notna()].sort_values(["P_topq_42d", "P_topq_63d", "avg_score", "n_signatures", "ticker"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked.insert(1, "tier", pd.cut(ranked.P_topq_42d, TIER_EDGES, labels=[6, 5, 4, 3, 2, 1]).astype(int))
    ranked["best_signature"] = ranked.best_signature_plain
    cols = ["rank", "tier", "ticker", "Company", "Sector", "beta_bucket", "beta_252", "P_topq_42d", "P_topq_63d", "basis", "n_signatures", "best_signature", "avg_score", "top_states"]
    out_dir = pd.io.common.stringify_path(out_dir); out_dir = RESULTS_DIR if out_dir == str(RESULTS_DIR) else __import__("pathlib").Path(out_dir)
    ranked[cols].to_csv(out_dir / "THE_LIST.csv", index=False)
    # signatures_today.csv (columns the page, pdfs and checks read)
    st = el.rename(columns={"avg_score": "comp_p"})[["ticker", "sector", "beta_bucket", "comp_p", "n_fire", "best_signature", "best_signature_plain", "best_p_conf", "best_n_conf",
                                                      "top5_signatures", "liftsum", "bestp", "ls_iso_q42", "ls_iso_q63", "bp_iso_q42", "bp_iso_q63"]].sort_values("n_fire", ascending=False)
    st.to_csv(out_dir / "signatures_today.csv", index=False)
    # universe_scores_smooth.csv (every name in the universe, with the reason when not scored)
    C = pd.read_parquet(CACHE_DIR / "Close.parquet")
    u = uni[["Company", "Sector", "SectorETF"]].copy(); u.index.name = "ticker"
    u["sessions_of_history"] = [int(C[t].notna().sum()) if t in C.columns else 0 for t in u.index]
    u["eligible_today"] = u.index.isin(el.ticker)
    t = el.set_index("ticker")
    for c in ["beta_bucket", "state_score", "beta_252", "avg_score", "top_states", "iso_q42", "iso_h42", "iso_q63", "iso_h63", "n_fire"]:
        u[c] = t[c].reindex(u.index)
    u["note"] = ["" if pd.notna(r.state_score) else (f"insufficient history ({r.sessions_of_history} sessions; 252 needed)" if r.sessions_of_history < 252
                 else "fails liquidity / price eligibility today or no completed session") for _, r in u.iterrows()]
    u.insert(0, "asof", asof.date())
    u["model"] = M["meta"]["sha256"]
    u.reset_index().to_csv(out_dir / "universe_scores_smooth.csv", index=False)
    # text list
    from .volindex import readings_at
    v = readings_at(asof); reg = v["move_band"]
    lines = [f"THE LIST, {asof.date()} close. VIX {v['vix']:.1f}  VXN {v['vxn']:.1f}  IWM 20d rv {v['iwm_rv20']:.1f}  MOVE {v['move']:.0f} ({reg}). Rate-sensitive sectors (utilities, REITs, staples, financials) measured on {reg} sessions; all other sectors on every session.",
             f" P42 / P63 = probability (realised out of sample, {RECENT_START[:4]} onward, COVID Feb-Jun 2020 excluded) that the next 42 / 63 sessions are a top-quartile",
             "smooth climb vs the whole universe (Sharpe + max drawdown + straightness + up-day share). Baseline 25%. Every name ranked.",
             f"model frozen {M['meta']['frozen_utc']} UTC, fitted through {M['meta']['fitted_through']}, {M['meta']['n_signatures']} confirmed signatures, id {M['meta']['sha256']}.", "",
             "Tier 1 >= 45%, Tier 2 40-45%, Tier 3 35-40%, Tier 4 30-35%, Tier 5 25-30%, Tier 6 below baseline.", "",
             f"{'#':>3} tier {'tkr':<6} {'sector':<22} {'beta':<5} {'P42':>5} {'P63':>5} {'basis':<22} {'sigs':>4}  strongest confirmed signature firing tonight"]
    for _, r in ranked.iterrows():
        lines.append(f"{int(r['rank']):>3}   {int(r.tier)}  {r.ticker:<6} {str(r.Sector)[:22]:<22} {str(r.beta_bucket):<5} {r.P_topq_42d*100:>4.1f}% {r.P_topq_63d*100:>4.1f}% {r.basis:<22} {int(r.n_signatures):>4}  {r.best_signature}")
    unr = u[u.state_score.isna()]
    lines += ["", "not ranked:"] + [f"    {tk:<6} {r.note}" for tk, r in unr.iterrows()]
    (out_dir / "THE_LIST.txt").write_text("\n".join(lines))
    print(f"scored {len(ranked)} names as of {asof.date()} from frozen model {M['meta']['sha256']}; tiers {ranked.tier.value_counts().sort_index().tolist()}", file=sys.stderr)
    return ranked


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default="research"); ap.add_argument("--out", default=str(RESULTS_DIR)); ap.add_argument("--asof", default=None)
    a = ap.parse_args(); main(a.rows, a.out, a.asof)
