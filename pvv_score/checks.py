"""Hard-failing consistency checks on a finished run. Exit 1 on any failure, so a chain cannot publish a list that fails.
Run after pdfs: python -m pvv_score.checks"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .config import CACHE_DIR, RESULTS_DIR
from .livecomp import live_composite_percentile
from .volindex import today_readings

R = RESULTS_DIR
FAIL = []


def check(ok: bool, msg: str):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def main(frozen: bool = False):
    L = pd.read_csv(R / "THE_LIST.csv"); U = pd.read_csv(R / "universe_scores_smooth.csv"); S = pd.read_csv(R / "signatures_today.csv").set_index("ticker")
    asof = pd.Timestamp(U["asof"].iloc[0])
    print(f"checks for the {asof.date()} list: {len(L)} ranked of {len(U)}" + ("  [frozen model, nightly rows]" if frozen else ""))
    unr = U[U.state_score.isna()]
    print("  unranked: " + "; ".join(f"{r.ticker} ({r.note})" for _, r in unr.iterrows()))

    from .signatures import QUINTILE_FACTORS, FIXED, build_conditions
    rows_path = CACHE_DIR / ("recent_rows.parquet" if frozen else "research_long.parquet")
    if frozen:
        print("[A] tonight's composite percentile recomputed from the frozen weights")
        import json
        from .composite import tercile_series, score_rows
        W = json.load(open(R / "model" / "weights.json"))
        fac = sorted({f for layer in ("state", "level") for b in W[layer].values() for f in b})
        rows = pq.read_table(rows_path, columns=list(dict.fromkeys(["date", "ticker", "eligible", "beta_252"] + fac)), filters=[("date", "==", asof)]).to_pandas()
        rows = rows[rows.eligible.astype(bool)].reset_index(drop=True); rows["beta_bucket"] = tercile_series(rows)
        comp = {}
        for layer in ("state", "level"):
            raw = pd.Series(np.nan, index=rows.index)
            for b, wd in W[layer].items():
                m = (rows.beta_bucket == b).values
                if m.any() and wd:
                    raw[m] = score_rows(rows[m], pd.Series(wd, dtype=float)).values
            comp[layer] = raw
        live = ((comp["state"].rank(pct=True) + comp["level"].rank(pct=True)) / 2).round(8).where(comp["state"].notna() & comp["level"].notna())
        live.index = rows.ticker
        check(U["model"].iloc[0] == json.load(open(R / "model" / "model.json"))["sha256"], "list carries the frozen model id")
    else:
        print("[A] tonight's composite percentile is on the calibration tables' scale")
        live = live_composite_percentile(asof)
    j = L.merge(live.rename("comp_p"), left_on="ticker", right_index=True)
    check(len(j) == len(L), f"every ranked name has a composite tonight ({len(j)} of {len(L)})")
    check(np.allclose(j.avg_score, j.comp_p, atol=1e-6), f"list Score == composite percentile (max diff {np.abs(j.avg_score - j.comp_p).max():.2e})")

    print("[B] tonight's signature firing recomputed from the raw factor rows")
    need = list(dict.fromkeys(["date", "ticker", "eligible"] + QUINTILE_FACTORS + sorted({v[0] for v in FIXED.values()})))
    rows = pq.read_table(rows_path, columns=need, filters=[("date", "==", asof)]).to_pandas()
    rows = rows[rows.eligible.astype(bool)].reset_index(drop=True)
    conds = build_conditions(rows)
    conf = pd.read_csv(R / "model" / "signatures.csv") if frozen else pd.read_csv(R / "signatures_all.csv")
    conf = conf if frozen else conf[conf.confirmed]
    n = len(rows); cnt = np.zeros(n, int); bestp = np.zeros(n); liftsum = np.zeros(n)
    for _, r in conf.iterrows():
        m = np.ones(n, bool)
        for p in r.signature.split(" & "):
            m &= conds[p]
        cnt += m; liftsum += m * (r.lift_conf - 1); bestp = np.maximum(bestp, m * r.p_conf)
    chk = pd.DataFrame({"ticker": rows.ticker, "n_re": cnt, "bp_re": bestp, "ls_re": liftsum}).set_index("ticker").join(S[["n_fire", "bestp", "liftsum"]], how="inner")
    check(len(chk) == len(L), f"signature table covers every ranked name ({len(chk)} of {len(L)})")
    check(int((chk.n_re != chk.n_fire).sum()) == 0, f"signature counts match for every name ({int((chk.n_re != chk.n_fire).sum())} mismatches)")
    check(np.allclose(chk.bp_re, chk.bestp, atol=1e-9) and np.allclose(chk.ls_re, chk.liftsum, atol=1e-9), "best-signature probability and lift-sum match")
    check(int((L.n_signatures > 0).sum()) == int((chk.n_re > 0).sum()), "names firing on the list == recomputed")

    print("[C] probability assembly, tiers, ordering")
    m = L.merge(U[["ticker", "iso_q42"]], on="ticker").merge(S[["ls_iso_q42", "bp_iso_q42"]], left_on="ticker", right_index=True, how="left")
    p = np.nanmax(np.vstack([m.iso_q42, m.ls_iso_q42.where(m.n_signatures > 0), m.bp_iso_q42.where(m.n_signatures > 0)]), axis=0)
    check(int((np.abs(p - m.P_topq_42d) > 1e-4).sum()) == 0, "P42 == max(composite cell, lift-sum curve, best-signature curve) for every name")
    tier_re = pd.cut(L.P_topq_42d, [-1, .25, .30, .35, .40, .45, 2], labels=[6, 5, 4, 3, 2, 1]).astype(int)
    check(int((tier_re != L.tier).sum()) == 0, "tiers follow the P42 bands")
    check(bool((L.P_topq_42d.diff().dropna() <= 1e-12).all()), "rank order is P42 descending")
    check(bool(L.P_topq_42d.between(0, 1).all() and L.P_topq_63d.between(0, 1).all()), "probabilities in [0, 1]")
    check(int(((L.basis.str.startswith("signatures")) & (L.n_signatures == 0)).sum()) == 0, "no 'signatures' basis without a signature firing")
    check(int(L.ticker.duplicated().sum()) == 0, "no duplicate tickers")
    check(int(L.beta_252.isna().sum()) == 0, "every ranked name has a 252d beta")
    bb = L.beta_bucket.value_counts(); check(bb.max() - bb.min() <= 3, f"beta terciles balanced {bb.to_dict()}")
    from .composite import tercile_series
    r = U[U.state_score.notna()].copy(); r["date"] = asof
    check(int((tercile_series(r).values != r.beta_bucket.values).sum()) == 0, "tonight's beta buckets follow the same tercile rule as the historical rows")

    print("[D] price data at asof")
    C = pd.read_parquet(CACHE_DIR / "Close.parquet"); V = pd.read_parquet(CACHE_DIR / "Volume.parquet")
    O = pd.read_parquet(CACHE_DIR / "Open.parquet"); H = pd.read_parquet(CACHE_DIR / "High.parquet"); Lo = pd.read_parquet(CACHE_DIR / "Low.parquet")
    tk = [t for t in L.ticker if t in C.columns]
    check(len(tk) == len(L), "every ranked name is in the price panel")
    check(C.index.max() == asof, f"panel ends on asof ({C.index.max().date()} vs {asof.date()})")
    last = C[tk].apply(lambda s: s.dropna().index.max())
    check(int((last < asof).sum()) == 0, f"every ranked name has a close on asof ({list(last[last < asof].index)[:8]})")
    r1 = C.loc[asof, tk] / C.shift(1).loc[asof, tk] - 1
    check(int((r1.abs() > 0.4).sum()) == 0, f"no |1d return| > 40% on asof ({list(r1[r1.abs() > 0.4].index)})")
    check(int((V.loc[asof, tk] <= 0).sum()) == 0, f"no zero-volume bar on asof ({list(V.loc[asof, tk][V.loc[asof, tk] <= 0].index)})")
    stub = [t for t in tk if O.loc[asof, t] == H.loc[asof, t] == Lo.loc[asof, t] == C.loc[asof, t]]
    check(len(stub) == 0, f"no flat OHLC stub bars on asof ({stub})")
    v = today_readings()
    check(pd.Timestamp(v["date"]) >= asof - pd.Timedelta(days=4), f"vol indices fresh (last {v['date']}, asof {asof.date()})")
    check(all(v[k] is not None and v[k] > 0 for k in ["vix", "vxn", "move", "iwm_rv20"]), f"VIX/VXN/MOVE/IWM rv present {({k: v[k] for k in ['vix', 'vxn', 'move', 'iwm_rv20']})}")

    print("[E] page, PDFs and CSV built from the same list")
    html = (R / "THE_LIST.html").read_text(); D = json.loads(re.search(r"const D=(\[.*?\]);\n", html, re.S).group(1))
    check(len(D) == len(L) and np.allclose(sorted(x["p42"] for x in D), sorted(L.P_topq_42d)), "html rows and probabilities == csv")
    check(f"{asof.date()}" in html, "html header carries asof")
    mt = {f: os.path.getmtime(R / f) for f in ["signatures_today.csv", "universe_scores_smooth.csv", "THE_LIST.csv", "THE_LIST.html", "THE_LIST.pdf", "THE_LIST_by_mktcap.pdf"]}
    check(mt["signatures_today.csv"] <= mt["THE_LIST.csv"] <= mt["THE_LIST.html"] <= mt["THE_LIST.pdf"] <= mt["THE_LIST_by_mktcap.pdf"], "artifacts built in order signatures -> csv -> html -> pdfs")
    mc = pd.read_csv(R / "THE_LIST_by_mktcap.csv"); check(len(mc) == len(L) and np.allclose(sorted(mc.P_topq_42d), sorted(L.P_topq_42d)), "market-cap list is the same list")

    print("[F] calibration tables and model")
    B = pd.read_csv(R / "buylist_probability_table.csv")
    check(int(B.n.min()) >= 500 or B.pooled_from.notna().any(), f"composite cells rest on >= 500 cases or are pooled (min n {int(B.n.min())})")
    dep = pd.read_csv(R / "signature_depth_oos.csv", index_col=0)
    check(bool(dep.P_topq42.is_monotonic_increasing) and int(dep.n.min()) >= 500, "signature depth table monotone with >= 500 cases per step")
    check(int(len(conf)) > 0 and bool((conf.n_conf >= 150).all()), f"{len(conf)} confirmed signatures, all with >= 150 confirmation cases")
    import json
    cc = json.load(open(R / "model" / "calib_composite.json")); cs = json.load(open(R / "model" / "calib_signature.json"))
    check(all(len(cc[b][c]["x"]) >= 1 for b in cc for c in cc[b]) and all(len(cs["curves"][c][q]["x"]) >= 1 for c in cs["curves"] for q in cs["curves"][c]), "frozen calibration curves present for every bucket and quintile")
    check(all(v["min_n"] >= 500 for b in cc for v in cc[b].values()) and all(v["min_n"] >= 500 for c in cs["curves"].values() for v in c.values()), "every frozen curve was fitted with >= 500 cases per step")
    if (R / "movers_today.csv").exists():
        mv = pd.read_csv(R / "movers_today.csv")
        check(set(mv.ticker) == set(L.ticker), "movers table covers exactly the ranked names")

    print()
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED:\n  - " + "\n  - ".join(FAIL)); sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main(frozen="--frozen" in sys.argv)
