"""THE LIST. One probability per ticker: P(next 42 sessions are a top-quartile smooth climb vs the universe), realised
out of sample in the 2024+ window for the rule that applies to the name:
  * if confirmed signatures fire on the name tonight: the signature-depth cell (how many fire) by beta bucket
  * otherwise: the composite (beta bucket x score tier) cell
Both are realised frequencies on the same outcome, so they sort together. Baseline 25%."""
import numpy as np
import pandas as pd
from .config import RESULTS_DIR, RECENT_START

TIERS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
TIER_LABELS = ["<50", "50-70", "70-80", "80-90", "90-95", "95-100"]
DEPTH_BINS = [-1, 0, 2, 5, 10, 20, 50, 10000]
DEPTH_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "50+"]


def main():
    u = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")
    asof = u["asof"].iloc[0]
    comp = pd.read_csv(RESULTS_DIR / "buylist_probability_table.csv")
    comp_cell = {(r.beta_bucket, r.tier): r for _, r in comp.iterrows()}
    sig = pd.read_csv(RESULTS_DIR / "signatures_today.csv").set_index("ticker")
    depth = pd.read_csv(RESULTS_DIR / "signature_depth_oos.csv", index_col=0)
    rule = pd.read_csv(RESULTS_DIR / "signature_rule_oos.csv", index_col=0)
    u["tier"] = pd.cut(u.avg_score, TIERS, labels=TIER_LABELS, include_lowest=True).astype(str)
    u["n_signatures"] = u.ticker.map(sig["n_fire"]).fillna(0).astype(int)
    u["best_signature"] = u.ticker.map(sig["best_signature_plain"]).fillna("")
    u["depth_bin"] = pd.cut(u.n_signatures, DEPTH_BINS, labels=DEPTH_LABELS).astype(str)
    p42, p63, basis = [], [], []
    ls42 = u.ticker.map(sig["ls_iso_q42"]); ls63 = u.ticker.map(sig["ls_iso_q63"])
    bp42 = u.ticker.map(sig["bp_iso_q42"]); bp63 = u.ticker.map(sig["bp_iso_q63"])
    for i, r in u.iterrows():
        if pd.isna(r.state_score):
            p42.append(np.nan); p63.append(np.nan); basis.append("unscored"); continue
        cands = []
        if pd.notna(r.get("iso_q42", np.nan)):
            cands.append((float(r.iso_q42), float(r.iso_q63), f"composite {r.avg_score:.2f}"))
        if r.n_signatures > 0 and pd.notna(ls42.iloc[i]):
            cands.append((float(ls42.iloc[i]), float(ls63.iloc[i]), f"signatures x{int(r.n_signatures)}"))
        if r.n_signatures > 0 and pd.notna(bp42.iloc[i]):
            cands.append((float(bp42.iloc[i]), float(bp63.iloc[i]), f"signatures x{int(r.n_signatures)}"))
        if not cands:
            p42.append(np.nan); p63.append(np.nan); basis.append("no cell"); continue
        best = max(cands, key=lambda x: x[0])
        p42.append(best[0]); p63.append(best[1])
        others = sorted({x[2].split(" ")[0] for x in cands if x[2] != best[2]})
        basis.append(best[2] + (f" (+{others[0]})" if others else ""))
    u["P_topq_42d"] = np.round(p42, 4); u["P_topq_63d"] = np.round(p63, 4); u["basis"] = basis
    ranked = u[u.P_topq_42d.notna()].sort_values(["P_topq_42d", "P_topq_63d", "avg_score", "n_signatures"], ascending=False).reset_index(drop=True)
    ranked = ranked.rename(columns={"tier": "score_tier"})
    ranked.insert(0, "rank", ranked.index + 1)
    ranked.insert(1, "tier", pd.cut(ranked.P_topq_42d, [-1, .25, .30, .35, .40, .45, 2], labels=[6, 5, 4, 3, 2, 1]).astype(int))
    cols = ["rank", "tier", "ticker", "Company", "Sector", "beta_bucket", "beta_252", "P_topq_42d", "P_topq_63d", "basis", "n_signatures", "best_signature", "avg_score", "top_states"]
    ranked[cols].to_csv(RESULTS_DIR / "THE_LIST.csv", index=False)
    from .regime import today_regime, today_move
    from .volindex import today_readings
    v = today_readings(); reg = today_regime()
    lines = [f"THE LIST, {asof} close. VIX {v['vix']:.1f}  VXN {v['vxn']:.1f}  IWM 20d rv {v['iwm_rv20']:.1f}  MOVE {v['move']:.0f} ({reg}). Rate-sensitive sectors (utilities, REITs, staples, financials) measured on {reg} sessions; all other sectors on every session.",
             f" P42 / P63 = probability (realised out of sample, {RECENT_START[:4]} onward, COVID Feb-Jun 2020 excluded) that the next 42 / 63 sessions are a top-quartile",
             "smooth climb vs the whole universe (Sharpe + max drawdown + straightness + up-day share). Baseline 25%. Every name ranked.",
             "basis = which rule produced the number: 'signatures xN' = N confirmed conjunction signatures fire tonight (probability from their lift-weighted strength or the best one); 'composite s' = multi-factor score s. Probabilities are read off monotone out-of-sample calibration curves (each step >= 500 cases).", "",
             "Tier 1 >= 45%, Tier 2 40-45%, Tier 3 35-40%, Tier 4 30-35%, Tier 5 25-30%, Tier 6 below baseline.", "",
             f"{'#':>3} tier {'tkr':<6} {'sector':<22} {'beta':<5} {'P42':>5} {'P63':>5} {'basis':<22} {'sigs':>4}  strongest confirmed signature firing tonight"]
    for _, r in ranked.iterrows():
        lines.append(f"{int(r['rank']):>3}   {int(r.tier)}  {r.ticker:<6} {str(r.Sector)[:22]:<22} {str(r.beta_bucket):<5} {r.P_topq_42d*100:>4.1f}% {r.P_topq_63d*100:>4.1f}% {r.basis:<22} {int(r.n_signatures):>4}  {r.best_signature}")
    unsc = u[u.P_topq_42d.isna()]
    lines += ["", "not ranked:"] + [f"    {r.ticker:<6} {r.note if isinstance(r.note, str) else ''}" for _, r in unsc.iterrows()]
    (RESULTS_DIR / "THE_LIST.txt").write_text("\n".join(lines))
    print("\n".join(lines[:65]))


if __name__ == "__main__":
    main()
