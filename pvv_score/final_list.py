"""THE LIST. One probability per ticker: P(next 42 sessions are a top-quartile smooth climb vs the universe), realised
out of sample in the 2024+ window for the rule that applies to the name:
  * if confirmed signatures fire on the name tonight: the signature-depth cell (how many fire) by beta bucket
  * otherwise: the composite (beta bucket x score tier) cell
Both are realised frequencies on the same outcome, so they sort together. Baseline 25%."""
import numpy as np
import pandas as pd
from .config import RESULTS_DIR

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
    for _, r in u.iterrows():
        if pd.isna(r.state_score):
            p42.append(np.nan); p63.append(np.nan); basis.append("unscored"); continue
        if r.n_signatures > 0 and r.depth_bin in depth.index:
            p42.append(depth.loc[r.depth_bin, "P_topq42"]); p63.append(depth.loc[r.depth_bin, "P_topq63"]); basis.append(f"signatures x{r.n_signatures}")
        else:
            c = comp_cell.get((r.beta_bucket, r.tier))
            p42.append(c.p_top_q_42 if c is not None else np.nan); p63.append(c.p_top_q_63 if c is not None else np.nan); basis.append(f"composite {r.tier}")
    u["P_topq_42d"] = np.round(p42, 3); u["P_topq_63d"] = np.round(p63, 3); u["basis"] = basis
    ranked = u[u.P_topq_42d.notna()].sort_values(["P_topq_42d", "P_topq_63d", "n_signatures", "avg_score"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    cols = ["rank", "ticker", "Company", "Sector", "beta_bucket", "beta_252", "P_topq_42d", "P_topq_63d", "basis", "n_signatures", "best_signature", "avg_score", "top_states"]
    ranked[cols].to_csv(RESULTS_DIR / "THE_LIST.csv", index=False)
    from .regime import today_regime
    reg = today_regime()
    lines = [f"THE LIST, {asof} close. Market regime today: {reg}; all probabilities measured on {reg} sessions.",
             f" P42 / P63 = probability (realised out of sample, 2024 onward, same regime) that the next 42 / 63 sessions are a top-quartile",
             "smooth climb vs the whole universe (Sharpe + max drawdown + straightness + up-day share). Baseline 25%. Every name ranked.",
             "basis = which rule produced the number: 'signatures xN' = N confirmed conjunction signatures fire tonight; 'composite <tier>' = multi-factor score tier.", "",
             f"{'#':>3} {'tkr':<6} {'sector':<22} {'beta':<5} {'P42':>5} {'P63':>5} {'basis':<16} {'sigs':>4}  strongest confirmed signature firing tonight"]
    for _, r in ranked.iterrows():
        lines.append(f"{int(r['rank']):>3} {r.ticker:<6} {str(r.Sector)[:22]:<22} {str(r.beta_bucket):<5} {r.P_topq_42d:>5.2f} {r.P_topq_63d:>5.2f} {r.basis:<16} {int(r.n_signatures):>4}  {r.best_signature}")
    unsc = u[u.P_topq_42d.isna()]
    lines += ["", "not ranked:"] + [f"    {r.ticker:<6} {r.note if isinstance(r.note, str) else ''}" for _, r in unsc.iterrows()]
    (RESULTS_DIR / "THE_LIST.txt").write_text("\n".join(lines))
    print("\n".join(lines[:65]))


if __name__ == "__main__":
    main()
