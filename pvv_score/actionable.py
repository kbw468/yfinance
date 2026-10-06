"""Universe sorted by actionability: avg of state percentile and level percentile, ties by state. CSV + plain text."""
import pandas as pd
from .config import RESULTS_DIR


def main():
    u = pd.read_csv(RESULTS_DIR / "universe_scores.csv")
    asof = u["asof"].iloc[0]
    unsc = u[u.state_score.isna()]
    u = u[u.state_score.notna()].sort_values(["avg_score", "state_score_pooled"], ascending=False).reset_index(drop=True)
    u.insert(0, "rank", u.index + 1)
    cols = ["rank", "ticker", "Company", "Sector", "beta_bucket", "avg_score", "p_top_half_42d", "p_top_quartile_42d", "xs_sharpe_42_vs_universe",
            "p_top_half_63d", "state_score_pooled", "level_score", "both_agree", "top_states", "mom_12_1", "dist_52w_high", "beta_252", "rv20"]
    cols = [c for c in cols if c in u.columns]
    out = u[cols].round(3)
    out.to_csv(RESULTS_DIR / "actionable_sorted.csv", index=False)
    lines = [f"PVV actionability sort, as of {asof} close. {len(out)} scored names, best first.",
             "score = avg of state pct (behaviour unusual for the name) and level pct (persistent-strength regime). both = both >= 0.8.",
             "pTopHalf / pTopQ = out-of-sample calibrated probability (2022-10 onward) that the name's next-42d Sharpe lands in the top half / top quartile of the universe. xsSh = mean 42d Sharpe vs universe mean for names in this score bin.", "",
             f"{'#':>3} {'tkr':<6} {'sector':<22} {'beta':<5} {'score':>5} {'pTopHalf':>8} {'pTopQ':>5} {'xsSh':>6} {'state':>5} {'level':>5} both  top state readings"]
    for _, r in out.iterrows():
        lines.append(f"{int(r['rank']):>3} {r.ticker:<6} {str(r.Sector)[:22]:<22} {str(r.beta_bucket):<5} {r.avg_score:>5.2f} {r.p_top_half_42d:>8.3f} {r.p_top_quartile_42d:>5.3f} {r.xs_sharpe_42_vs_universe:>+6.2f} {r.state_score_pooled:>5.2f} {r.level_score:>5.2f} {'Y' if r.both_agree else '-'}     {r.top_states}")
    lines += ["", "not scored:"] + [f"    {r.ticker:<6} {r.note}" for _, r in unsc.iterrows()]
    (RESULTS_DIR / "actionable_sorted.txt").write_text("\n".join(lines))
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
