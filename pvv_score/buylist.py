"""The buy list: names ranked by probability that their next 42 sessions trace a smooth, drawdown-light climb
relative to the universe (the smooth-path objective). Reads universe_scores_smooth.csv + calibration_smooth_recent.csv."""
import numpy as np
import pandas as pd
from .config import RESULTS_DIR

BINS = 20


def main():
    u = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")
    asof = u["asof"].iloc[0]
    cal = pd.read_csv(RESULTS_DIR / "calibration_smooth_recent.csv", index_col=0)
    b = np.ceil(u.avg_score * BINS).clip(1, BINS)
    u["p_smooth_top_half_42d"] = b.map(cal["p_top_half_42d"]).round(3)
    u["p_smooth_top_quartile_42d"] = b.map(cal["p_top_quartile_42d"]).round(3)
    u["p_smooth_top_half_63d"] = b.map(cal["p_top_half_63d"]).round(3)
    u["p_smooth_top_quartile_63d"] = b.map(cal["p_top_quartile_63d"]).round(3)
    scored = u[u.state_score.notna()].sort_values(["avg_score", "state_score_pooled"], ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)
    cols = ["rank", "ticker", "Company", "Sector", "beta_bucket", "avg_score", "p_smooth_top_half_42d", "p_smooth_top_quartile_42d",
            "p_smooth_top_half_63d", "p_smooth_top_quartile_63d", "state_score_pooled", "level_score", "both_agree", "top_states"]
    scored[cols].to_csv(RESULTS_DIR / "buylist.csv", index=False)
    u.to_csv(RESULTS_DIR / "universe_scores_smooth.csv", index=False)
    lines = [f"BUY LIST as of {asof} close. Objective: smooth, drawdown-light appreciation over the next 21-63 sessions.",
             "P42 / Q42 = out-of-sample probability (2022-10 onward) that the name's next 42-session path ranks in the top half / top quartile of the universe on",
             "            a combined Sharpe + max-drawdown + straightness + up-day-share score. Universe baseline 50% / 25%. P63 / Q63 same at 63 sessions.",
             "both = behaviour-unusual-for-the-name gate AND persistent-regime gate both pass (>= 0.8).", "",
             f"{'#':>3} {'tkr':<6} {'sector':<22} {'beta':<5} {'score':>5} {'P42':>5} {'Q42':>5} {'P63':>5} {'Q63':>5} both  what the tape is doing"]
    for _, r in scored.iterrows():
        lines.append(f"{int(r['rank']):>3} {r.ticker:<6} {str(r.Sector)[:22]:<22} {str(r.beta_bucket):<5} {r.avg_score:>5.2f} {r.p_smooth_top_half_42d:>5.2f} {r.p_smooth_top_quartile_42d:>5.2f} {r.p_smooth_top_half_63d:>5.2f} {r.p_smooth_top_quartile_63d:>5.2f} {'Y' if r.both_agree else '-'}     {r.top_states}")
    unsc = u[u.state_score.isna()]
    lines += ["", "not scored:"] + [f"    {r.ticker:<6} {r.note}" for _, r in unsc.iterrows()]
    (RESULTS_DIR / "buylist.txt").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
