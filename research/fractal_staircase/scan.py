"""Scan the latest session with everything the backtest validated.

Signals per name (all known on the scan date):
  qb_rungs    quiet-base ladder rungs passed (0-4); qb_strict = the tight version
  rules_hit   how many of the top out-of-sample rules the name satisfies
  model_pct   walk-forward model score, percentile within today's universe
  thesis      the stated thesis ladder (for reference, with its backtested lift)
  chart_aug   'already staircasing' state (the reference chart in August)
Writes results/scan_<date>.csv and prints the ranked list.
"""
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA_DIR, RES_DIR, best_model_tag

N_RULES = 12


def parse(rule):
    out = []
    for part in rule.split(" & "):
        m = re.match(r"(.+?)(<=|>=|==)(-?[\d.]+)$", part.strip())
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def holds(df, conds):
    m = np.ones(len(df), bool)
    for col, op, v in conds:
        x = df[col].to_numpy()
        m &= (x <= v) if op == "<=" else (x >= v) if op == ">=" else (x == v)
    return m


def main():
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    rules = pd.read_csv(f"{RES_DIR}/rules_y63.csv")
    rules = rules[(rules["test_weeks"] >= 150) & (rules["test_n"] >= 800)].sort_values("test_lift", ascending=False)
    top_rules = rules.head(N_RULES).reset_index(drop=True)
    parsed = [parse(r) for r in top_rules["rule"]]
    rule_cols = {c for p in parsed for c, _, _ in p}
    p_cols = [c for c in rule_cols if c.startswith("p_")]

    show = ["date", "ticker", "sector", "industry", "mcap", "mcap_bucket", "logp", "roc21", "roc63", "roc126", "acc21",
            "ddh252", "mar63", "rng63", "mdd63", "mdd126", "vroc21", "vacc21", "rc21_252", "rv10_126", "vlroc21", "udv21",
            "kt63", "er63", "stair_b63", "bo_hold63", "ind_d_roc63", "q_mar63", "q_mdd126", "q_roc126", "q_mdd63",
            "q_roc63", "q_vroc21", "q_rv10_126", "q_rng63"]
    cols = list(dict.fromkeys(show + [c for c in rule_cols if c in names] + [c[2:] for c in p_cols]))
    full = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=cols)
    last_date = full["date"].max()
    last = full[full["date"] == last_date].copy()
    hist = full[full["date"] >= "2006-01-01"]
    for c in p_cols:  # percentile of today's value within the full history (as in the rule search)
        h = np.sort(hist[c[2:]].dropna().to_numpy())
        last[c] = np.searchsorted(h, last[c[2:]].to_numpy(), side="right") / len(h)

    qb = [last["q_mar63"] <= 0.2, last["q_mdd126"] <= 0.3, last["q_roc126"] <= 0.4, last["ddh252"] > -0.105]
    rung = np.zeros(len(last), int)
    alive = np.ones(len(last), bool)
    for c in qb:
        alive &= c.fillna(False).to_numpy()
        rung += alive
    last["qb_rungs"] = rung
    last["qb_strict"] = ((last["q_mar63"] <= 0.2) & (last["q_mdd126"] <= 0.2) & (last["q_roc126"] <= 0.3) & (last["ddh252"] > -0.05))
    hits = np.zeros(len(last), int)
    which = [[] for _ in range(len(last))]
    for i, p in enumerate(parsed):
        m = holds(last, p)
        hits += m
        for j in np.flatnonzero(m):
            which[j].append(f"R{i + 1}")
    last["rules_hit"] = hits
    last["rules_list"] = [",".join(w) for w in which]
    last["thesis"] = ((last["roc63"] > 0) & (last["vroc21"] < 0) & (last["rv10_126"] > 0) & (last["q_mdd63"] < 0.5)
                      & (last["acc21"] > 0) & (last["vacc21"] < 0) & (last["vlroc21"] > 0))
    last["chart_aug"] = (last["stair_b63"] == 1) & (last["vroc21"] < 0) & (last["rv10_126"] > 0)

    tag = best_model_tag()
    if tag:
        print("model used:", tag)
        mdl = pd.read_parquet(f"{DATA_DIR}/latest_{tag}.parquet", columns=["ticker", "score", "pct", "top_drivers"])
        last = last.merge(mdl.rename(columns={"pct": "model_pct", "score": "model_score"}), on="ticker", how="left")
    else:
        last["model_pct"] = np.nan
        last["model_score"] = np.nan
        last["top_drivers"] = ""
    last["price"] = np.exp(last["logp"])
    last["signal_count"] = (last["qb_rungs"] == 4).astype(int) + last["qb_strict"].astype(int) + \
        (last["rules_hit"] > 0).astype(int) + (last["model_pct"] >= 0.9).astype(int)
    last = last.sort_values(["signal_count", "rules_hit", "qb_rungs", "model_pct"], ascending=False)
    out_cols = ["ticker", "sector", "industry", "mcap", "price", "signal_count", "qb_rungs", "qb_strict", "rules_hit",
                "rules_list", "model_pct", "thesis", "chart_aug", "roc21", "roc63", "roc126", "ddh252", "mar63", "rng63",
                "mdd126", "vroc21", "rv10_126", "vlroc21", "top_drivers"]
    last[out_cols].to_csv(f"{RES_DIR}/scan_{last_date:%Y-%m-%d}.csv", index=False)
    top_rules.to_csv(f"{RES_DIR}/scan_rules_used.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 40)
    print("scan date", last_date.date(), "names", len(last))
    print("quiet base full:", int((last["qb_rungs"] == 4).sum()), " strict:", int(last["qb_strict"].sum()),
          " any top rule:", int((last["rules_hit"] > 0).sum()), " thesis:", int(last["thesis"].sum()),
          " chart_aug:", int(last["chart_aug"].sum()))
    print(top_rules[["rule", "test_n", "test_hit", "test_lift", "test_weeks"]].round(3).to_string())
    print(last[last["signal_count"] >= 1][out_cols[:16]].head(60).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
