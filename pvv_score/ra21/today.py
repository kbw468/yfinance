"""Score the latest session: final model fitted on every labelled session (every 3rd, COVID out), today's names ranked.
P(superior) = binned isotonic map (>= 500 cases per step) from the walk-forward out-of-sample score percentile (2017+) to the
realised superior rate. Drivers = the three largest positive tree contributions (TreeSHAP), in plain language.
Writes results/ra21/today.csv and today.txt."""
import json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from ..calib import BinnedIsotonic
from ..data_io import load_universe
from .common import D, OUT
from .model import prepare, PARAMS, TARGET
from .plain import name, rule as plain_rule
from .rules import conditions as rule_conditions


def main(top=40):
    T, sf, mf = prepare(); feats = sf + mf
    asof = T.date.max()
    tr = T[TARGET].notna() & T.thin
    m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], T.loc[tr, TARGET])
    O = pd.read_parquet(D / "oos.parquet")
    X = T[["date", "ticker", TARGET]].merge(O[["date", "ticker", "score"]], on=["date", "ticker"])
    X = X[X.score.notna() & X[TARGET].notna()]
    X["pct"] = X.groupby("date").score.rank(pct=True)
    cal = BinnedIsotonic(500).fit(X.pct.to_numpy(), X[TARGET].to_numpy())
    now = T[T.date == asof].copy()
    now["score"] = m.predict_proba(now[feats])[:, 1]
    now["pct"] = now.score.rank(pct=True)
    now["P_superior_21"] = cal.predict(now.pct.to_numpy())
    contrib = m.predict(now[feats], pred_contrib=True)[:, :-1]
    drivers = []
    for row in contrib:
        idx = np.argsort(-row)[:3]
        drivers.append("; ".join(name(feats[i][2:] if feats[i].startswith("r_") else feats[i]) for i in idx if row[i] > 0))
    now["drivers"] = drivers
    uni = load_universe()
    now["company"] = now.ticker.map(uni.Company)
    # confirmed rules firing tonight
    R = pd.read_csv(OUT / "rules_confirmed.csv") if (OUT / "rules_confirmed.csv").exists() else pd.DataFrame(columns=["rule"])
    if len(R):
        raw = T[T.date == asof].reset_index(drop=True)
        cuts = json.load(open(OUT / "rules_meta.json"))["breadth_cuts"]
        C = rule_conditions(raw, [c[2:] for c in sf], cuts)
        hits = [[] for _ in range(len(raw))]
        for _, r in R.iterrows():
            mm = np.ones(len(raw), bool)
            for leg in r.rule.split(" & "): mm &= C[leg]
            for j in np.flatnonzero(mm): hits[j].append(r)
        now["rules_firing"] = [len(h) for h in hits]
        now["best_rule"] = [plain_rule(max(h, key=lambda r: r.p_conf).rule) if h else "" for h in hits]
    out = now.sort_values("P_superior_21", ascending=False)[["ticker", "company", "sector", "P_superior_21", "pct", "beta_l1_252"] + (["rules_firing", "best_rule"] if len(R) else []) + ["drivers"]]
    out.to_csv(OUT / "today.csv", index=False)
    lines = [f"Superior 21-session trade candidates, {asof.date()} close (base rate 20%). P = realised superior rate of names at this score percentile, walk-forward 2017+."]
    for _, r in out.head(top).iterrows():
        lines.append(f"{r.ticker:<6} {str(r.sector)[:18]:<18} P {r.P_superior_21*100:4.1f}%  beta {r.beta_l1_252:4.2f}  {r.drivers}")
    (OUT / "today.txt").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
