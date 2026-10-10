"""Book-level test of everything found: model top 20, transparent characteristics top 20, the V-recovered sector-leader setup,
every eligible name, and SPY, over 2017-2019 and 2021 onward (the COVID window is out of every fit, so it is out of the books).
Same trade rules everywhere (next open, 8% stop, 21 sessions, 10 bp round trip). Writes results/ra21/books.txt."""
import json
import numpy as np
import pandas as pd
from ..data_io import load_panel
from ..clean import clean_panel
from .common import load, D, OUT, MARKET
from .book import run, stats
from .rules import conditions

PERIODS = [("2017-01-01", "2019-12-31"), ("2021-01-01", "2023-12-31"), ("2024-01-01", None)]


def main():
    panel, _ = clean_panel(load_panel(), verbose=False)
    spy = panel["Close"]["SPY"]
    O = pd.read_parquet(D / "oos.parquet")
    need = ["mdd_126", "mean_dd_63", "xs_sec_126", "up_capture_252", "rs_off_high_252"]
    T = load(need + MARKET + ["sup_21", "sector"], start="2017-01-01").merge(O, on=["date", "ticker"], how="left")
    cuts = json.load(open(OUT / "rules_meta.json"))["breadth_cuts"]
    C = conditions(T, need, cuts)
    lead = C["xs_sec_126:TOP"] | C["rs_off_high_252:TOP"] | C["up_capture_252:TOP"]
    T["vrec"] = np.where(C["mdd_126:BOT"] & C["mean_dd_63:TOP"] & lead, 1.0, np.nan)
    T["vrec_x"] = T.vrec.where(~T.sector.isin(["Energy", "Basic Materials", "Utilities"]))
    T["all"] = 1.0
    W = {k: T.pivot(index="date", columns="ticker", values=c) for k, c in [("model top 20", "score_stock_only"), ("V-recovered leaders", "vrec"), ("V-recovered leaders ex commodity/utility", "vrec_x"), ("every eligible name", "all")]}
    lines = []
    for a, b in PERIODS:
        res = {}
        for k, S in W.items():
            v = run(S, panel, top_n=None if k != "model top 20" else 20, start=a, end=b)
            res[k] = stats(v, spy)
            if k.startswith("V-recovered"):
                held = S.loc[a:b].notna().sum(axis=1)
                res[k]["signal days with a name"] = float((held > 0).mean()); res[k]["median names when firing"] = float(held[held > 0].median())
        s = spy.loc[a:b] / spy.loc[a:b].iloc[0]
        res["SPY held"] = stats(s, spy)
        tab = pd.DataFrame(res).T
        lines += [f"\n{a[:4]} to {(b or 'latest')[:10]}:", tab.round(3).to_string()]
        print(lines[-2], "\n", lines[-1], flush=True)
    (OUT / "books.txt").write_text("Books: next open, 8% stop, 21 sessions, 10 bp round trip, 21 staggered sleeves, daily marks\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
