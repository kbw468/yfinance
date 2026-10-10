"""Q4 2026 candidate lists for the three best-tested baskets, sorted by market cap.

usage: python candidates.py OUT UNIVERSE_CSV
Reads OUT/screen_2026.csv and OUT/megacap_2018.json; writes OUT/q4_2026_candidates.csv and
OUT/quintiles_2026.json (1Y return cutoffs for each fifth of the list).
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT, UNIV = Path(sys.argv[1]), Path(sys.argv[2])
s = pd.read_csv(OUT / "screen_2026.csv", index_col=0)
u = pd.read_csv(UNIV)
u["t"] = u.Ticker.str.replace(".", "-", regex=False)
s = s.join(u.set_index("t")[["Company", "Sector", "Market Cap"]])
s["fifth"] = pd.qcut(s.R12, 5, labels=False) + 1

cut = s.groupby("fifth").R12.agg(["min", "max", "size"])
(OUT / "quintiles_2026.json").write_text(json.dumps(
    {int(k): {"min": float(r["min"]), "max": float(r["max"]), "n": int(r["size"])} for k, r in cut.iterrows()}))

mega = pd.DataFrame(json.loads((OUT / "megacap_2018.json").read_text())["top100_now"])
mega_np = set(mega[mega.bucket == "1Y neg + 1M pos"].ticker)
a = (s.fifth == 2) & (s.R1 < 0)
b = (s.fifth <= 2) & (s.R1 < -0.10)
c = s.index.isin(mega_np)
s["basket"] = [",".join(x for x, f in zip("ABC", flags) if f) for flags in zip(a, b, c)]
out = s[s.basket != ""].sort_values("Market Cap", ascending=False)
out.index.name = "ticker"
out = out.reset_index()[["basket", "ticker", "Company", "Sector", "Market Cap", "R12", "R1", "QTD"]]
out.columns = ["basket", "ticker", "company", "sector", "mcap_musd", "ret_1y_sep30", "ret_sep", "qtd_oct9"]
out.to_csv(OUT / "q4_2026_candidates.csv", index=False)
print(out.basket.value_counts().to_dict())
print(cut.round(4).to_string())
