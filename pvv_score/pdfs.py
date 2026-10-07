"""Render THE LIST to two PDFs (ranked; sorted by market cap) via headless Chromium."""
import html
import subprocess
import pandas as pd
from .config import RESULTS_DIR, UNIVERSE_CSV

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
TIER_BG = {1: "#1f5fbf", 2: "#4f82d4", 3: "#a9c1e8", 4: "#cbd9f2", 5: "#e6edf9", 6: "#ffffff"}
TIER_FG = {1: "#fff", 2: "#fff", 3: "#1a1f2b", 4: "#1a1f2b", 5: "#1a1f2b", 6: "#5b6475"}


def cap(x):
    return "" if pd.isna(x) else (f"${x/1e6:.2f}T" if x >= 1e6 else f"${x/1e3:.0f}B")


def render(df, sig, title, lead, mcap_first):
    rows = []
    for _, r in df.iterrows():
        t = int(r.tier)
        best = sig["best_signature_plain"].get(r.ticker, "") if r.ticker in sig.index else ""
        best = best if isinstance(best, str) else ""
        nm = sig["near_miss_n"].get(r.ticker, 0) if r.ticker in sig.index else 0
        nmp = sig["near_miss_piece"].get(r.ticker, "") if r.ticker in sig.index else ""
        nmp = nmp if isinstance(nmp, str) else ""
        tail = best.replace(" AND ", " + ") if best else "none firing"
        cells = [f'<td class="n">{cap(r.mcap_m)}</td>', f'<td>{int(r["rank"])}</td>', f'<td><b>{t}</b></td>', f'<td><b>{r.ticker}</b></td>', f'<td>{r.Index}</td>',
                 f'<td>{html.escape(str(r.Sector))}</td>', f'<td>{r.beta_bucket}</td>', f'<td class="n">{r.P_topq_42d*100:.1f}%</td>', f'<td class="n">{r.P_topq_63d*100:.1f}%</td>',
                 f'<td class="n">{r.avg_score:.2f}</td>', f'<td class="n">{int(r.n_signatures)}</td>', f'<td class="s">{html.escape(tail)}</td>']
        hdr = ["Mkt cap", "Rank", "Tier", "Ticker", "Index", "Sector", "Beta", "P 42d", "P 63d", "Score", "sigs", "Strongest confirmed signature firing"]
        if not mcap_first:
            cells = cells[1:4] + [cells[0]] + cells[4:]
        rows.append(f'<tr style="background:{TIER_BG[t]};color:{TIER_FG[t]}">' + "".join(cells) + "</tr>")
    if not mcap_first:
        hdr = hdr[1:4] + [hdr[0]] + hdr[4:]
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page{{size:A4 landscape;margin:10mm}} body{{font-family:Helvetica,Arial,sans-serif;font-size:8px;color:#1a1f2b}}
h1{{font-size:16px;margin:0 0 4px}} p{{margin:0 0 6px;font-size:8.5px;color:#444}}
table{{border-collapse:collapse;width:100%}} th{{background:#e8edf6;text-align:left;padding:3px 4px;font-size:8px}} td{{padding:2px 4px;border-top:1px solid #d9dde6;vertical-align:top}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} td.s{{font-size:7.2px}} tr{{page-break-inside:avoid}} .leg span{{display:inline-block;padding:1px 6px;margin-right:4px;border:1px solid #ccc}}
</style></head><body><h1>{title}</h1><p>{lead}</p>
<p class="leg"><b>Tiers (P 42d):</b> <span style="background:#1f5fbf;color:#fff">Tier 1 &ge; 45%</span><span style="background:#4f82d4;color:#fff">Tier 2 40–45%</span><span style="background:#a9c1e8">Tier 3 35–40%</span><span style="background:#cbd9f2">Tier 4 30–35%</span><span style="background:#e6edf9">Tier 5 25–30%</span><span>Tier 6 below baseline</span> &nbsp; sigs = confirmed three-condition signatures firing.</p>
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in hdr)}</tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""


def main():
    L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv")
    sig = pd.read_csv(RESULTS_DIR / "signatures_today.csv").set_index("ticker")
    asof = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")["asof"].iloc[0]
    mc = pd.read_csv(UNIVERSE_CSV)[["Ticker", "Market Cap", "Index"]]
    mc["Ticker"] = mc.Ticker.str.replace(".", "-", regex=False)
    L = L.merge(mc.rename(columns={"Ticker": "ticker", "Market Cap": "mcap_m"}), on="ticker", how="left")
    lead = (f"{len(L)} names (S&P 500 + S&P 400). P = probability that the next 42 / 63 sessions trace a top-quartile smooth climb against the whole "
            "universe (Sharpe, max drawdown, straightness, up-day share), read off a monotone out-of-sample calibration (2024 onward, today's market regime; "
            "every step of the curve rests on at least 500 historical cases). Baseline 25%. Score = multi-factor composite (0-1). sigs = confirmed three-condition "
            "signatures firing. Below the signature region, names are ordered by Score; in low beta the composite carried no out-of-sample edge, so those "
            "probabilities are flat by evidence, not by omission.")
    tmp = RESULTS_DIR / "_tmp"
    tmp.mkdir(exist_ok=True)
    for name, df, mf, title in [("THE_LIST", L.sort_values("rank"), False, f"THE LIST, {asof} close"),
                                ("THE_LIST_by_mktcap", L.sort_values("mcap_m", ascending=False), True, f"THE LIST, {asof} close, sorted by market cap")]:
        h = tmp / f"{name}.html"
        h.write_text(render(df, sig, title, lead, mf))
        subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={RESULTS_DIR / (name + '.pdf')}", f"file://{h}"],
                       capture_output=True)
    L.sort_values("mcap_m", ascending=False).to_csv(RESULTS_DIR / "THE_LIST_by_mktcap.csv", index=False)
    print("wrote THE_LIST.pdf and THE_LIST_by_mktcap.pdf")


if __name__ == "__main__":
    main()
