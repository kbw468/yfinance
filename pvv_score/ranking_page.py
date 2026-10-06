"""Sortable HTML page of the full-universe scores (results/universe_scores.csv)."""
import json
import pandas as pd
from .config import RESULTS_DIR

COLS = [("ticker", "Ticker", "s"), ("Sector", "Sector", "s"), ("beta_bucket", "Beta bucket", "s"),
        ("state_score_pooled", "State score (universe pct)", "n"), ("state_score", "State score (within beta)", "n"),
        ("level_score", "Level score (identity-heavy)", "n"), ("avg_score", "Avg (state+level)/2", "n"), ("p_top_half_42d", "P(top half 42d)", "n"), ("p_top_quartile_42d", "P(top quartile 42d)", "n"), ("both_agree", "Both >= 0.8", "s"), ("vs_sector_score", "Vs sector (within-sector pct)", "n"), ("read", "Read", "s"), ("ml_score", "ML pct", "n"),
        ("z_mom_12_1", "z 12-1 mom", "n"), ("z_rs_lead_126", "z RS lead", "n"), ("z_dist_52w_high", "z dist 52w hi", "n"),
        ("z_rv20_pct_252", "z RV20 pct", "n"), ("z_bbw_pct_252", "z BB width", "n"), ("z_vol_dry_20_250", "z vol dry-up", "n"),
        ("z_updown_vol_ratio_50", "z up/dn vol", "n"), ("z_obv_price_div_63", "z OBV lead", "n"), ("z_dn_up_vol_asym_63", "z dn/up vol asym", "n"),
        ("z_corr_spy_63", "z corr SPY", "n"), ("z_idio_vol_63", "z idio vol", "n"), ("beta_252", "beta 252", "n"), ("rv20", "RV20", "n"),
        ("top_states", "Top state readings", "s"), ("note", "Note", "s")]


def main():
    df = pd.read_csv(RESULTS_DIR / "universe_scores.csv")
    asof = df["asof"].iloc[0]
    rows = []
    for _, r in df.iterrows():
        rows.append([None if (isinstance(r[c], float) and pd.isna(r[c])) else (round(float(r[c]), 2) if k == "n" else str(r[c])) for c, _, k in COLS])
    data = json.dumps({"cols": [[c, h, k] for c, h, k in COLS], "rows": rows})
    html = f"""<title>PVV Universe Scores</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: toolbar + one full-width sortable table; deutan-safe blue/orange */
:root{{--bg:#f7f8fa;--fg:#1a1f2b;--muted:#5b6475;--rule:#d9dde6;--accent:#1f5fbf;--accent2:#d9731a;--th:#e8edf6;--row:#eef1f6;--pos:#1f5fbf;--neg:#d9731a;
--sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#13161d;--fg:#e6e9ef;--muted:#9aa3b5;--rule:#2b3140;--accent:#6ea3f2;--accent2:#f0a35e;--th:#1f2735;--row:#1c212c;--pos:#6ea3f2;--neg:#f0a35e;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#13161d;--fg:#e6e9ef;--muted:#9aa3b5;--rule:#2b3140;--accent:#6ea3f2;--accent2:#f0a35e;--th:#1f2735;--row:#1c212c;--pos:#6ea3f2;--neg:#f0a35e;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;margin:0;padding-block:24px 48px;padding-inline:16px}}
h1{{font-size:1.4rem;margin:0 0 4px}} .sub{{color:var(--muted);font-size:.9rem;margin:0 0 14px;max-width:90ch;line-height:1.5}}
.bar{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}}
input,select{{font:inherit;padding:6px 8px;border:1px solid var(--rule);border-radius:4px;background:var(--bg);color:var(--fg)}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px}}
table{{border-collapse:collapse;font-family:var(--mono);font-size:.76rem;font-variant-numeric:tabular-nums;white-space:nowrap;min-width:100%}}
th{{background:var(--th);text-align:left;padding:6px 8px;cursor:pointer;position:sticky;top:0;user-select:none}}
th.on{{color:var(--accent)}} td{{padding:4px 8px;border-top:1px solid var(--rule)}} tr:nth-child(even) td{{background:var(--row)}}
td.pos{{color:var(--pos);font-weight:500}} td.neg{{color:var(--neg);font-weight:500}} td.s{{white-space:normal;max-width:34ch}}
.count{{color:var(--muted);font-size:.85rem}}
</style>
<h1>PVV universe scores</h1>
<p class="sub">All {len(df)} tickers as of the {asof} close. <b>State score</b> is the identity-neutral ranking: every factor is the name's deviation from its own trailing 3-year norm (z), combined with weights validated walk-forward inside beta terciles. A chronically strong name scores near the middle unless its current price / volume / volatility behaviour is unusual for it. <b>Level score</b> is the earlier identity-heavy composite, shown for comparison. z columns are in standard deviations vs the name's own history; blue = above its norm, orange = below. <b>Recommended sort for actionable names today:</b> <code>Avg (state+level)/2</code> descending, or filter <code>Both >= 0.8</code> = True (behaviour unusual for the name AND the name is in a persistent-strength regime). Click a header to sort; type to filter.</p>
<div class="bar"><input id="q" placeholder="filter ticker / sector / note" size="34"><select id="bb"><option value="">all beta buckets</option><option>low</option><option>mid</option><option>high</option></select><span class="count" id="n"></span></div>
<div class="tw"><table id="t"><thead><tr id="h"></tr></thead><tbody id="b"></tbody></table></div>
<script>
const D={data};let sortCol=3,sortDir=-1;
const h=document.getElementById('h'),b=document.getElementById('b'),q=document.getElementById('q'),bb=document.getElementById('bb'),n=document.getElementById('n');
D.cols.forEach((c,i)=>{{const th=document.createElement('th');th.textContent=c[1];th.onclick=()=>{{if(sortCol===i)sortDir*=-1;else{{sortCol=i;sortDir=c[2]==='n'?-1:1}}render()}};h.appendChild(th)}});
function render(){{
  const f=q.value.toLowerCase(),bk=bb.value;
  let rows=D.rows.filter(r=>(!bk||r[2]===bk)&&(!f||[r[0],r[1],r[11],r[26],r[27]].join(' ').toLowerCase().includes(f)));
  rows.sort((x,y)=>{{const a=x[sortCol],c=y[sortCol];if(a==null&&c==null)return 0;if(a==null)return 1;if(c==null)return -1;return (a<c?-1:a>c?1:0)*sortDir}});
  [...h.children].forEach((th,i)=>th.className=i===sortCol?'on':'');
  b.innerHTML=rows.map(r=>'<tr>'+r.map((v,i)=>{{const k=D.cols[i][2];let cls=k==='s'?'s':'';if(k==='n'&&D.cols[i][0].startsWith('z_')&&v!=null){{if(v>=1)cls='pos';else if(v<=-1)cls='neg'}}return `<td class="${{cls}}">${{v==null?'':v}}</td>`}}).join('')+'</tr>').join('');
  n.textContent=rows.length+' of '+D.rows.length+' tickers';
}}
q.oninput=render;bb.onchange=render;render();
</script>
"""
    out = RESULTS_DIR / "universe_scores.html"
    out.write_text(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
