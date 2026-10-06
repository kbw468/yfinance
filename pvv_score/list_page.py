"""THE LIST as a page: probability tiers shaded, signature conditions as chips colored by factor family."""
import json
import pandas as pd
from .config import RESULTS_DIR, CACHE_DIR
from .signatures import FIXED
from .signature_rule import PLAIN

FAMILY_OVERRIDE = {"beta_252": "relative", "corr_spy_63": "relative", "up_capture_63": "relative", "capture_spread_63": "relative", "rs_lead_126": "relative", "rs_spy_63": "relative",
                   "idio_vol_63": "volatility", "sharpe_126": "price"}


def family(cond: str, reg: dict) -> str:
    if cond in FIXED:
        col = FIXED[cond][0]
        return "volume" if col.startswith(("vol", "dvol")) else "volatility" if col.startswith("rv") else "price"
    f = cond.split(":")[0]
    f = f[2:] if f.startswith("z_") else f
    if f in FAMILY_OVERRIDE:
        return FAMILY_OVERRIDE[f]
    g = reg.get(f, "price")
    return {"vol": "volatility", "volume": "volume", "relative": "relative", "run": "price", "price": "price", "interaction": "price", "roc": "price"}.get(g, "price")


def chip(cond: str, reg: dict) -> dict:
    if cond in FIXED:
        return {"t": cond, "f": family(cond, reg)}
    f, side = cond.split(":")
    z = f.startswith("z_")
    base = f[2:] if z else f
    return {"t": f"{PLAIN.get(base, base)}{' vs own norm' if z else ''} {'TOP' if side == 'TOP' else 'BOTTOM'} 20%", "f": family(cond, reg)}


def main():
    reg = pd.read_csv(CACHE_DIR / "factor_registry.csv", index_col=0)["group"].to_dict()
    L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv")
    sig = pd.read_csv(RESULTS_DIR / "signatures_today.csv").set_index("ticker")
    asof = pd.read_csv(RESULTS_DIR / "universe_scores_smooth.csv")["asof"].iloc[0]
    rows = []
    for _, r in L.iterrows():
        sigs = []
        raw = sig["top5_signatures"].get(r.ticker, "")
        if isinstance(raw, str) and raw:
            for item in raw.split(" || "):
                parts = item.split("|")
                conds = parts[0].split(" & ")
                sigs.append({"p": float(parts[1]), "n": int(parts[2]), "c": [chip(c, reg) for c in conds]})
        rows.append({"rank": int(r["rank"]), "t": r.ticker, "co": r.Company, "sec": r.Sector, "bb": r.beta_bucket, "b": round(float(r.beta_252), 2) if pd.notna(r.beta_252) else None,
                     "p42": float(r.P_topq_42d), "p63": float(r.P_topq_63d), "basis": r.basis, "ns": int(r.n_signatures), "sigs": sigs, "states": r.top_states if isinstance(r.top_states, str) else ""})
    data = json.dumps(rows)
    html = f"""<title>THE LIST</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: toolbar, legend, one table; rows expand to show signatures. deutan-safe: blue / orange / charcoal / grey */
:root{{--bg:#f7f8fa;--fg:#1a1f2b;--muted:#5b6475;--rule:#d9dde6;--th:#e8edf6;--row:#eef1f6;
--price:#1f5fbf;--volume:#d9731a;--volatility:#2b2f3a;--relative:#8a93a6;
--t1:#1f5fbf;--t2:#4f82d4;--t3:#86a9e3;--t4:#b9cdef;--t5:#dde6f7;--t6:transparent;
--sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#13161d;--fg:#e6e9ef;--muted:#9aa3b5;--rule:#2b3140;--th:#1f2735;--row:#1c212c;--price:#6ea3f2;--volume:#f0a35e;--volatility:#c9ced9;--relative:#6b7488;--t1:#2f6fd0;--t2:#28589f;--t3:#234572;--t4:#1f3550;--t5:#1a2838;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#13161d;--fg:#e6e9ef;--muted:#9aa3b5;--rule:#2b3140;--th:#1f2735;--row:#1c212c;--price:#6ea3f2;--volume:#f0a35e;--volatility:#c9ced9;--relative:#6b7488;--t1:#2f6fd0;--t2:#28589f;--t3:#234572;--t4:#1f3550;--t5:#1a2838;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;margin:0;padding-block:24px 48px;padding-inline:16px}}
h1{{font-size:1.4rem;margin:0 0 4px}} .sub{{color:var(--muted);font-size:.9rem;margin:0 0 12px;max-width:100ch;line-height:1.5}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 12px;font-size:.85rem}}
.chip{{display:inline-block;padding:2px 7px;border-radius:3px;font-size:.74rem;font-family:var(--mono);margin:1px 2px;color:#fff;white-space:nowrap}}
.chip.price{{background:var(--price)}} .chip.volume{{background:var(--volume)}} .chip.volatility{{background:var(--volatility)}} .chip.relative{{background:transparent;color:var(--fg);border:1px solid var(--relative)}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]) .chip.volatility{{color:#13161d}}}} :root[data-theme="dark"] .chip.volatility{{color:#13161d}}
.bar{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px}}
input,select{{font:inherit;padding:6px 8px;border:1px solid var(--rule);border-radius:4px;background:var(--bg);color:var(--fg)}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px}}
table{{border-collapse:collapse;font-size:.82rem;font-variant-numeric:tabular-nums;min-width:100%}}
th{{background:var(--th);text-align:left;padding:7px 9px;cursor:pointer;position:sticky;top:0;user-select:none;white-space:nowrap}}
td{{padding:5px 9px;border-top:1px solid var(--rule);vertical-align:top}}
tr.main{{cursor:pointer}} tr.main:hover td{{background:var(--row)}}
td.p{{font-family:var(--mono);font-weight:600;text-align:right}}
.tier1{{background:var(--t1);color:#fff}} .tier2{{background:var(--t2);color:#fff}} .tier3{{background:var(--t3)}} .tier4{{background:var(--t4)}} .tier5{{background:var(--t5)}}
tr.detail td{{background:var(--row);font-size:.8rem;padding:8px 12px 10px 40px}}
.sigline{{margin:3px 0}} .sigp{{font-family:var(--mono);color:var(--muted);margin-right:8px}}
.states{{color:var(--muted);font-size:.78rem;white-space:normal}}
.count{{color:var(--muted);font-size:.85rem}}
</style>
<h1>THE LIST</h1>
<p class="sub">{asof} close. Every name ranked by the probability that its next 42 / 63 sessions trace a top-quartile smooth climb against the whole universe (Sharpe, max drawdown, straightness, up-day share). Probabilities are realised out-of-sample frequencies, 2024 onward, same market regime as today. Baseline 25%. Click a row to see the confirmed signatures firing on it; each signature is three conditions that must all hold, colored by what they measure.</p>
<div class="legend"><b>Condition family:</b> <span class="chip price">price / trend / structure</span> <span class="chip volume">volume / participation</span> <span class="chip volatility">volatility</span> <span class="chip relative">relative to market (beta, correlation, capture, RS)</span>
<span style="margin-left:14px"><b>Probability tiers:</b></span> <span class="chip tier1">&ge; 45%</span> <span class="chip tier2">40–45%</span> <span class="chip tier3" style="color:var(--fg)">35–40%</span> <span class="chip tier4" style="color:var(--fg)">30–35%</span> <span class="chip tier5" style="color:var(--fg)">25–30%</span> <span style="color:var(--muted)">unshaded &lt; 25%</span></div>
<div class="bar"><input id="q" placeholder="filter ticker / sector" size="28"><select id="bb"><option value="">all beta</option><option>low</option><option>mid</option><option>high</option></select><select id="sg"><option value="">all names</option><option value="1">signatures firing</option><option value="50">50+ signatures</option></select><span class="count" id="n"></span></div>
<div class="tw"><table><thead><tr id="h"></tr></thead><tbody id="b"></tbody></table></div>
<script>
const D={data};
const COLS=[["rank","#","n"],["t","Ticker","s"],["sec","Sector","s"],["bb","Beta","s"],["b","b252","n"],["p42","P 42d","n"],["p63","P 63d","n"],["ns","# signatures","n"],["basis","Basis","s"],["best","Strongest signature firing","x"]];
let sortCol=0,sortDir=1,open=new Set();
const h=document.getElementById('h'),b=document.getElementById('b'),q=document.getElementById('q'),bb=document.getElementById('bb'),sg=document.getElementById('sg'),n=document.getElementById('n');
COLS.forEach((c,i)=>{{const th=document.createElement('th');th.textContent=c[1];th.onclick=()=>{{if(c[2]==='x')return;if(sortCol===i)sortDir*=-1;else{{sortCol=i;sortDir=c[2]==='n'&&c[0]!=='rank'?-1:1}}render()}};h.appendChild(th)}});
const tier=p=>p>=.45?'tier1':p>=.40?'tier2':p>=.35?'tier3':p>=.30?'tier4':p>=.25?'tier5':'';
const chips=s=>s.c.map(x=>`<span class="chip ${{x.f}}">${{x.t}}</span>`).join(' AND ');
function render(){{
  const f=q.value.toLowerCase(),k=bb.value,g=sg.value;
  let rows=D.filter(r=>(!k||r.bb===k)&&(!g||r.ns>=+g)&&(!f||(r.t+' '+r.sec+' '+r.co).toLowerCase().includes(f)));
  rows.sort((x,y)=>{{const c=COLS[sortCol][0];let a=x[c],d=y[c];if(a==null)return 1;if(d==null)return -1;return (a<d?-1:a>d?1:0)*sortDir}});
  [...h.children].forEach((th,i)=>th.style.color=i===sortCol?'var(--price)':'');
  b.innerHTML=rows.map(r=>{{
    const main=`<tr class="main" data-t="${{r.t}}"><td>${{r.rank}}</td><td><b>${{r.t}}</b></td><td>${{r.sec}}</td><td>${{r.bb}}</td><td class="p">${{r.b==null?'':r.b.toFixed(2)}}</td><td class="p ${{tier(r.p42)}}">${{(r.p42*100).toFixed(0)}}%</td><td class="p ${{tier(r.p63)}}">${{(r.p63*100).toFixed(0)}}%</td><td class="p">${{r.ns}}</td><td>${{r.basis}}</td><td>${{r.sigs.length?chips(r.sigs[0]):'<span class="states">'+r.states+'</span>'}}</td></tr>`;
    if(!open.has(r.t))return main;
    const det=r.sigs.length?r.sigs.map(s=>`<div class="sigline"><span class="sigp">P ${{(s.p*100).toFixed(0)}}% n=${{s.n}}</span>${{chips(s)}}</div>`).join(''):'<div class="states">No confirmed signature fires. Probability from the multi-factor composite tier.</div>';
    return main+`<tr class="detail"><td colspan="10">${{det}}<div class="states" style="margin-top:6px">State readings vs own norm: ${{r.states}}</div></td></tr>`;
  }}).join('');
  n.textContent=rows.length+' of '+D.length+' names';
  b.querySelectorAll('tr.main').forEach(tr=>tr.onclick=()=>{{const t=tr.dataset.t;open.has(t)?open.delete(t):open.add(t);render()}});
}}
q.oninput=render;bb.onchange=render;sg.onchange=render;render();
</script>
"""
    (RESULTS_DIR / "THE_LIST.html").write_text(html)
    print("wrote", RESULTS_DIR / "THE_LIST.html", len(rows), "rows")


if __name__ == "__main__":
    main()
