"""Build dashboard.html from dash_data.json (produced by snapshot.py).

Usage: python build_dashboard.py [path/to/dash_data.json] [out.html]
Static page, no network calls at view time. Blue/orange palette (deutan-safe).
"""
import json, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else 'dash_data.json'
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'dashboard.html')
DATA = json.load(open(src))
tick_src = os.path.join(os.path.dirname(os.path.abspath(src)), 'tick_data.json')
DATA['tick'] = json.load(open(tick_src)) if os.path.exists(tick_src) else None
rv_src = os.path.join(os.path.dirname(os.path.abspath(src)), 'rv_data.json')
DATA['rv'] = json.load(open(rv_src)) if os.path.exists(rv_src) else None
sc_src = os.path.join(os.path.dirname(os.path.abspath(src)), 'score_data.json')
DATA['score'] = json.load(open(sc_src)) if os.path.exists(sc_src) else None

RULE_META = {
 'SETUP_rates':   ('SETUP',  'Yields ripping, VIX asleep',          'TNX 5d chg z > 1 and VIX 5d ROC z < -0.3', 'P(5% DD/21d) 0.25 vs 0.17 base. 56% of tops, median 18d lead to the -5% break.'),
 'SETUP_complac': ('SETUP',  'Complacency',                          'VIX 21d ROC z < -1, VIX 5d ROC z < -1, SKEW 10d ROC z < -0.5', 'P 0.24. Fwd21 +0.29% vs +0.86%.'),
 'ONSET_impulse': ('ONSET',  'First impulse after compression',      'VIX 21d ROC z < -0.3 and VIX 5d ROC z > 1', 'P 0.27. Fwd 5d and 10d negative.'),
 'ONSET_vvixlag': ('ONSET',  'VIX spike, VVIX not confirming',       'VIX 5d ROC z > 1 and VVIX/VIX 5d ROC z < -1', 'P 0.27. 47% of tops, median 11d lead. If VVIX/VIX ROC z > -0.3 instead, it is a dip (P 0.155, fwd21 +1.4%).'),
 'ONSET_movelag': ('ONSET',  'Equity vol out, bond vol compressing', 'Impulse-after-compression and MOVE 5d ROC z < 0', 'P 0.43 (n=14). Fwd21 -0.46%.'),
 'ONSET_ovxdiv':  ('ONSET',  'Equity vol up, oil vol down',          'VIX 5d ROC z > 0.3 and OVX 5d ROC z < -1', 'P 0.31. Fwd21 -0.15%.'),
 'CONT_2ndleg':   ('CONT',   'Second leg',                           'VIX 3d ROC z > 1, VIX/VIX3M 3d ROC z > 1, SPY 7%+ off 63d high', 'P 0.39. Highest drawdown odds of any rule.'),
 'FAIL_yieldsup': ('FAIL',   'Rally failure: vol fading, yields up', 'VIX 10d ROC z < -1, TNX 5d chg z > 0.5, SPY 5%+ off high', 'Fwd21 -1.04%, fwd42 -0.36%. The only negative-return rule.'),
 'CAP_alldims':   ('CAP',    'Capitulation: every dimension at once','VIX, GVZ 5d ROC z > 1, MOVE 5d ROC z > 0.5, TNX 5d chg z < -0.5', 'Fwd21 +2.50%, fwd63 +4.22%. Fires median day -1 vs trough. 77% positive.'),
 'CAP_vix_gvz':   ('CAP',    'Gold vol confirming equity vol',       'VIX 5d ROC z > 1 and GVZ 5d ROC z > 1', 'Fwd21 +2.07%, fwd63 +4.22%. 38% of troughs at day -2.'),
 'CAP_vvixout':   ('CAP',    'VVIX outrunning VIX in a spike',       'VIX 5d ROC z > 1.5 and VVIX/VIX 5d ROC z > 0', 'Fwd21 +3.99%, fwd63 +7.60%. 9 of 9 positive.'),
 'ONCONF_collapse':('ONCONF','Flip confirmed',                       'VIX 5d ROC z < -1, VVIX/VIX 5d ROC z > 0.3, VIX/VIX3M 5d ROC z < -0.5', 'Median day +4 after trough. Confirmation, not edge.'),
}
DATA['rule_meta'] = RULE_META

HTML = r"""<title>Vol Tape Regime</title>
<meta name="description" content="ROC-based risk-on/off onset read of 13 volatility and rate indices against 38 ETFs">
<style>
/* Layout: summary strip, then rule board, then ROC tape, then SPY chart with fire markers, then playbook. Dense terminal, left-aligned. */
:root{
  --bg:#f7f7f4; --surface:#ffffff; --ink:#1b1f26; --ink2:#4b5563; --muted:#8a919c; --line:#e3e5e8; --grid:#eef0f2;
  --blue:#1d4ed8; --orange:#ea580c; --teal:#0891b2; --violet:#7c3aed;
  --blue-soft:#dbe7ff; --orange-soft:#ffe4d1; --teal-soft:#d6f3f8; --violet-soft:#ece2fc;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141517; --surface:#1c1e22; --ink:#ebedf0; --ink2:#b3b8c2; --muted:#7c838e; --line:#2c3037; --grid:#23262b;
  --blue:#3b82f6; --orange:#ea580c; --teal:#0891b2; --violet:#8b5cf6;
  --blue-soft:#1b2a4a; --orange-soft:#4a2a14; --teal-soft:#123a44; --violet-soft:#2e2150; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#141517; --surface:#1c1e22; --ink:#ebedf0; --ink2:#b3b8c2; --muted:#7c838e; --line:#2c3037; --grid:#23262b;
  --blue:#3b82f6; --orange:#ea580c; --teal:#0891b2; --violet:#8b5cf6;
  --blue-soft:#1b2a4a; --orange-soft:#4a2a14; --teal-soft:#123a44; --violet-soft:#2e2150; color-scheme:dark}
*{box-sizing:border-box}
.wrap>*{min-width:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-inline:16px;display:grid;gap:22px}
h1{font-family:var(--mono);font-weight:600;font-size:20px;letter-spacing:-0.01em;margin:0;text-wrap:balance}
h2{font-family:var(--mono);font-weight:600;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:0 0 10px}
.sub{color:var(--muted);font-family:var(--mono);font-size:12px}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
header{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px;min-width:0}
.tile .k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-family:var(--mono)}
.tile .v{font-size:22px;font-weight:600;margin-top:2px}
.tile .d{font-size:12px;color:var(--ink2);margin-top:2px}
.state{display:inline-block;padding:2px 8px;border-radius:999px;font-family:var(--mono);font-size:12px;font-weight:600;border:1px solid var(--line)}
.state.live{background:var(--orange-soft);color:var(--orange);border-color:var(--orange)}
.state.recent{background:var(--blue-soft);color:var(--blue);border-color:var(--blue)}
.state.quiet{color:var(--muted)}
.board{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:10px}
.rule{background:var(--surface);border:1px solid var(--line);border-left-width:4px;border-radius:6px;padding:10px 12px;min-width:0;display:grid;gap:4px}
.rule.SETUP,.rule.ONSET,.rule.CONT,.rule.FAIL{border-left-color:var(--orange)}
.rule.CAP,.rule.ONCONF{border-left-color:var(--blue)}
.rule .t{display:flex;justify-content:space-between;gap:8px;align-items:center}
.rule .tag{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--ink2)}
.rule .name{font-weight:600}
.rule .def{font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
.rule .stat{font-size:12px;color:var(--muted)}
.rule .last{font-family:var(--mono);font-size:11.5px;color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
td:first-child,th:first-child{text-align:left;font-family:var(--mono);font-weight:600;position:sticky;left:0;background:var(--surface)}
tr:last-child td{border-bottom:none}
.z{display:inline-block;min-width:44px;padding:1px 5px;border-radius:3px}
.z.hot{background:var(--orange-soft);color:var(--orange);font-weight:600}
.z.cold{background:var(--blue-soft);color:var(--blue);font-weight:600}
.chartcard{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--ink2);margin-bottom:6px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:var(--c)}
svg{display:block;width:100%;height:auto;max-width:100%}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:11.5px;color:var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;z-index:9;white-space:nowrap}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
select,button{font:inherit;font-family:var(--mono);font-size:12px;background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:5px 8px}
button.on{border-color:var(--blue);color:var(--blue);background:var(--blue-soft)}
.pb{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:720px){.pb{grid-template-columns:1fr}}
.pb .col{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px;min-width:0}
.bar{display:grid;grid-template-columns:52px 1fr 54px 40px;gap:8px;align-items:center;font-size:12px;padding:2px 0}
.bar .trk{height:10px;position:relative;background:var(--grid);border-radius:2px}
.bar .fill{position:absolute;top:0;height:100%;border-radius:2px}
.note{font-size:12.5px;color:var(--ink2);max-width:70ch}
.heat td{padding:3px 6px;text-align:center;font-family:var(--mono);font-size:11px;min-width:46px}
.heat th{padding:4px 6px;text-align:center;font-size:10px}
.heat td:first-child{text-align:left}
.hc{display:inline-block;width:100%;border-radius:3px;padding:2px 0}
.lead{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.lead .col{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px;min-width:0}
.lead h3{font-family:var(--mono);font-size:12px;font-weight:600;margin:0 0 6px;color:var(--ink2)}
.lead ol{margin:0;padding-left:18px;font-family:var(--mono);font-size:12px}
.lead li{display:flex;justify-content:space-between;gap:8px}
:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<div class="wrap">
<header>
  <div><h1>Vol Tape Regime</h1><div class="sub">13 vol/rate indices, ROC z vs own trailing 252d. As of <span id="asof" class="num"></span>. Backtest 1990 to date, Feb-Jul 2020 excluded.</div></div>
  <div class="sub">orange = risk-off side rules, blue = risk-on side rules</div>
</header>

<section id="scoresec">
  <h2>Exposure dial, 0 to 100 (drawdown-first)</h2>
  <div class="chartcard" style="display:grid;grid-template-columns:minmax(220px,1fr) 2fr;gap:16px;align-items:center">
    <div><div class="k sub" style="letter-spacing:.06em;text-transform:uppercase">today</div><div id="scorebig" class="num" style="font-size:56px;font-weight:600;line-height:1"></div><div id="scorelabel" class="sub" style="margin-top:4px"></div><div id="scoreparts" style="margin-top:10px;font-family:var(--mono);font-size:11.5px;color:var(--ink2);display:grid;gap:2px"></div></div>
    <div><svg id="scorechart" viewBox="0 0 700 170" role="img" aria-label="Exposure dial, last two years"></svg><div class="sub" id="scorebuckets" style="margin-top:6px"></div></div>
  </div>
</section>

<section>
  <h2>Now</h2>
  <div class="tiles" id="tiles"></div>
</section>

<section>
  <h2>Rule board</h2>
  <div class="board" id="board"></div>
</section>

<section>
  <h2>ROC tape (z vs own trailing 252d; orange ≥ +1, blue ≤ -1)</h2>
  <div class="tablewrap"><table id="tape"></table></div>
  <div class="tablewrap" style="margin-top:10px"><table id="ratios"></table></div>
</section>

<section>
  <h2>SPY with rule fires, last 2 years</h2>
  <div class="chartcard">
    <div class="row" id="chartctl"></div>
    <div class="legend"><span style="--c:var(--ink2)">SPY</span><span style="--c:var(--orange)">risk-off side fire</span><span style="--c:var(--blue)">risk-on side fire</span></div>
    <svg id="spychart" viewBox="0 0 1100 380" role="img" aria-label="SPY close with rule fire markers"></svg>
  </div>
</section>

<section>
  <h2>ROC z history, selected pairs</h2>
  <div class="chartcard">
    <div class="row" id="zctl"></div>
    <div class="legend" id="zlegend"></div>
    <svg id="zchart" viewBox="0 0 1100 300" role="img" aria-label="ROC z-score history"></svg>
  </div>
</section>

<section>
  <h2>Ticker playbook: median 10d return relative to SPY after a rule's first fire</h2>
  <div class="row"><label for="pbsel" class="sub">rule</label><select id="pbsel"></select><span class="sub" id="pbn"></span></div>
  <div class="pb" id="pb"></div>
  <p class="note">Bars are median relative return (%) over the 10 sessions after the fire; the right-hand number is the share of fires where the ticker beat SPY. Orange = lagged SPY, blue = led SPY.</p>
</section>

<section id="ticksec">
  <h2>Ticker-side ROC: pair rules (ticker relative ROC × index ROC)</h2>
  <div class="board" id="tickboard"></div>
</section>

<section>
  <h2>Relative-to-SPY ROC tape, 38 tickers (z vs own trailing 252d)</h2>
  <div class="tablewrap"><table id="ticktape"></table></div>
  <p class="note">Historical context columns: median relative ROC5 z three days before the 36 risk-off peaks, on the peak day, on the 40 trough days, and three days after. P(off) is the chance of a 5% SPY drawdown in 21d when the ticker's 21d relative ROC z is in its top vs bottom quintile (base 0.174).</p>
</section>

<section>
  <h2>Ticker × index ROC matrix: median 10d forward return relative to SPY (%) when the index 5d ROC z is above +1 (up) or below -1 (down)</h2>
  <div class="tablewrap"><table id="heat" class="heat"></table></div>
  <p class="note">Blue = ticker beats SPY after that index move, orange = lags. Read a row to see what a ticker does when each dimension of the vol complex moves; read a column to see who to hold when it does.</p>
</section>

<section>
  <h2>Stress-regime ROC beta: % change in ticker/SPY ratio per 100% move in the index (per 1 pct-pt for TNX/TYX), VIX pct &gt; 75</h2>
  <div class="tablewrap"><table id="beta" class="heat"></table></div>
</section>

<section id="rvsec">
  <h2>Realized-vol layer: implied vs realized, realized term structure, crossed with index ROC</h2>
  <div class="tiles" id="rvtiles"></div>
  <div class="board" id="rvboard" style="margin-top:10px"></div>
</section>

<section>
  <h2>Native implied / realized pairs (index ÷ ticker 21d realized vol)</h2>
  <div class="tablewrap"><table id="ivrv"></table></div>
  <p class="note">Low pct = implied cheap relative to that ticker's realized. HYG and XLE bottom quintiles carry P(5% SPY DD) of 0.31 and 0.27 vs 0.17 base. ROC z &lt; -1 while the index ROC z &gt; 1 means realized is outrunning implied in a spike (continuation); ROC z &gt; 1 in a spike means fear premium (dip).</p>
</section>

<section>
  <h2>Realized-vol tape, 38 tickers + SPY</h2>
  <div class="tablewrap"><table id="rvtape"></table></div>
  <p class="note">RV10/RV21 = realized term structure (short over 1-month), z vs own trailing 252d. rel RV = ticker 21d realized ÷ SPY 21d realized, with the 5d ROC z of that ratio. Right-hand four columns: the ticker's median 10d return relative to SPY historically when its own realized was expanding (z &gt; 1) or compressed (z &lt; -0.5) while VIX 5d ROC z was &gt; 1 (up) or &lt; -1 (down).</p>
</section>
</div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const RM=D.rule_meta; const N=D.dates.length; const last=N-1;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
document.getElementById('asof').textContent=D.asof;
const f2=x=>x==null?'':(x>0?'+':'')+x.toFixed(2);
// ---- tiles
const live=Object.keys(D.rules).filter(k=>D.rules[k][last]);
const recent=Object.keys(D.rules).filter(k=>!D.rules[k][last]&&D.rules[k].slice(-10).some(v=>v));
const lv=k=>D.levels.find(r=>r.series===k);
const dims=Object.keys(D.dims).filter(k=>D.dims[k].slice(-5).some(v=>v));
const tiles=[
 ['SPY',D.spy[last].toFixed(2),`${f2(D.spy_ret5)}% 5d · ${f2(D.spy_dd63)}% off 63d high`],
 ['VIX',lv('VIX').level.toFixed(2),`pct ${Math.round(lv('VIX').pct252*100)} · 5d ROC z ${f2(lv('VIX').z5)}`],
 ['VVIX / VIX ROC z',f2(D.ratios.find(r=>r.ratio==='VVIX/VIX').roc5z),'the spike discriminator (5d)'],
 ['MOVE',lv('MOVE').level.toFixed(1),`pct ${Math.round(lv('MOVE').pct252*100)} · 21d ROC z ${f2(lv('MOVE').z21)}`],
 ['TNX',lv('TNX').level.toFixed(2)+'%',`pct ${Math.round(lv('TNX').pct252*100)} · 5d z ${f2(lv('TNX').z5)} · 21d z ${f2(lv('TNX').z21)}`],
 ['Rules live',live.length?live.map(k=>RM[k][1]).join(', '):'none',recent.length?`last 10d: ${recent.map(k=>RM[k][1]).join(', ')}`:'none in last 10 sessions'],
 ['Dimension tags, 5d',dims.length?dims.join(', '):'none','which dimensions printed with VIX'],
];
document.getElementById('tiles').innerHTML=tiles.map(([k,v,d])=>`<div class="tile"><div class="k">${k}</div><div class="v num">${v}</div><div class="d">${d}</div></div>`).join('');
// ---- board
const board=document.getElementById('board');
board.innerHTML=Object.keys(RM).map(k=>{
  const [tag,name,def,stat]=RM[k]; const arr=D.rules[k];
  const st=arr[last]?'live':arr.slice(-10).some(v=>v)?'recent':'quiet';
  const hist=D.rule_history[k]; const lastFire=hist.length?hist[hist.length-1]:'never';
  const cnt=hist.length;
  return `<div class="rule ${tag}"><div class="t"><span class="tag">${tag}</span><span class="state ${st}">${st==='live'?'LIVE':st==='recent'?'last 10d':'quiet'}</span></div><div class="name">${name}</div><div class="def">${def}</div><div class="stat">${stat}</div><div class="last">last first-fire ${lastFire} · ${cnt} first-fires total</div></div>`;
}).join('');
// ---- tape table
const zc=v=>v==null?'<span class="z"></span>':`<span class="z ${v>=1?'hot':v<=-1?'cold':''}">${f2(v)}</span>`;
const tape=document.getElementById('tape');
tape.innerHTML=`<tr><th>series</th><th>level</th><th>pct 252d</th><th>ROC 1d</th><th>ROC 3d</th><th>ROC 5d</th><th>ROC 10d</th><th>ROC 21d</th><th>z 3d</th><th>z 5d</th><th>z 10d</th><th>z 21d</th></tr>`+
 D.levels.map(r=>`<tr><td>${r.series}</td><td class="num">${r.level}</td><td class="num">${Math.round(r.pct252*100)}</td><td class="num">${f2(r.roc1)}</td><td class="num">${f2(r.roc3)}</td><td class="num">${f2(r.roc5)}</td><td class="num">${f2(r.roc10)}</td><td class="num">${f2(r.roc21)}</td><td class="num">${zc(r.z3)}</td><td class="num">${zc(r.z5)}</td><td class="num">${zc(r.z10)}</td><td class="num">${zc(r.z21)}</td></tr>`).join('');
document.getElementById('ratios').innerHTML=`<tr><th>ratio</th><th>level</th><th>pct 252d</th><th>ROC z 3d</th><th>ROC z 5d</th></tr>`+
 D.ratios.map(r=>`<tr><td>${r.ratio}</td><td class="num">${r.level}</td><td class="num">${Math.round(r.pct252*100)}</td><td class="num">${zc(r.roc3z)}</td><td class="num">${zc(r.roc5z)}</td></tr>`).join('');
// ---- SPY chart
const OFFRULES=['SETUP_rates','SETUP_complac','ONSET_impulse','ONSET_vvixlag','ONSET_movelag','ONSET_ovxdiv','CONT_2ndleg','FAIL_yieldsup'];
const ONRULES=['CAP_alldims','CAP_vix_gvz','CAP_vvixout','ONCONF_collapse'];
let shown=new Set([...OFFRULES,...ONRULES]);
const ctl=document.getElementById('chartctl');
[...OFFRULES,...ONRULES].forEach(k=>{const b=document.createElement('button');b.id='btn_'+k;b.textContent=RM[k][1];b.className='on';b.onclick=()=>{shown.has(k)?shown.delete(k):shown.add(k);b.classList.toggle('on');drawSpy();};ctl.appendChild(b);});
const tip=document.getElementById('tip');
function showTip(e,html){tip.innerHTML=html;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px';}
function hideTip(){tip.style.display='none';}
function drawSpy(){
  const svg=document.getElementById('spychart'); const W=1100,H=380,L=56,R=16,T=14,B=30;
  const ys=D.spy; const mn=Math.min(...ys)*0.99, mx=Math.max(...ys)*1.01;
  const x=i=>L+(W-L-R)*i/(N-1), y=v=>T+(H-T-B)*(1-(v-mn)/(mx-mn));
  let s='';
  const ticks=5; for(let t=0;t<=ticks;t++){const v=mn+(mx-mn)*t/ticks;s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${css('--grid')}" stroke-width="1"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="${css('--muted')}" font-family="${css('--mono')}">${v.toFixed(0)}</text>`;}
  for(let i=0;i<N;i+=63){s+=`<text x="${x(i)}" y="${H-10}" font-size="11" fill="${css('--muted')}" text-anchor="middle" font-family="${css('--mono')}">${D.dates[i].slice(0,7)}</text>`;}
  s+=`<path d="${ys.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join('')}" fill="none" stroke="${css('--ink2')}" stroke-width="1.6"/>`;
  // fire markers: first-fires only, risk-off above the line, risk-on below
  const marks=[];
  for(const k of shown){const a=D.rules[k];for(let i=1;i<N;i++){if(a[i]&&!a[i-1])marks.push({i,k,off:OFFRULES.includes(k)});}}
  marks.forEach(m=>{const cx=x(m.i),cy=y(ys[m.i]); const col=m.off?css('--orange'):css('--blue'); const dy=m.off?-14:14;
    s+=`<g class="mk" data-i="${m.i}" data-k="${m.k}"><circle cx="${cx}" cy="${cy+dy}" r="9" fill="transparent"/><path d="${m.off?`M${cx} ${cy+dy-5}L${cx+5} ${cy+dy+4}L${cx-5} ${cy+dy+4}Z`:`M${cx} ${cy+dy+5}L${cx+5} ${cy+dy-4}L${cx-5} ${cy+dy-4}Z`}" fill="${col}" stroke="${css('--surface')}" stroke-width="1.5"/></g>`;});
  // last point
  s+=`<circle cx="${x(last)}" cy="${y(ys[last])}" r="4" fill="${css('--ink')}" stroke="${css('--surface')}" stroke-width="2"/>`;
  svg.innerHTML=s;
  svg.querySelectorAll('.mk').forEach(g=>{g.addEventListener('mousemove',e=>{const i=+g.dataset.i;showTip(e,`${D.dates[i]}<br>${RM[g.dataset.k][1]}<br>SPY ${ys[i].toFixed(2)} · VIX ${D.vix[i].toFixed(2)}`);});g.addEventListener('mouseleave',hideTip);});
  svg.onmousemove=e=>{if(e.target.closest('.mk'))return;const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;const i=Math.max(0,Math.min(N-1,Math.round((px-L)/(W-L-R)*(N-1))));showTip(e,`${D.dates[i]}<br>SPY ${ys[i].toFixed(2)} · VIX ${D.vix[i].toFixed(2)}`);};
  svg.onmouseleave=hideTip;
}
drawSpy();
// ---- z chart
const ZSETS={'VIX vs VVIX/VIX (onset discriminator)':['VIX','VVIX/VIX'],'VIX vs TNX (setup)':['VIX','TNX'],'VIX vs MOVE (confirm / dimension)':['VIX','MOVE'],'VIX, GVZ, MOVE (capitulation)':['VIX','GVZ','MOVE'],'VIX 21d trend vs 5d impulse':['VIX21','VIX'],'VIX vs VIX/VIX3M (break)':['VIX','VIX/VIX3M'],'VIX vs OVX':['VIX','OVX'],'VIX vs VIX9D/VIX':['VIX','VIX9D/VIX'],'VIX vs SKEW (10d)':['VIX','SKEW']};
const zctl=document.getElementById('zctl'); const zsel=document.createElement('select'); zsel.id='zsel';
Object.keys(ZSETS).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;zsel.appendChild(o);}); zctl.appendChild(zsel);
const ZCOL=['--blue','--orange','--teal','--violet'];
function drawZ(){
  const keys=ZSETS[zsel.value]; const svg=document.getElementById('zchart'); const W=1100,H=300,L=40,R=120,T=12,B=28;
  const x=i=>L+(W-L-R)*i/(N-1); const lo=-3.5,hi=3.5; const y=v=>T+(H-T-B)*(1-(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo));
  let s='';
  [-3,-2,-1,0,1,2,3].forEach(v=>{s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${v===0?css('--line'):css('--grid')}" stroke-width="${v===0?1.5:1}"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="${css('--muted')}" font-family="${css('--mono')}">${v}</text>`;});
  s+=`<rect x="${L}" y="${y(hi)}" width="${W-L-R}" height="${y(1)-y(hi)}" fill="${css('--orange')}" opacity=".06"/><rect x="${L}" y="${y(-1)}" width="${W-L-R}" height="${y(lo)-y(-1)}" fill="${css('--blue')}" opacity=".06"/>`;
  for(let i=0;i<N;i+=63){s+=`<text x="${x(i)}" y="${H-8}" font-size="11" fill="${css('--muted')}" text-anchor="middle" font-family="${css('--mono')}">${D.dates[i].slice(0,7)}</text>`;}
  keys.forEach((k,j)=>{const a=D.z[k];let d='',pen=false;for(let i=0;i<N;i++){if(a[i]==null){pen=false;continue;}d+=(pen?'L':'M')+x(i).toFixed(1)+' '+y(a[i]).toFixed(1);pen=true;}s+=`<path d="${d}" fill="none" stroke="${css(ZCOL[j])}" stroke-width="1.6"/>`;const li=[...a].reverse().findIndex(v=>v!=null);if(li>=0){const i=N-1-li;s+=`<text x="${x(i)+6}" y="${y(a[i])+4}" font-size="11" fill="${css('--ink2')}" font-family="${css('--mono')}">${k} ${f2(a[i])}</text>`;}});
  svg.innerHTML=s;
  document.getElementById('zlegend').innerHTML=keys.map((k,j)=>`<span style="--c:var(${ZCOL[j]})">${k}${k==='SKEW'?' (10d ROC z)':k==='VIX/VIX3M'?' (3d ROC z)':k==='VIX21'?' (21d ROC z)':k==='TNX'?' (5d chg z)':' (5d ROC z)'}</span>`).join('');
  svg.onmousemove=e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;const i=Math.max(0,Math.min(N-1,Math.round((px-L)/(W-L-R)*(N-1))));showTip(e,`${D.dates[i]}<br>`+keys.map(k=>`${k} ${f2(D.z[k][i])}`).join('<br>'));};
  svg.onmouseleave=hideTip;
}
zsel.onchange=drawZ; drawZ();
// ---- playbook
const pbsel=document.getElementById('pbsel'); Object.keys(D.playbook).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=RM[k]?RM[k][1]:k;pbsel.appendChild(o);});
function drawPB(){
  const p=D.playbook[pbsel.value]; const n=p.tickers.length; const half=Math.ceil(n/2);
  document.getElementById('pbn').textContent=`${Math.max(...p.n)} fires`;
  const mx=Math.max(...p.rel10.map(v=>Math.abs(v||0)))||1;
  const row=i=>{const v=p.rel10[i]||0;const w=Math.abs(v)/mx*50;const left=v<0?50-w:50;const col=v<0?'var(--orange)':'var(--blue)';return `<div class="bar"><span class="num" style="font-weight:600">${p.tickers[i]}</span><div class="trk"><div class="fill" style="left:${left}%;width:${w}%;background:${col}"></div><div style="position:absolute;left:50%;top:-2px;height:14px;width:1px;background:var(--line)"></div></div><span class="num" style="text-align:right">${f2(v)}%</span><span class="num" style="color:var(--muted);text-align:right">${p.hit10[i]==null?'':Math.round(p.hit10[i]*100)+'%'}</span></div>`;};
  document.getElementById('pb').innerHTML=`<div class="col">${Array.from({length:half},(_,i)=>row(i)).join('')}</div><div class="col">${Array.from({length:n-half},(_,i)=>row(i+half)).join('')}</div>`;
}
pbsel.onchange=drawPB; drawPB();
// ---- ticker side
const TK=D.tick;
if(TK){
  const tb=document.getElementById('tickboard');
  tb.innerHTML=Object.entries(TK.tick_rules).map(([k,r])=>{const st=r.live?'live':r.recent?'recent':'quiet';const bear=r.P_off!=null&&r.P_off>0.174;
    return `<div class="rule ${bear?'ONSET':'CAP'}"><div class="t"><span class="tag">PAIR</span><span class="state ${st}">${st==='live'?'LIVE':st==='recent'?'last 10d':'quiet'}</span></div><div class="name">${k.replace(/_/g,' ')}</div><div class="def">${r.def}</div><div class="stat">${r.note} P(5% DD/21d) ${r.P_off==null?'n/a':r.P_off.toFixed(2)} vs 0.17 · fwd21 ${r.fwd21==null?'':f2(r.fwd21)+'%'} · n=${r.n}</div><div class="last">last first-fire ${r.last}</div></div>`;}).join('');
  const Lx=TK.lead; const pr=TK.pred;
  const rows=TK.tape.slice().sort((a,b)=>(a.z5??0)-(b.z5??0));
  document.getElementById('ticktape').innerHTML=`<tr><th>ticker</th><th>rel 5d %</th><th>rel 10d %</th><th>rel 21d %</th><th>z 5d</th><th>z 10d</th><th>z 21d</th><th>peak -3</th><th>peak 0</th><th>trough 0</th><th>trough +3</th><th>P(off) top/bottom Q</th></tr>`+
    rows.map(r=>`<tr><td>${r.t}</td><td class="num">${f2(r.rel5)}</td><td class="num">${f2(r.rel10)}</td><td class="num">${f2(r.rel21)}</td><td class="num">${zc(r.z5)}</td><td class="num">${zc(r.z10)}</td><td class="num">${zc(r.z21)}</td><td class="num">${f2(Lx.peak_m3[r.t])}</td><td class="num">${f2(Lx.peak_0[r.t])}</td><td class="num">${f2(Lx.trough_0[r.t])}</td><td class="num">${f2(Lx.trough_p3[r.t])}</td><td class="num">${pr[r.t]?pr[r.t].topQ.toFixed(2)+' / '+pr[r.t].bottomQ.toFixed(2):''}</td></tr>`).join('');
  function heat(id,M,scale){
    const mx=scale; const cell=v=>{if(v==null)return '<td></td>';const a=Math.min(1,Math.abs(v)/mx);const col=v>0?css('--blue'):css('--orange');const bg=`color-mix(in srgb, ${col} ${Math.round(a*70)}%, var(--surface))`;const ink=a>0.5?'#fff':'var(--ink)';return `<td><span class="hc" style="background:${bg};color:${ink}">${f2(v)}</span></td>`;};
    document.getElementById(id).innerHTML=`<tr><th>ticker</th>${M.cols.map(c=>`<th>${c}</th>`).join('')}</tr>`+M.rows.map((r,i)=>`<tr><td>${r}</td>${M.vals[i].map(cell).join('')}</tr>`).join('');
  }
  heat('heat',TK.mat,1.0); heat('beta',TK.beta,12);
}
// ---- exposure dial
const SC=D.score;
if(SC){
  const v=SC.score; document.getElementById('scorebig').textContent=v;
  const bk=SC.buckets.find(b=>{const [lo,hi]=b.b.split('-').map(Number);return v>=lo&&v<=hi;});
  const names={'0-30':'drawdown regime','31-45':'reduced','46-55':'neutral','56-65':'constructive','66-75':'long','76-100':'max long'};
  const lab=bk?`${bk.b}: ${names[bk.b]}. Since 2008 this band had ${Math.round(bk.P_off*100)}% odds of a 5% drop within 21d, mean next 21d ${bk.fwd21>0?'+':''}${bk.fwd21.toFixed(1)}%, mean 63d max drawdown ${bk.DD63.toFixed(1)}%.`+(SC.cap_active?' Drawdown cap active (SPY 5%+ off its high, no confirmation yet).':'') : '';
  document.getElementById('scorelabel').textContent=lab;
  const parts=Object.entries(SC.components).sort((a,b)=>a[1]-b[1]);
  document.getElementById('scoreparts').innerHTML=`<div>base ${SC.base}</div>`+parts.map(([k,p])=>`<div><span style="color:${p<0?'var(--orange)':'var(--blue)'};font-weight:600">${p>0?'+':''}${p}</span> ${k.replace(/_/g,' ')}</div>`).join('');
  const svg=document.getElementById('scorechart'); const W=700,H=170,Lm=34,Rm=10,T=10,B=24; const h=SC.hist; const n=h.length;
  const x=i=>Lm+(W-Lm-Rm)*i/(n-1), y=q=>T+(H-T-B)*(1-q/100);
  let g='';
  [0,30,45,55,65,75,100].forEach(q=>{g+=`<line x1="${Lm}" x2="${W-Rm}" y1="${y(q)}" y2="${y(q)}" stroke="${css('--grid')}"/><text x="${Lm-5}" y="${y(q)+4}" text-anchor="end" font-size="10" fill="${css('--muted')}" font-family="${css('--mono')}">${q}</text>`;});
  g+=`<rect x="${Lm}" y="${y(100)}" width="${W-Lm-Rm}" height="${y(65)-y(100)}" fill="${css('--blue')}" opacity=".06"/><rect x="${Lm}" y="${y(45)}" width="${W-Lm-Rm}" height="${y(0)-y(45)}" fill="${css('--orange')}" opacity=".06"/>`;
  for(let i=0;i<n;i+=63){g+=`<text x="${x(i)}" y="${H-6}" font-size="10" fill="${css('--muted')}" text-anchor="middle" font-family="${css('--mono')}">${D.dates[D.dates.length-n+i].slice(0,7)}</text>`;}
  g+=`<path d="${h.map((q,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(q).toFixed(1)).join('')}" fill="none" stroke="${css('--ink2')}" stroke-width="1.5"/><circle cx="${x(n-1)}" cy="${y(h[n-1])}" r="4" fill="${v<=45?css('--orange'):css('--blue')}" stroke="${css('--surface')}" stroke-width="2"/>`;
  svg.innerHTML=g;
  svg.onmousemove=e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;const i=Math.max(0,Math.min(n-1,Math.round((px-Lm)/(W-Lm-Rm)*(n-1))));showTip(e,`${D.dates[D.dates.length-n+i]}<br>dial ${h[i]}`);}; svg.onmouseleave=hideTip;
  document.getElementById('scorebuckets').innerHTML=`Drawdown-first tuning. Holding SPY at dial/100 since 2008: ${SC.strategy.ann}% a year, max drawdown ${SC.strategy.maxDD}%, vs buy-and-hold ${SC.buyhold.ann}% and ${SC.buyhold.maxDD}%. By band, P(5% DD in 21d) / mean fwd 21d: `+SC.buckets.map(b=>`${b.b}: ${b.P_off.toFixed(2)} / ${b.fwd21>0?'+':''}${b.fwd21.toFixed(1)}%`).join(' · ');
}
// ---- realized vol layer
const RVD=D.rv;
if(RVD){
  const spy=RVD.rv_tape.find(r=>r.t==='SPY'); const ivspy=RVD.ivrv.find(r=>r.pair==='SPY:VIX/RV'); const ivhyg=RVD.ivrv.find(r=>r.pair==='HYG:VIX/RV'); const ivtlt=RVD.ivrv.find(r=>r.pair==='TLT:MOVE/RV');
  document.getElementById('rvtiles').innerHTML=[
    ['SPY RV10 / RV21',`${spy.rv10} / ${spy.rv21}`,`term structure ${spy.rvr} · z ${f2(spy.rvr_z)}`],
    ['VIX / SPY RV21',ivspy.level.toFixed(2),`pct ${Math.round(ivspy.pct252*100)} · ROC5 z ${f2(ivspy.roc5z)}`],
    ['VIX / HYG RV21',ivhyg.level.toFixed(2),`pct ${Math.round(ivhyg.pct252*100)} · bottom-Q P(off) 0.31`],
    ['MOVE / TLT RV21',ivtlt.level.toFixed(2),`pct ${Math.round(ivtlt.pct252*100)} · ROC5 z ${f2(ivtlt.roc5z)}`],
    ['Realized-vol breadth',`${Math.round(RVD.breadth.now*100)}%`,`share of 38 with RV10/RV21 > 1.2 · z ${f2(RVD.breadth.z)} · peaks at +1..+3 after lows`],
  ].map(([k,v,d])=>`<div class="tile"><div class="k">${k}</div><div class="v num">${v}</div><div class="d">${d}</div></div>`).join('');
  document.getElementById('rvboard').innerHTML=Object.entries(RVD.rv_rules).map(([k,r])=>{const st=r.live?'live':r.recent?'recent':'quiet';const bear=r.P_off!=null&&r.P_off>0.174;
    return `<div class="rule ${bear?'ONSET':'CAP'}"><div class="t"><span class="tag">RV</span><span class="state ${st}">${st==='live'?'LIVE':st==='recent'?'last 10d':'quiet'}</span></div><div class="name">${k.replace(/_/g,' ')}</div><div class="def">${r.def}</div><div class="stat">${r.note} P(5% DD/21d) ${r.P_off==null?'n/a':r.P_off.toFixed(2)} vs 0.17 · fwd21 ${r.fwd21==null?'':f2(r.fwd21)+'%'} · n=${r.n}</div><div class="last">last first-fire ${r.last}</div></div>`;}).join('');
  document.getElementById('ivrv').innerHTML=`<tr><th>pair</th><th>level</th><th>pct 252d</th><th>level z</th><th>ROC5 z</th></tr>`+RVD.ivrv.map(r=>`<tr><td>${r.pair}</td><td class="num">${r.level==null?'':r.level.toFixed(2)}</td><td class="num">${r.pct252==null?'':Math.round(r.pct252*100)}</td><td class="num">${zc(r.lvl_z)}</td><td class="num">${zc(r.roc5z)}</td></tr>`).join('');
  const rows=RVD.rv_tape.slice().sort((a,b)=>(b.rvr_z??-9)-(a.rvr_z??-9));
  document.getElementById('rvtape').innerHTML=`<tr><th>ticker</th><th>RV10</th><th>RV21</th><th>RV10/RV21</th><th>z</th><th>RV10 ROC5 z</th><th>rel RV</th><th>rel RV ROC5 z</th><th>P(off) RVR top/bot Q</th><th>exp · VIX up</th><th>exp · VIX dn</th><th>cmp · VIX up</th><th>cmp · VIX dn</th></tr>`+
    rows.map(r=>`<tr><td>${r.t}</td><td class="num">${r.rv10??''}</td><td class="num">${r.rv21??''}</td><td class="num">${r.rvr??''}</td><td class="num">${zc(r.rvr_z)}</td><td class="num">${zc(r.rv10_roc5z)}</td><td class="num">${r.relrv==null?'':r.relrv.toFixed(2)}</td><td class="num">${zc(r.relrv_roc5z)}</td><td class="num">${r.P_top==null?'':r.P_top.toFixed(2)+' / '+r.P_bot.toFixed(2)}</td><td class="num">${f2(r.exp_vixup)}</td><td class="num">${f2(r.exp_vixdn)}</td><td class="num">${f2(r.cmp_vixup)}</td><td class="num">${f2(r.cmp_vixdn)}</td></tr>`).join('');
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{drawSpy();drawZ();if(TK){heat('heat',TK.mat,1.0);heat('beta',TK.beta,12);}});
</script>
"""
html = HTML.replace('__DATA__', json.dumps(DATA).replace('</', '<\\/'))
open(out, 'w').write(html)
print('wrote', out, len(html)//1024, 'KB')
