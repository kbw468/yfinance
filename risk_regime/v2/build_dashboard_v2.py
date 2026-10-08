#!/usr/bin/env python3
"""v2 dashboard: static HTML from v2_data.json. Usage: python3 build_dashboard_v2.py v2_data.json out.html
Blue / orange only (deutan-safe), light and dark, phone-safe, print-styled for the PDF."""
import json, sys, os
src, out = sys.argv[1], sys.argv[2]
D = json.load(open(src))
HTML = r"""<title>Vol Tape Regime v2</title>
<meta name="description" content="Out-of-sample volatility-regime reading: drawdown, rally and vol-expansion probabilities from 15 vol and rate indices and 38 ETFs">
<style>
:root{--bg:#f7f7f4;--surface:#ffffff;--ink:#1b1f26;--ink2:#4b5563;--muted:#8a919c;--line:#e3e5e8;--grid:#eef0f2;--blue:#1d4ed8;--orange:#ea580c;--teal:#0891b2;--blue-soft:#dbe7ff;--orange-soft:#ffe4d1;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141517;--surface:#1c1e22;--ink:#ebedf0;--ink2:#b3b8c2;--muted:#7c838e;--line:#2c3037;--grid:#23262b;--blue:#3b82f6;--orange:#ea580c;--teal:#0891b2;--blue-soft:#1b2a4a;--orange-soft:#4a2a14;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141517;--surface:#1c1e22;--ink:#ebedf0;--ink2:#b3b8c2;--muted:#7c838e;--line:#2c3037;--grid:#23262b;--blue:#3b82f6;--orange:#ea580c;--teal:#0891b2;--blue-soft:#1b2a4a;--orange-soft:#4a2a14;color-scheme:dark}
*{box-sizing:border-box} .wrap>*{min-width:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-inline:16px;display:grid;gap:22px}
h1{font-family:var(--mono);font-weight:600;font-size:20px;margin:0} h2{font-family:var(--mono);font-weight:600;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:0 0 10px}
.sub{color:var(--muted);font-family:var(--mono);font-size:12px} .num{font-family:var(--mono);font-variant-numeric:tabular-nums} .note{font-size:12.5px;color:var(--ink2);max-width:78ch}
header{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px} .tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px;min-width:0}
.tile .k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-family:var(--mono)} .tile .v{font-size:22px;font-weight:600;margin-top:2px;font-family:var(--mono)} .tile .d{font-size:12px;color:var(--ink2);margin-top:2px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px;min-width:0}
.grid2{display:grid;grid-template-columns:minmax(240px,1fr) 2fr;gap:16px;align-items:start} .grid2>div{min-width:0} @media (max-width:720px){.grid2{grid-template-columns:1fr}}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px} .grid3>div{min-width:0}
table{border-collapse:collapse;width:100%;font-size:12.5px} .tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
th,td{padding:5px 8px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap} th{font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
td:first-child,th:first-child{text-align:left;font-family:var(--mono);font-weight:600;position:sticky;left:0;background:var(--surface)} tr:last-child td{border-bottom:none} tr.hl td{background:var(--blue-soft)} tr.pre td{background:var(--orange-soft)}
.z{display:inline-block;min-width:44px;padding:1px 5px;border-radius:3px} .z.hot{background:var(--orange-soft);color:var(--orange);font-weight:600} .z.cold{background:var(--blue-soft);color:var(--blue);font-weight:600}
.bar{display:grid;grid-template-columns:minmax(150px,1.4fr) 1fr 50px 44px;gap:8px;align-items:center;font-size:12px;padding:2px 0} .bar .trk{height:10px;position:relative;background:var(--grid);border-radius:2px} .bar .fill{position:absolute;top:0;height:100%;border-radius:2px}
.state{display:inline-block;padding:2px 10px;border-radius:999px;font-family:var(--mono);font-size:12px;font-weight:600;border:1px solid var(--line)} .state.out{background:var(--orange-soft);color:var(--orange);border-color:var(--orange)} .state.in{background:var(--blue-soft);color:var(--blue);border-color:var(--blue)}
svg{display:block;width:100%;height:auto;max-width:100%} .legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--ink2);margin-bottom:6px} .legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:var(--c)}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:11.5px;color:var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;z-index:9;white-space:nowrap}
:focus-visible{outline:2px solid var(--blue);outline-offset:2px} @media (prefers-reduced-motion: reduce){*{transition:none!important}}
@media print{@page{size:Letter;margin:10mm} body{font-size:10px} .wrap{gap:12px;padding:0;max-width:none} .pb{page-break-before:always} section{page-break-inside:auto} tr{page-break-inside:avoid} .tablewrap{overflow:visible;border:none} table{font-size:8px} th,td{padding:2px 4px;white-space:normal} td:first-child,th:first-child{position:static;white-space:nowrap} .grid2,.grid3{grid-template-columns:1fr} #drivers.grid3{grid-template-columns:repeat(3,1fr)} .tiles{grid-template-columns:repeat(4,1fr)} .tile .v{font-size:16px} .tip{display:none!important} svg{max-height:240px} .bar{font-size:10px} .note{max-width:none}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<div class="wrap">
<header><div><h1>Vol Tape Regime v2</h1><div class="sub">15 vol and rate indices and 38 ETFs, rank-based features, monotone trees refit yearly; every number on this page is out of sample. As of <span id="asof" class="num"></span>. Feb to Jul 2020 excluded.</div></div><div class="sub">orange = risk-off side, blue = risk-on side</div></header>

<section id="reading"><h2>Reading</h2>
<div class="card grid2">
  <div><div id="statebig" style="font-family:var(--mono);font-size:30px;font-weight:600;line-height:1.1"></div><div id="statesub" class="sub" style="margin-top:4px"></div><div id="dialbig" class="num" style="font-size:44px;font-weight:600;line-height:1;margin-top:10px"></div><div id="diallabel" class="sub" style="margin-top:4px;max-width:40ch"></div></div>
  <div class="tiles" id="probtiles"></div>
</div>
<p class="note" id="readingnote"></p>
</section>

<section><h2>Drawdown probability, last two years, with the trailing thresholds the rule uses</h2>
<div class="card"><div class="legend"><span style="--c:var(--orange)">P(5% drawdown within 21 sessions), out of sample</span><span style="--c:var(--blue)">trailing 95th / 80th percentile</span><span style="--c:var(--ink2)">SPY</span><span style="--c:var(--orange-soft)">OUT</span></div><svg id="pchart" viewBox="0 0 1100 320" role="img" aria-label="Out-of-sample drawdown probability with thresholds and SPY"></svg></div>
</section>

<section><h2>Why today: the features inside each model and where they sit against their own history</h2>
<div class="grid3" id="drivers"></div>
<p class="note">Each probability comes from 15 features chosen inside the last training window. Bars show each feature's trailing-rank distance from its median, signed by the direction the model is constrained to use: orange pushes the probability up, blue pushes it down.</p>
</section>

<section class="pb"><h2>What each probability meant, out of sample, by decile (2005 to date)</h2>
<div class="grid3" id="deciles"></div>
</section>

<section><h2>Two-state stay-in rules on the probabilities, fully out of sample, 2006 to date</h2>
<div class="tablewrap"><table id="rules"></table></div>
<p class="note" id="rulesnote"></p>
<div class="grid2" style="margin-top:12px"><div class="tablewrap"><table id="yearly"></table></div><div class="card"><div class="sub">OUT spells of the reference rule (SPY move while out)</div><div id="spells" style="font-family:var(--mono);font-size:12px;margin-top:6px;line-height:1.6"></div></div></div>
<div class="tablewrap" style="margin-top:12px"><table id="addbook"></table></div>
<p class="note" id="addnote"></p>
<div class="tablewrap" style="margin-top:12px"><table id="p2020"></table></div>
<p class="note" id="p2020note"></p>
<div class="tablewrap" style="margin-top:12px"><table id="early"></table></div>
<p class="note" id="earlynote"></p>
</section>

<section class="pb"><h2>15% declines: the loss that matters</h2>
<div class="tiles" id="t15"></div>
<p class="note" id="note15"></p>
<div class="tablewrap" style="margin-top:10px"><table id="events15"></table></div>
<p class="note">Every SPY peak-to-trough decline of 15% or more since 2005 (zigzag on closes; a 10% rebound or a new high closes an event; 2020 excluded). For each out-of-sample probability: where it sat against its own trailing three years at the peak, and the first session at or after the peak where it crossed its trailing 95th percentile, with the decline already in place by then.</p>
<div class="grid2" style="margin-top:12px"><div class="card" id="drv15"></div><div class="tablewrap"><table id="dec15"></table></div></div>
<div class="tablewrap" style="margin-top:10px"><table id="rules15"></table></div>
<p class="note" id="rules15note"></p>
<div class="tablewrap" style="margin-top:10px"><table id="onset"></table></div>
<p class="note">Onset rules: a vol-expansion alert (P_vol above its trailing 90th percentile at some point in the last 42 sessions) that becomes OUT only once SPY sits a given percent below its 63-session closing high, against the same price trigger with no vol filter. Re-entry on calmed probabilities, on a new 42-session closing high, or on either. "taken" is the mean share of the eleven 15% declines the book took; "false" counts OUT spells not followed by any 10% decline.</p>
</section>

<section class="pb"><h2>Robustness: model capacity, raw single features, year-block bootstrap, and a permutation test of the whole pipeline</h2>
<div class="tablewrap"><table id="variants"></table></div>
<div class="tiles" id="nulltiles" style="margin-top:10px"></div>
<p class="note">The permutation test shifts the label series by a random offset of at least a year and reruns the entire walk-forward pipeline, feature selection included; the resulting AUC is what selection and fitting produce from nothing.</p>
</section>

<section><h2>Why version 1 was retired: every v1 rule by sample half</h2>
<div class="tablewrap"><table id="audit"></table></div>
</section>

<section class="pb"><h2>Features that carried the same sign in every era (to 2012, 2013-19, 2020-26)</h2>
<div class="grid3" id="screen"></div>
</section>

<section><h2>Vol episodes and multifractal state</h2>
<div class="grid2"><div class="tablewrap"><table id="episodes"></table></div><div class="tablewrap"><table id="mf"></table></div></div>
<p class="note">Episodes: realized = SPY 21-day realized vol crossing above its trailing-504 80th percentile, ending below the 50th; implied = VIX level rank likewise. Multifractal: rolling 504-session MF-DFA; h2 is the Hurst exponent of returns, width = h(-4) - h(4), alpha_abs the DFA exponent of absolute returns; vr5 and vr21 are Lo-MacKinlay variance ratios.</p>
</section>

<section><h2>Tape today, as trailing ranks (0 = lowest of the last two years, 1 = highest)</h2>
<div class="tiles" id="tape"></div>
</section>

<section class="pb"><h2>Ticker layer: behaviour around 76 vol-episode starts and under high dispersion</h2>
<div class="grid2"><div class="tablewrap"><table id="lead"></table></div><div class="tablewrap"><table id="cond"></table></div></div>
</section>
<footer class="note" style="border-top:1px solid var(--line);padding-top:8px">Method: features are rates of change, ratios and trailing-504 percentile ranks built from daily OHLCV (close-to-close, Parkinson and Garman-Klass realized vol; volume and the volatility of volume; range compression; cross-sectional dispersion and correlation; implied over realized; equal-weight breadth; MF-DFA spectra). Models: 15 features chosen inside each training window by same-signed rank correlation in both halves, depth-2 gradient-boosted trees constrained to be monotone in that sign, refit each January on data whose labels end before the test year. Thresholds are trailing quantiles of past out-of-sample values. KRE-based features are excluded from the candidate set by the user's direction. Nothing here assumes a distribution.</footer>
</div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent); const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); const f2=x=>x==null?'':(x>0?'+':'')+Number(x).toFixed(2); const f1=x=>x==null?'':(x>0?'+':'')+Number(x).toFixed(1); const p3=x=>x==null?'':Number(x).toFixed(3); const pc=x=>x==null?'':Math.round(x*100);
document.getElementById('asof').textContent=D.asof+(D.stale&&D.stale.length?' · stale: '+D.stale.map(s=>s.series+' ('+s.last+')').join(', '):'');
const P=D.probs; const isOut=D.state==='OUT'; const col=isOut?'var(--orange)':'var(--blue)';
const pct=P.P_off.trailing_pct; const dial=Math.round(100*(1-pct));
document.getElementById('statebig').innerHTML=`<span style="color:${col}">${D.state}</span>`;
document.getElementById('statesub').textContent=`since ${D.state_since} · rule: OUT when the drawdown probability is above its trailing-756 95th percentile, back IN once below the 80th`;
document.getElementById('dialbig').textContent=dial;
document.getElementById('diallabel').textContent=`stay-in dial = 100 minus the trailing percentile of the drawdown probability; the rule goes OUT at 5 or below and returns IN above 20`;
const dec=(p)=>P[p].deciles[P[p].today_decile];
document.getElementById('probtiles').innerHTML=[
 ['P(5% drawdown in 21d)',p3(P.P_off.today),`trailing pct ${pc(P.P_off.trailing_pct)} · decile ${P.P_off.today_decile+1}: historically ${pc(dec('P_off').P_off)}% drew down 5%, mean next month ${f2(dec('P_off').fwd21)}%`],
 ['P(5% rally in 21d)',p3(P.P_on.today),`trailing pct ${pc(P.P_on.trailing_pct)} · decile ${P.P_on.today_decile+1}: historically ${pc(dec('P_on').P_on)}% rallied 5%, ${pc(dec('P_on').P_off)}% drew down 5%`],
 ['P(vol expands 1.5x in 21d)',p3(P.P_vol.today),`trailing pct ${pc(P.P_vol.trailing_pct)} · decile ${P.P_vol.today_decile+1}: historically ${pc(dec('P_vol').P_volexp)}% expanded, mean next month ${f2(dec('P_vol').fwd21)}%`],
].map(([k,v,d])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('');
document.getElementById('readingnote').textContent=`${D.inside_episode?'Inside a vol episode.':'Not inside a vol episode.'} The drawdown probability sits at the ${pc(pct)}th percentile of its own last three years; the rally probability at the ${pc(P.P_on.trailing_pct)}th; the vol-expansion probability at the ${pc(P.P_vol.trailing_pct)}th. High readings of all three travel together (high vol is a two-tailed state), which is why the stay-in rule keys on the drawdown probability alone and re-enters quickly.`;
// ---- chart
(function(){const svg=document.getElementById('pchart');const W=1100,H=320,L=44,R=60,T=14,B=28;const n=D.dates.length;const x=i=>L+(W-L-R)*i/(n-1);const h=P.P_off.hist,q95=P.P_off.q95,q80=P.P_off.q80,st=D.state_hist,spy=D.spy;
 const mx=Math.max(...h.filter(v=>v!=null),...q95.filter(v=>v!=null))*1.05;const y=v=>T+(H-T-B)*(1-v/mx);const smn=Math.min(...spy)*0.98,smx=Math.max(...spy)*1.02;const ys=v=>T+(H-T-B)*(1-(v-smn)/(smx-smn));let s='';
 for(let i=0;i<n;i++){if(st[i]===0)s+=`<rect x="${x(i)-(W-L-R)/(n-1)/2}" y="${T}" width="${(W-L-R)/(n-1)+0.5}" height="${H-T-B}" fill="${css('--orange')}" opacity=".10"/>`;}
 [0,0.1,0.2,0.3,0.4,0.5].filter(v=>v<=mx).forEach(v=>{s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${css('--grid')}"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="${css('--muted')}" font-family="${css('--mono')}">${v.toFixed(1)}</text>`;});
 for(let i=0;i<n;i+=63){s+=`<text x="${x(i)}" y="${H-8}" font-size="11" fill="${css('--muted')}" text-anchor="middle" font-family="${css('--mono')}">${D.dates[i].slice(0,7)}</text>`;}
 const path=(arr,yf)=>{let d='',pen=false;for(let i=0;i<n;i++){if(arr[i]==null){pen=false;continue;}d+=(pen?'L':'M')+x(i).toFixed(1)+' '+yf(arr[i]).toFixed(1);pen=true;}return d;};
 s+=`<path d="${path(spy,ys)}" fill="none" stroke="${css('--ink2')}" stroke-width="1.2" opacity=".7"/>`;
 s+=`<path d="${path(q95,y)}" fill="none" stroke="${css('--blue')}" stroke-width="1" stroke-dasharray="4 3"/><path d="${path(q80,y)}" fill="none" stroke="${css('--blue')}" stroke-width="1" stroke-dasharray="2 3"/>`;
 s+=`<path d="${path(h,y)}" fill="none" stroke="${css('--orange')}" stroke-width="1.8"/>`;
 const li=h.length-1; s+=`<circle cx="${x(li)}" cy="${y(h[li])}" r="4" fill="${css('--orange')}" stroke="${css('--surface')}" stroke-width="2"/><text x="${x(li)+6}" y="${y(h[li])+4}" font-size="11" fill="${css('--ink2')}" font-family="${css('--mono')}">${p3(h[li])}</text>`;
 s+=`<text x="${W-R+4}" y="${ys(spy[li])+4}" font-size="11" fill="${css('--ink2')}" font-family="${css('--mono')}">SPY ${spy[li].toFixed(0)}</text>`;
 svg.innerHTML=s; const tip=document.getElementById('tip');
 svg.onmousemove=e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;const i=Math.max(0,Math.min(n-1,Math.round((px-L)/(W-L-R)*(n-1))));tip.innerHTML=`${D.dates[i]}<br>P_off ${p3(h[i])} · q95 ${p3(q95[i])} · q80 ${p3(q80[i])}<br>SPY ${spy[i]} · ${st[i]?'IN':'OUT'}`;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px';}; svg.onmouseleave=()=>{tip.style.display='none';};})();
// ---- drivers
document.getElementById('drivers').innerHTML=['P_off','P_on','P_vol'].map(p=>`<div class="card"><div class="sub" style="margin-bottom:6px">${{P_off:'drawdown probability',P_on:'rally probability',P_vol:'vol-expansion probability'}[p]} · ${p3(P[p].today)}</div>`+P[p].drivers.slice(0,10).map(d=>{const v=d.push;const w=Math.abs(v)/2*100;const left=v<0?50-w:50;const c=v>0?'var(--orange)':'var(--blue)';return `<div class="bar"><span class="num" style="font-size:11px">${esc(d.feature)}</span><div class="trk"><div class="fill" style="left:${left}%;width:${w}%;background:${c}"></div><div style="position:absolute;left:50%;top:-2px;height:14px;width:1px;background:var(--line)"></div></div><span class="num" style="text-align:right">${d.rank.toFixed(2)}</span><span class="num" style="color:var(--muted);text-align:right">${d.sign>0?'+':'-'}</span></div>`;}).join('')+`</div>`).join('');
// ---- deciles
document.getElementById('deciles').innerHTML=['P_off','P_on','P_vol'].map(p=>`<div class="tablewrap"><table><tr><th>${p} decile</th><th>range</th><th>fwd 21d</th><th>mean DD</th><th>P(5% DD)</th><th>P(5% up)</th><th>P(vol x1.5)</th></tr>`+P[p].deciles.map((d,i)=>`<tr class="${i===P[p].today_decile?'hl':''}"><td>${i+1}</td><td class="num">${p3(d.p_lo)}-${p3(d.p_hi)}</td><td class="num">${f2(d.fwd21)}%</td><td class="num">${f1(d.fwdDD21)}%</td><td class="num">${p3(d.P_off)}</td><td class="num">${p3(d.P_on)}</td><td class="num">${p3(d.P_volexp)}</td></tr>`).join('')+'</table></div>').join('');
// ---- rules
const R=D.rules; document.getElementById('rules').innerHTML=`<tr><th>rule</th><th>ann %</th><th>vol %</th><th>sharpe</th><th>max DD (log) %</th><th>ulcer</th><th>exposure</th><th>spells</th><th>ann 05-12</th><th>ann 13-19</th><th>ann 20-26</th><th>DD 05-12</th><th>DD 13-19</th><th>DD 20-26</th><th>today</th></tr>`+R.map(r=>`<tr class="${r.rule===D.headline_rule?'hl':(r.rule===D.preregistered_rule?'pre':'')}"><td>${esc(r.rule)}</td><td class="num">${r.ann}</td><td class="num">${r.vol}</td><td class="num">${r.sharpe}</td><td class="num">${r.maxDD}</td><td class="num">${r.ulcer}</td><td class="num">${r.exposure}</td><td class="num">${r.spells}</td><td class="num">${r['ann 2005-12']}</td><td class="num">${r['ann 2013-19']}</td><td class="num">${r['ann 2020-26']}</td><td class="num">${r['maxDD 2005-12']}</td><td class="num">${r['maxDD 2013-19']}</td><td class="num">${r['maxDD 2020-26']}</td><td>${r.today}</td></tr>`).join('');
document.getElementById('rulesnote').innerHTML=`Blue row: the reference rule shown above. Orange row: the rule that was pre-registered before the grid was run (vol expansion likely with no rebound setup) and failed, because an expected vol expansion is a low-drawdown state out of sample. The reference rule was chosen from the pre-specified grid after the grid was run, and its re-entry threshold was moved from the 60th to the 80th percentile after the slow version sat out 2009 and 2016; both versions are in the table. Drawdowns are in log units (-80 log = -55% price).`;
const Y=D.yearly&&D.yearly.C95; if(Y){document.getElementById('yearly').innerHTML=`<tr><th>year</th>${Y.map(r=>`<th>${String(r.year).slice(0,4)}</th>`).join('')}</tr><tr><td>book %</td>${Y.map(r=>`<td class="num">${r['book %']}</td>`).join('')}</tr><tr><td>SPY %</td>${Y.map(r=>`<td class="num">${r['SPY %']}</td>`).join('')}</tr><tr><td>diff</td>${Y.map(r=>`<td class="num"><span class="z ${r.diff>=5?'cold':(r.diff<=-5?'hot':'')}">${f1(r.diff)}</span></td>`).join('')}</tr>`;}
const S=D.spells&&D.spells.C95; if(S){document.getElementById('spells').innerHTML=S.map(s=>`${s.from} to ${s.to} · ${s.sessions}d · <span style="color:${s.spy_while_out>0?'var(--orange)':'var(--blue)'}">${f1(s.spy_while_out)}%</span>`).join('<br>');}
// ---- fifteen
const FT=D.fifteen; if(FT){const a=FT.dd15_63,b=FT.dd15_126,da=a.deciles[a.today_decile],db=b.deciles[b.today_decile];const md=(FT.models||[]).find(m=>m.model==='dd15_63 K15 d2 mono')||{};
 const tile=([k,v,d])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
 document.getElementById('t15').innerHTML=[
  ['P(15% decline within 63 sessions)',p3(a.today),`trailing pct ${pc(a.trailing_pct)} · decile ${a.today_decile+1}: historically ${pc(da.hit)}% saw one (base ${pc(a.base)}%), mean worst close within 63d ${f1(da.worst)}%`],
  ['P(15% decline within 126 sessions)',p3(b.today),`trailing pct ${pc(b.trailing_pct)} · decile ${b.today_decile+1}: historically ${pc(db.hit)}% saw one (base ${pc(b.base)}%)`],
  ['dedicated 15%/63 model, out of sample',p3(md['auc all']),`AUC by era ${p3(md['auc 2005-12'])} / ${p3(md['auc 2013-19'])} / ${p3(md['auc 2020-26'])} · year-bootstrap CI ${md.ci95||''} · top quintile hit ${p3((md.quintiles||[])[4])}`],
  ['permutation test, 15%/63 pipeline',FT.null?`${p3(FT.null.real.all)} vs ${p3(FT.null.null_mean)}`:'running',FT.null?`real AUC vs the mean of ${FT.null.n} null pipelines (sd ${p3(FT.null.null_sd)}, max ${p3(FT.null.null_max)}) · z ${Number(FT.null.z).toFixed(1)}`:'results land on the next refresh']].map(tile).join('');
 const T=FT.timing||{};const tsum=k=>T[k]?`${T[k].at_peak_already_extreme} already extreme at the peak, ${T[k].crossed_before_5pct} crossed before SPY was 5% down, ${T[k].crossed_after_5pct} crossed later, ${T[k].never} never`:'';
 document.getElementById('note15').textContent=`Of the ${(T['P_off (5%/21)']||{}).events||11} declines: drawdown probability (5%/21): ${tsum('P_off (5%/21)')}. Dedicated 15%/63 probability: ${tsum('P15/63 (new)')}. Vol-expansion probability: ${tsum('P_vol')}. The "already extreme" cases are the 2008-09 legs, where the previous leg had put every probability at its ceiling.`;
 const EK=['P_off (5%/21)','P15/63 (new)','P_vol'];document.getElementById('events15').innerHTML=`<tr><th>peak</th><th>trough</th><th>depth %</th><th>sessions</th>${EK.map(k=>`<th>${esc(k)} pct at peak</th><th>first above 95th</th>`).join('')}</tr>`+(FT.events||[]).map(e=>`<tr><td>${e.peak}</td><td class="num">${e.trough}</td><td class="num">${e['depth%']}</td><td class="num">${e.sessions}</td>${EK.map(k=>{const f=String(e[k+' first>95th']);const hot=f==='never'||/SPY -(1[0-9]|[5-9])\./.test(f);return `<td class="num">${p3(e[k+' @peak'])}</td><td class="num"><span class="z ${hot?'hot':(f.includes('+0s')?'':'cold')}">${esc(f)}</span></td>`;}).join('')}</tr>`).join('');
 document.getElementById('drv15').innerHTML=`<div class="sub" style="margin-bottom:6px">15%/63 probability · ${p3(a.today)} · features inside the last fit and where they sit (orange pushes up)</div>`+a.drivers.slice(0,10).map(d=>{const v=d.push;const w=Math.abs(v)/2*100;const left=v<0?50-w:50;const c=v>0?'var(--orange)':'var(--blue)';return `<div class="bar"><span class="num" style="font-size:11px">${esc(d.feature)}</span><div class="trk"><div class="fill" style="left:${left}%;width:${w}%;background:${c}"></div><div style="position:absolute;left:50%;top:-2px;height:14px;width:1px;background:var(--line)"></div></div><span class="num" style="text-align:right">${d.rank.toFixed(2)}</span><span class="num" style="color:var(--muted);text-align:right">${d.sign>0?'+':'-'}</span></div>`;}).join('');
 document.getElementById('dec15').innerHTML=`<tr><th>P15/63 decile</th><th>range</th><th>P(15% decline in 63d)</th><th>mean worst close 63d</th><th>mean fwd 63d</th></tr>`+a.deciles.map((d,i)=>`<tr class="${i===a.today_decile?'hl':''}"><td>${i+1}</td><td class="num">${p3(d.p_lo)}-${p3(d.p_hi)}</td><td class="num">${p3(d.hit)}</td><td class="num">${f1(d.worst)}%</td><td class="num">${f1(d.fwd)}%</td></tr>`).join('');
 const RR=(FT.rules||[]).slice().sort((x,y)=>x['mean taken']-y['mean taken']);const keep=RR.filter((r,i)=>i<14||r.rule==='SPY buy and hold'||r.rule.startsWith('C: OUT P_off>q0.95, IN P_off<q0.8'));
 document.getElementById('rules15').innerHTML=`<tr><th>rule</th><th>ann %</th><th>sharpe</th><th>max DD (log)</th><th>exposure</th><th>spells</th><th>false</th><th>mean taken</th><th>worst</th><th>avoided inside 15% events</th><th>given up outside</th><th>taken per event (${(FT.events_scored||[]).length})</th></tr>`+keep.map(r=>`<tr class="${r.rule.startsWith('C: OUT P_off>q0.95, IN P_off<q0.8')?'hl':''}"><td>${esc(r.rule)}</td><td class="num">${r.ann}</td><td class="num">${r.sharpe}</td><td class="num">${r.maxDD}</td><td class="num">${r.exposure}</td><td class="num">${r.spells}</td><td class="num">${r['spells w/o any 10% decline']}</td><td class="num">${r['mean taken']}</td><td class="num">${r['worst taken']}</td><td class="num">${r['return avoided inside 15% events, total %']}</td><td class="num">${r['return given up outside 15% events, total %']}</td><td class="num" style="font-size:10px">${(r['taken per 15% event']||[]).map(x=>x==null?'':Number(x).toFixed(2)).join(' ')}</td></tr>`).join('');
 document.getElementById('rules15note').textContent=`The fourteen rules that took the least of the eleven 15% declines (events: ${(FT.events_scored||[]).join('; ')}), plus the reference rule and buy-and-hold. "taken" 1.0 = took the whole decline, 0 = sidestepped it, above 1 = whipsawed into more than the decline. Totals are log-return points over 2006 to date. Every rule that sidesteps more of the declines gives up more outside them; none clears buy-and-hold by more than the reference rule does.`;
 const ON=FT.onset||[];document.getElementById('onset').innerHTML=`<tr><th>price trigger</th><th>re-entry</th><th>control ann %</th><th>sharpe</th><th>max DD</th><th>spells</th><th>false</th><th>taken</th><th>with vol alert ann %</th><th>sharpe</th><th>max DD</th><th>spells</th><th>false</th><th>taken</th></tr>`+ON.map(o=>`<tr><td>${esc(o.trigger)}</td><td class="num">${esc(o.reentry)}</td><td class="num">${o.control.ann}</td><td class="num">${o.control.sharpe}</td><td class="num">${o.control.maxDD}</td><td class="num">${o.control.spells}</td><td class="num">${o.control.false}</td><td class="num">${o.control.taken}</td><td class="num">${o.vol.ann}</td><td class="num">${o.vol.sharpe}</td><td class="num">${o.vol.maxDD}</td><td class="num">${o.vol.spells}</td><td class="num">${o.vol.false}</td><td class="num">${o.vol.taken}</td></tr>`).join('');}
// ---- ADD as its own book
const AB=D.addbook; if(AB){const ex=p=>p.path==='ex-2020';const SL=AB.sleeve.filter(ex),BK=AB.book.filter(ex);
 document.getElementById('addbook').innerHTML=`<tr><th>ADD as a sleeve (long only on ADD sessions, rule IN; ex-2020)</th><th>ann %</th><th>sharpe</th><th>max DD</th><th>sessions long</th><th>share of IN sessions</th><th>mean daily bp while long</th><th>spells</th></tr>`+SL.map(r=>`<tr class="${r.book.includes('unconditional')?'hl':''}"><td>${esc(r.book)}</td><td class="num">${r.ann}</td><td class="num">${r.sharpe}</td><td class="num">${r.maxDD}</td><td class="num">${r['sessions long']}</td><td class="num">${r['share of IN sessions']}</td><td class="num">${r['mean daily bp while long']}</td><td class="num">${r.spells}</td></tr>`).join('')+`<tr><th>ADD as a no-leverage book (base weight when IN, 100% on ADD, 0 when OUT; ex-2020)</th><th>ann %</th><th>sharpe</th><th>max DD</th><th>mean weight</th><th></th><th></th><th></th></tr>`+BK.map(r=>`<tr class="${r.book.startsWith('reference')?'hl':''}"><td>${esc(r.book)}</td><td class="num">${r.ann}</td><td class="num">${r.sharpe}</td><td class="num">${r.maxDD}</td><td class="num">${r['mean weight']}</td><td></td><td></td><td></td></tr>`).join('');
 const a=SL.find(r=>r.book.includes('q.80, held'))||{},u=SL.find(r=>r.book.includes('unconditional'))||{},b75=BK.find(r=>r.book.startsWith('75% when IN, no ADD'))||{},b80=BK.find(r=>r.book.includes('75% when IN, 100% on ADD: P_on >= q.80, held'))||{},ref=BK.find(r=>r.book.startsWith('reference'))||{};
 document.getElementById('addnote').textContent=`The rally condition as its own book. The sessions it selects (rally probability at or above its trailing 80th percentile, held until it falls below the 60th) earn ${a['mean daily bp while long']} basis points a day against ${u['mean daily bp while long']} for all IN sessions, on ${Math.round(a['share of IN sessions']*100)}% of them. There is no free lunch without leverage: a book that holds 75% when IN and goes to 100% only on ADD makes ${b80.ann}% against ${b75.ann}% for the reserve never deployed and ${ref.ann}% for always full. It beats sitting on cash, not holding the position already held, so it is not on the instruction page: a note for deploying new money only.`;}
// ---- the 2020 path
const P20=D.path2020; if(P20){const pick=P20.filter(r=>r.rule==='SPY buy and hold'||r.rule.startsWith('C: OUT P_off>q0.9')||r.rule.startsWith('F: OUT P_off>q0.95')||r.rule.startsWith('C: OUT P_off>q0.98, IN P_off<q0.8'));
 document.getElementById('p2020').innerHTML=`<tr><th>rule</th><th>ex-2020 ann %</th><th>max DD log</th><th>price %</th><th>incl-2020 ann %</th><th>max DD log</th><th>price %</th><th>share of Feb 19 to Mar 23 2020 taken</th><th>first OUT in the crash</th><th>back IN</th><th>2020 book %</th><th>2020 SPY %</th></tr>`+pick.map(r=>`<tr class="${r.rule===D.headline_rule?'hl':''}"><td>${esc(r.rule)}</td><td class="num">${r['ex-2020 ann']}</td><td class="num">${r['ex-2020 maxDD log']}</td><td class="num">${r['ex-2020 maxDD price']}</td><td class="num">${r['incl-2020 ann']}</td><td class="num">${r['incl-2020 maxDD log']}</td><td class="num">${r['incl-2020 maxDD price']}</td><td class="num">${r['2020 crash taken']}</td><td class="num">${esc(r['first OUT in crash'])}</td><td class="num">${esc(r['back IN'])}</td><td class="num">${r['2020 book %']}</td><td class="num">${r['2020 SPY %']}</td></tr>`).join('');
 const h=P20.find(r=>r.rule===D.headline_rule)||{};document.getElementById('p2020note').textContent=`The 2020 path. Every statistic on this page excludes Feb to Jul 2020 by design; here the frozen rules, fits and thresholds unchanged, are scored with the window included. Reference rule: max drawdown ${h['ex-2020 maxDD log']} log excluding the window and ${h['incl-2020 maxDD log']} log including it; it went OUT ${h['first OUT in crash']}, took ${h['2020 crash taken']} of the Feb 19 to Mar 23 decline, and came back IN ${h['back IN']}.`;}
// ---- early record (1999-2005)
const EA=D.early; if(EA){const K=Object.keys(EA.tables);const t0=EA.tables[K[0]],t2=EA.tables[K[2]];const pick=['SPY buy and hold','reference C: OUT P_off>=q.95, IN P_off<q.8','C strict: OUT P_off>q.95, IN P_off<q.8','price 10% below 63d high | IN new 42d high','vol alert & price 10% | IN new 42d high','price 7% below 63d high | IN new 42d high','price 5% below 63d high | IN new 42d high'];
 const row=(n)=>{const a=t0.find(r=>r.rule===n)||{},b=t2.find(r=>r.rule===n)||{};return `<tr class="${n.startsWith('reference')?'hl':''}"><td>${esc(n)}</td><td class="num">${a.ann}</td><td class="num">${a.maxDD}</td><td class="num">${a.spells}</td><td class="num" style="font-size:10px">${(a['taken per 15% event']||[]).map(x=>Number(x).toFixed(2)).join(' ')}</td><td class="num">${b.ann}</td><td class="num">${b.sharpe}</td><td class="num">${b.maxDD}</td><td class="num">${b.spells}</td><td class="num">${b['mean taken']}</td></tr>`;};
 document.getElementById('early').innerHTML=`<tr><th>rule</th><th>1999-2005 ann %</th><th>max DD</th><th>spells</th><th>taken: 2000-02 legs (-27, -26, -32, -19)</th><th>1999-2026 ann %</th><th>sharpe</th><th>max DD</th><th>spells</th><th>mean taken, 15 events</th></tr>`+pick.map(row).join('');
 const yr=EA.yearly.map(y=>`${y.year} ${f1(y['book %'])} vs ${f1(y['SPY %'])}`).join(' · ');
 document.getElementById('earlynote').textContent=`The record before 2006, refit yearly from 1998 on the 1993+ history (little beyond SPY, the VIX family and rates exists then; drawdown-model AUC 1998-2004 ${p3(EA.auc_early.P_off)}). Reference rule, book vs SPY by year: ${yr}. The rule took the 2000-02 bear almost in full and sidestepped only its last leg. The strict variant (above, not at or above, the threshold) differs on the ${EA.ties} sessions since 2006 where the discrete tree output equals its own trailing 95th percentile; the May 2022 exit is one of them.`;}
// ---- variants + null
const V=D.variants2||[]; document.getElementById('variants').innerHTML=`<tr><th>target</th><th>model</th><th>AUC 05-12</th><th>AUC 13-19</th><th>AUC 20-26</th><th>AUC all</th><th>hit rate by predicted quintile (low to high)</th></tr>`+V.map(v=>`<tr class="${String(v.model).includes('production')?'hl':''}"><td>${v.target}</td><td>${esc(v.model)}</td><td class="num">${v['auc 2005-12']}</td><td class="num">${v['auc 2013-19']}</td><td class="num">${v['auc 2020-26']}</td><td class="num">${v['auc all']}</td><td class="num">${(v.quintiles||[]).map(x=>Number(x).toFixed(2)).join(' · ')}</td></tr>`).join('');
const NL=D.null; document.getElementById('nulltiles').innerHTML=NL?Object.entries(NL).map(([k,v])=>`<div class="tile"><div class="k">${k} permutation test</div><div class="v">${p3(v.real.all)} vs ${p3(v.null_mean)}</div><div class="d">real pooled AUC vs the mean of ${v.n} null pipelines (sd ${p3(v.null_sd)}, max ${p3(v.null_max)}) · z ${Number(v.z).toFixed(1)}</div></div>`).join(''):'<div class="tile"><div class="k">permutation test</div><div class="v">running</div><div class="d">results land on the next refresh</div></div>';
// ---- audit
const A=D.audit_v1||[]; document.getElementById('audit').innerHTML=`<tr><th>group</th><th>v1 rule</th><th>side</th><th>fires</th><th>P all</th><th>P 2008-17</th><th>P 2018-26</th><th>fwd21 2018-26</th><th>halves</th></tr>`+A.map(a=>`<tr><td>${a.group}</td><td>${esc(a.rule)}</td><td>${a.kind}</td><td class="num">${a.n_all}</td><td class="num">${a.P_all}</td><td class="num">${a['2008-17 P']}</td><td class="num">${a['2018-26 P']}</td><td class="num">${a['2018-26 fwd21']}</td><td><span class="z ${String(a.halves).includes('FAIL')?'hot':''}">${a.halves}</span></td></tr>`).join('');
// ---- screen
const SC=D.screen; const lst=(t,arr,neg)=>`<div class="card"><div class="sub" style="margin-bottom:6px">${t}</div>`+arr.map(f=>`<div class="bar" style="grid-template-columns:minmax(150px,1.6fr) 1fr 56px"><span class="num" style="font-size:11px">${esc(f.feature)}</span><div class="trk"><div class="fill" style="left:0;width:${Math.min(100,Math.abs(f.cons)/0.2*100)}%;background:${neg?'var(--orange)':'var(--blue)'}"></div></div><span class="num" style="text-align:right">${f2(f.cons)}</span></div>`).join('')+'</div>';
document.getElementById('screen').innerHTML=lst('high value precedes deeper drawdowns (consistent IC)',SC.risky,true)+lst('high value precedes a safer tape',SC.safe,false)+lst('high value precedes bigger rallies',SC.rally,false);
// ---- episodes + mf
document.getElementById('episodes').innerHTML=`<tr><th>start</th><th>end</th><th>kind</th><th>sessions</th><th>SPY DD inside %</th><th>SPY start to end %</th></tr>`+D.episodes.slice().reverse().map(e=>`<tr><td>${e.start}</td><td class="num">${e.end}</td><td class="num">${e.kind}</td><td class="num">${e.sessions}</td><td class="num">${f1(e.spy_dd)}</td><td class="num">${f1(e.spy_ret)}</td></tr>`).join('');
const MF=D.multifractal; document.getElementById('mf').innerHTML=`<tr><th>series</th><th>h2</th><th>rank</th><th>width</th><th>rank</th><th>alpha |r|</th><th>rank</th><th>VR5</th><th>VR21</th></tr>`+Object.entries(MF).map(([k,v])=>`<tr><td>${k}</td><td class="num">${p3(v.h2.value)}</td><td class="num">${p3(v.h2.rank)}</td><td class="num">${p3(v.width.value)}</td><td class="num">${p3(v.width.rank)}</td><td class="num">${p3(v.alpha_abs.value)}</td><td class="num">${p3(v.alpha_abs.rank)}</td><td class="num">${p3(v.vr5.value)}</td><td class="num">${p3(v.vr21.value)}</td></tr>`).join('');
// ---- tape
const TP=D.tape; const NAMES={'VIX:lvl_rank504':'VIX level','VIX:roc5_rank':'VIX 5d ROC','VIX:roc21_rank':'VIX 21d ROC','VVIX/VIX:roc5_rank':'VVIX/VIX 5d ROC','VIX/VIX3M:roc5_rank':'VIX/VIX3M 5d ROC','MOVE:lvl_rank504':'MOVE level','MOVE:roc5_rank':'MOVE 5d ROC','TNX:roc21_rank':'10y yield 21d change','SPY:cc21':'SPY realized 21d (%)','SPY:pk21':'SPY Parkinson 21d (%)','SPY:cc21_rank':'SPY realized level','SPY:cc21_63_rank':'SPY RV 21/63','SPY:pk21_cc21_rank':'SPY Parkinson / close-to-close','SPY:VIX/cc21_rank':'VIX over realized','SPY:VIX_prem_chg5_rank':'VIX premium 5d change','SPY:vol_rank63':'SPY volume vs 63d','SPY:volvol21_rank':'SPY volatility of volume','SPY:compress5_63_rank':'SPY range compression','SPY:amihud21_rank':'SPY illiquidity','HYG:pk21_rank':'HYG Parkinson vol','HYG:signedvol21':'HYG signed volume 21d','HYG:VIX/cc21_rank':'VIX over HYG realized','XLV:rel63_rank':'XLV relative 63d','XS:disp5_rank':'dispersion 5d','XS:avgcorr21_rank':'avg correlation 21d','XS:avgcorr21_chg5_rank':'avg correlation 5d change','XS:rv_breadth_rank':'realized-vol breadth','XS:vol_breadth_rank':'volume breadth','XS:compress_breadth':'compression breadth','XS:breadth_up21':'ROC breadth 21d','RSP/SPY:roc21_rank':'RSP/SPY 21d','RSP/SPY:roc63_rank':'RSP/SPY 63d'};
document.getElementById('tape').innerHTML=Object.entries(TP).map(([k,v])=>{const isRank=k.includes('rank')||k.startsWith('XS:breadth')||k.startsWith('XS:compress');const c=isRank?(v>=0.9?'hot':(v<=0.1?'cold':'')):'';return `<div class="tile"><div class="k">${NAMES[k]||k}</div><div class="v"><span class="z ${c}">${isRank?Number(v).toFixed(2):Number(v).toFixed(2)}</span></div><div class="d">${isRank?'trailing rank':'level'}</div></div>`;}).join('');
// ---- tickers
const LD=D.tick_lead||[]; document.getElementById('lead').innerHTML=`<tr><th>ticker</th><th>rel ROC5 rank -5</th><th>-1</th><th>0</th><th>+3</th><th>+10</th><th>fwd 10d rel %</th><th>fwd 21d rel %</th><th>n</th></tr>`+LD.map(r=>`<tr><td>${r.ticker}</td><td class="num">${r['rel5rank@-5']}</td><td class="num">${r['rel5rank@-1']}</td><td class="num">${r['rel5rank@+0']}</td><td class="num">${r['rel5rank@+3']}</td><td class="num">${r['rel5rank@+10']}</td><td class="num"><span class="z ${r['fwd10rel%@0']<=-0.5?'hot':(r['fwd10rel%@0']>=0.3?'cold':'')}">${f2(r['fwd10rel%@0'])}</span></td><td class="num">${f2(r['fwd21rel%@0'])}</td><td class="num">${r.n}</td></tr>`).join('');
const CD=(D.tick_cond||{})['XS:disp5_rank']||[]; document.getElementById('cond').innerHTML=`<tr><th>ticker</th><th>low dispersion</th><th>mid</th><th>high dispersion</th></tr>`+CD.map(r=>`<tr><td>${r.ticker}</td><td class="num">${f2(r.low)}</td><td class="num">${f2(r.mid)}</td><td class="num">${f2(r.high)}</td></tr>`).join('');
</script>
"""
html = HTML.replace('__DATA__', json.dumps(D).replace('</', '<\\/'))
open(out, 'w').write(html); print('wrote', out, len(html) // 1024, 'KB')
