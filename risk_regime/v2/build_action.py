#!/usr/bin/env python3
"""The action page: one instruction, HOLD or REDUCE, from the v2 reading. Usage: python3 build_action.py v2_data.json out.html

Logic, fixed from the out-of-sample record and nothing else:
  REDUCE  the reference rule is OUT: the drawdown probability reached its trailing 95th percentile and has not yet
          fallen back below the 80th. The only exit signal with a record (kept the book out of four of the five
          2008-09 legs; fires about 14 times in 20 years).
  REDUCE  VIX outrunning VVIX: VIX's 21-day rate of change and the VIX/VVIX 21-day rate of change are both in the top
          10% of their trailing two years; stays live until VIX's 21-day rate of change falls back below its median.
          Adopted from vvsig.py after its record was seen (see the evidence page).
  HOLD    everything else. Full size, nothing to do. Quiet-tape dips are not sold: the record priced that at 3 to 5
          points a year.
ADD was removed from the instruction: scored as its own book (addbook.py) it beats sitting on cash but not holding
the position already held, so it justifies no size change on a full position. It stays on the evidence page as a
note for new money.
"""
import json, sys
src, out = sys.argv[1], sys.argv[2]; D = json.load(open(src)); P = D['probs']
state = D['state']; p_on = P['P_on']['trailing_pct']; p_off = P['P_off']['trailing_pct']; p_vol = P['P_vol']['trailing_pct']; dial = round(100 * (1 - p_off))
pc = lambda x: round(x * 100)
def ordn(x):
    n = round(x * 100); return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
def bucket_note(p, key, hit_word):
    d = P[p]['deciles']; i = P[p]['today_decile']; means = [x['fwd21'] for x in d]; worst = int(min(range(10), key=lambda k: means[k])); best = int(max(range(10), key=lambda k: means[k]))
    txt = f"decile {i+1} of 10: {pc(d[i][key])}% {hit_word}, mean next month {d[i]['fwd21']:+.1f}%"
    if i == worst: txt += f" (the weakest bucket in this table; the curve is not monotone, deciles {', '.join(str(k+1) for k in range(10) if means[k] > 0)} average positive)"
    elif i == best: txt += " (the strongest bucket in this table)"
    return txt
VV = D.get('vvsig', {}).get('today', {}); vv_live = bool(VV.get('live')); vr = VV.get('vix_roc21_rank'); rr = VV.get('ratio_roc21_rank')
legs = []
if state == 'OUT': legs.append(f'The drawdown probability hit its trailing 95th percentile and is still above the 80th (now the {ordn(p_off)}); that leg clears when it drops below the 80th.')
if vv_live: legs.append(f"VIX is outrunning VVIX since {VV.get('since')}: VIX's 21-day rate of change is at the {ordn(vr)} percentile and the VIX/VVIX ratio's at the {ordn(rr)}; that leg clears when VIX's 21-day rate of change falls below its median.")
if legs: action, why = 'REDUCE', ' '.join(legs)
else: action, why = 'HOLD', f"Rule is IN, dial {dial}. Drawdown probability at the {ordn(p_off)} percentile; VIX 21-day rate of change at the {ordn(vr)}, VIX/VVIX at the {ordn(rr)}. Full size. REDUCE needs the drawdown probability at its 95th, or both rates of change at their 90th." if vr is not None else f'Rule is IN, dial {dial}. Drawdown probability at the {ordn(p_off)} percentile. Full size.'
dec = lambda p: P[p]['deciles'][P[p]['today_decile']]
tiles = [
    ('5% drawdown within a month', P['P_off']['today'], f"{ordn(p_off)} percentile of its trailing three years · {bucket_note('P_off', 'P_off', 'drew down 5%')}"),
    ('5% rally within a month', P['P_on']['today'], f"{ordn(p_on)} percentile · {bucket_note('P_on', 'P_on', 'rallied 5%')}"),
    ('vol expands 1.5x within a month', P['P_vol']['today'], f"{ordn(p_vol)} percentile · {bucket_note('P_vol', 'P_volexp', 'expanded')}"),
]
col = {'REDUCE': 'var(--orange)', 'HOLD': 'var(--ink)'}[action]
html = f"""<title>Risk Action</title>
<meta name="description" content="One instruction from the vol tape regime system: hold or reduce">
<style>
:root{{--bg:#f7f7f4;--surface:#fff;--ink:#1b1f26;--ink2:#4b5563;--muted:#8a919c;--line:#e3e5e8;--blue:#1d4ed8;--orange:#ea580c;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#141517;--surface:#1c1e22;--ink:#ebedf0;--ink2:#b3b8c2;--muted:#7c838e;--line:#2c3037;--blue:#3b82f6;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#141517;--surface:#1c1e22;--ink:#ebedf0;--ink2:#b3b8c2;--muted:#7c838e;--line:#2c3037;--blue:#3b82f6;color-scheme:dark}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.45}}
.wrap{{max-width:720px;margin:0 auto;padding:28px 16px 48px;display:grid;gap:18px}}
.sub{{color:var(--muted);font-family:var(--mono);font-size:12px}}
.action{{font-family:var(--mono);font-weight:700;font-size:clamp(64px,18vw,120px);line-height:1;letter-spacing:.02em;color:{col}}}
.why{{font-size:17px;color:var(--ink2);max-width:60ch}}
.dial{{font-family:var(--mono);font-size:14px;color:var(--ink2)}} .dial b{{font-size:28px;color:var(--ink)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px}} .tile .k{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-family:var(--mono)}} .tile .v{{font-family:var(--mono);font-size:24px;font-weight:600;margin-top:2px}} .tile .d{{font-size:12.5px;color:var(--ink2);margin-top:4px}}
.rules{{font-size:13px;color:var(--ink2);border-top:1px solid var(--line);padding-top:12px}} .rules b{{font-family:var(--mono)}}
a{{color:var(--blue)}}
</style>
<div class="wrap">
<div class="sub">Vol tape regime · as of {D['asof']} close · rule {('IN since ' + D['state_since']) if state == 'IN' else ('OUT since ' + D['state_since'])}</div>
<div class="action">{action}</div>
<div class="why">{why}</div>
<div class="dial">stay-in dial <b>{dial}</b> · OUT at 5 or below, back IN above 20 · {'inside a vol episode' if D.get('inside_episode') else 'no vol episode in progress'}</div>
<div class="dial">VIX outrunning VVIX <b style="font-size:20px;color:{'var(--orange)' if vv_live else 'var(--ink)'}">{'LIVE since ' + str(VV.get('since')) if vv_live else 'quiet'}</b> · VIX {VV.get('vix')} ({VV.get('vix_roc21_pct'):+.1f}% in 21 sessions, {ordn(vr) if vr is not None else '-'} percentile) · VIX/VVIX {VV.get('ratio_roc21_pct'):+.1f}% ({ordn(rr) if rr is not None else '-'}) · fires at the 90th on both</div>
<div class="tiles">{''.join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v:.3f}</div><div class="d">{d}</div></div>' for k, v, d in tiles)}</div>
<div class="sub">The decile lines are history for the bucket each probability sits in today. The instruction does not act on them: REDUCE waits for the drawdown probability to reach its trailing 95th percentile, or for VIX to outrun VVIX. A below-average bucket with HOLD printed is a normal reading.</div>
<div class="rules"><b>REDUCE</b> when the drawdown probability reaches its trailing 95th percentile, until it falls below the 80th; or when VIX's 21-day rate of change and the VIX/VVIX ratio's are both in their top 10%, until VIX's falls back below its median. <b>HOLD</b> otherwise; dips are not sold. Everything is out of sample, 2005 to date, refit yearly. Evidence: <a href="https://claude.ai/artifact/KuXFhbnKXaRjJ7dYXW2ci4">full dashboard</a>.</div>
</div>
"""
open(out, 'w').write(html); print('wrote', out, '|', action, '|', why)
