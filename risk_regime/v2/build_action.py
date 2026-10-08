#!/usr/bin/env python3
"""The action page: one instruction, ADD / HOLD / REDUCE, from the v2 reading. Usage: python3 build_action.py v2_data.json out.html

Logic, fixed from the out-of-sample record and nothing else:
  REDUCE  the reference rule is OUT: the drawdown probability reached its trailing 95th percentile and has not yet
          fallen back below the 80th. The only exit signal with a record (kept the book out of four of the five
          2008-09 legs; fires about 14 times in 20 years).
  ADD     the rule is IN and the rally probability is in the top fifth of its trailing three years: historically a
          5% rally within a month followed 41% of the time against a 17% base, five sigma above its null.
  HOLD    everything else. Full size, nothing to do. Quiet-tape dips are not sold: the record priced that at 3 to 5
          points a year.
"""
import json, sys
src, out = sys.argv[1], sys.argv[2]; D = json.load(open(src)); P = D['probs']
state = D['state']; p_on = P['P_on']['trailing_pct']; p_off = P['P_off']['trailing_pct']; p_vol = P['P_vol']['trailing_pct']; dial = round(100 * (1 - p_off))
if state == 'OUT': action, why = 'REDUCE', f'The drawdown probability hit its trailing 95th percentile and is still above the 80th (now the {round(p_off*100)}th). Stay reduced until it drops below the 80th.'
elif p_on >= 0.8: action, why = 'ADD', f'Rule is IN and the rally probability is in the top fifth of its range ({round(p_on*100)}th percentile). Historically a 5% rally inside a month followed 41% of the time against a 17% base.'
else: action, why = 'HOLD', f'Rule is IN, dial {dial}. Drawdown probability at the {round(p_off*100)}th percentile, rally probability at the {round(p_on*100)}th. Full size, nothing to do.'
dec = lambda p: P[p]['deciles'][P[p]['today_decile']]
pc = lambda x: round(x * 100)
tiles = [
    ('5% drawdown within a month', P['P_off']['today'], f"{pc(p_off)}th percentile · this decile: {pc(dec('P_off')['P_off'])}% drew down 5%, mean next month {dec('P_off')['fwd21']:+.1f}%"),
    ('5% rally within a month', P['P_on']['today'], f"{pc(p_on)}th percentile · this decile: {pc(dec('P_on')['P_on'])}% rallied 5%, mean next month {dec('P_on')['fwd21']:+.1f}%"),
    ('vol expands 1.5x within a month', P['P_vol']['today'], f"{pc(p_vol)}th percentile · this decile: {pc(dec('P_vol')['P_volexp'])}% expanded, mean next month {dec('P_vol')['fwd21']:+.1f}%"),
]
col = {'REDUCE': 'var(--orange)', 'ADD': 'var(--blue)', 'HOLD': 'var(--ink)'}[action]
html = f"""<title>Risk Action</title>
<meta name="description" content="One instruction from the vol tape regime system: add, hold or reduce">
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
<div class="tiles">{''.join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v:.3f}</div><div class="d">{d}</div></div>' for k, v, d in tiles)}</div>
<div class="rules"><b>REDUCE</b> when the drawdown probability reaches its trailing 95th percentile, until it falls below the 80th. <b>ADD</b> when the rule is IN and the rally probability is in the top fifth of its range. <b>HOLD</b> otherwise; dips are not sold. Everything is out of sample, 2005 to date, refit yearly. Evidence: <a href="https://claude.ai/artifact/KuXFhbnKXaRjJ7dYXW2ci4">full dashboard</a>.</div>
</div>
"""
open(out, 'w').write(html); print('wrote', out, '|', action, '|', why)
