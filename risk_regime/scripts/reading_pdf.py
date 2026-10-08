#!/usr/bin/env python3
"""Render the current Vol Tape Regime reading as a printable PDF.

Reads the five result JSONs in risk_regime/results/ (dash_data, tick_data, rv_data,
score_data, breadth_data), writes results/VOL_TAPE_READING.pdf (Letter, 5 pages).
Blue / orange only (deutan-safe). Usage: python3 scripts/reading_pdf.py [results_dir]
"""
import json, os, sys, datetime, zoneinfo, html as H

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
R = os.path.abspath(R)
ld = lambda f: json.load(open(os.path.join(R, f)))
D, T, V, S, B = ld('dash_data.json'), ld('tick_data.json'), ld('rv_data.json'), ld('score_data.json'), ld('breadth_data.json')
last = len(D['dates']) - 1
NY = zoneinfo.ZoneInfo('America/New_York')
gen = datetime.datetime.now(NY).strftime('%Y-%m-%d %H:%M %Z')

RULE_META = {
 'SETUP_rates':   ('SETUP',  'Yields ripping, VIX asleep',          'TNX 5d chg z > 1 and VIX 5d ROC z < -0.3', 'P(5% DD/21d) 0.25 vs 0.17 base; 56% of tops, median 18d lead'),
 'SETUP_complac': ('SETUP',  'Complacency',                          'VIX 21d ROC z < -1, VIX 5d ROC z < -1, SKEW 10d ROC z < -0.5', 'P 0.24; fwd21 +0.29% vs +0.86%'),
 'ONSET_impulse': ('ONSET',  'First impulse after compression',      'VIX 21d ROC z < -0.3 and VIX 5d ROC z > 1', 'P 0.27; fwd 5d and 10d negative'),
 'ONSET_vvixlag': ('ONSET',  'VIX spike, VVIX not confirming',       'VIX 5d ROC z > 1 and VVIX/VIX 5d ROC z < -1', 'P 0.27; 47% of tops, median 11d lead'),
 'ONSET_movelag': ('ONSET',  'Equity vol out, bond vol compressing', 'Impulse-after-compression and MOVE 5d ROC z < 0', 'P 0.43 (n=14); fwd21 -0.46%'),
 'ONSET_ovxdiv':  ('ONSET',  'Equity vol up, oil vol down',          'VIX 5d ROC z > 0.3 and OVX 5d ROC z < -1', 'P 0.31; fwd21 -0.15%'),
 'CONT_2ndleg':   ('CONT',   'Second leg',                           'VIX 3d ROC z > 1, VIX/VIX3M 3d ROC z > 1, SPY 7%+ off 63d high', 'P 0.39; highest drawdown odds'),
 'FAIL_yieldsup': ('FAIL',   'Rally failure: vol fading, yields up', 'VIX 10d ROC z < -1, TNX 5d chg z > 0.5, SPY 5%+ off high', 'fwd21 -1.04%; the only negative-return rule'),
 'CAP_alldims':   ('CAP',    'Capitulation: every dimension at once','VIX, GVZ 5d ROC z > 1, MOVE 5d ROC z > 0.5, TNX 5d chg z < -0.5', 'fwd21 +2.50%, fwd63 +4.22%; median day -1 vs trough'),
 'CAP_vix_gvz':   ('CAP',    'Gold vol confirming equity vol',       'VIX 5d ROC z > 1 and GVZ 5d ROC z > 1', 'fwd21 +2.07%, fwd63 +4.22%'),
 'CAP_vvixout':   ('CAP',    'VVIX outrunning VIX in a spike',       'VIX 5d ROC z > 1.5 and VVIX/VIX 5d ROC z > 0', 'fwd21 +3.99%, fwd63 +7.60%; 9 of 9 positive'),
 'ONCONF_collapse':('ONCONF','Flip confirmed',                       'VIX 5d ROC z < -1, VVIX/VIX 5d ROC z > 0.3, VIX/VIX3M 5d ROC z < -0.5', 'median day +4 after trough; confirmation, not edge'),
}
PLAIN = {
 'ONCONF_collapse': 'the flip confirmation', 'KRE_banks_vs_yields': 'regional banks weak against rising yields',
 'HYG_credit_vol_cheap_VIX': 'equity implied vol cheap against credit realized vol', 'rates_pressure_vix_asleep': 'month-long rates pressure with the VIX asleep',
 'vix_floor_vvix_floor': 'VIX and VVIX both on the floor', 'complacency_both_compressed': 'VIX compressed over both the week and the month',
 'vol_collapsing_from_high': 'vol collapsing from a high', 'drawdown_no_capitulation': 'a drawdown without a capitulation print',
 'SETUP_rates': 'yields ripping while the VIX sleeps', 'SETUP_complac': 'the complacency setup', 'ONSET_impulse': 'the first vol impulse after compression',
 'ONSET_vvixlag': 'a VIX spike that VVIX does not confirm', 'ONSET_movelag': 'equity vol breaking out while bond vol compresses', 'ONSET_ovxdiv': 'equity vol up with oil vol down',
 'CONT_2ndleg': 'the second leg', 'FAIL_yieldsup': 'the rally-failure tell', 'CAP_alldims': 'capitulation on every dimension', 'CAP_vix_gvz': 'gold vol confirming equity vol',
 'CAP_vvixout': 'VVIX outrunning VIX in a spike', 'VVIX_confirms_spike': 'VVIX confirming the spike', 'HYG_credit_crack': 'credit cracking relative while vol spikes',
 'TLT_duration_bid_calm': 'a duration bid while vol falls', 'XLU_defensive_bid': 'utilities bid with yields up', 'IWM_beta_chase': 'a small-cap beta chase into a spike',
 'XLU_month_lead': 'utilities leading over the month', 'SPY_realized_outruns_implied': 'realized vol outrunning implied in a spike', 'SPY_implied_outruns_realized': 'implied outrunning realized in a spike',
 'QQQ_realized_outruns_VXN': 'Nasdaq realized outrunning implied', 'XLE_energy_vol_cheap_OVX': 'oil vol cheap against energy realized', 'TLT_MOVE_leads_realized': 'bond implied spiking over compressed bond realized',
 'HYG_realized_expanding_spike': 'credit realized expanding into the spike', 'RVX_premium_collapse': 'the VIX premium over realized collapsing',
 'GLD_realized_collapse_GVZdown': 'gold realized collapsing faster than gold implied', 'XOP_realized_collapse_OVXdown': 'energy realized collapsing faster than oil implied',
 'RSP_bleed_month_yields_up': 'equal weight losing a month into a yield rip', 'RSP_bleed_quarter_yields_up': 'equal weight losing a quarter into a yield rip',
 'QQQE_bleed_quarter_yields_up': 'equal-weight Nasdaq losing a quarter into a yield rip', 'EW_rotation_inside_decline': 'equal weight outperforming inside a decline',
 'RSP_broadening_vol_compressing': 'broadening while vol compresses',
}
plain = lambda k: PLAIN.get(k, k.replace('_', ' '))
f2 = lambda x, s=True: '' if x is None else (('+' if (s and x > 0) else '') + f'{x:.2f}')
f1 = lambda x, s=True: '' if x is None else (('+' if (s and x > 0) else '') + f'{x:.1f}')
pc = lambda x: '' if x is None else f'{round(x*100)}'
def ordn(x):
    n = round(x*100); suf = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'); return f'{n}{suf}'
WORDS = ['zero','one','two','three','four','five','six','seven','eight','nine']
wn = lambda n: WORDS[n] if 0 <= n <= 9 else str(n)
lv = {r['series']: r for r in D['levels']}
rt = {r['ratio']: r for r in D['ratios']}
e = H.escape

def say_list(items):
    items = list(items)
    if not items: return ''
    if len(items) == 1: return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]

def status(live, recent):
    return ('live', 'LIVE') if live else (('recent', 'last 10d') if recent else ('quiet', 'quiet'))

# ---------- boards
main_rows = []
for k, (tag, name, dfn, stat) in RULE_META.items():
    arr = D['rules'][k]; live = bool(arr[last]); recent = (not live) and any(arr[-10:])
    hist = D['rule_history'][k]; main_rows.append((tag, name, dfn, stat, live, recent, hist[-1] if hist else 'never', len(hist)))
main_live = [RULE_META[k][1] for k in RULE_META if D['rules'][k][last]]
main_recent = [RULE_META[k][1] for k in RULE_META if not D['rules'][k][last] and any(D['rules'][k][-10:])]
pair_live = [k for k, v in T['tick_rules'].items() if v['live']]; pair_recent = [k for k, v in T['tick_rules'].items() if v['recent'] and not v['live']]
rv_live = [k for k, v in V['rv_rules'].items() if v['live']]; rv_recent = [k for k, v in V['rv_rules'].items() if v['recent'] and not v['live']]
br_live = [k for k, v in B['cells'].items() if v['live']]; br_recent = [k for k, v in B['cells'].items() if v['recent'] and not v['live']]

# ---------- plain-English summary
st = S['states'][S['state']]; comps = sorted(S['components'].items(), key=lambda kv: kv[1])
comp_txt = say_list([f"{'plus' if p > 0 else 'minus'} {abs(int(p))} for {plain(k)}" for k, p in comps])
vix, vvix, move, tnx, tyx = lv['VIX'], lv['VVIX'], lv['MOVE'], lv['TNX'], lv['TYX']
vvr = rt['VVIX/VIX']['roc5z']
if vix['z5'] > 1:
    disc = ('VVIX is keeping pace, which historically reads as a dip, not an onset.' if vvr > -0.3 else 'VVIX is lagging, which is the onset configuration.')
else:
    disc = 'There is no spike to discriminate. If the VIX jumps from here, the question is whether VVIX keeps pace: lagging is onset, keeping up is a dip.'
setup_live = bool(D['rules']['SETUP_rates'][last]); rates_ctx = 'rates_pressure_vix_asleep' in S['components']
spy_rv = next(r for r in V['rv_tape'] if r['t'] == 'SPY'); iv = {p['pair']: p for p in V['ivrv']}
rs = next(p for p in B['pairs'] if p['pair'] == 'RSP/SPY'); qq = next((p for p in B['pairs'] if p['pair'] == 'QQQE/QQQ'), None)
dims_on = [k for k, v in D['dims'].items() if any(v[-5:])]
stale = D.get('stale') or []
paras = []
paras.append(f"As of the {D['asof']} close. " + (f"Stale series: {say_list([s['series']+' (last '+s['last']+')' for s in stale])}. " if stale else "Every series posted this session. ") +
    f"The environment dial reads {S['score']} and the state is {S['state']}, {S['days_in_state']} sessions in. Base {S['base']}, {comp_txt}. "
    f"OUT needs {S['thresholds']['out_in']} or below; back IN above {S['thresholds']['out_exit']}. In the {S['state']} state since 2008 the odds of a five percent drop within a month were {round(st['P_off21']*100)} percent and of a ten percent drop within a quarter {round(st['P10_63']*100)} percent, with a mean next quarter of {f1(st['fwd63'])} percent.")
def board_sentence(name, live, recent, pl):
    if not live and not recent: return f"The {name} is empty: nothing live, nothing fired in the last ten sessions."
    s = f"On the {name}, " + (f"live: {say_list([pl(k) for k in live])}" if live else "nothing is live")
    s += (f"; fired in the last ten sessions: {say_list([pl(k) for k in recent])}." if recent else ".")
    return s
paras.append(' '.join([board_sentence('main rule board', main_live, main_recent, lambda k: k), board_sentence('ticker pair board', pair_live, pair_recent, plain), board_sentence('realized-vol board', rv_live, rv_recent, plain)]))
paras.append(f"VIX {vix['level']:.2f}, {ordn(vix['pct252'])} percentile of the year, five-day rate of change z {f2(vix['z5'])}, twenty-one-day {f2(vix['z21'])}. VVIX-to-VIX five-day rate of change z {f2(vvr)}, with VVIX itself at the {ordn(vvix['pct252'])} percentile. {disc}")
paras.append(f"Ten-year at {tnx['level']:.2f} percent, {ordn(tnx['pct252'])} percentile, five-day change z {f2(tnx['z5'])}, twenty-one-day {f2(tnx['z21'])}. Thirty-year at {tyx['level']:.2f}, {ordn(tyx['pct252'])} percentile, five-day {f2(tyx['z5'])}, twenty-one-day {f2(tyx['z21'])}. "
    + ("The five-day rates setup is live." if setup_live else "The five-day rates setup is not live.") + (" The month-long rates pressure context is live and counting in the dial." if rates_ctx else "")
    + f" Bond vol is at the {ordn(move['pct252'])} percentile, five-day z {f2(move['z5'])}, twenty-one-day {f2(move['z21'])}.")
paras.append(f"Realized: SPY ten-day realized {spy_rv['rv10']}, twenty-one-day {spy_rv['rv21']}, term structure {spy_rv['rvr']} at z {f2(spy_rv['rvr_z'])}. Realized-vol breadth {round(V['breadth']['now']*100)} percent of the thirty-eight with short realized expanding, z {f2(V['breadth']['z'])}. VIX minus realized premium five-day change z {f2(V['rvroc'].get('VIX-RV premium 5d chg z'))}. VIX over high-yield realized vol sits at the {ordn(iv['HYG:VIX/RV']['pct252'])} percentile"
    + (", the cheap-credit-vol bucket that carried about thirty percent odds of a five percent drawdown inside a month." if iv['HYG:VIX/RV']['pct252'] <= 0.2 else "."))
paras.append(f"Breadth, context only and not in the dial. Equal-weight S&P {f1(rs['chg21'])} percent against cap-weight over twenty-one sessions, ratio rate of change z {f2(rs['roc21z'])}, and {f1(rs['chg63'])} percent over sixty-three, z {f2(rs['roc63z'])}."
    + (f" Equal-weight Nasdaq {f1(qq['chg21'])} and {f1(qq['chg63'])} percent, z {f2(qq['roc21z'])} and {f2(qq['roc63z'])}." if qq else '')
    + f" {wn(round(B['composite']['share']*9)).capitalize()} of nine sectors have equal weight ahead over the month. "
    + (f"Live cells: {say_list([plain(k) + (' since ' + B['cells'][k]['since'] if B['cells'][k].get('since') else '') for k in br_live])}. Historically those cells carried " + say_list([f"{round(B['cells'][k]['P_days']*100)} percent" for k in br_live]) + " odds of a five percent drawdown inside a month." if br_live else "No breadth cell is live."))
if dims_on: paras.append(f"Dimension tags printed in the last five sessions: {say_list([k.replace('rateled','rate-led').replace('ftq','flight to quality').replace('equityonly','equity only').replace('goldvol','gold vol') for k in dims_on])}.")
paras.append(f"Backtest context for the dial: fully invested when IN and in cash when OUT since 2008 returned {S['binary']['ann']} percent a year at {S['binary']['vol']} percent vol with a maximum drawdown of {S['binary']['maxDD']} percent, against buy-and-hold {S['buyhold']['ann']} percent at {S['buyhold']['vol']} percent vol and {S['buyhold']['maxDD']} percent. About {S['changes_per_year']} state changes a year.")

# ---------- HTML
def chip(live, recent):
    c, t = status(live, recent); return f'<span class="chip {c}">{t}</span>'
def z(v):
    if v is None: return '<td class="n"></td>'
    c = 'hot' if v >= 1 else ('cold' if v <= -1 else ''); return f'<td class="n"><span class="z {c}">{f2(v)}</span></td>'
rows_main = ''.join(f'<tr><td><span class="tag {tag}">{tag}</span> {e(name)}</td><td>{chip(live, recent)}</td><td class="mono">{e(dfn)}</td><td class="muted">{e(stat)}</td><td class="mono">{lastf} · {n}</td></tr>' for tag, name, dfn, stat, live, recent, lastf, n in main_rows)
def rows_generic(rules, note_key='note'):
    out = ''
    for k, r in rules.items():
        p = r.get('P_days', r.get('P_off')); odds = f"P {p:.2f} vs 0.17" if p is not None else ''
        fw = r.get('fwd21'); odds += f" · fwd21 {f2(fw)}%" if fw is not None else ''
        lastf = (('live since ' + r['since'] + ' · ') if r.get('live') and r.get('since') else '') + 'last ' + str(r.get('last', ''))
        out += f'<tr><td>{e(plain(k))}</td><td>{chip(r["live"], r["recent"])}</td><td class="mono">{e(r["def"])}</td><td class="muted">{e(odds)} · n={r["n"]}</td><td class="mono">{e(lastf)}</td></tr>'
    return out
tape_rows = ''.join(f'<tr><td>{e(r["series"])}</td><td class="n">{r["level"]}</td><td class="n">{pc(r["pct252"])}</td><td class="n">{f2(r["roc5"])}</td><td class="n">{f2(r["roc21"])}</td>{z(r["z3"])}{z(r["z5"])}{z(r["z10"])}{z(r["z21"])}</tr>' for r in D['levels'])
ratio_rows = ''.join(f'<tr><td>{e(r["ratio"])}</td><td class="n">{r["level"]}</td><td class="n">{pc(r["pct252"])}</td>{z(r["roc3z"])}{z(r["roc5z"])}</tr>' for r in D['ratios'])
br_rows = ''.join(f'<tr><td>{e(p["pair"])}</td><td class="n">{f2(p["chg21"])}</td><td class="n">{f2(p["chg63"])}</td>{z(p["roc5z"])}{z(p["roc21z"])}{z(p["roc63z"])}{z(p["accel5z"])}<td class="n">{"" if p["P_bot63"] is None else f"{p["P_bot63"]:.2f} / {p["P_top63"]:.2f}"}</td></tr>' for p in sorted(B['pairs'], key=lambda p: (p['roc21z'] if p['roc21z'] is not None else 0)))
tiles = [
 ('SPY', f"{D['spy'][last]:.2f}", f"{f2(D['spy_ret5'])}% 5d · {f2(D['spy_dd63'])}% off 63d high"),
 ('VIX', f"{vix['level']:.2f}", f"pct {pc(vix['pct252'])} · 5d ROC z {f2(vix['z5'])} · 21d {f2(vix['z21'])}"),
 ('VVIX / VIX ROC z', f2(vvr), f"5d · VVIX pct {pc(vvix['pct252'])}"),
 ('MOVE', f"{move['level']:.1f}", f"pct {pc(move['pct252'])} · 5d z {f2(move['z5'])} · 21d z {f2(move['z21'])}"),
 ('TNX', f"{tnx['level']:.2f}%", f"pct {pc(tnx['pct252'])} · 5d z {f2(tnx['z5'])} · 21d z {f2(tnx['z21'])}"),
 ('TYX', f"{tyx['level']:.2f}%", f"pct {pc(tyx['pct252'])} · 5d z {f2(tyx['z5'])} · 21d z {f2(tyx['z21'])}"),
]
rv_tiles = [
 ('SPY RV10 / RV21', f"{spy_rv['rv10']} / {spy_rv['rv21']}", f"term structure {spy_rv['rvr']} · z {f2(spy_rv['rvr_z'])}"),
 ('VIX / SPY RV21', f"{iv['SPY:VIX/RV']['level']:.2f}", f"pct {pc(iv['SPY:VIX/RV']['pct252'])} · ROC5 z {f2(iv['SPY:VIX/RV']['roc5z'])}"),
 ('VIX / HYG RV21', f"{iv['HYG:VIX/RV']['level']:.2f}", f"pct {pc(iv['HYG:VIX/RV']['pct252'])} · bottom-Q P(off) 0.31"),
 ('MOVE / TLT RV21', f"{iv['TLT:MOVE/RV']['level']:.2f}", f"pct {pc(iv['TLT:MOVE/RV']['pct252'])} · ROC5 z {f2(iv['TLT:MOVE/RV']['roc5z'])}"),
 ('Realized-vol breadth', f"{round(V['breadth']['now']*100)}%", f"share of 38 with RV10/RV21 > 1.2 · z {f2(V['breadth']['z'])}"),
] + [(k, f2(v), 'realized ROC, z vs own 252d') for k, v in V['rvroc'].items()]
br_tiles = [
 ('RSP / SPY, 21d', f"{f2(rs['chg21'])}%", f"ratio ROC z {f2(rs['roc21z'])} · 5d z {f2(rs['roc5z'])}"),
 ('RSP / SPY, 63d', f"{f2(rs['chg63'])}%", f"ratio ROC z {f2(rs['roc63z'])}"),
] + ([('QQQE / QQQ, 21d', f"{f2(qq['chg21'])}%", f"ratio ROC z {f2(qq['roc21z'])}"), ('QQQE / QQQ, 63d', f"{f2(qq['chg63'])}%", f"ratio ROC z {f2(qq['roc63z'])}")] if qq else []) + [
 ('Sectors EW > CW, 21d', f"{round(B['composite']['share']*100)}%", f"of 9 · z {f2(B['composite']['share_z'])}"),
 ('TNX 21d chg z', f2(B['tnx21z']), f"SPY 21d return z {f2(B['spy21z'])}"),
]
tile_html = lambda ts: ''.join(f'<div class="tile"><div class="k">{e(k)}</div><div class="v">{e(v)}</div><div class="d">{e(d)}</div></div>' for k, v, d in ts)
comp_rows = ''.join(f'<tr><td>{e(plain(k))}</td><td class="n {"neg" if p < 0 else "pos"}">{"+" if p > 0 else ""}{int(p)}</td></tr>' for k, p in comps)
state_col = 'var(--orange)' if S['state'] == 'OUT' else 'var(--blue)'

HTML = f'''<!doctype html><html><head><meta charset="utf-8"><title>Vol Tape Regime reading {D['asof']}</title>
<style>
:root{{--ink:#1b1f26;--ink2:#4b5563;--muted:#7c838e;--line:#d9dde3;--grid:#eef0f2;--blue:#1d4ed8;--orange:#ea580c;--blue-soft:#dbe7ff;--orange-soft:#ffe4d1}}
@page{{size:Letter;margin:13mm 12mm}}
*{{box-sizing:border-box}} body{{font-family:Helvetica,Arial,sans-serif;font-size:10.5px;color:var(--ink);margin:0;line-height:1.38}}
.mono{{font-family:Menlo,Consolas,"DejaVu Sans Mono",monospace;font-size:9.5px}} .n{{font-family:Menlo,Consolas,"DejaVu Sans Mono",monospace;text-align:right;font-variant-numeric:tabular-nums}}
h1{{font-size:18px;margin:0;font-family:Menlo,Consolas,monospace}} h2{{page-break-after:avoid;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:14px 0 6px;font-family:Menlo,Consolas,monospace;border-bottom:1px solid var(--line);padding-bottom:3px}}
.sub{{color:var(--muted);font-size:10px;font-family:Menlo,Consolas,monospace}} .muted{{color:var(--muted)}}
header{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid var(--ink);padding-bottom:6px}}
.env{{display:grid;grid-template-columns:150px 1fr 1fr;gap:14px;align-items:start;margin-top:8px;page-break-inside:avoid}}
.dial{{font-family:Menlo,Consolas,monospace}} .dial .state{{font-size:26px;font-weight:700;color:{state_col}}} .dial .num{{font-size:40px;font-weight:700;line-height:1}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:3px 6px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}} th{{font-family:Menlo,Consolas,monospace;font-size:8.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:500}}
th.n{{text-align:right}} tr{{page-break-inside:avoid}}
.tiles{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;page-break-inside:avoid}} .tiles.five{{grid-template-columns:repeat(5,1fr)}}
.tile{{border:1px solid var(--line);border-radius:4px;padding:5px 7px}} .tile .k{{font-size:8px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-family:Menlo,Consolas,monospace}} .tile .v{{font-size:15px;font-weight:700;font-family:Menlo,Consolas,monospace}} .tile .d{{font-size:8.5px;color:var(--ink2)}}
.chip{{display:inline-block;padding:0 6px;border-radius:999px;font-family:Menlo,Consolas,monospace;font-size:8.5px;font-weight:700;border:1px solid var(--line);color:var(--muted);white-space:nowrap}}
.chip.live{{background:var(--orange-soft);color:var(--orange);border-color:var(--orange)}} .chip.recent{{background:var(--blue-soft);color:var(--blue);border-color:var(--blue)}}
.tag{{font-family:Menlo,Consolas,monospace;font-size:8px;letter-spacing:.05em;color:var(--ink2)}} .tag.CAP,.tag.ONCONF{{color:var(--blue)}} .tag.SETUP,.tag.ONSET,.tag.CONT,.tag.FAIL{{color:var(--orange)}}
.z{{display:inline-block;min-width:38px;padding:0 4px;border-radius:3px;text-align:right}} .z.hot{{background:var(--orange-soft);color:var(--orange);font-weight:700}} .z.cold{{background:var(--blue-soft);color:var(--blue);font-weight:700}}
.neg{{color:var(--orange);font-weight:700}} .pos{{color:var(--blue);font-weight:700}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}}
.summary p{{margin:0 0 7px;font-size:11px;line-height:1.5}}
.pb{{page-break-before:always}} .avoid{{page-break-inside:avoid}}
footer{{margin-top:12px;border-top:1px solid var(--line);padding-top:5px;color:var(--muted);font-size:8.5px}}
</style></head><body>
<header><div><h1>Vol Tape Regime · daily reading</h1><div class="sub">13 vol and rate indices, ROC z vs own trailing 252d, against 38 ETFs · backtest 1990 to date, Feb to Jul 2020 excluded</div></div>
<div class="sub" style="text-align:right">as of <b>{D['asof']}</b> close<br>generated {gen}{('<br><span style="color:var(--orange)">stale: ' + e(', '.join(s['series']+' ('+s['last']+')' for s in stale)) + '</span>') if stale else ''}</div></header>

<h2>Environment</h2>
<div class="env">
  <div class="dial"><div class="state">{S['state']}</div><div class="num">{S['score']}</div><div class="sub">{S['days_in_state']} sessions in state · OUT at ≤ {S['thresholds']['out_in']}, back IN above {S['thresholds']['out_exit']}</div></div>
  <div><table><tr><th>component</th><th class="n">pts</th></tr><tr><td>base</td><td class="n">{S['base']}</td></tr>{comp_rows}</table></div>
  <div><table><tr><th>state</th><th class="n">time</th><th class="n">P(5% DD/21d)</th><th class="n">P(10% DD/63d)</th><th class="n">fwd 63d</th><th class="n">63d max DD</th></tr>
  {''.join(f"<tr><td>{nm}</td><td class='n'>{round(S['states'][nm]['share']*100)}%</td><td class='n'>{S['states'][nm]['P_off21']:.2f}</td><td class='n'>{S['states'][nm]['P10_63']:.2f}</td><td class='n'>{f1(S['states'][nm]['fwd63'])}%</td><td class='n'>{f1(S['states'][nm]['DD63'])}%</td></tr>" for nm in ('IN','OUT'))}</table>
  <div class="sub" style="margin-top:4px">stay-in book since 2008: {S['binary']['ann']}% a year, {S['binary']['vol']}% vol, max DD {S['binary']['maxDD']}% · buy and hold {S['buyhold']['ann']}%, {S['buyhold']['vol']}% vol, max DD {S['buyhold']['maxDD']}%</div></div>
</div>

<h2>Now</h2>
<div class="tiles">{tile_html(tiles)}</div>

<h2>Main rule board</h2>
<table><tr><th>rule</th><th>status</th><th>definition (ROC z)</th><th>what it meant historically</th><th>last first-fire · total</th></tr>{rows_main}</table>

<h2 class="pb">ROC tape · z vs own trailing 252d (orange ≥ +1, blue ≤ -1)</h2>
<div class="two">
<table><tr><th>series</th><th class="n">level</th><th class="n">pct</th><th class="n">ROC 5d %</th><th class="n">ROC 21d %</th><th class="n">z 3d</th><th class="n">z 5d</th><th class="n">z 10d</th><th class="n">z 21d</th></tr>{tape_rows}</table>
<table><tr><th>ratio</th><th class="n">level</th><th class="n">pct</th><th class="n">ROC z 3d</th><th class="n">ROC z 5d</th></tr>{ratio_rows}</table>
</div>

<h2>Ticker pair rules (ticker relative ROC × index ROC)</h2>
<table><tr><th>rule</th><th>status</th><th>definition</th><th>odds</th><th>last first-fire</th></tr>{rows_generic(T['tick_rules'])}</table>

<h2 class="pb">Realized-vol layer</h2>
<div class="tiles five">{tile_html(rv_tiles[:5])}</div>
<div class="tiles" style="margin-top:6px;grid-template-columns:repeat(4,1fr)">{tile_html(rv_tiles[5:])}</div>
<table style="margin-top:8px"><tr><th>rule</th><th>status</th><th>definition</th><th>odds</th><th>last first-fire</th></tr>{rows_generic(V['rv_rules'])}</table>

<h2 class="pb">Breadth · equal weight vs cap weight (context, not in the dial)</h2>
<div class="tiles">{tile_html(br_tiles)}</div>
<table style="margin-top:8px"><tr><th>cell</th><th>status</th><th>definition</th><th>odds, all days</th><th>since · last scored first-fire</th></tr>{rows_generic(B['cells'])}</table>
<table style="margin-top:8px"><tr><th>pair (EW / CW)</th><th class="n">21d %</th><th class="n">63d %</th><th class="n">ROC 5d z</th><th class="n">ROC 21d z</th><th class="n">ROC 63d z</th><th class="n">accel z</th><th class="n">P(off) 63d bot / top Q</th></tr>{br_rows}</table>

<h2 class="pb">Spoken summary</h2>
<div class="summary">{''.join(f'<p>{e(p)}</p>' for p in paras)}</div>
<footer>Rates of change are log changes over 1 to 21 sessions, z-scored against each series' own trailing 252 sessions. P(5% DD/21d) is the share of days after which SPY fell 5% or more within 21 sessions; base rate 0.174. The dial is base 60 plus rule points that persist for each rule's window; the state is OUT at 20 or below and returns to IN only above 35. Breadth and volume are reported as context and carry no weight in the dial. Source: Yahoo via the yfinance fork; ^VIX1Y and ^RVX are not served and are excluded.</footer>
</body></html>'''

pdf_path = os.path.join(R, 'VOL_TAPE_READING.pdf')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    try: b = p.chromium.launch()
    except Exception:
        exe = os.environ.get('CHROMIUM_PATH', '/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell'); b = p.chromium.launch(executable_path=exe)
    pg = b.new_page(); pg.set_content(HTML, wait_until='load'); pg.emulate_media(media='print')
    pg.pdf(path=pdf_path, format='Letter', print_background=True, prefer_css_page_size=True)
    b.close()
print('wrote', pdf_path, os.path.getsize(pdf_path) // 1024, 'KB')
print('\n'.join(paras))
