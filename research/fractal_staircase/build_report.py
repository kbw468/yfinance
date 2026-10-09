"""Assemble the HTML report from results/*.csv (inline SVG charts, no libraries).

Palette: blue / orange only (deutan-safe), validated with the dataviz validator.
Orange = more likely than the same-week universe, blue = less likely, grey = 1.0.
"""
import html
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))
OUT = os.environ.get("FS_REPORT", os.path.join(HERE, "report.html"))


def rd(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def esc(x):
    return html.escape(str(x))


def pct(x, d=1):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def lift_class(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nz"
    return "up" if v >= 1.05 else "dn" if v <= 0.95 else "nz"


def chip(v, d=2):
    return f'<span class="lift {lift_class(v)}">{num(v, d)}</span>'


# ------------------------------------------------------------------ svg charts
def hbar(items, lo=0.6, hi=None, ref=1.0, w=1000, row=24, label_w=420, title=""):
    """Horizontal lift bars from the 1.0 reference. items: (label, value, tone, tip);
    value None makes the label a group header."""
    vals = [v for _, v, _, _ in items if v is not None and v == v]
    hi = hi or max(1.5, max(vals) * 1.08)
    lo = min(lo, min(vals) * 0.95)
    pw = w - label_w - 56
    x = lambda v: label_w + (v - lo) / (hi - lo) * pw
    h = row * len(items) + 34
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    step = 0.1 if hi - lo < 1.2 else 0.2
    for t in np.arange(math.ceil(lo / step) * step, hi + 1e-9, step):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="6" y2="{h - 24}" class="grid"/>')
        s.append(f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for i, (lab, v, tone, tip) in enumerate(items):
        y = 8 + i * row
        if v is None:
            s.append(f'<text x="8" y="{y + row * 0.62}" class="ghead">{esc(lab)}</text>')
            continue
        x0, x1 = sorted([x(ref), x(v)])
        s.append(f'<text x="{label_w - 10}" y="{y + row * 0.62}" class="lab" text-anchor="end">{esc(lab)}</text>')
        s.append(f'<rect x="{x0:.1f}" y="{y + 4}" width="{max(x1 - x0, 1.5):.1f}" height="{row - 9}" rx="3" class="{tone}">'
                 f'<title>{esc(lab)}: {v:.2f}{(" · " + esc(tip)) if tip else ""}</title></rect>')
        tx = x(v) + (6 if v >= ref else -6)
        s.append(f'<text x="{tx:.1f}" y="{y + row * 0.62}" class="val" text-anchor="{"start" if v >= ref else "end"}">{v:.2f}</text>')
    s.append(f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="2" y2="{h - 22}" class="refline"/>')
    s.append("</svg>")
    return "".join(s)


def decile_bars(vals, title, lo=0.75, hi=1.25, w=260, h=130):
    """Ten bars (decile 1 = lowest feature value) drawn from the 1.0 line."""
    pl, pr, pt, pb = 26, 6, 18, 20
    pw, ph = w - pl - pr, h - pt - pb
    y = lambda v: pt + (hi - min(max(v, lo), hi)) / (hi - lo) * ph
    bw = pw / 10
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart small" role="img" aria-label="{esc(title)}">',
         f'<text x="{pl}" y="12" class="ptitle">{esc(title)}</text>']
    for t in (0.8, 1.0, 1.2):
        s.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="{"refline" if t == 1.0 else "grid"}"/>')
        s.append(f'<text x="{pl - 4}" y="{y(t) + 3:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for i, v in enumerate(vals):
        if v != v:
            continue
        y0, y1 = sorted([y(1.0), y(v)])
        tone = "fo" if v >= 1 else "fb"
        s.append(f'<rect x="{pl + i * bw + 1.5:.1f}" y="{y0:.1f}" width="{bw - 3:.1f}" height="{max(y1 - y0, 1):.1f}" rx="2" class="{tone}">'
                 f'<title>decile {i + 1}: lift {v:.2f}</title></rect>')
    s.append(f'<text x="{pl}" y="{h - 5}" class="tick">low</text><text x="{w - pr}" y="{h - 5}" class="tick" text-anchor="end">high</text>')
    s.append("</svg>")
    return "".join(s)


def heat(grid, row_name, col_name, title, w=330):
    """5x5 diverging heatmap of lift: blue < 1 < orange, grey at 1."""
    cell, pl, pt = 50, 70, 26
    h = pt + cell * 5 + 34
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart heat" role="img" aria-label="{esc(title)}">',
         f'<text x="{pl}" y="14" class="ptitle">{esc(title)}</text>']
    for _, r in grid.iterrows():
        i, j, v = int(r["row_q"]) - 1, int(r["col_q"]) - 1, r["lift"]
        t = max(-1.0, min(1.0, (v - 1) / 0.35))
        pole = "var(--orange)" if t > 0 else "var(--blue)"
        mix = abs(t) * 100
        ink = "on-strong" if abs(t) > 0.6 else "lab"
        x0, y0 = pl + j * cell, pt + (4 - i) * cell
        s.append(f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{cell - 2}" height="{cell - 2}" rx="3" '
                 f'style="fill:color-mix(in oklab, {pole} {mix:.0f}%, var(--div-mid))">'
                 f'<title>{esc(row_name)} Q{i + 1}, {esc(col_name)} Q{j + 1}: lift {v:.2f}, hit {r["hit"] * 100:.1f}%, n={int(r["n"])}</title></rect>')
        s.append(f'<text x="{x0 + cell / 2}" y="{y0 + cell / 2 + 4}" class="{ink} cellv" text-anchor="middle">{v:.2f}</text>')
    for k in range(5):
        s.append(f'<text x="{pl - 6}" y="{pt + (4 - k) * cell + cell / 2 + 4}" class="tick" text-anchor="end">Q{k + 1}</text>')
        s.append(f'<text x="{pl + k * cell + cell / 2}" y="{pt + 5 * cell + 14}" class="tick" text-anchor="middle">Q{k + 1}</text>')
    s.append(f'<text x="{pl + 2.5 * cell}" y="{h - 4}" class="axl" text-anchor="middle">{esc(col_name)} (low → high)</text>')
    s.append(f'<text x="12" y="{pt + 2.5 * cell}" class="axl" text-anchor="middle" transform="rotate(-90 12 {pt + 2.5 * cell})">{esc(row_name)}</text>')
    s.append("</svg>")
    return "".join(s)


def year_bars(df, w=1000, h=200):
    df = df.dropna(subset=["lift"])
    pl, pr, pt, pb = 34, 8, 12, 24
    pw, ph = w - pl - pr, h - pt - pb
    hi = max(2.6, df["lift"].max() * 1.05)
    y = lambda v: pt + (hi - v) / hi * ph
    bw = pw / len(df)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Quiet-base lift by year">']
    step = 0.5 if hi <= 3 else 1.0 if hi <= 6 else 2.0
    ticks = sorted(set([1.0] + [round(t, 2) for t in np.arange(step, hi + 1e-9, step)]))
    for t in ticks:
        if t <= hi:
            s.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="{"refline" if t == 1.0 else "grid"}"/>')
            s.append(f'<text x="{pl - 4}" y="{y(t) + 3:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for i, (_, r) in enumerate(df.iterrows()):
        v = r["lift"]
        y0, y1 = sorted([y(1.0), y(v)])
        faint = " faint" if r["n"] < 300 else ""
        s.append(f'<rect x="{pl + i * bw + 2:.1f}" y="{y0:.1f}" width="{bw - 4:.1f}" height="{max(y1 - y0, 1):.1f}" rx="2" class="{"fo" if v >= 1 else "fb"}{faint}">'
                 f'<title>{int(r["value"])}: lift {v:.2f}, hit {r["hit"] * 100:.1f}%, {int(r["n"])} obs</title></rect>')
        s.append(f'<text x="{pl + i * bw + bw / 2:.1f}" y="{h - 8}" class="tick" text-anchor="middle">{str(int(r["value"]))[2:]}</text>')
    s.append("</svg>")
    return "".join(s)


def hist(vals, w=1000, h=170, lo=0.9, hi=2.0, bins=22):
    pl, pr, pt, pb = 34, 8, 10, 22
    pw, ph = w - pl - pr, h - pt - pb
    cnt, edges = np.histogram(np.clip(vals, lo, hi - 1e-9), bins=bins, range=(lo, hi))
    top = cnt.max()
    x = lambda v: pl + (v - lo) / (hi - lo) * pw
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Distribution of quiet-base lifts across 400 threshold settings">']
    for t in np.arange(lo, hi + 1e-9, 0.2):
        s.append(f'<text x="{x(t):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{t:.1f}</text>')
    for c, a, b in zip(cnt, edges[:-1], edges[1:]):
        bh = c / top * ph
        s.append(f'<rect x="{x(a) + 1:.1f}" y="{pt + ph - bh:.1f}" width="{x(b) - x(a) - 2:.1f}" height="{bh:.1f}" rx="2" class="{"fo" if a >= 1 else "fb"}">'
                 f'<title>lift {a:.2f}–{b:.2f}: {c} settings</title></rect>')
    s.append(f'<line x1="{x(1.0):.1f}" x2="{x(1.0):.1f}" y1="{pt - 4}" y2="{pt + ph}" class="refline"/>')
    s.append("</svg>")
    return "".join(s)


def dumbbell(rows, w=1000, row=26):
    """Per test block: base hit rate (blue) and model top-10 hit rate (orange)."""
    pl, pr = 90, 60
    hi = max(max(r[1], r[2]) for r in rows) * 1.15
    pw = w - pl - pr
    x = lambda v: pl + v / hi * pw
    h = row * len(rows) + 30
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Walk-forward test blocks: base vs model top 10">']
    for t in np.arange(0, hi, 0.05):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{h - 22}" class="grid"/>')
        s.append(f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t * 100:.0f}%</text>')
    for i, (lab, b, t10) in enumerate(rows):
        yy = 8 + i * row + row / 2
        s.append(f'<text x="{pl - 10}" y="{yy + 4}" class="lab" text-anchor="end">{esc(lab)}</text>')
        s.append(f'<line x1="{x(b):.1f}" x2="{x(t10):.1f}" y1="{yy}" y2="{yy}" class="conn"/>')
        s.append(f'<circle cx="{x(b):.1f}" cy="{yy}" r="5" class="fb ring"><title>{esc(lab)} base {b * 100:.1f}%</title></circle>')
        s.append(f'<circle cx="{x(t10):.1f}" cy="{yy}" r="5" class="fo ring"><title>{esc(lab)} model top 10 {t10 * 100:.1f}%</title></circle>')
        s.append(f'<text x="{max(x(b), x(t10)) + 10:.1f}" y="{yy + 4}" class="val">{t10 / b:.2f}×</text>')
    s.append("</svg>")
    return "".join(s)


def spark(points, w=160, h=36):
    p = np.asarray(points, float)
    xs = np.linspace(2, w - 2, len(p))
    ys = h - 3 - p * (h - 6)
    d = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    return f'<svg viewBox="0 0 {w} {h}" class="spark"><polyline points="{d}" class="sline"/></svg>'


# ------------------------------------------------------------------ table helper
def table(df, cols, heads=None, fmt=None, cls="", wide=()):
    fmt = fmt or {}
    heads = heads or cols
    out = [f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in heads) + "</tr></thead><tbody>"]
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            s = f(v) if f else esc(v)
            tds.append(f'<td class="w">{s}</td>' if c in wide else f"<td>{s}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


REGIME_LABELS = {
    "mkt_med_mar21": "Universe median volatility", "mkt_rsp_spy_roc63": "Equal-weight minus cap-weight, 63d (RSP − SPY)",
    "mkt_med_vc21_126": "Universe median vol compression 21/126", "mkt_med_rv10_126": "Universe median relative volume",
    "mkt_vix": "VIX level", "mkt_vvix_vix": "VVIX ÷ VIX", "mkt_brd_up63": "Share of names up over 63d",
    "mkt_med_roc63": "Universe median ROC 63d", "mkt_med_vroc21": "Universe median vol ROC 21d",
    "mkt_brd_eff": "Share of names in efficient uptrends", "mkt_spy_roc63": "SPY ROC 63d",
    "stair_prev63": "Share of names already staircasing (63d)", "mkt_vix_term": "VIX ÷ VIX3M",
    "mkt_brd_hi": "Share of names within 2% of 63d high", "mkt_vix_pct252": "VIX percentile in its 1y range",
    "mkt_iwm_spy_roc63": "Small minus large caps, 63d (IWM − SPY)", "d_brd_up63": "Change in breadth, 4 weeks",
    "mkt_med_roc21": "Universe median ROC 21d", "d_stair_prev63": "Change in staircase share, 4 weeks",
    "stair_prev21": "Share of names already staircasing (21d)", "mkt_vix_roc21": "VIX ROC 21d",
    "mkt_spy_roc21": "SPY ROC 21d", "mkt_spy_ddh252": "SPY distance from 52-week high",
}

LABELS = {
    "roc63": "Price ROC 63d", "roc21": "Price ROC 21d", "roc126": "Price ROC 126d", "acc21": "Price ROC acceleration 21d",
    "vroc21": "Volatility ROC 21d", "vroc63": "Volatility ROC 63d", "vacc21": "Volatility ROC acceleration",
    "vc21_126": "Vol compression 21/126", "rc21_252": "Range compression 21/252", "rv10_126": "Relative volume 10/126",
    "vlroc21": "Volume ROC 21d", "vlacc21": "Volume ROC acceleration", "udv63": "Up/down volume 63d",
    "pvv21": "Triad: price↑ vol↓ volume↑", "mar63": "Volatility level (median daily move)", "rng63": "63-day price range",
    "mdd126": "6-month max drawdown", "mdd63": "63-day max drawdown", "ddh252": "Distance below 52-week high",
    "er63": "Path efficiency 63d", "er_min3": "Efficiency, worst of 21/63/126", "kt63": "Kendall trend 63d",
    "kt_min": "Kendall trend, worst of 3 scales", "hexp126": "Scaling exponent 126d", "hexp252": "Scaling exponent 252d",
    "tmpl_min": "Match to your chart (best of 3 scales)", "mem_y63_long": "Staircase memory (5y)",
    "bo_hold63": "Holding above prior base", "hl_steps63": "Higher-low steps 63d", "co126": "Co-movement with SPY",
    "rel_dn126": "Excess return on SPY down days", "stair_b63": "Already in a 63d staircase (no / yes)",
    "stair_b21": "Already in a 21d staircase (no / yes)", "stair_scales": "Staircasing at 21/63/126 bars (0 → all 3)", "dn63": "Drift-to-noise 63d",
    "pv_spread63": "Price ROC minus vol ROC 63d", "vv_spread21": "Volume ROC minus vol ROC 21d",
}


def main():
    hyp = rd("hypothesis_ladders.csv")
    uni63 = rd("univariate_y63.csv")
    uni21 = rd("univariate_y21.csv")
    grid = rd("quietbase_grid.csv")
    conf = rd("quietbase_confirmers.csv")
    seg = rd("quietbase_segments.csv")
    rules = rd("rules_y63.csv")
    rnull = rd("rules_null_y63.csv")
    rreg = rd("rules_y63_with_regime.csv")
    ic = rd("ic_all.csv")
    reg = rd("regime.csv")
    sh126 = rd("shapes126_y63.csv")
    tri = rd("triads_y63.csv")
    scan_files = sorted(f for f in os.listdir(RES) if f.startswith("scan_20"))
    scan = rd(scan_files[-1]) if scan_files else None
    scan_date = scan_files[-1][5:15] if scan_files else ""
    if os.environ.get("FS_SCAN_NOTE"):
        scan_date = f"{scan_date} ({os.environ['FS_SCAN_NOTE']})"
    used_rules = rd("scan_rules_used.csv")
    from common import best_model_tag as _bmt
    _tag = _bmt()
    prof = rd(f"model_profile_{_tag}.csv") if _tag else None

    base = hyp[hyp["ladder"] == "base"].iloc[0]
    qb = hyp[hyp["ladder"] == "quiet_base"].reset_index(drop=True)
    th = hyp[hyp["ladder"] == "thesis"].reset_index(drop=True)
    ths = hyp[hyp["ladder"] == "thesis_strong"].reset_index(drop=True)
    aug = hyp[hyp["ladder"] == "chart_in_august"].reset_index(drop=True)
    qb_full = qb.iloc[-1]
    strict = grid.sort_values("lift", ascending=False).iloc[0]
    grid["era_min"] = grid[[c for c in grid.columns if c.startswith("lift_20")]].min(axis=1)
    yr = seg[seg["split"] == "year"].copy()
    yr["value"] = yr["value"].astype(float)
    yr = yr.sort_values("value")
    y25 = yr.loc[yr["value"] == 2025, "lift"]
    y26 = yr.loc[yr["value"] == 2026, "lift"]
    ok_rules = rules[(rules["test_weeks"] >= 150) & (rules["test_n"] >= 800)].sort_values("test_lift", ascending=False)

    # ---------------------------------------------------------------- sections
    S = []
    icx = ic.set_index("feature")
    roc_like = [f for f in icx.index if f[2:].startswith(("roc", "acc", "vroc", "vacc", "rroc", "racc", "vlroc", "vlacc", "rv", "udv", "pv", "vv", "pvv", "dq_", "d_"))]
    roc_ic_max = icx.loc[roc_like, "ic_y63"].abs().max()
    # the read
    m_top10 = prof.set_index(prof.columns[0]).loc["top10"] if prof is not None else None
    m_all = prof.set_index(prof.columns[0]).loc["all"] if prof is not None else None
    n_qb_now = int((scan["qb_rungs"] == 4).sum()) if scan is not None else 0
    n_rule_now = int((scan["rules_hit"] > 0).sum()) if scan is not None else 0
    pw = rd("prototype_watch.csv")
    proto_bullet, proto_section = "", ""
    if pw is not None and scan is not None:
        w = pw.set_index("segment").loc["Prototype watch (filters applied)"]
        pp = rd("prototype_profiles.csv")
        ppa = pp[(pp["target"] == "proto") & (pp["cohort"] == "all")].set_index("profile")
        n_watch = int(scan["proto_watch"].sum()) if "proto_watch" in scan.columns else 0
        proto_bullet = (f"<li><b>MSI 2024 is the prototype, and quiet profiles precede runs of that grade.</b> MSI-grade runs "
                        f"(+25% or more over 126 sessions, no pullback deeper than 6%) start in 0.72% of stock-weeks; MSI's Apr–Oct 2024 stretch ranks in the top 0.1%. "
                        f"For this target the quiet base lifts the odds {num(ppa.loc['quiet base (scan)', 'lift'])}×, the JNJ/MSI base {num(ppa.loc['JNJ/MSI base', 'lift'])}×, "
                        f"TD's re-base {num(ppa.loc['TD re-base', 'lift'])}×. Prototype watch (any of the three, $2B+, no regional banks, no pinned names): "
                        f"<b>{num(w['lift_all'])}×</b> ({num(w['lift_2006_16'])} in 2006–16, {num(w['lift_2017_26'])} in 2017–26), {pct(w['hit_all'])} hit vs 0.72%. {n_watch} names today.</li>")
        proto_section = prototype_section(scan, pw, pp, scan_date)
    read = f"""
<section id="read" class="read">
  <h2>The read</h2>
  <ol class="bluf">
    <li><b>Your thesis has no edge as a predictor.</b> Price rising + volatility compressing + relative volume rising + below-median drawdown + price and compression accelerating + volume ROC rising:
      lift {num(th['lift_y63'].min())}–{num(th['lift_y63'].max())} at every rung (1.00 = same-week universe). The strong version (top-quintile momentum, bottom-quintile vol ROC, top-40% relative volume, bottom-40% drawdown) scores <b>{num(ths['lift_y63'].iloc[-1])}</b>.</li>
    <li><b>Buying the staircase once it's visible underperforms.</b> Names already in a 63-session staircase (your chart in August) continue at lift <b>{num(aug['lift_y63'].iloc[0])}</b>; with volatility compressing and relative volume rising, <b>{num(aug['lift_y63'].iloc[2])}</b>.</li>
    <li><b>What precedes the staircase is a quiet base</b>, your chart's April–June stretch: bottom-20% volatility, shallow 6-month drawdown, flat or lagging 6-month return, within ~10% of the 52-week high.
      Lift <b>{num(qb_full['lift_y63'])}</b> (eras {num(qb_full['lift_2006'])} / {num(qb_full['lift_2013'])} / {num(qb_full['lift_2020'])}), hit rate {pct(qb_full['hit_y63'])} vs {pct(base['hit_y63'])},
      {pct(qb_full['p_up63'], 0)} up after 63 sessions, median forward max drawdown {pct(qb_full['med_fdd63'])} vs {pct(base['med_fdd63'])}. Tight version: lift <b>{num(strict['lift'])}</b>, hit {pct(strict['hit'])}, {pct(strict['p_up63'], 0)} up, {num(strict['per_week'], 1)} names a week.</li>
    {proto_bullet}
    <li><b>The rule search found the same coil on its own:</b> the tightest 63-day range (bottom 10%), a bottom-quintile 6-month return, few new higher highs, and a fading trend. Out of sample (2017–2026): lift {num(ok_rules['test_lift'].iloc[:12].min())}–{num(ok_rules['test_lift'].iloc[:12].max())} for the top 12 rules, hit {pct(ok_rules['test_hit'].iloc[:12].min())}–{pct(ok_rules['test_hit'].iloc[:12].max())}.</li>
    <li><b>Volatility level is the one strong input, and it predicts drawdown, not gain.</b> Weekly rank correlation of volatility with the forward 63-session max drawdown: {num(-icx.loc['q_mar63', 'ic_neg_fdd63'])}; with the forward 63-session gain: {num(icx.loc['q_mar63', 'ic_fr63'], 3)}.
      No rate-of-change feature in price, volume or volatility reaches a rank correlation above {num(roc_ic_max, 3)} (absolute) with the staircase hit.</li>
    {f'<li><b>Walk-forward model:</b> the top 10 names a week hit {pct(m_top10["p_y63"])} vs {pct(m_all["p_y63"])} for the universe, out of sample 2010–2025; its lift comes mostly from drawdown control (P(max DD &lt; 5%) {pct(m_top10["p_dd63_lt5"])} vs {pct(m_all["p_dd63_lt5"])}).</li>' if m_top10 is not None else ''}
    <li><b>Calendar:</b> quiet-base signals from December to April run above lift 1.4 in every era. September–November signals were weak in the 2020s (October 0.30, November 0.32), though October was 1.27 in 2006–12. The 21-session version of the search finds the same coil at about half the edge (best out-of-sample lift ≈1.2).</li>
    <li><b>Today ({scan_date}):</b> {n_qb_now} names pass the full quiet base, {n_rule_now} pass at least one top out-of-sample rule. The quiet base ran at lift {num(y25.iloc[0]) if len(y25) else 'n/a'} in 2025 and {num(y26.iloc[0]) if len(y26) else 'n/a'} for 2026 starts through July.</li>
  </ol>
</section>"""
    S.append(read)
    rank_html = rank_section(scan_date)
    if rank_html:
        S.append(rank_html)

    # current scan
    if scan is not None:
        sc = scan.copy()
        sc["mcap_b"] = sc["mcap"].map(lambda m: f"${m / 1000:.1f}B" if m >= 1000 else f"${m:.0f}M")
        main_list = sc[(sc["qb_rungs"] == 4) | (sc["rules_hit"] > 0)].copy()
        model_list = sc[(sc["model_pct"] >= 0.99) & ~((sc["qb_rungs"] == 4) | (sc["rules_hit"] > 0))].sort_values("model_pct", ascending=False).copy()
        indt = rd("quietbase_industry.csv")
        ind_map = {} if indt is None else {r["industry"]: (r["lift_1st"], r["lift_2nd"]) for _, r in indt.iterrows()}
        main_list["ind_hist"] = main_list["industry"].map(lambda i: f"{chip(ind_map[i][0])} {chip(ind_map[i][1])}" if i in ind_map else '<span class="muted">n/a</span>')
        main_list["why"] = main_list.apply(lambda r: " ".join(filter(None, [
            '<span class="tag o">quiet base</span>' if r["qb_rungs"] == 4 else "",
            '<span class="tag o">strict</span>' if bool(r["qb_strict"]) else "",
            f'<span class="tag o">rules {esc(r["rules_list"])}</span>' if r["rules_hit"] > 0 else "",
            f'<span class="tag b">model {r["model_pct"] * 100:.1f} pct</span>' if r["model_pct"] == r["model_pct"] and r["model_pct"] >= 0.99 else "",
            '<span class="tag n">&lt;$2B: setup fails here</span>' if r["mcap"] < 2000 else "",
        ])), axis=1)
        main_list = main_list.sort_values(["signal_count", "rules_hit", "qb_rungs", "model_pct"], ascending=False)
        cal_top = rd(f"model_calibration_{_tag}.csv") if _tag else None
        cal_note = ""
        if cal_top is not None:
            ct = cal_top.set_index(cal_top.columns[0])
            top_rows = ct.iloc[-2:]
            hit_top = (top_rows["p_y63"] * top_rows["n"]).sum() / top_rows["n"].sum()
            cal_note = f"Historical hit for the model's top 1% each week: {pct(hit_top)} vs {pct(ct['p_y63'].mul(ct['n']).sum() / ct['n'].sum())} for all names."
        mcols = ["ticker", "sector", "industry", "mcap_b", "roc126", "roc63", "roc21", "ddh252", "mar63", "mdd126", "model_pct", "top_drivers"]
        mheads = ["Ticker", "Sector", "Industry", "Mkt cap", "ROC 126d", "ROC 63d", "ROC 21d", "From 52w high", "Median daily move", "6m max DD", "Model pct", "Largest drivers"]
        cols = ["ticker", "sector", "industry", "mcap_b", "price", "why", "ind_hist", "roc126", "roc63", "roc21", "ddh252", "mar63", "rng63", "mdd126", "vroc21", "rv10_126", "model_pct"]
        heads = ["Ticker", "Sector", "Industry", "Mkt cap", "Price", "Signals", "Industry quiet-base lift 06–16 / 16–26", "ROC 126d", "ROC 63d", "ROC 21d", "From 52w high", "Median daily move", "63d range", "6m max DD", "Vol ROC 21d", "Rel vol 10/126", "Model pct"]
        f = {"price": lambda v: f"{v:.2f}", "why": lambda v: v, "ind_hist": lambda v: v, "roc126": pct, "roc63": pct, "roc21": pct, "ddh252": pct,
             "mar63": lambda v: pct(v, 2), "rng63": pct, "mdd126": pct, "vroc21": pct, "rv10_126": pct,
             "model_pct": lambda v: "" if v != v else f"{v * 100:.1f}", "ticker": lambda v: f"<b>{esc(v)}</b>"}
        rules_tbl = ""
        if used_rules is not None:
            ur = used_rules.copy()
            ur["id"] = [f"R{i + 1}" for i in range(len(ur))]
            rules_tbl = table(ur, ["id", "rule", "test_hit", "test_lift", "test_weeks"], ["Rule", "Conditions (q_ = rank in the universe that week, 0–1)", "Hit 2017–26", "Lift 2017–26", "Weeks fired"],
                              {"test_hit": pct, "test_lift": lambda v: chip(v), "rule": lambda v: f"<code>{esc(v)}</code>"}, cls="compact", wide=("rule",))
        thesis_now = sc[sc["thesis"] == True]
        aug_now = sc[sc["chart_aug"] == True]
        th_cols = ["ticker", "sector", "industry", "mcap_b", "roc63", "roc21", "vroc21", "rv10_126", "mdd126", "model_pct"]
        th_heads = ["Ticker", "Sector", "Industry", "Mkt cap", "ROC 63d", "ROC 21d", "Vol ROC 21d", "Rel vol 10/126", "6m max DD", "Model pct"]
        S.append(f"""
<section id="now">
  <h2>Setups on the tape, {scan_date}</h2>
  <p class="lede">Every name below meets a condition that beat its own week's universe out of sample. Signals: <span class="tag o">quiet base</span> = all four rungs;
  <span class="tag o">rules</span> = matches top out-of-sample rules (listed under the table); <span class="tag b">model</span> = also in the walk-forward model's top 1% today.
  Industry column: that industry's quiet-base lift in 2006–16 and 2016–26 (n/a = fewer than 400 observations).
  Historical odds for the quiet base: hit {pct(qb_full['hit_y63'])}, {pct(qb_full['p_up63'], 0)} up at 63 sessions, median forward max drawdown {pct(qb_full['med_fdd63'])}.</p>
  {table(main_list, cols, heads, f, cls="scan", wide=("why",))}
  <details><summary>The rules referenced above ({len(used_rules) if used_rules is not None else 0})</summary>{rules_tbl}</details>
  <details><summary>Walk-forward model, top 1% today that are not in the list above ({len(model_list)})</summary>
    <p class="note">{cal_note} The model's out-of-sample edge (top-10 lift ≈1.28) is smaller than the quiet base's and the rules'. Its current top names lean on size (dollar volume) and market-state inputs.</p>
    {table(model_list, mcols, mheads, f | {"top_drivers": lambda v: f"<code>{esc(v)}</code>"}, cls="compact", wide=("top_drivers",))}</details>
  <details><summary>Names matching your full thesis today ({len(thesis_now)}): backtested lift {num(th['lift_y63'].iloc[-1])}</summary>
    {table(thesis_now.sort_values('model_pct', ascending=False), th_cols, th_heads, f, cls="compact")}</details>
  <details><summary>Names already staircasing with vol compressing and rel volume rising ({len(aug_now)}): backtested lift {num(aug['lift_y63'].iloc[2])}</summary>
    {table(aug_now.sort_values('model_pct', ascending=False), th_cols, th_heads, f, cls="compact")}</details>
</section>""")

    if proto_section:
        S.append(proto_section)
    msi_section = msi_like_section(scan_date)
    if msi_section:
        S.append(msi_section)

    # thesis vs quiet base
    def rows_of(df):
        return [(r["rung"], r["lift_y63"], "fo" if r["lift_y63"] >= 1 else "fb", f"{r['per_week']:.1f}/wk, hit {r['hit_y63'] * 100:.1f}%") for _, r in df.iterrows()]
    items = [("Your thesis", None, "", "")] + rows_of(th) + [("Strong version", None, "", "")] + rows_of(ths) + \
        [("Your chart in August (already staircasing)", None, "", "")] + rows_of(aug) + [("Quiet base", None, "", "")] + rows_of(qb)
    lad_cols = ["ladder", "rung", "per_week", "hit_y21", "hit_y63", "lift_y21", "lift_y63", "med_fr63", "med_fdd63", "p_up63", "lift_2006", "lift_2013", "lift_2020"]
    lad_heads = ["Ladder", "Rung (cumulative)", "Names/wk", "Hit 21", "Hit 63", "Lift 21", "Lift 63", "Med fwd 63d", "Med fwd max DD", "Up at 63d", "Lift 06–12", "Lift 13–19", "Lift 20–26"]
    lf = {"per_week": lambda v: f"{v:.1f}", "hit_y21": pct, "hit_y63": pct, "lift_y21": chip, "lift_y63": chip, "med_fr63": pct,
          "med_fdd63": pct, "p_up63": lambda v: pct(v, 0), "lift_2006": chip, "lift_2013": chip, "lift_2020": chip}
    S.append(f"""
<section id="thesis">
  <h2>Your thesis vs. the quiet base</h2>
  <p class="lede">Each rung adds one condition to the one above. Lift = hits ÷ the hits the same rows would have scored at their own week's universe hit rate, so bull-market weeks give no free credit.
  Hit = +10% or better within 63 sessions with max drawdown ≤ ⅓ of the gain (21-session version: +5%). Universe hit rate {pct(base['hit_y63'])} (63) / {pct(base['hit_y21'])} (21).</p>
  <figure><figcaption>Lift of each rung (orange beats the same-week universe, blue trails it)</figcaption>{hbar(items, lo=0.8, hi=1.45, title="Thesis and quiet-base ladder lifts")}</figure>
  {table(hyp, lad_cols, lad_heads, lf, cls="compact", wide=("rung",))}
</section>""")

    # ROC deciles
    u = uni63.set_index("feature")
    u21 = uni21.set_index("feature")
    roc_feats = ["roc21", "roc63", "acc21", "vroc21", "vroc63", "vacc21", "vc21_126", "rc21_252", "rv10_126", "vlroc21", "vlacc21", "udv63", "pvv21", "pv_spread63", "vv_spread21", "dn63"]
    lvl_feats = ["mar63", "rng63", "mdd126", "ddh252"]

    def panel(fe, src):
        if fe not in src.index:
            return ""
        vals = [src.loc[fe].get(f"L{i}", np.nan) for i in range(10)]
        return decile_bars(vals, LABELS.get(fe, fe))
    S.append(f"""
<section id="roc">
  <h2>Rates of change in price, volume and volatility, one at a time</h2>
  <p class="lede">Lift by weekly decile of each feature (decile 1 = lowest value that week), 63-session hit. Orange bars beat the week's universe, blue bars trail it. 2006–2026, {int(base['n']):,} stock-weeks.</p>
  <div class="sm">{''.join(panel(fe, u) for fe in roc_feats)}</div>
  <h3>Levels, not rates of change, carry the signal</h3>
  <div class="sm">{''.join(panel(fe, u) for fe in lvl_feats)}</div>
  <h3>Same panels, 21-session hit</h3>
  <div class="sm">{''.join(panel(fe, u21) for fe in ["roc21", "vroc21", "rv10_126", "vlroc21", "mar63", "rng63"])}</div>
</section>""")

    # vol x ROC heatmaps
    hm = []
    for a, b, t in [("q_mar63", "q_roc63", "Volatility level × price ROC 63d"), ("q_mar63", "q_vroc21", "Volatility level × volatility ROC 21d"),
                    ("q_mar63", "q_rv10_126", "Volatility level × relative volume"), ("q_mar63", "q_acc21", "Volatility level × price ROC acceleration"),
                    ("q_rng63", "q_roc126", "63-day range × 6-month ROC"), ("q_mdd126", "q_roc126", "6-month max DD × 6-month ROC")]:
        g = rd(f"grid_{a}__{b}.csv")
        if g is not None:
            hm.append(heat(g, LABELS.get(a[2:], a), LABELS.get(b[2:], b), t))
    S.append(f"""
<section id="grids">
  <h2>Interactions</h2>
  <p class="lede">5×5 weekly quintiles, lift per cell (row Q1 = lowest). The edge sits in the low-volatility / tight-range / lagging corner. Within low volatility, rising rates of change move the cell toward 1.0, not away from it.</p>
  <div class="heats">{''.join(hm)}</div>
  {triad_block(tri)}
</section>""")

    # confirmers on quiet base
    citems = [(r["confirmer"], r["lift_change"], "fo" if r["lift_change"] >= 1 else "fb", f"lift {r['lift']:.2f}, {r['per_week']:.1f}/wk")
              for _, r in conf.iloc[1:].sort_values("lift_change", ascending=False).iterrows()]
    S.append(f"""
<section id="confirm">
  <h2>Confirming factors on top of the quiet base</h2>
  <p class="lede">Lift of the quiet base with one confirmer added, relative to the quiet base alone (1.00 = no change; base alone {num(conf.iloc[0]['lift'])}).</p>
  <figure>{hbar(citems, lo=0.9, hi=1.1, title="Confirmer lift change")}</figure>
  {table(conf, ["confirmer", "per_week", "hit", "lift", "lift21", "med_fr63", "med_fdd63", "p_up63", "lift_2006", "lift_2013", "lift_2020"],
         ["Added to quiet base", "Names/wk", "Hit 63", "Lift 63", "Lift 21", "Med fwd 63d", "Med fwd max DD", "Up at 63d", "Lift 06–12", "Lift 13–19", "Lift 20–26"],
         {"per_week": lambda v: f"{v:.1f}", "hit": pct, "lift": chip, "lift21": chip, "med_fr63": pct, "med_fdd63": pct, "p_up63": lambda v: pct(v, 0), "lift_2006": chip, "lift_2013": chip, "lift_2020": chip}, cls="compact", wide=("confirmer",))}
</section>""")

    # robustness + segments
    segt = seg[seg["split"] != "year"].copy()
    S.append(f"""
<section id="robust">
  <h2>Quiet base: robustness, sectors, size, years</h2>
  <p class="lede">400 neighbouring threshold settings (volatility cut 10–40%, 6-month drawdown cut 20–50%, 6-month return cut 30–100%, distance from 52-week high 5%–any).
  {pct((grid['lift'] > 1.15).mean(), 0)} beat 1.15, {pct((grid['lift'] > 1.25).mean(), 0)} beat 1.25, {pct((grid['era_min'] > 1.1).mean(), 0)} beat 1.10 in all three eras.</p>
  <figure><figcaption>Distribution of lift across the 400 settings</figcaption>{hist(grid['lift'].to_numpy())}</figure>
  <figure><figcaption>Lift by year (quiet base, all four rungs; 2026 = starts through July)</figcaption>{year_bars(yr)}</figure>
  {table(segt, ["split", "value", "per_week", "hit", "lift", "base_lift_segment", "med_fr63", "med_fdd63", "p_up63"],
         ["Split", "Segment", "Names/wk", "Hit 63", "Quiet-base lift", "Segment lift (no filter)", "Med fwd 63d", "Med fwd max DD", "Up at 63d"],
         {"per_week": lambda v: f"{v:.1f}", "hit": pct, "lift": chip, "base_lift_segment": chip, "med_fr63": pct, "med_fdd63": pct, "p_up63": lambda v: pct(v, 0)}, cls="compact")}
  <p class="note">Market-cap buckets use today's cap (the only cap in the file); the dollar-volume quintiles are point-in-time.</p>
  {industry_block()}
  {month_block()}
</section>""")

    # fractal section
    fr_feats = ["er63", "er_min3", "kt63", "kt_min", "hexp126", "hexp252", "tmpl_min", "hl_steps63", "bo_hold63", "mem_y63_long", "stair_b21", "stair_b63", "stair_scales"]
    frows = []
    for fe in fr_feats:
        if fe in u.index:
            r = u.loc[fe]
            frows.append(dict(feature=LABELS.get(fe, fe), rho=r["rho"], low=r["lift_low"], high=r["lift_high"], best=r["best_lift"], era=r["era_min"]))
    frt = pd.DataFrame(frows)
    shapes_html = ""
    if sh126 is not None:
        s2 = sh126.rename(columns={sh126.columns[0]: "shape"}).copy()
        pts = [c for c in s2.columns if c.startswith("p") and c[1:].isdigit()]
        s2 = s2.sort_values("lift_test", ascending=False)
        cards = []
        for _, r in pd.concat([s2.head(4), s2.tail(2)]).iterrows():
            cards.append(f'<div class="shape">{spark(r[pts].to_numpy(float))}<div><b>Shape {int(r["shape"])}</b><br>'
                         f'train {num(r["lift_train"])} · test {chip(r["lift_test"])}</div></div>')
        shapes_html = "".join(cards)
    S.append(f"""
<section id="fractal">
  <h2>Fractal and geometry tests</h2>
  <p class="lede">Multi-scale path efficiency (net move ÷ distance travelled), Kendall trend at 21/63/126 bars and their worst-of-three agreement, a median-based scaling exponent (how |k-day moves| grow with k),
  shape match to your chart at 40/80/160 bars, higher-low steps, holding above the prior base, staircase memory, and the already-in-a-staircase state.</p>
  {table(frt, ["feature", "rho", "low", "high", "best", "era"], ["Feature", "Monotonic (rank ρ)", "Lift, lowest bucket", "Lift, highest bucket", "Best bucket", "Best bucket, worst era"],
         {"rho": lambda v: num(v), "low": chip, "high": chip, "best": chip, "era": chip}, cls="compact")}
  <h3>Shape alphabet: 30 prototype 126-bar paths (k-medians on min-max scaled paths, learned on 2006–2016)</h3>
  <p class="lede">Best four by out-of-sample lift (2017–2026) and the worst two. Shapes that held up: rally then a long plateau (11, 16) and a plateau or rally that ends in a pullback (19, 26). Shapes that failed: decline then flatline (5, 22). Lift = raw hit rate vs the era's base.</p>
  <div class="shapes">{shapes_html}</div>
</section>""")

    # rules
    rn = rnull["lift"].max() if rnull is not None else np.nan
    r21 = rd("rules_y21.csv")
    r21_html = ""
    if r21 is not None:
        ok21 = r21[(r21["test_weeks"] >= 150) & (r21["test_n"] >= 800)].sort_values("test_lift", ascending=False)
        r21_html = ("<h3>Same search, 21-session hit</h3>" + table(ok21.head(8), ["rule", "train_lift", "test_n", "test_hit", "test_lift", "test_weeks"],
                    ["Rule", "Train lift", "Test obs", "Test hit", "Test lift", "Test weeks"],
                    {"rule": lambda v: f"<code>{esc(v)}</code>", "train_lift": lambda v: num(v), "test_hit": pct, "test_lift": chip}, cls="compact", wide=("rule",)) +
                    f'<p class="note">Same coil (tightest 63-day range, price not extended above its prior base), weaker at the shorter horizon: median out-of-sample lift across all {len(r21)} rules {num(r21["test_lift"].median())}; the training winners that leaned on sector momentum fell to ≈1.0.</p>')
    rr_top = rreg.sort_values("train_score", ascending=False).head(5) if rreg is not None else None
    S.append(f"""
<section id="rules">
  <h2>Rule search</h2>
  <p class="lede">Beam search over {1957:,} threshold conditions, up to 4 per rule, on 2006–2016 only; scored by a conservative Beta-posterior bound on the hit rate divided by the same-week universe expectation, counts shrunk for overlapping forward windows.
  Scored again on 2017-04 → 2026-07, which the search never saw. Shuffling labels within each week and re-running the search produced training lifts up to {num(rn)}: a training lift below that means nothing, so the out-of-sample column is the one to read.</p>
  {table(ok_rules.head(15), ["rule", "train_lift", "test_n", "test_hit", "test_lift", "test_weeks"],
         ["Rule", "Train lift", "Test obs", "Test hit", "Test lift", "Test weeks"],
         {"rule": lambda v: f"<code>{esc(v)}</code>", "train_lift": lambda v: num(v), "test_hit": pct, "test_lift": chip}, cls="compact", wide=("rule",))}
  <p class="note">All {len(rules)} rules kept: median out-of-sample lift {num(rules['test_lift'].median())}; 4-condition rules {num(rules[rules['depth'] == 4]['test_lift'].median())}.
  A first pass that also allowed market-regime conditions reached training lift {num(rr_top['train_lift'].max()) if rr_top is not None else ''} and fell to {num(rr_top['test_lift'].min()) if rr_top is not None else ''}–{num(rr_top['test_lift'].max()) if rr_top is not None else ''} out of sample: regime conditions overfit and were dropped.</p>
  {r21_html}
</section>""")

    # model
    from common import MODEL_TAGS, best_model_tag, family
    tag = best_model_tag()
    comp = []
    for t, name in MODEL_TAGS.items():
        p = rd(f"model_profile_{t}.csv")
        if p is None:
            continue
        p = p.set_index(p.columns[0])
        comp.append(dict(model=name + (" (used for the scan)" if t == tag else ""), base=p.loc["all", "p_y63"], top5=p.loc["top5", "p_y63"],
                         top10=p.loc["top10", "p_y63"], top25=p.loc["top25", "p_y63"], lift10=p.loc["top10", "p_y63"] / p.loc["all", "p_y63"],
                         dd5=p.loc["top10", "p_dd63_lt5"], dd5_all=p.loc["all", "p_dd63_lt5"], mdd=p.loc["top10", "med_fdd63"], up=p.loc["top10", "p_up63"]))
    if tag:
        prof = rd(f"model_profile_{tag}.csv")
        cal = rd(f"model_calibration_{tag}.csv")
        imp = rd(f"model_importance_{tag}.csv")
        cal = cal.rename(columns={cal.columns[0]: "bucket"})

        def nice_bucket(sv):
            lo_, hi_ = [float(x) for x in str(sv).strip("([]").split(",")]
            return f"{max(lo_, 0) * 100:g}–{hi_ * 100:g}"
        cal["bucket"] = cal["bucket"].map(nice_bucket)
        im = imp.rename(columns={imp.columns[0]: "feature", imp.columns[1]: "gain"})
        im["family"] = im["feature"].map(family)
        fam = im.groupby("family")["gain"].sum().sort_values(ascending=False).reset_index()
        blocks = []
        o = pd.read_parquet(os.path.join(DATA_DIR, f"oos_{tag}.parquet"), columns=["date", "score", "y63"])
        o["rk"] = o.groupby("date")["score"].rank(ascending=False, method="first")
        o["blk"] = ((o["date"].dt.year - 2010) // 2) * 2 + 2010
        for bk, g in o.groupby("blk"):
            blocks.append((f"{bk}–{str(bk + 1)[2:]}", g["y63"].mean(), g.loc[g["rk"] <= 10, "y63"].mean()))
        S.append(f"""
<section id="model">
  <h2>Walk-forward model</h2>
  <p class="lede">Gradient-boosted trees, retrained every two years on data whose 63-session outcomes were complete a quarter before each test block; out of sample 2010–2025.
  Three versions: Bernoulli log-loss with stock-level inputs only, the same with market-regime inputs added, and a LambdaRank version that ranks names within each week.</p>
  {table(pd.DataFrame(comp), ["model", "base", "top5", "top10", "top25", "lift10", "dd5", "dd5_all", "mdd", "up"],
         ["Model", "Universe hit", "Top 5 hit", "Top 10 hit", "Top 25 hit", "Top 10 lift", "Top 10: max DD < 5%", "Universe: max DD < 5%", "Top 10: med fwd max DD", "Top 10: up at 63d"],
         {"base": pct, "top5": pct, "top10": pct, "top25": pct, "lift10": chip, "dd5": pct, "dd5_all": pct, "mdd": pct, "up": lambda v: pct(v, 0)}, cls="compact", wide=("model",))}
  <figure><figcaption><span class="key b"></span> universe hit rate &nbsp; <span class="key o"></span> model top 10 per week ({esc(MODEL_TAGS[tag]).lower()})</figcaption>{dumbbell(blocks)}</figure>
  <h3>Calibration: score percentile within the week → realised outcomes</h3>
  {table(cal, ["bucket", "n", "p_y63", "p_up63", "p_dd63_lt5", "med_fr63", "med_fdd63"], ["Score percentile", "Obs", "Hit 63", "Up at 63d", "Max DD < 5%", "Med fwd 63d", "Med fwd max DD"],
         {"n": lambda v: f"{int(v):,}", "p_y63": pct, "p_up63": lambda v: pct(v, 0), "p_dd63_lt5": pct, "med_fr63": pct, "med_fdd63": pct}, cls="compact")}
  <h3>Where the model finds its signal (share of split gain)</h3>
  {table(fam, ["family", "gain"], ["Feature family", "Share"], {"gain": pct}, cls="compact narrow")}
</section>""")

    # regime
    if reg is not None:
        rg = reg.copy()
        rg["state"] = rg["state"].map(lambda k: REGIME_LABELS.get(k, k))
        S.append(f"""
<section id="regime">
  <h2>When staircases cluster</h2>
  <p class="lede">The weekly universe hit rate ranges from 0% to 47%. Market-state variables explain little of it (rank ρ across weeks ≤ 0.1 in absolute value). Today's readings are shown as percentiles of 2006–2026.</p>
  {table(rg.head(14), ["state", "rho_y63", "y63_low", "y63_mid", "y63_high", "today_pctile"], ["Market state", "Rank ρ with weekly hit", "Hit, low third", "Hit, mid third", "Hit, high third", "Today (percentile)"],
         {"rho_y63": lambda v: num(v), "y63_low": pct, "y63_mid": pct, "y63_high": pct, "today_pctile": lambda v: pct(v, 0)}, cls="compact", wide=("state",))}
</section>""")

    # method
    S.append(f"""
<section id="method">
  <h2>Method</h2>
  <ul class="meth">
    <li><b>Data:</b> Yahoo Finance daily OHLCV (split/dividend adjusted), 2004-01 → {scan_date}, for the 2,367 tickers in your file; 2,293 pass a $2M median daily dollar-volume screen at some point. One observation per name per week; 1,086 weeks; 1.39M labelled stock-weeks from 2006.</li>
    <li><b>Hit definitions:</b> 63-session staircase = forward log return ≥ +10% and max peak-to-trough drawdown along the path ≤ ⅓ of that gain. 21-session = +5%, same ⅓ rule. Your chart from Aug 3 to Oct 8 (+11%, worst pullback ≈2%) qualifies.</li>
    <li><b>No Gaussian machinery anywhere:</b> volatility = median absolute daily move and median high–low range; trend = Kendall rank statistic and Theil–Sen (median) slopes; relative volume on medians; drawdowns direct from price; all comparisons on weekly percentile ranks; uncertainty from Beta posteriors, week-shuffle nulls, quarter-block resampling and era splits. No standard deviation, z-score, OLS, Sharpe, t-test or moving average.</li>
    <li><b>Regime-neutral lift:</b> hits ÷ Σ(each row's own-week universe hit rate). A filter that only fires in strong markets scores 1.00.</li>
    <li><b>Out of sample:</b> rule search trained 2006–2016, tested 2017-04 → 2026-07; model retrained every two years with a quarter embargo.</li>
    <li><b>Universe:</b> today's constituents only, so names that left the index are missing. Lifts compare names within the same weeks, so this hits all of them equally.</li>
    <li><b>Code:</b> <code>research/fractal_staircase/</code> on branch <code>claude/russell-1000-fractal-search-lpl5bz</code>.</li>
  </ul>
</section>""")

    toc = "".join(f'<a href="#{i}">{t}</a>' for i, t in [("read", "Read"), ("rank", "Ranked"), ("now", "Setups now"), ("proto", "Prototype watch"), ("msi", "MSI look-alikes"), ("thesis", "Thesis vs base"), ("roc", "Rates of change"),
                                                          ("grids", "Interactions"), ("confirm", "Confirmers"), ("robust", "Robustness"), ("fractal", "Fractal"),
                                                          ("rules", "Rules"), ("model", "Model"), ("regime", "Regime"), ("method", "Method")])
    page = TEMPLATE.replace("{{TOC}}", toc).replace("{{BODY}}", "\n".join(S)).replace("{{DATE}}", scan_date)
    with open(OUT, "w") as fh:
        fh.write(page)
    print("wrote", OUT, f"{len(page) / 1024:.0f} KB")


def prototype_section(scan, pw, pp, scan_date):
    yrs = rd("prototype_watch_years.csv")
    sf = rd("prototype_single_features.csv")
    ppa = pp[pp["target"] == "proto"].copy()
    keep = ["quiet base (scan)", "JNJ/MSI base", "TD re-base", "quiet base, any 6m return", "already staircasing 63d"]
    ppa = ppa[ppa["profile"].isin(keep)]
    w = scan[scan["proto_watch"] == True].copy()
    w["mcap_b"] = w["mcap"].map(lambda m: f"${m / 1000:.1f}B")
    w = w.sort_values("mcap", ascending=False)
    pinned = ", ".join(scan.loc[scan["pinned"] == True, "ticker"].tolist()) if "pinned" in scan.columns else ""
    yr_html = ""
    if yrs is not None:
        y = yrs.rename(columns={"year": "value"})
        y["hit"] = y["hits"] / y["n"]
        yr_html = f'<figure><figcaption>Prototype watch lift by year (faded = fewer than 300 observations)</figcaption>{year_bars(y, h=200)}</figure>'
    vol = ""
    if sf is not None:
        v = sf.set_index("feature")
        vol = (f"Volatility is the gate: the quietest decile carries {num(v.loc['q_mar63', 'best_lift'])}× the odds and the loudest decile "
               f"{num(v.loc['q_mar63', 'lift_d10'], 2)}×. Tight 63-day range (decile 1): {num(v.loc['q_rng63', 'best_lift'])}×. "
               f"Shallow 6-month drawdown (decile 1): {num(v.loc['q_mdd126', 'best_lift'])}×.")
    prof_tbl = table(ppa, ["profile", "cohort", "per_week", "hit", "lift", "lift_2006_16", "lift_2017_26", "cohort_lift", "up126", "med_dd126"],
                     ["Profile", "Cohort", "Names/wk", "Hit", "Lift", "2006–16", "2017–26", "Cohort lift", "Up at 126", "Med worst DD 126"],
                     {"per_week": lambda v: f"{v:.1f}", "hit": pct, "lift": chip, "lift_2006_16": chip, "lift_2017_26": chip,
                      "cohort_lift": chip, "up126": lambda v: pct(v, 0), "med_dd126": pct}, cls="compact", wide=("profile",))
    seg_tbl = table(pw, ["segment", "per_week", "lift_all", "lift_2006_16", "lift_2017_26", "hit_all", "up126", "med_fr126", "med_dd126"],
                    ["Segment", "Names/wk", "Lift", "2006–16", "2017–26", "Hit", "Up at 126", "Med 126-session return", "Med worst DD 126"],
                    {"per_week": lambda v: f"{v:.1f}", "lift_all": chip, "lift_2006_16": chip, "lift_2017_26": chip, "hit_all": pct,
                     "up126": lambda v: pct(v, 0), "med_fr126": pct, "med_dd126": pct}, cls="compact", wide=("segment",))
    names_tbl = table(w, ["ticker", "sector", "industry", "mcap_b", "proto_profile", "q_mar63", "q_rng63", "q_mdd126", "q_roc126", "ddh252", "model_pct"],
                      ["Ticker", "Sector", "Industry", "Mkt cap", "Profile", "Volatility pct", "63d range pct", "6m DD pct", "6m return pct", "From 52w high", "Model pct"],
                      {"ticker": lambda v: f"<b>{esc(v)}</b>", "q_mar63": lambda v: f"{v * 100:.1f}", "q_rng63": lambda v: f"{v * 100:.1f}",
                       "q_mdd126": lambda v: f"{v * 100:.1f}", "q_roc126": lambda v: f"{v * 100:.0f}", "ddh252": pct,
                       "model_pct": lambda v: "" if v != v else f"{v * 100:.1f}"}, cls="compact")
    return f"""
<section id="proto">
  <h2>Prototype watch: MSI-grade staircases</h2>
  <p class="lede">MSI from Oct 27 2023 to the Nov 8 2024 peak: +87% in 260 sessions, worst pullback 5.9%. Best 126-session stretch, Apr 18 → Oct 16 2024: +41% with a 3.8% worst pullback,
  no close more than 5% below its running high, new highs on 44% of days. That stretch ranks in the top 0.1% of all 1.37M stock-weeks since 2006 by gain over worst pullback.
  The target here: +25% or more over 126 sessions with no pullback deeper than 6% (0.72% of stock-weeks; 1.46% at $50B+).</p>
  <p class="lede">{vol} Every quiet profile from your case studies works for this target, including the ones that didn't work for the 63-session hit.</p>
  {prof_tbl}
  <h3>Filters that held in both halves</h3>
  {seg_tbl}
  <p class="note">Pinned = volatility and 63-day range both in the bottom 1% of the universe, the signature of a stock held near a pending deal price. Pinned today: {esc(pinned)}.</p>
  {yr_html}
  <h3>On the watch, {scan_date} ({len(w)} names)</h3>
  {names_tbl}
  <p class="note">Percentile columns are ranks in the universe that week (0 = quietest / tightest / shallowest / weakest). Hits on this target cluster in years (2010–12, 2016, 2019, 2021, 2023–24).</p>
</section>"""


def rank_section(scan_date):
    files = sorted(f for f in os.listdir(RES) if f.startswith("rank_20"))
    tw, ta = rd("rank_tests.csv"), rd("rank_tests_all.csv")
    if not files or tw is None or ta is None:
        return ""
    r = rd(files[-1])
    r["mcap_b"] = r["mcap"].map(lambda v: f"${v / 1000:.1f}B")
    pinned = r[(r["deal_pinned"] == True) & ((r["watch"] == True) | (r["analog"] >= 3.0))]["ticker"].tolist()
    bt = rd("rank_band_test.csv")
    band_note = ""
    if bt is not None:
        bw = bt[bt["group"] == "watch"].set_index("band")
        lo, hi = bw.loc["under 3%"], bw.loc["3% or wider"]
        band_note = (f"20-session high–low band under 3% = held at a deal price or stalled: inside the watch those weeks ran at {num(lo['lift'])}× "
                     f"({num(lo['lift_2006_16'])} / {num(lo['lift_2017_26'])}, {int(lo['weeks']):,} stock-weeks) vs {num(hi['lift'])}× for 3% or wider, so they are dropped.")
    r = r[r["tier"].notna()].sort_values("order")
    r["order"] = r["order"].astype(int)
    r["look"] = r.apply(lambda x: " ".join(filter(None, [
        '<span class="tag o">MSI 2023 base</span>' if x["msi_dist"] <= 0.05 else "",
        '<span class="tag b">MSI 2024 run</span>' if x["gap126"] <= 0.13 else ""])), axis=1)
    r["runs"] = r.apply(lambda x: f"{int(x['box_hits'])} / {int(x['box_hit_names'])}", axis=1)
    r["profile"] = r["proto_profile"].fillna("")
    t = tw.set_index(["key", "third"])["lift"]
    tl = ta[ta["analog"] == "tier"].set_index("group")
    tier_note = lambda k: next((f"{num(v['lift'])}× ({num(v['lift_2017_21'])} / {num(v['lift_2022_26'])})" for g, v in tl.iterrows() if g.startswith(f"{k}:")), "")
    cols = ["order", "ticker", "industry", "mcap_b", "profile", "analog", "analog_2006_16", "analog_2017_26", "runs", "look",
            "msi_dist", "gap126", "gain126", "worst_pullback126", "ddh252", "model_pct"]
    heads = ["#", "Ticker", "Industry", "Mkt cap", "Profile", "Analog odds", "2006–16", "2017–26", "Past runs / names", "MSI look",
             "Gap to MSI 2023 base (pct pts)", "Shape gap to MSI 2024 run", "Last 126 sessions", "Worst pullback", "From 52w high", "Model pct"]
    f = {"ticker": lambda v: f"<b>{esc(v)}</b>", "analog": lambda v: chip(v), "analog_2006_16": lambda v: chip(v), "analog_2017_26": lambda v: chip(v),
         "look": lambda v: v, "msi_dist": lambda v: f"{v * 100:.1f}", "gap126": lambda v: num(v, 3), "gain126": pct, "worst_pullback126": pct,
         "ddh252": pct, "model_pct": lambda v: "" if v != v else f"{v * 100:.0f}"}
    blocks = []
    for k, title in ((1, "Tier 1: on the watch, analog odds 2.5 or more"), (2, "Tier 2: on the watch, analog odds under 2.5"),
                     (3, "Tier 3: off the watch, analog odds 3 or more")):
        s = r[r["tier"] == k]
        if len(s):
            blocks.append(f"<h3>{title} ({len(s)}): backtest {tier_note(k)}</h3>" + table(s, cols, heads, f, cls="compact"))
    tw_tbl = table(tw.assign(key=tw["key"].map({"analog": "Analog odds (built 2006–16)", "msi_dist": "Closeness to MSI 2023 base",
                                                "gap126": "Closeness to MSI 2024 run shape (126)", "gap260": "Closeness to MSI Oct 2023–Nov 2024 shape (260)",
                                                "score": "Walk-forward model score"}),
                             third=tw["third"].map({1: "worst third", 2: "middle", 3: "best third"})),
                   ["key", "third", "weeks", "hit", "lift", "lift_2017_21", "lift_2022_26"],
                   ["Key, ranked inside each week's watch", "Third", "Stock-weeks", "Hit", "Lift", "2017–21", "2022–26"],
                   {"weeks": lambda v: f"{int(v):,}", "hit": pct, "lift": chip, "lift_2017_21": chip, "lift_2022_26": chip}, cls="compact", wide=("key",))
    ta_tbl = table(ta, ["analog", "group", "weeks", "hit", "lift", "lift_2017_21", "lift_2022_26"],
                   ["Analog odds (built 2006–16)", "Names", "Stock-weeks", "Hit", "Lift", "2017–21", "2022–26"],
                   {"weeks": lambda v: f"{int(v):,}", "hit": pct, "lift": chip, "lift_2017_21": chip, "lift_2022_26": chip}, cls="compact")
    return f"""
<section id="rank">
  <h2>Ranked: signal strength, then MSI likeness, {scan_date}</h2>
  <p class="lede">Signal strength = analog odds: every past stock-week since 2006 in the same state (volatility, 63-day range, 6-month drawdown and 6-month return percentiles,
  distance from the 52-week high; $2B+, no regional banks, no pinned names), MSI-grade hits ÷ what the same weeks' universe would have scored, shrunk toward 1.
  Tested before use: built from 2006–16 only, then applied to 2017-07 → 2026-04. Across all eligible names the odds rise bin by bin (under 0.5: {num(ta.loc[(ta['analog'] == '[0.0, 0.5)') & (ta['group'] == 'all'), 'lift'].iloc[0])}×;
  3–3.5: {num(ta.loc[(ta['analog'] == '[3.0, 3.5)') & (ta['group'] == 'all'), 'lift'].iloc[0])}×); inside each week's watch, best third {num(t[('analog', 3)])}× vs worst {num(t[('analog', 1)])}×.
  Tiers use the steps in those test bins: inside the watch the odds jump at 2.5, outside it they reach the watch's range at 3.</p>
  <p class="lede">MSI likeness, tested the same way: closeness to MSI's Jun–Oct 2023 base (the setup before its 2024 run) leaves the odds flat ({num(t[('msi_dist', 3)])} / {num(t[('msi_dist', 2)])} / {num(t[('msi_dist', 1)])}),
  so it orders names inside each tier at no cost. Closeness to MSI's 2024 run shape lowers them (closest third {num(t[('gap126', 3)])}× vs furthest {num(t[('gap126', 1)])}×): a name that already
  looks like MSI mid-run has spent part of the move, so that look is tagged, not ranked. <span class="tag o">MSI 2023 base</span> = within 5 percentile points of MSI's base state on average;
  <span class="tag b">MSI 2024 run</span> = 126-session shape gap 0.13 or less (universe median ≈0.27).</p>
  {"".join(blocks)}
  <p class="note">Past runs / names = MSI-grade runs inside the analog box and the distinct tickers behind them. {band_note} Left out today: {esc(', '.join(pinned)) or 'none'}.</p>
  <details><summary>How each ranking key did inside the watch, out of sample</summary>{tw_tbl}</details>
  <details><summary>Analog odds as a stand-alone screen, all eligible names, every 4th week 2017-07 → 2026-04</summary>{ta_tbl}</details>
</section>"""


def msi_like_section(scan_date):
    files = sorted(f for f in os.listdir(RES) if f.startswith("msi_like_20"))
    od = rd("msi_like_odds.csv")
    if not files or od is None:
        return ""
    m = rd(files[-1])
    m["mcap_b"] = m["mcap"].map(lambda v: f"${v / 1000:.1f}B")
    run = m[m["in_run"] == True].sort_values("gap126")
    close = m[(m["worst_pullback126"] <= 0.08) & (m["mcap"] >= 2000) & (m["pinned"] != True)].sort_values("gap126").head(10)
    cols = ["ticker", "sector", "industry", "mcap_b", "gain126", "worst_pullback126", "gap126", "gap260", "q_mar63", "q_mdd126", "ddh252", "pinned"]
    heads = ["Ticker", "Sector", "Industry", "Mkt cap", "Last 126 sessions", "Worst pullback", "Shape gap vs MSI Apr–Oct 2024",
             "Shape gap vs MSI Oct 2023–Nov 2024", "Volatility pct", "6m DD pct", "From 52w high", "Pinned"]
    f = {"ticker": lambda v: f"<b>{esc(v)}</b>", "gain126": pct, "worst_pullback126": pct, "gap126": lambda v: num(v, 3),
         "gap260": lambda v: num(v, 3), "q_mar63": lambda v: f"{v * 100:.1f}", "q_mdd126": lambda v: f"{v * 100:.1f}", "ddh252": pct,
         "pinned": lambda v: "yes" if v else ""}
    odt = table(od.iloc[:3], ["outcome", "weeks", "hit", "base", "lift", "lift_2006_16", "lift_2017_26"],
                ["After a stock looks like MSI mid-run", "Stock-weeks", "Hit", "Base", "Lift", "2006–16", "2017–26"],
                {"weeks": lambda v: f"{int(v):,}", "hit": pct, "base": pct, "lift": chip, "lift_2006_16": chip, "lift_2017_26": chip}, cls="compact", wide=("outcome",))
    last = od.iloc[3]
    return f"""
<section id="msi">
  <h2>MSI 2024 look-alikes, {scan_date}</h2>
  <p class="lede">Two forms. Before the run: MSI's base (the JNJ/MSI base inside the prototype watch above). During the run: last 126 sessions up 20%+ with a worst
  pullback of 6% or less, volatility in the bottom 20%, within 5% of the 52-week high. Shape gap = mean distance between scaled price paths and MSI's own (0 = identical;
  the universe median is about {num(m['gap126'].median(), 2)}).</p>
  <h3>Running like MSI now</h3>
  {table(run, cols, heads, f, cls="compact")}
  <h3>Closest 126-session shapes to MSI's Apr–Oct 2024 run ($2B+, worst pullback 8% or less, not pinned)</h3>
  {table(close, cols, heads, f, cls="compact")}
  {odt}
  <p class="note">Next 63 sessions after the mid-run state: {pct(last['hit'], 0)} up, median return {pct(last['lift'])}, median worst pullback {pct(last['lift_2006_16'])}.
  The base before the run carries better odds (prototype watch 2.85×) than the run itself (about 1.9×).</p>
</section>"""


def industry_block():
    t = rd("quietbase_industry.csv")
    if t is None:
        return ""
    return "<h3>By industry (industries with 400+ quiet-base observations)</h3>" + table(
        t, ["industry", "sector", "names", "n", "hit", "lift", "lift_1st", "lift_2nd", "ind_base_lift", "med_fdd63", "p_up63"],
        ["Industry", "Sector", "Names", "Obs", "Hit 63", "Quiet-base lift", "2006–16", "2016–26", "Industry lift (no filter)", "Med fwd max DD", "Up at 63d"],
        {"n": lambda v: f"{int(v):,}", "hit": pct, "lift": chip, "lift_1st": chip, "lift_2nd": chip, "ind_base_lift": chip, "med_fdd63": pct,
         "p_up63": lambda v: pct(v, 0)}, cls="compact", wide=("industry",))


def month_block():
    m = rd("month.csv")
    if m is None:
        return ""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m["name"] = m["month"].map(lambda i: names[int(i) - 1])
    return "<h3>By calendar month of the signal</h3>" + table(
        m, ["name", "n", "lift_all", "lift_2006", "lift_2013", "lift_2020", "lowvol_lift"],
        ["Month", "Obs", "Quiet-base lift", "2006–12", "2013–19", "2020–26", "Low volatility alone"],
        {"n": lambda v: f"{int(v):,}", "lift_all": chip, "lift_2006": chip, "lift_2013": chip, "lift_2020": chip, "lowvol_lift": chip}, cls="compact") + \
        '<p class="note">December–April is above 1.4 in all three eras. The September–November weakness comes from the 2020s (October 0.30, November 0.32); in 2006–12 October was 1.27.</p>'


def triad_block(tri):
    if tri is None:
        return ""
    allc = tri[tri["subset"] == "all"]
    low = tri[tri["subset"] == "low_vol_third"]
    return f"""<p class="note">Price ROC × volatility ROC × volume ROC cubes (8 combinations × 125 cells): across the whole universe {int((allc['era_min'] > 1.1).sum())} of {len(allc)} cells beat 1.10 in all three eras;
    inside the low-volatility third, {int((low['era_min'] > 1.1).sum())} of {len(low)} do, and the best of those have price ROC in the bottom quintile (lift up to {num(low['lift'].max())}).</p>"""


TEMPLATE = """<title>Staircase Precursor Study</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* layout: one reading column, research-note density; tables scroll inside their own frame */
:root {
  --bg: #f4f5f7; --surface: #ffffff; --ink: #11151b; --ink-2: #4b5260; --muted: #7d8492;
  --rule: #dcdfe5; --grid: #eceef2; --blue: #2a78d6; --orange: #eb6834; --div-mid: #eef0f3;
  --blue-wash: #e3eefb; --orange-wash: #fde9e0; --on-strong: #ffffff;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1013; --surface: #161a1f; --ink: #eef1f5; --ink-2: #b6bdc8; --muted: #8790a0;
  --rule: #2b3039; --grid: #222730; --blue: #3987e5; --orange: #d95926; --div-mid: #2a2f37;
  --blue-wash: #16263b; --orange-wash: #3a1f14; --on-strong: #ffffff; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1013; --surface: #161a1f; --ink: #eef1f5; --ink-2: #b6bdc8; --muted: #8790a0;
  --rule: #2b3039; --grid: #222730; --blue: #3987e5; --orange: #d95926; --div-mid: #2a2f37;
  --blue-wash: #16263b; --orange-wash: #3a1f14; --on-strong: #ffffff; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--f-body); margin: 0; }
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; }
header.top { display: grid; gap: 6px; padding-bottom: 18px; border-bottom: 1px solid var(--rule); }
header.top .eyebrow { font: 500 12px/1.2 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font: 600 clamp(28px, 4vw, 40px)/1.08 var(--f-display); margin: 0; letter-spacing: -.01em; text-wrap: balance; }
h2 { font: 600 22px/1.2 var(--f-display); margin: 0 0 8px; text-wrap: balance; }
h3 { font: 600 16px/1.3 var(--f-display); margin: 18px 0 6px; color: var(--ink-2); }
.sub { color: var(--ink-2); max-width: 75ch; margin: 0; }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px 14px; padding-block: 12px; font: 500 13px var(--f-mono); position: sticky; top: env(safe-area-inset-top, 0px); background: var(--bg); z-index: 2; border-bottom: 1px solid var(--rule); }
nav.toc a { color: var(--ink-2); text-decoration: none; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--orange); outline: none; text-decoration: underline; }
section { padding-block: 26px; border-bottom: 1px solid var(--rule); display: grid; gap: 12px; min-width: 0; }
.lede { color: var(--ink-2); margin: 0; max-width: 92ch; }
.note { color: var(--muted); font-size: 13px; margin: 0; max-width: 92ch; }
.read { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 20px 22px; margin-top: 20px; }
ol.bluf { margin: 0; padding-left: 22px; display: grid; gap: 10px; }
ol.bluf li { max-width: 98ch; }
.tw { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { font: 500 11px/1.25 var(--f-mono); text-transform: uppercase; letter-spacing: .04em; color: var(--muted); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; position: sticky; top: 0; background: var(--surface); }
td { padding: 7px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
table.compact td { white-space: nowrap; }
td.w { white-space: normal !important; min-width: 200px; }
table.scan td { white-space: nowrap; }

table.narrow { width: auto; min-width: 320px; }
code { font: 12px/1.4 var(--f-mono); color: var(--ink-2); white-space: normal; }
.lift { font: 500 12px var(--f-mono); padding: 1px 6px; border-radius: 3px; }
.lift.up { background: var(--orange-wash); color: var(--ink); box-shadow: inset 2px 0 0 var(--orange); }
.lift.dn { background: var(--blue-wash); color: var(--ink); box-shadow: inset 2px 0 0 var(--blue); }
.lift.nz { color: var(--ink-2); }
.tag { display: inline-block; font: 500 11px/1.6 var(--f-mono); padding: 0 6px; border-radius: 3px; margin: 1px 2px 1px 0; white-space: nowrap; }
.tag.o { background: var(--orange-wash); box-shadow: inset 2px 0 0 var(--orange); }
.tag.b { background: var(--blue-wash); box-shadow: inset 2px 0 0 var(--blue); }
.tag.n { color: var(--muted); border: 1px solid var(--rule); }
.muted { color: var(--muted); }
details { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 12px; }
details > summary { cursor: pointer; font-weight: 500; }
details[open] > summary { margin-bottom: 10px; }
details .tw { border: 0; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; min-width: 0; }
figcaption { font: 500 12px var(--f-mono); color: var(--muted); margin-bottom: 6px; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 12px; }
.sm { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 240px), 1fr)); gap: 10px; }
.sm svg, .heats svg { box-sizing: border-box; }
.sm svg { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 6px; }
.heats { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); gap: 10px; }
.heats svg { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 6px; }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-family: var(--f-mono); }
.tick { font-size: 10px; fill: var(--muted); }
.lab { font-size: 11.5px; fill: var(--ink-2); font-family: var(--f-body); }
.val { font-size: 11px; fill: var(--ink); }
.ptitle { font-size: 11.5px; fill: var(--ink); font-family: var(--f-body); font-weight: 600; }
.axl { font-size: 10.5px; fill: var(--muted); }
.cellv { font-size: 12px; font-weight: 500; }
.on-strong { fill: var(--on-strong); }
.grid { stroke: var(--grid); stroke-width: 1; }
.refline { stroke: var(--ink-2); stroke-width: 1.2; }
.conn { stroke: var(--rule); stroke-width: 2; }
.fb { fill: var(--blue); }
.fo { fill: var(--orange); }
.ring { stroke: var(--surface); stroke-width: 2; }
.faint { opacity: .45; }
.ghead { font: 600 12px var(--f-body); fill: var(--ink); }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: -1px; margin-right: 4px; }
.key.b { background: var(--blue); } .key.o { background: var(--orange); }
.shapes { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 250px), 1fr)); gap: 10px; }
.shape { display: flex; gap: 10px; align-items: center; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; font-size: 12.5px; }
.spark { width: 140px; height: 36px; flex: none; }
.sline { fill: none; stroke: var(--orange); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
ul.meth { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 98ch; }
@media (prefers-reduced-motion: no-preference) { nav.toc a { transition: color .15s; } }
</style>
<div class="wrap">
<header class="top">
  <div class="eyebrow">Backtest · 2,367-ticker universe · Yahoo daily OHLCV 2004 → {{DATE}}</div>
  <h1>Staircase Precursor Study</h1>
  <p class="sub">What comes before a smooth, low-drawdown 21–63 session advance, tested on every name in your file. Each claim below is a backtested number: lift against the same week's universe, checked out of sample and across eras.</p>
</header>
<nav class="toc">{{TOC}}</nav>
{{BODY}}
</div>
"""

if __name__ == "__main__":
    main()
