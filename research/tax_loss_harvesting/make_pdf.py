"""Render the 2018+ Q4 tax-loss study as a PDF report (reportlab + matplotlib).

usage: python make_pdf.py SINCE_JSON DASH_DIR OUT.pdf
SINCE_JSON comes from study_since.py; DASH_DIR holds live.json and screen.json from export_dashboard.py.
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

SINCE, DASH, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FONT_DIR = Path("/usr/share/fonts/truetype/crosextra")

pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "Carlito-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "Carlito-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", str(FONT_DIR / "Carlito-Italic.ttf")))
pdfmetrics.registerFont(TTFont("Serif", str(FONT_DIR / "Caladea-Regular.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-Bold")
for f in ("Carlito-Regular.ttf", "Carlito-Bold.ttf"):
    font_manager.fontManager.addfont(str(FONT_DIR / f))

INK, MUTED, RULE, PANEL = "#1b1b19", "#66655f", "#d6d5cf", "#f3f2ee"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29b"   # deutan-safe: blue / orange / neutral
plt.rcParams.update({
    "font.family": "Carlito", "font.size": 8.5, "axes.edgecolor": RULE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

B = json.loads(SINCE.read_text())
START, LAST = B["start"], B["last_q4"]
YEARS = list(range(START, LAST + 1))
SPAN = f"{START}–{LAST}"
QT = pd.DataFrame(B["quarters"])
WN = pd.DataFrame(B["windows"])
SP = pd.DataFrame(B["spreads"]).set_index(["window", "pair"])
HM = pd.DataFrame(B["halfmonth"])
MA = pd.DataFrame(B["monthly_alpha"])
CP = pd.DataFrame(B["cum_paths"])
RBD = pd.DataFrame(B["rebound"]).set_index("group")
VOL = pd.DataFrame(B["volume"])
TR = pd.DataFrame(B["trough"]).set_index("year")
ROB = pd.DataFrame(B["robust"])
PY = {k: {int(y): v for y, v in d.items()} for k, d in B["panel_years"].items()}
LIVE = pd.DataFrame(json.loads((DASH / "live.json").read_text())).set_index("group")
SC = pd.DataFrame(json.loads((DASH / "screen.json").read_text()))

LABELS = {
    "NEG_1Y": "1Y negative", "POS_1Y": "1Y positive", "NEG_1M": "1M negative", "POS_1M": "1M positive",
    "NEG1Y_NEG1M": "1Y neg + 1M neg", "NEG1Y_POS1M": "1Y neg + 1M pos", "POS1Y_NEG1M": "1Y pos + 1M neg",
    "POS1Y_POS1M": "1Y pos + 1M pos", "Q1_1Y": "1Y quintile 1 (worst)", "Q2_1Y": "1Y quintile 2",
    "Q3_1Y": "1Y quintile 3", "Q4_1Y": "1Y quintile 4", "Q5_1Y": "1Y quintile 5 (best)",
    "NEG_1Y_SMALL": "1Y neg, low liquidity", "POS_1Y_SMALL": "1Y pos, low liquidity",
    "NEG_1Y_LARGE": "1Y neg, high liquidity", "POS_1Y_LARGE": "1Y pos, high liquidity",
    "ALL": "All names (equal weight)", "SPY": "SPY", "DEEP_LT_-30": "1Y return −30% or worse",
}
PAIR_LAB = {"NEG1Y_NEG1M-POS1Y_POS1M": "1Y+1M neg − 1Y+1M pos", "NEG_1Y-POS_1Y": "1Y neg − 1Y pos",
            "NEG_1M-POS_1M": "1M neg − 1M pos", "Q1_1Y-Q5_1Y": "Worst − best 1Y quintile"}
WIN_LAB = {"Oct1_15": "Oct 1–15", "Oct15_Dec15": "Oct 15–Dec 15", "Oct15_Dec31": "Oct 15–Dec 31",
           "Q4": "Full Q4", "Jan": "January"}


def minus(s):
    return s.replace("-", "−")


def bad(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def pct(v, d=1, sign=False):
    return "—" if bad(v) else minus(f"{v * 100:{'+' if sign else ''}.{d}f}%")


def num(v, d=2, sign=False):
    return "—" if bad(v) else minus(f"{v:{'+' if sign else ''}.{d}f}")


def frac(h, n=None):
    n = n or len(YEARS)
    return f"{int(round(h * n))} of {n}"


def money(v):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:.1f}{suf}"
    return f"${v:.0f}"


def sp(window, pair, f="mean"):
    return SP.loc[(window, pair), f]


def wn(window, group, f):
    return WN[(WN.window == window) & (WN.group == group)][f].iloc[0]


def qt(quarter, group, f):
    return QT[(QT.quarter == quarter) & (QT.group == group)][f].iloc[0]


def hm(pair, window, f="mean"):
    return HM[(HM.pair == pair) & (HM.window == window)][f].iloc[0]


# ---------------------------------------------------------------- styles
ST = {
    "title": ParagraphStyle("title", fontName="Serif", fontSize=26, leading=30, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Serif", fontSize=16, leading=20, textColor=INK, spaceBefore=6, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.5, leading=13, textColor=INK, spaceBefore=8,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=10, leading=14, textColor=INK, spaceAfter=6),
    "note": ParagraphStyle("note", fontName="Sans", fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=10, leading=14, textColor=INK, leftIndent=12,
                             bulletIndent=0, spaceAfter=5),
    "bluf": ParagraphStyle("bluf", fontName="Sans", fontSize=11, leading=15.5, textColor=INK),
    "k": ParagraphStyle("k", fontName="Sans", fontSize=8, leading=10, textColor=MUTED),
    "v": ParagraphStyle("v", fontName="Serif", fontSize=20, leading=24, textColor=INK),
    "s": ParagraphStyle("s", fontName="Sans", fontSize=8, leading=10.5, textColor=MUTED),
}
W = letter[0] - 1.2 * inch


def P(t, s="body"):
    return Paragraph(t, ST[s])


def fig_image(fig, width=W):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    return img


def table(rows, col_widths, align_left=(0,), bold_rows=(), font=8, zebra=False, rule_rows=()):
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, -1), "Sans", font), ("FONT", (0, 0), (-1, 0), "Sans-Bold", font),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    style += [("ALIGN", (c, 0), (c, -1), "LEFT") for c in align_left]
    style += [("FONT", (0, r), (-1, r), "Sans-Bold", font) for r in bold_rows]
    style += [("LINEABOVE", (0, r), (-1, r), 0.8, colors.HexColor(INK)) for r in rule_rows]
    if zebra:
        style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor(PANEL)) for r in range(1, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def tiles(items, cols=2):
    gap = 8
    cw = (W - gap * (cols - 1)) / cols
    widths = sum([[cw] + ([gap] if i < cols - 1 else []) for i in range(cols)], [])
    cells = [[P(k, "k"), Spacer(1, 2), P(v, "v"), Spacer(1, 2), P(s, "s")] for k, v, s in items]
    grid = []
    for i in range(0, len(cells), cols):
        line = []
        for j, c in enumerate(cells[i:i + cols]):
            line += [c] + ([""] if j < cols - 1 else [])
        grid.append(line)
    t = Table(grid, colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 9),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 10),
          ("RIGHTPADDING", (0, 0), (-1, -1), 10)]
    for ri in range(len(grid)):
        for ci in range(0, len(widths), 2):
            st += [("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor(PANEL)),
                   ("BOX", (ci, ri), (ci, ri), 0.5, colors.HexColor(RULE))]
    t.setStyle(TableStyle(st))
    return t


def vs(a, b):
    return f"{a} <font size=11 color='{MUTED}'>vs</font> {b}"


def pct_axis(ax, d=1):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v * 100:+.{d}f}%")))
    ax.axhline(0, color="#8c8b85", lw=0.8, zorder=1)


# ---------------------------------------------------------------- charts
HW = ["Oct1_15", "Oct16_31", "Nov1_15", "Nov16_30", "Dec1_15", "Dec16_31", "Jan1_15", "Jan16_31"]
HW_LAB = ["Oct\n1–15", "Oct\n16–31", "Nov\n1–15", "Nov\n16–30", "Dec\n1–15", "Dec\n16–31", "Jan\n1–15", "Jan\n16–31"]
SERIES = [("NEG1Y-POS1Y", "1Y neg − 1Y pos", GRAY), ("NEG1M-POS1M", "1M neg − 1M pos", BLUE),
          ("NN-PP", "1Y+1M neg − 1Y+1M pos", ORANGE)]


def chart_halfmonth():
    fig, ax = plt.subplots(figsize=(7.3, 2.1))
    x = np.arange(len(HW))
    bw = 0.26
    for i, (pair, lab, c) in enumerate(SERIES):
        ax.bar(x + (i - 1) * bw, [hm(pair, w) for w in HW], bw * 0.9, color=c, zorder=2, label=lab)
    ax.set_xticks(x, HW_LAB)
    pct_axis(ax)
    ax.legend(ncol=3, loc="upper right", bbox_to_anchor=(1, 1.16))
    return fig_image(fig)


def chart_paths():
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.2), sharey=True)
    ticks, labs = [1, 32, 62, 93, 123], ["Oct 1", "Nov 1", "Dec 1", "Jan 1", "Jan 31"]
    for ax, (pair, title) in zip(axes, [("NEG_1Y-POS_1Y", "1Y neg − 1Y pos"),
                                        ("NEG1Y_NEG1M-POS1Y_POS1M", "1Y+1M neg − 1Y+1M pos")]):
        d = CP[CP.pair == pair]
        for y in YEARS:
            dy = d[d.year == y].sort_values("day")
            ax.plot(dy.day, dy.rel, color=GRAY, lw=0.8, alpha=0.75, zorder=2)
        avg = d.groupby("day").rel.mean()
        ax.plot(avg.index, avg.values, color=ORANGE, lw=2.4, zorder=3, solid_capstyle="round")
        ax.set_xticks(ticks, labs)
        ax.set_xlim(0, 124)
        pct_axis(ax, 0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    axes[1].legend(handles=[plt.Line2D([], [], color=ORANGE, lw=2.4, label=f"{SPAN} average"),
                            plt.Line2D([], [], color=GRAY, lw=0.8, label="Single year")],
                   loc="upper left")
    fig.tight_layout(w_pad=2)
    return fig_image(fig)


def chart_yearly():
    spec = [(lambda y: sp("Oct1_15", "NEG_1Y-POS_1Y", f"y{y}"), "Oct 1–15: 1Y neg − 1Y pos", 1),
            (lambda y: sp("Oct15_Dec15", "NEG1Y_NEG1M-POS1Y_POS1M", f"y{y}"), "Oct 15–Dec 15: 1Y+1M neg − pos", 0),
            (lambda y: sp("Jan", "NEG_1Y-POS_1Y", f"y{y}"), "January: 1Y neg − 1Y pos", 0),
            (lambda y: VOL[(VOL.window == "Dec21_31") & (VOL.year == y)].footprint.iloc[0],
             "Dec 21–31 volume: losers over winners", 0)]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.0))
    for ax, (fn, title, d) in zip(axes.flat, spec):
        vals = [fn(y) for y in YEARS]
        ax.bar(YEARS, vals, 0.62, color=[BLUE if v >= 0 else ORANGE for v in vals], zorder=2)
        for y, v in zip(YEARS, vals):
            ax.text(y, v + (0.004 if v >= 0 else -0.004), minus(f"{v * 100:+.{d}f}"), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.5, color=MUTED)
        pct_axis(ax, 0)
        ax.set_xticks(YEARS, [str(y) for y in YEARS])
        ax.margins(y=0.18)
        ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")
    fig.legend(handles=[Patch(color=BLUE, label="Positive"), Patch(color=ORANGE, label="Negative")], ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 1.03))
    fig.tight_layout(h_pad=1.6, w_pad=2, rect=(0, 0, 1, 0.97))
    return fig_image(fig)


def chart_calendar():
    fig, ax = plt.subplots(figsize=(7.3, 2.0))
    x = np.arange(1, 13)
    bw = 0.26
    ax.axvspan(9.5, 12.5, color="#efeee9", zorder=0)
    for i, (pair, lab, c) in enumerate(SERIES):
        d = MA[MA.pair == pair].sort_values("month")
        ax.bar(x + (i - 1) * bw, d.alpha, bw * 0.9, color=c, zorder=2, label=lab)
    ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    pct_axis(ax)
    ax.legend(ncol=3, loc="upper right", bbox_to_anchor=(1, 1.16))
    return fig_image(fig)


# ---------------------------------------------------------------- tables
RISK_GROUPS = ["NEG1Y_NEG1M", "NEG_1Y", "NEG_1M", "NEG1Y_POS1M", "POS1Y_NEG1M", "POS_1M", "POS_1Y", "POS1Y_POS1M",
               "Q1_1Y", "Q2_1Y", "Q3_1Y", "Q4_1Y", "Q5_1Y", "NEG_1Y_SMALL", "POS_1Y_SMALL", "NEG_1Y_LARGE",
               "POS_1Y_LARGE", "ALL", "SPY"]


def risk_table():
    d = QT[QT.quarter == "Q4"].set_index("group")
    rows, bold = [["Bucket", "Names", "Avg Q4", "Median", "Vol", "Sharpe", "Sortino", "Beta", "Alpha", "Beat SPY",
                   "Avg DD", "Worst"]], []
    for g in RISK_GROUPS:
        r = d.loc[g]
        rows.append([LABELS[g], f"{r.avg_n:,.0f}", pct(r["mean"], 1, True), pct(r["median"], 1, True),
                     pct(r.ann_vol), num(r.sharpe), num(r.sortino), num(r.beta_spy), pct(r.alpha_spy_ann, 1, True),
                     frac(r.hit_vs_spy), pct(r.avg_mdd), pct(r.worst, 1, True)])
        if g == "SPY":
            bold.append(len(rows) - 1)
    return table(rows, [112] + [37.5] * 11, bold_rows=bold, zebra=True)


def quarter_table():
    gs = ["NEG1Y_NEG1M", "NEG_1Y", "NEG_1M", "POS_1M", "POS_1Y", "POS1Y_POS1M", "Q1_1Y", "Q5_1Y", "ALL", "SPY"]
    rows = [["Bucket", "Q1 Sharpe", "Q2 Sharpe", "Q3 Sharpe", "Q4 Sharpe", "Q1 alpha", "Q2 alpha", "Q3 alpha",
             "Q4 alpha"]]
    for g in gs:
        rows.append([LABELS[g]] + [num(qt(q, g, "sharpe")) for q in ("Q1", "Q2", "Q3", "Q4")]
                    + [pct(qt(q, g, "alpha_spy_ann"), 1, True) for q in ("Q1", "Q2", "Q3", "Q4")])
    return table(rows, [118] + [50.9] * 8, bold_rows=[len(rows) - 1], zebra=True)


def year_table():
    cols = [("Q4\n1Y N−P", lambda y: sp("Q4", "NEG_1Y-POS_1Y", f"y{y}")),
            ("Q4\n1M N−P", lambda y: sp("Q4", "NEG_1M-POS_1M", f"y{y}")),
            ("Oct 1–15\n1Y N−P", lambda y: sp("Oct1_15", "NEG_1Y-POS_1Y", f"y{y}")),
            ("Oct 15–Dec 15\nNN−PP", lambda y: sp("Oct15_Dec15", "NEG1Y_NEG1M-POS1Y_POS1M", f"y{y}")),
            ("Oct 15–Dec 31\n1M N−P", lambda y: sp("Oct15_Dec31", "NEG_1M-POS_1M", f"y{y}")),
            ("Dec 1–15\n1Y N−P", lambda y: PY["Dec1_15"][y]),
            ("Jan\n1Y N−P", lambda y: sp("Jan", "NEG_1Y-POS_1Y", f"y{y}")),
            ("Dec 21–31\nvolume", lambda y: VOL[(VOL.window == "Dec21_31") & (VOL.year == y)].footprint.iloc[0])]
    rows = [["Year"] + [c for c, _ in cols] + ["Loser\ntrough"]]
    data = {c: [fn(y) for y in YEARS] for c, fn in cols}
    for i, y in enumerate(YEARS):
        trough = pd.Timestamp(TR.loc[y, "trough_date"]).strftime("%b %-d")
        rows.append([str(y)] + [pct(data[c][i], 1, True) for c, _ in cols] + [trough])
    arr = {c: pd.Series(v) for c, v in data.items()}
    rows.append(["Mean"] + [pct(arr[c].mean(), 2, True) for c, _ in cols] + [""])
    rows.append(["t"] + [num(arr[c].mean() / (arr[c].std(ddof=1) / np.sqrt(len(arr[c]))), 1, True)
                         for c, _ in cols] + [""])
    rows.append(["Positive"] + [frac((arr[c] > 0).mean()) for c, _ in cols] + [""])
    n = len(rows)
    return table(rows, [42] + [55.5] * 8 + [39], bold_rows=[n - 3], rule_rows=[n - 3])


def window_table(window, groups=None):
    groups = groups or ["NEG1Y_NEG1M", "NEG_1Y", "NEG_1M", "Q1_1Y", "ALL", "SPY", "POS_1M", "POS_1Y", "POS1Y_POS1M",
                        "Q5_1Y"]
    rows, bold = [["Bucket", "Avg", "Median", "Vol", "Sharpe", "Sortino", "Beta", "Alpha", "Up yrs", "Beat SPY",
                   "Worst"]], []
    for g in groups:
        r = WN[(WN.window == window) & (WN.group == g)].iloc[0]
        rows.append([LABELS[g], pct(r["mean"], 1, True), pct(r["median"], 1, True), pct(r.ann_vol), num(r.sharpe),
                     num(r.sortino), num(r.beta_spy), pct(r.alpha_spy_ann, 1, True), frac(r.pct_pos),
                     frac(r.hit_vs_spy), pct(r.worst, 1, True)])
        if g == "SPY":
            bold.append(len(rows) - 1)
    return table(rows, [118] + [40.7] * 10, bold_rows=bold, zebra=True)


def spread_table():
    rows = [["Window", "Pair", "Mean", "Median", "t", "Positive", "Worst yr", "Best yr", "Ex-2020", "t ex-2020"]]
    rule = []
    for w in ["Oct1_15", "Oct15_Dec15", "Oct15_Dec31", "Q4", "Jan"]:
        rule.append(len(rows))
        for i, p in enumerate(PAIR_LAB):
            r = SP.loc[(w, p)]
            rows.append([WIN_LAB[w] if i == 0 else "", PAIR_LAB[p], pct(r["mean"], 2, True), pct(r["median"], 2, True),
                         num(r.t, 1, True), frac(r.hit), pct(r.worst, 1, True), pct(r.best, 1, True),
                         pct(r.mean_ex2020, 2, True), num(r.t_ex2020, 1, True)])
    return table(rows, [66, 117, 42, 42, 32, 46, 46, 46, 46, 42], align_left=(0, 1), rule_rows=rule[1:])


def rebound_table():
    rows, bold = [["Bucket (re-sorted Dec 15)", "Avg", "Median", "Vol", "Sharpe", "Sortino", "Beta", "Alpha", "Up yrs",
                   "Beat SPY"]], []
    for g in ["DEEP_LT_-30", "Q1_1Y", "NEG_1Y", "ALL", "SPY", "POS_1Y", "Q5_1Y"]:
        r = RBD.loc[g]
        rows.append([LABELS[g], pct(r["mean"], 1, True), pct(r["median"], 1, True), pct(r.ann_vol), num(r.sharpe),
                     num(r.sortino), num(r.beta_spy), pct(r.alpha_spy_ann, 1, True), frac(r.pct_pos),
                     frac(r.hit_vs_spy)])
        if g == "SPY":
            bold.append(len(rows) - 1)
    return table(rows, [130] + [43.9] * 9, bold_rows=bold, zebra=True)


def robust_table():
    order = ["Oct1_15", "Oct15_Dec15", "Oct15_Dec31", "Jan"]
    d = ROB.copy()
    d["o"] = d.window.map(order.index)
    d = d.sort_values(["o", "sort", "floor"], ascending=[True, False, True])
    rows = [["Window", "Sort", "Min $ vol", "NN−PP", "t", "Positive", "NN−PP med", "1Y N−P", "t", "NN−PP ex-20",
             "t ex-20"]]
    for (_, so, fl), g in d.groupby(["o", "sort", "floor"], sort=False):
        a, x = g[~g.ex2020].iloc[0], g[g.ex2020].iloc[0]
        rows.append([WIN_LAB[a.window], "Sep 30" if so == "sep30" else "Oct 15", money(fl), pct(a.nn_pp, 2, True),
                     num(a.nn_pp_t, 1, True), frac(a.nn_pp_hit), pct(a.nn_pp_med, 2, True), pct(a.n_p, 2, True),
                     num(a.n_p_t, 1, True), pct(x.nn_pp, 2, True), num(x.nn_pp_t, 1, True)])
    return table(rows, [66, 40, 44, 46, 34, 44, 52, 46, 34, 60, 39], align_left=(0, 1), zebra=True)


def screen_table():
    rows = [["#", "Ticker", "Company", "Sector", "Mkt cap", "$ vol/day", "1Y", "1M (Sep)", "QTD"]]
    for i, r in enumerate(SC.itertuples(), 1):
        name = r.company if len(str(r.company)) <= 30 else str(r.company)[:29] + "…"
        rows.append([str(i), r.ticker, name, r.sector, money(r.mcap_m * 1e6), money(r.dollar_vol), pct(r.r12),
                     pct(r.r1), pct(r.qtd, 1, True)])
    return table(rows, [24, 40, 140, 104, 48, 50, 40, 42, 37], align_left=(1, 2, 3), font=7, zebra=True)


# ---------------------------------------------------------------- numbers used in text
nnpp = "NEG1Y_NEG1M-POS1Y_POS1M"
K = {
    "sh_n1y": qt("Q4", "NEG_1Y", "sharpe"), "sh_p1y": qt("Q4", "POS_1Y", "sharpe"),
    "sh_n1m": qt("Q4", "NEG_1M", "sharpe"), "sh_p1m": qt("Q4", "POS_1M", "sharpe"),
    "sh_nn": qt("Q4", "NEG1Y_NEG1M", "sharpe"), "sh_pp": qt("Q4", "POS1Y_POS1M", "sharpe"),
    "pp_beat": qt("Q4", "POS1Y_POS1M", "hit_vs_spy"),
    "q4_np": sp("Q4", "NEG_1Y-POS_1Y"), "q4_np_t": sp("Q4", "NEG_1Y-POS_1Y", "t"),
    "q4_1m": sp("Q4", "NEG_1M-POS_1M"), "q4_1m_t": sp("Q4", "NEG_1M-POS_1M", "t"),
    "q4_1m_hit": sp("Q4", "NEG_1M-POS_1M", "hit"),
    "oct_np": sp("Oct1_15", "NEG_1Y-POS_1Y"), "oct_np_t": sp("Oct1_15", "NEG_1Y-POS_1Y", "t"),
    "oct_np_hit": sp("Oct1_15", "NEG_1Y-POS_1Y", "hit"), "oct_q": sp("Oct1_15", "Q1_1Y-Q5_1Y"),
    "oct_nn": sp("Oct1_15", nnpp),
    "w_nn": sp("Oct15_Dec15", nnpp), "w_nn_t": sp("Oct15_Dec15", nnpp, "t"), "w_nn_hit": sp("Oct15_Dec15", nnpp, "hit"),
    "w_nn_worst": sp("Oct15_Dec15", nnpp, "worst"), "w_nn_x": sp("Oct15_Dec15", nnpp, "mean_ex2020"),
    "w_1m31": sp("Oct15_Dec31", "NEG_1M-POS_1M"), "w_1m31_t": sp("Oct15_Dec31", "NEG_1M-POS_1M", "t"),
    "w_1m31_hit": sp("Oct15_Dec31", "NEG_1M-POS_1M", "hit"), "w_1m31_worst": sp("Oct15_Dec31", "NEG_1M-POS_1M", "worst"),
    "nn_ret": wn("Oct15_Dec15", "NEG1Y_NEG1M", "mean"), "nn_sh": wn("Oct15_Dec15", "NEG1Y_NEG1M", "sharpe"),
    "nn_al": wn("Oct15_Dec15", "NEG1Y_NEG1M", "alpha_spy_ann"),
    "pp_ret": wn("Oct15_Dec15", "POS1Y_POS1M", "mean"), "pp_sh": wn("Oct15_Dec15", "POS1Y_POS1M", "sharpe"),
    "spy_ret": wn("Oct15_Dec15", "SPY", "mean"), "spy_sh": wn("Oct15_Dec15", "SPY", "sharpe"),
    "nov_nn": hm("NN-PP", "Nov1_15"), "dec_nn": hm("NN-PP", "Dec1_15"), "dec_nn_hit": hm("NN-PP", "Dec1_15", "hit"),
    "dec2_np": hm("NEG1Y-POS1Y", "Dec16_31"),
    "jan_np": sp("Jan", "NEG_1Y-POS_1Y"), "jan_np_hit": sp("Jan", "NEG_1Y-POS_1Y", "hit"), "jan_nn": sp("Jan", nnpp),
    "rb_neg_al": RBD.loc["NEG_1Y", "alpha_spy_ann"], "rb_neg_sh": RBD.loc["NEG_1Y", "sharpe"],
    "rb_pos_sh": RBD.loc["POS_1Y", "sharpe"], "rb_q5_sh": RBD.loc["Q5_1Y", "sharpe"],
    "rb_deep_sh": RBD.loc["DEEP_LT_-30", "sharpe"],
}
vd = VOL[VOL.window == "Dec21_31"].set_index("year").footprint
half = len(YEARS) // 2
K["vol_a"], K["vol_b"] = vd.loc[YEARS[:half]].mean(), vd.loc[YEARS[half:]].mean()
nov_troughs = [y for y in YEARS if pd.Timestamp(TR.loc[y, "trough_date"]).month == 11]
oct_m = MA[(MA.pair == "NEG1Y-POS1Y") & (MA.month == 10)].iloc[0]
lv = LIVE

# ---------------------------------------------------------------- story
story = [P(f"Q4 Tax-Loss Seasonality, {SPAN}", "title"),
         P(f"2,367 tickers · {len(YEARS)} fourth quarters, {START} to {LAST} · {B['n_stock_years']:,} stock-years · "
           "live read through the Oct 9, 2026 close", "sub")]
bluf = (f"<b>The read.</b> On the 1Y sort alone, Q4 is a tie: losers and winners land at Sharpe "
        f"{num(K['sh_n1y'])} vs {num(K['sh_p1y'])}. The edge is in the 1M sort and the timing. September losers beat "
        f"September winners in Q4 by {pct(K['q4_1m'], 2, True)} (t {num(K['q4_1m_t'], 1, True)}, positive "
        f"{frac(K['q4_1m_hit'])} years). 1Y losers get sold Oct 1–15 ({pct(K['oct_np'], 2, True)}, lagging "
        f"{frac(1 - K['oct_np_hit'])} years), then lead into mid-December. January adds nothing.<br/><br/>"
        f"<b>Setup:</b> long the 1Y-negative + 1M-negative basket (Sep 30 sort). <b>Entry</b> Oct 15. <b>Exit</b> "
        f"Dec 15. <b>Invalidation</b>: basket minus the 1Y-positive + 1M-positive basket below "
        f"{pct(K['w_nn_worst'], 1)} by Dec 15, the worst {SPAN} print.")
box = Table([[P(bluf, "bluf")]], colWidths=[W])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                         ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(BLUE)),
                         ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                         ("LEFTPADDING", (0, 0), (-1, -1), 13), ("RIGHTPADDING", (0, 0), (-1, -1), 13)]))
story += [box, Spacer(1, 12), tiles([
    (f"Q4 Sharpe, 1M losers vs 1M winners · {SPAN}", vs(num(K["sh_n1m"]), num(K["sh_p1m"])),
     f"1Y sort: {num(K['sh_n1y'])} vs {num(K['sh_p1y'])} · 1Y+1M: {num(K['sh_nn'])} vs {num(K['sh_pp'])}"),
    ("Oct 1–15, 1Y losers minus winners", pct(K["oct_np"], 2, True),
     f"t {num(K['oct_np_t'], 1, True)} · lagged {frac(1 - K['oct_np_hit'])} years · worst vs best quintile "
     f"{pct(K['oct_q'], 2, True)}"),
    ("Oct 15–Dec 15, 1Y neg + 1M neg basket",
     f"{pct(K['nn_ret'], 1, True)} <font size=11 color='{MUTED}'>Sharpe</font> {num(K['nn_sh'])}",
     f"1Y pos + 1M pos {pct(K['pp_ret'], 1, True)} (Sharpe {num(K['pp_sh'])}) · SPY {pct(K['spy_ret'], 1, True)} "
     f"(Sharpe {num(K['spy_sh'])})"),
    ("January, 1Y losers minus winners", pct(K["jan_np"], 2, True),
     f"ahead {frac(K['jan_np_hit'])} years · Dec 15→Jan 31 loser alpha {pct(K['rb_neg_al'], 1, True)} annualized"),
]), Spacer(1, 10), P("Findings", "h2")]
findings = [
    f"<b>1Y sort: no Q4 edge.</b> Sharpe {num(K['sh_n1y'])} vs {num(K['sh_p1y'])}; Q4 spread "
    f"{pct(K['q4_np'], 2, True)}, t {num(K['q4_np_t'], 1, True)}.",
    f"<b>1M sort: the clean Q4 edge.</b> September losers minus September winners {pct(K['q4_1m'], 2, True)} in Q4 "
    f"(t {num(K['q4_1m_t'], 1, True)}, positive {frac(K['q4_1m_hit'])}); Oct 15–Dec 31 {pct(K['w_1m31'], 2, True)} "
    f"(t {num(K['w_1m31_t'], 1, True)}, positive {frac(K['w_1m31_hit'])}, worst year {pct(K['w_1m31_worst'], 2, True)}).",
    f"<b>Double winners lag in Q4.</b> 1Y+1M positive Sharpe {num(K['sh_pp'])}, beat SPY in {frac(K['pp_beat'])} Q4s. "
    f"Double losers {num(K['sh_nn'])}.",
    f"<b>Oct 1–15 is the flush.</b> 1Y losers minus winners {pct(K['oct_np'], 2, True)} (t {num(K['oct_np_t'], 1, True)}), "
    f"1Y+1M {pct(K['oct_nn'], 2, True)}, worst vs best quintile {pct(K['oct_q'], 2, True)}. October is also the worst "
    f"calendar month for the 1Y spread after SPY beta ({pct(oct_m.alpha, 2, True)}).",
    f"<b>Mid-October to mid-December is the recovery.</b> 1Y+1M neg minus pos, Oct 15–Dec 15: "
    f"{pct(K['w_nn'], 2, True)} (t {num(K['w_nn_t'], 1, True)}, positive {frac(K['w_nn_hit'])}, worst "
    f"{pct(K['w_nn_worst'], 2, True)}, ex-2020 {pct(K['w_nn_x'], 2, True)}). Basket alpha {pct(K['nn_al'], 1, True)} "
    f"annualized. Nov 1–15 ({pct(K['nov_nn'], 2, True)}) and Dec 1–15 ({pct(K['dec_nn'], 2, True)}, positive "
    f"{frac(K['dec_nn_hit'])}) carry it; Dec 16–31 is flat ({pct(K['dec2_np'], 2, True)} on the 1Y sort).",
    f"<b>The relative low prints in November.</b> The loser-vs-winner path bottomed in November in "
    f"{len(nov_troughs)} of {len(YEARS)} years ({', '.join(str(y) for y in nov_troughs)}).",
    f"<b>January is dead.</b> 1Y losers minus winners {pct(K['jan_np'], 2, True)}, 1Y+1M {pct(K['jan_nn'], 2, True)}. "
    f"Dec 15 re-sort to Jan 31: 1Y losers Sharpe {num(K['rb_neg_sh'])} vs winners {num(K['rb_pos_sh'])}; best quintile "
    f"{num(K['rb_q5_sh'])} beats −30%-or-worse losers {num(K['rb_deep_sh'])}.",
    f"<b>Year-end loser volume is rising.</b> Dec 21–31 loser volume over winner volume averaged "
    f"{pct(K['vol_a'], 1, True)} in {YEARS[0]}–{YEARS[half - 1]} and {pct(K['vol_b'], 1, True)} in "
    f"{YEARS[half]}–{YEARS[-1]}, with no matching price drag in late December.",
]
story += [Paragraph(f, ST["bullet"], bulletText="•") for f in findings]

story += [PageBreak(), P(f"Q4 risk-adjusted returns by bucket, {SPAN}", "h1"),
          P("Buckets formed at the last September close, equal weight, buy-and-hold to Dec 31. Annualized stats pool "
            "every Q4 trading day across the years. Alpha and beta vs SPY; Sharpe and Sortino over 13-week T-bills. "
            "Avg Q4 / Median / Beat SPY / Worst are per-year figures. Liquidity terciles use 63-day median dollar "
            "volume at formation.", "note"),
          risk_table(), Spacer(1, 6),
          P(f"Is Q4 special? Same buckets in every quarter, Q1 {START} to Q3 2026 (Q4 through {LAST})", "h2"),
          quarter_table()]

story += [PageBreak(), P("Where in Q4 losers lag and lead", "h1"),
          P(f"Sep 30 sort held through January; each bar is the {SPAN} average of yearly losers-minus-winners returns in "
            "that half-month.", "note"),
          chart_halfmonth(), Spacer(1, 4),
          KeepTogether([P("Cumulative relative return since Sep 30 (losers' growth over winners' growth)", "h2"),
                        chart_paths()]), Spacer(1, 4),
          KeepTogether([P(f"Every calendar month, Jan {START} to Sep 2026 (alpha after SPY beta, Q4 shaded)", "h2"),
                        P("Same sorts re-formed at every month-end; next-month spread regressed on SPY with month "
                          "dummies.", "note"), chart_calendar()])]

story += [PageBreak(), P("Year by year", "h1"),
          P("N−P = negative minus positive; NN−PP = 1Y+1M negative minus 1Y+1M positive. Volume = median Dec 21–31 "
            "volume vs the Jun–Aug base, losers relative to winners (Nov 30 sort). Trough = day the 1Y loser-vs-winner "
            "path (after SPY beta) bottomed between Oct 1 and Jan 31. Jan and trough dates fall in the following January.",
            "note"),
          year_table(), Spacer(1, 10), chart_yearly()]

story += [PageBreak(), P("Oct 15 → Dec 15 window (Sep 30 sort)", "h1"), window_table("Oct15_Dec15"),
          Spacer(1, 4), P("Oct 1–15", "h2"), window_table("Oct1_15"),
          Spacer(1, 4), P("Oct 15–Dec 31", "h2"), window_table("Oct15_Dec31")]

story += [PageBreak(), P("Pair spreads by window", "h1"),
          P("Yearly long-minus-short return of equal-weight buckets; t on the yearly series; Positive = years the long "
            "side won.", "note"),
          spread_table(), Spacer(1, 10),
          KeepTogether([P("Dec 15 → Jan 31 (1Y return re-sorted at Dec 15)", "h1"), rebound_table()])]

story += [PageBreak(), P(f"Robustness, {SPAN}", "h1"),
          P("Liquidity floor on 63-day median dollar volume. Oct 15 sort recomputes 1Y and 1M as of Oct 15 (point in "
            "time). Median = spread of per-year bucket medians. Ex-20 drops 2020.", "note"),
          robust_table(), Spacer(1, 12),
          P("Q4 2026 so far", "h1"),
          P("Sep 30, 2026 sort; returns Oct 1 through the Oct 9, 2026 close.", "note"),
          tiles([
              ("1Y negative names", f"{int(lv.loc['NEG_1Y', 'n']):,}",
               f"of which 1M also negative: {int(lv.loc['NEG1Y_NEG1M', 'n']):,}"),
              ("QTD, 1Y neg vs 1Y pos", vs(pct(lv.loc["NEG_1Y", "qtd_mean"], 2, True),
                                          pct(lv.loc["POS_1Y", "qtd_mean"], 2, True)), "Equal-weight mean"),
              ("QTD, 1Y+1M neg vs 1Y+1M pos", vs(pct(lv.loc["NEG1Y_NEG1M", "qtd_mean"], 2, True),
                                                pct(lv.loc["POS1Y_POS1M", "qtd_mean"], 2, True)),
               f"SPY {pct(lv.loc['SPY', 'qtd_mean'], 2, True)}"),
              ("Oct 1–15 flush", "Not yet" if lv.loc["NEG_1Y", "qtd_mean"] > lv.loc["POS_1Y", "qtd_mean"] else "Printing",
               f"1Y losers vs winners QTD, against the {SPAN} Oct 1–15 pattern"),
          ])]

story += [PageBreak(), P("Method", "h1")]
for t in [
    f"<b>Universe.</b> The 2,367 tickers in the uploaded finviz list. Daily dividend- and split-adjusted closes and "
    f"volume from Yahoo Finance via yfinance. Formation years {START}–{LAST} for the Q4 work; Q1 {START} through Q3 "
    f"2026 for the other-quarter comparison; Jan {START} through Sep 2026 for calendar months. "
    f"{B['n_stock_years']:,} stock-years pass the Sep 30 screen.",
    "<b>Signals.</b> Formed at the last trading day of September: 1Y = trailing 12-month total return, 1M = "
    "September return. Eligibility: a full 12 months of history and 63-day median dollar volume of at least $1M. "
    "Buckets are equal-weighted at formation and held without rebalancing. Daily returns are clipped to −80% / "
    "+150% to remove bad ticks.",
    "<b>Statistics.</b> Annualized return, vol, Sharpe and Sortino pool all daily returns inside the window across "
    "years. Sharpe and Sortino are over 13-week T-bills (^IRX). Alpha and beta are from daily returns vs SPY. "
    f"t-stats use one observation per year ({len(YEARS)} years). Calendar-month alphas regress the monthly spread on "
    "SPY with month dummies (HC1 errors).",
    "<b>Universe bias.</b> The list is today's survivors, so delisted losers are missing, which lifts loser returns. "
    "Comparing windows and quarters on the same universe nets most of it out.",
    "<b>Reproduce.</b> research/tax_loss_harvesting on branch claude/tax-loss-harvesting-analysis-emumq1 of "
    f"kbw468/yfinance: study_since.py DATA OUT {START}, then make_pdf.py.",
]:
    story.append(P(t))

story += [PageBreak(), P("Appendix: 1Y negative and 1M negative names, Sep 30, 2026", "h1"),
          P(f"All {len(SC):,} names, sorted by 63-day median dollar volume. 1Y and 1M as of Sep 30, 2026; QTD through "
            "Oct 9, 2026. Market cap from the uploaded list.", "note"), screen_table()]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(0.6 * inch, 0.42 * inch, f"Q4 Tax-Loss Seasonality, {SPAN} · data through Oct 9, 2026")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.42 * inch, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.65 * inch, title=f"Q4 Tax-Loss Seasonality, {SPAN}",
                        author="Claude", subject="Tax-loss harvesting seasonality backtest")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
