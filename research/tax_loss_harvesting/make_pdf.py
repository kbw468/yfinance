"""Render the 2018+ Q4 tax-loss study as a PDF report (reportlab + matplotlib).

usage: python make_pdf.py SINCE_JSON MEGA_JSON CELLS_CSV CANDIDATES_CSV QUINTILES_JSON RANK_JSON TIERS_CSV DASH_DIR
       OUT.pdf
RANK_JSON from rank_strategies.py; TIERS_CSV = today's September sort with market-cap tiers.
SINCE_JSON from study_since.py, MEGA_JSON from megacap.py, CELLS_CSV from cells.py, CANDIDATES_CSV and
QUINTILES_JSON from candidates.py; DASH_DIR holds live.json from export_dashboard.py.
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

SINCE, MEGA, CELLS, CAND, QUINT, RANK, TIERS, DASH, OUT = (Path(a) for a in sys.argv[1:10])
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
CELL = pd.read_csv(CELLS)
CELL = CELL[CELL.window == "Oct15_Dec15"].set_index("cell")
CANDS = pd.read_csv(CAND)
QU = {int(k): v for k, v in json.loads(QUINT.read_text()).items()}
RK = pd.DataFrame(json.loads(RANK.read_text())["rows"])
TNOW = pd.read_csv(TIERS)
M = json.loads(MEGA.read_text())
MB = {n: pd.DataFrame(M["top"][n]["buckets"]) for n in M["top"]}
MS = {n: pd.DataFrame(M["top"][n]["spreads"]).set_index(["window", "pair"]) for n in M["top"]}
MN = pd.DataFrame(M["names"])
MAGG = pd.DataFrame(M["names_by_bucket"]).set_index("bucket")
TOP_NOW = pd.DataFrame(M["top100_now"])
TINT = "#dbe8f8"   # light blue highlight for 1Y-negative years

LABELS = {
    "NEG_1Y": "1Y negative", "POS_1Y": "1Y positive", "NEG_1M": "1M negative", "POS_1M": "1M positive",
    "NEG1Y_NEG1M": "1Y neg + 1M neg", "NEG1Y_POS1M": "1Y neg + 1M pos", "POS1Y_NEG1M": "1Y pos + 1M neg",
    "POS1Y_POS1M": "1Y pos + 1M pos", "Q1_1Y": "1Y worst 20%", "Q2_1Y": "1Y worst 20–40%",
    "Q3_1Y": "1Y middle 20%", "Q4_1Y": "1Y best 20–40%", "Q5_1Y": "1Y best 20%",
    "NEG_1Y_SMALL": "1Y neg, low liquidity", "POS_1Y_SMALL": "1Y pos, low liquidity",
    "NEG_1Y_LARGE": "1Y neg, high liquidity", "POS_1Y_LARGE": "1Y pos, high liquidity",
    "ALL": "All names (equal weight)", "SPY": "SPY", "DEEP_LT_-30": "1Y return −30% or worse",
}
PAIR_LAB = {"NEG1Y_NEG1M-POS1Y_POS1M": "1Y+1M neg − 1Y+1M pos", "NEG_1Y-POS_1Y": "1Y neg − 1Y pos",
            "NEG_1M-POS_1M": "1M neg − 1M pos", "Q1_1Y-Q5_1Y": "1Y worst 20% − best 20%"}
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
            (lambda y: sp("Q4", "NEG_1M-POS_1M", f"y{y}"), "Full Q4: 1M neg − 1M pos", 1)]
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
               "Q1_1Y", "Q2_1Y", "Q3_1Y", "Q4_1Y", "Q5_1Y", "ALL", "SPY"]


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
            ("Jan\n1Y N−P", lambda y: sp("Jan", "NEG_1Y-POS_1Y", f"y{y}"))]
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
    return table(rows, [44] + [62] * 7 + [47], bold_rows=[n - 3], rule_rows=[n - 3])


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
    d = ROB[ROB.floor == ROB.floor.min()].copy()
    d["o"] = d.window.map(order.index)
    d = d.sort_values(["o", "sort", "floor"], ascending=[True, False, True])
    rows = [["Window", "Sort", "NN−PP", "t", "Positive", "NN−PP med", "1Y N−P", "t", "NN−PP ex-20", "t ex-20"]]
    for (_, so, fl), g in d.groupby(["o", "sort", "floor"], sort=False):
        a, x = g[~g.ex2020].iloc[0], g[g.ex2020].iloc[0]
        rows.append([WIN_LAB[a.window], "Sep 30" if so == "sep30" else "Oct 15", pct(a.nn_pp, 2, True),
                     num(a.nn_pp_t, 1, True), frac(a.nn_pp_hit), pct(a.nn_pp_med, 2, True), pct(a.n_p, 2, True),
                     num(a.n_p_t, 1, True), pct(x.nn_pp, 2, True), num(x.nn_pp_t, 1, True)])
    return table(rows, [74, 48, 50, 38, 48, 58, 50, 38, 64, 57], align_left=(0, 1), zebra=True)


def candidates_table():
    rows = [["#", "Ticker", "Company", "Sector", "Mkt cap", "1Y", "Sep", "QTD", "Basket"]]
    for i, r in enumerate(CANDS.itertuples(), 1):
        name = r.company if len(str(r.company)) <= 32 else str(r.company)[:31] + "…"
        rows.append([str(i), r.ticker, name, r.sector, money(r.mcap_musd * 1e6), pct(r.ret_1y_sep30), pct(r.ret_sep),
                     pct(r.qtd_oct9, 1, True), r.basket.replace(",", "+")])
    return table(rows, [24, 42, 150, 112, 50, 42, 42, 38, 25], align_left=(1, 2, 3), font=7, zebra=True)


CODE = {"1Y neg + 1M neg": "NN", "1Y neg + 1M pos": "NP", "1Y pos + 1M neg": "PN", "1Y pos + 1M pos": "PP"}


def mega_bucket_table(n, windows=("Q4", "Oct15_Dec15")):
    d = MB[n]
    gs = ["NEG1Y_NEG1M", "NEG1Y_POS1M", "NEG_1Y", "NEG_1M", "POS_1M", "POS_1Y", "POS1Y_NEG1M", "POS1Y_POS1M", "ALL",
          "SPY"]
    short = {"Q4": "Q4", "Oct15_Dec15": "Oct 15–\nDec 15"}
    rows = [["Bucket", "Names"] + sum([[f"{short[w]}\navg", "Sharpe", "Alpha", "Beat\nSPY"] for w in windows], [])]
    bold = []
    for g in gs:
        line = [LABELS[g]]
        for i, w in enumerate(windows):
            r = d[(d.window == w) & (d.group == g)].iloc[0]
            if i == 0:
                line.append(f"{r.avg_n:.0f}")
            line += [pct(r["mean"], 1, True), num(r.sharpe), pct(r.alpha_spy_ann, 1, True),
                     f"{int(round(r.hit_vs_spy * r.years))} of {int(r.years)}"]
        rows.append(line)
        if g == "SPY":
            bold.append(len(rows) - 1)
    return table(rows, [104, 34] + [48, 38, 44, 45] * len(windows), bold_rows=bold, zebra=True)


def mega_spread_table():
    rows = [["Window", "Pair", "Top 100 mean", "t", "Positive", "Worst yr", "Best yr", "Top 50 mean", "t", "Positive"]]
    rule = []
    for w in ["Oct1_15", "Oct15_Dec15", "Oct15_Dec31", "Q4", "Jan"]:
        rule.append(len(rows))
        for i, p in enumerate(["NEG_1Y-POS_1Y", "NEG1Y_NEG1M-POS1Y_POS1M", "NEG_1M-POS_1M"]):
            a, b = MS["100"].loc[(w, p)], MS["50"].loc[(w, p)]
            rows.append([WIN_LAB[w] if i == 0 else "", PAIR_LAB[p], pct(a["mean"], 2, True), num(a.t, 1, True),
                         frac(a.hit), pct(a.worst, 1, True), pct(a.best, 1, True), pct(b["mean"], 2, True),
                         num(b.t, 1, True), frac(b.hit)])
    return table(rows, [66, 112, 50, 30, 42, 44, 44, 50, 30, 42], align_left=(0, 1), rule_rows=rule[1:])


def mega_name_table():
    names = MN.ticker.drop_duplicates().tolist()
    rows = [["Ticker"] + [str(y) for y in YEARS] + ["Avg"]]
    style = []
    for ri, t in enumerate(names, 1):
        d = MN[MN.ticker == t].set_index("year")
        line = [t]
        for ci, y in enumerate(YEARS, 1):
            if y not in d.index:
                line.append("—")
                continue
            r = d.loc[y]
            line.append(f"{CODE[r.bucket]} {minus(f'{r.Q4_xs * 100:+.1f}')}")
            if r.r12 < 0:
                style.append(("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor(TINT)))
        line.append(minus(f"{d.Q4_xs.mean() * 100:+.1f}"))
        rows.append(line)
    t = table(rows, [44] + [55] * len(YEARS) + [41])
    t.setStyle(TableStyle(style))
    return t


def mega_agg_table():
    rows = [["Bucket at Sep 30", "Obs", "Q4 vs SPY", "Median", "Beat SPY", "Oct 1–15 vs SPY", "Oct 15–Dec 15 vs SPY",
             "Beat SPY", "Jan vs SPY"]]
    for b in ["1Y neg + 1M neg", "1Y neg + 1M pos", "1Y pos + 1M neg", "1Y pos + 1M pos"]:
        r = MAGG.loc[b]
        rows.append([f"{b} ({CODE[b]})", f"{int(r.n)}", pct(r.q4_xs, 1, True), pct(r.q4_xs_med, 1, True),
                     pct(r.q4_beat, 0), pct(r.oct_xs, 1, True), pct(r.w_xs, 1, True), pct(r.w_beat, 0),
                     pct(r.jan_xs, 1, True)])
    for lab, k in [("Any 1Y negative", "names_neg1y"), ("Any 1Y positive", "names_pos1y")]:
        r = M[k]
        rows.append([lab, f"{r['n']}", pct(r["q4_xs"], 1, True), "", pct(r["q4_beat"], 0), "", pct(r["w_xs"], 1, True),
                     pct(r["w_beat"], 0), ""])
    return table(rows, [118, 30, 52, 46, 46, 62, 76, 46, 49], zebra=True, rule_rows=[5])


def top_now_table():
    rows = [["#", "Ticker", "Company", "Sector", "Mkt cap", "1Y", "Sep", "QTD", "Bucket"]]
    style = []
    d = TOP_NOW.rename(columns={"Market Cap": "mcap"}).sort_values("mcap", ascending=False)
    for i, r in enumerate(d.itertuples(), 1):
        name = r.Company if len(str(r.Company)) <= 30 else str(r.Company)[:29] + "…"
        rows.append([str(i), r.ticker, name, r.Sector, money(r.mcap * 1e6) if not bad(r.mcap) else "—", pct(r.R12),
                     pct(r.R1), pct(r.QTD, 1, True), CODE[r.bucket]])
        if r.R12 < 0:
            style.append(("BACKGROUND", (8, i), (8, i), colors.HexColor(TINT)))
    t = table(rows, [24, 42, 140, 110, 52, 44, 42, 40, 31], align_left=(1, 2, 3), font=7)
    t.setStyle(TableStyle(style))
    return t


def fifths_table():
    names = {1: "Worst 20% (lowest 1-year returns)", 2: "Worst 20–40%", 3: "Middle 20%", 4: "Best 20–40%",
             5: "Best 20% (highest 1-year returns)"}
    rows = [["Group (ranked by 1-year return, Sep 30, 2026)", "1-year return range", "Stocks"]]
    for k in range(1, 6):
        q = QU[k]
        lo = "below" if k == 1 else pct(q["min"], 1, True)
        rng = f"below {pct(q['max'], 1, True)}" if k == 1 else (
            f"above {pct(q['min'], 1, True)}" if k == 5 else f"{lo} to {pct(q['max'], 1, True)}")
        rows.append([names[k], rng, f"{q['n']}"])
    return table(rows, [230, 150, 50], zebra=True)


def picks_table():
    a, b = CELL.loc["1Y Q2 + 1M neg"], CELL.loc["1Y Q1-Q2 + Sep below -10%"]
    c = MB["100"][(MB["100"].window == "Oct15_Dec15") & (MB["100"].group == "NEG1Y_POS1M")].iloc[0]
    n_now = {k: CANDS.basket.str.contains(k).sum() for k in "ABC"}
    spec = [("A  1Y worst 20–40%, Sep down", a, n_now["A"]), ("B  1Y worst 40%, Sep below −10%", b, n_now["B"]),
            ("C  Top-100, 1Y down, Sep up", c, n_now["C"]), ("All Sep losers", CELL.loc["1M neg (all)"], None),
            ("SPY", CELL.loc["SPY"], None), ("Double winners (1Y up, Sep up)", CELL.loc["PP (all)"], None)]
    rows = [["Basket (Oct 15–Dec 15, 2018–2025)", "Names now", "Avg", "Sharpe", "Alpha", "Beta", "Up yrs", "Beat SPY",
             "Worst vs SPY"]]
    for lab, r, n in spec:
        yrs = int(r["years"])
        up = int(r["up"]) if "up" in r else int(round(r["pct_pos"] * yrs))
        beat = int(r["beat_spy"]) if "beat_spy" in r else int(round(r["hit_vs_spy"] * yrs))
        rows.append([lab, "" if n is None else f"{n}", pct(r["mean"], 1, True), num(r["sharpe"]),
                     "—" if lab == "SPY" else pct(r["alpha_spy_ann"], 1, True), num(r["beta_spy"]), f"{up} of {yrs}",
                     "—" if lab == "SPY" else f"{beat} of {yrs}",
                     "—" if lab == "SPY" else pct(r["worst_vs_spy"], 1, True)])
    return table(rows, [150, 44, 40, 40, 44, 34, 44, 46, 58], zebra=True, bold_rows=[1], rule_rows=[4])


def names(code, k=40):
    g = CANDS[CANDS.basket.str.contains(code)]
    return ", ".join(g.ticker.head(k)), len(g)


SHORT = {
    "sep_ls_dec31": "Sept losers − Sept winners, Oct 15 → Dec 31", "sep_ls_q4": "Sept losers − Sept winners, full Q4",
    "sep_ls_dec15": "Sept losers − Sept winners, Oct 15 → Dec 15",
    "nn_pp": "Double losers − double winners, Oct 15 → Dec 15", "oct_rev": "1Y winners − 1Y losers, Oct 1–15",
    "long_a": "Basket A vs SPY, Oct 15 → Dec 15", "long_b": "Basket B vs SPY, Oct 15 → Dec 15",
    "long_sep": "Sept losers vs SPY, Oct 15 → Dec 15", "long_nn": "Double losers vs SPY, Oct 15 → Dec 15",
    "long_np": "1Y losers with Sept up vs SPY, Oct 15 → Dec 15", "long_jan": "1Y losers vs SPY, Dec 15 → Jan 31"}
TIER_LAB = {"All": "All caps", "Mega": "Mega (>$200B)", "Large": "Large ($10–200B)", "Mid": "Mid ($2–10B)",
            "Small": "Small (<$2B)"}
ORANGE_TINT = "#fbe1d4"


def rk(key, tier, f):
    r = RK[(RK.key == key) & (RK.tier == tier)]
    return r[f].iloc[0] if len(r) else np.nan


def ranked():
    d = RK[RK.years >= 7].copy()
    for c in ["t", "mean", "hit", "worst"]:
        d["r_" + c] = d[c].rank(ascending=False)
    d["score"] = d[["r_t", "r_mean", "r_hit", "r_worst"]].mean(axis=1)
    return d.sort_values(["score", "t"], ascending=[True, False])


def rank_table(n=14):
    d = ranked().head(n)
    rows = [["#", "Strategy", "Market cap", "Avg / yr", "t", "Positive", "Worst yr", "Names L / S"]]
    for i, r in enumerate(d.itertuples(), 1):
        nm = f"{r.avg_long_n:.0f}" + ("" if r.kind == "long only" else f" / {r.avg_short_n:.0f}")
        rows.append([str(i), SHORT[r.key], TIER_LAB[r.tier], pct(r.mean, 2, True), num(r.t, 1, True),
                     f"{int(round(r.hit * r.years))} of {int(r.years)}", pct(r.worst, 1, True), nm])
    t = table(rows, [18, 196, 76, 46, 32, 42, 48, 67], align_left=(1, 2), zebra=True)
    t.setStyle(TableStyle([("FONT", (0, 1), (-1, 1), "Sans-Bold", 8)]))
    return t


def grid_table():
    tiers = ["All", "Mega", "Large", "Mid", "Small"]
    order = ["sep_ls_dec31", "sep_ls_q4", "sep_ls_dec15", "nn_pp", "oct_rev", "long_b", "long_a", "long_sep",
             "long_nn", "long_np", "long_jan"]
    rows = [["Strategy"] + [TIER_LAB[t] for t in tiers]]
    style = []
    for ri, k in enumerate(order, 1):
        line = [SHORT[k]]
        for ci, t in enumerate(tiers, 1):
            m, tt, yrs = rk(k, t, "mean"), rk(k, t, "t"), rk(k, t, "years")
            if bad(m) or bad(yrs) or yrs < 4:
                line.append("too few")
                continue
            line.append(f"{pct(m, 1, True)}  t {num(tt, 1, True)}")
            if tt >= 2.5:
                style.append(("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor(TINT)))
            elif tt <= -2.0:
                style.append(("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor(ORANGE_TINT)))
        rows.append(line)
    t = table(rows, [190, 67, 67, 67, 67, 67], font=7.5)
    t.setStyle(TableStyle(style))
    return t


def tier_names(tier, side, k=15):
    g = TNOW[(TNOW.tier == tier) & (TNOW.side == side)].sort_values("mcap", ascending=False)
    return ", ".join(g.ticker.head(k)), len(g)


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
m100, m50 = MB["100"], MB["50"]


def mb(n, w, g, f):
    d = MB[n]
    return d[(d.window == w) & (d.group == g)][f].iloc[0]


MK = {
    "oct": MS["100"].loc[("Oct1_15", "NEG_1Y-POS_1Y"), "mean"], "oct_hit": MS["100"].loc[("Oct1_15", "NEG_1Y-POS_1Y"), "hit"],
    "q4_al_n": mb("100", "Q4", "NEG_1Y", "alpha_spy_ann"), "q4_al_p": mb("100", "Q4", "POS_1Y", "alpha_spy_ann"),
    "q4_sh_n": mb("100", "Q4", "NEG_1Y", "sharpe"), "q4_sh_p": mb("100", "Q4", "POS_1Y", "sharpe"),
    "q4_t": MS["100"].loc[("Q4", "NEG_1Y-POS_1Y"), "t"], "w_np": MS["100"].loc[("Oct15_Dec15", "NEG_1Y-POS_1Y"), "mean"],
    "w_np_t": MS["100"].loc[("Oct15_Dec15", "NEG_1Y-POS_1Y"), "t"],
    "q4_50": MS["50"].loc[("Q4", "NEG_1Y-POS_1Y"), "mean"], "w_50": MS["50"].loc[("Oct15_Dec15", "NEG_1Y-POS_1Y"), "mean"],
    "np_n": mb("100", "Oct15_Dec15", "NEG1Y_POS1M", "avg_n"), "np_ret": mb("100", "Oct15_Dec15", "NEG1Y_POS1M", "mean"),
    "np_beat": mb("100", "Oct15_Dec15", "NEG1Y_POS1M", "hit_vs_spy"),
    "np_yrs": int(mb("100", "Oct15_Dec15", "NEG1Y_POS1M", "years")),
    "spy_w": mb("100", "Oct15_Dec15", "SPY", "mean"),
    "neg_xs": M["names_neg1y"]["q4_xs"], "neg_beat": M["names_neg1y"]["q4_beat"], "neg_obs": M["names_neg1y"]["n"],
    "pos_xs": M["names_pos1y"]["q4_xs"], "pos_beat": M["names_pos1y"]["q4_beat"],
}
now = TOP_NOW.set_index("ticker")


def now_list(code):
    return ", ".join(t for t in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "ORCL", "NFLX", "LLY",
                                 "JPM", "WMT"] if t in now.index and CODE[now.loc[t, "bucket"]] == code)


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
bluf = (f"<b>The read.</b> The strongest signal is the September sort run as a long/short: own the stocks that fell "
        f"in September, short the ones that rose, Oct 15 → Dec 31. {pct(rk('sep_ls_dec31', 'All', 'mean'), 2, True)} a "
        f"year, t {num(rk('sep_ls_dec31', 'All', 't'), 1, True)}, positive "
        f"{frac(rk('sep_ls_dec31', 'All', 'hit'))} years, worst year {pct(rk('sep_ls_dec31', 'All', 'worst'), 1, True)}. "
        f"By market cap it holds in large ($10–200B: full Q4 positive {frac(rk('sep_ls_q4', 'Large', 'hit'))} years), "
        f"mid (t {num(rk('sep_ls_dec31', 'Mid', 't'), 1, True)}) and small caps "
        f"({pct(rk('sep_ls_dec31', 'Small', 'mean'), 1, True)} a year), and fails in mega-caps over $200B, where "
        f"September losers lagged SPY in {frac(1 - rk('long_sep', 'Mega', 'hit'))} years. Long-only baskets beat SPY on "
        f"average, mostly through beta. On the 1Y sort alone Q4 is a tie (Sharpe {num(K['sh_n1y'])} vs "
        f"{num(K['sh_p1y'])}); 1Y losers get sold Oct 1–15; January adds nothing.<br/><br/>"
        f"<b>Setup:</b> long September losers, short September winners, equal weight, $200B market cap and under. "
        f"<b>Entry</b> Oct 15. <b>Exit</b> Dec 31. <b>Invalidation</b>: spread below "
        f"{pct(rk('sep_ls_dec31', 'All', 'worst') - 0.0005, 1)} at Dec 31, worse than any year since {START}. "
        f"Ranking on page 2, names on page 3.")
box = Table([[P(bluf, "bluf")]], colWidths=[W])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                         ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(BLUE)),
                         ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                         ("LEFTPADDING", (0, 0), (-1, -1), 13), ("RIGHTPADDING", (0, 0), (-1, -1), 13)]))
q = QU
key_txt = (
    "<b>What the terms mean</b><br/>"
    "<b>1Y</b> = a stock's total return over the 12 months ending Sep 30. "
    "<b>1M</b> or <b>Sep</b> = its return in September alone.<br/>"
    "<b>Quintiles (fifths)</b> measure how a stock's 1-year return ranks against every other stock. Each Sep 30, "
    "all ~2,300 stocks are lined up from the lowest 1-year return to the highest and split into five equal groups "
    "of ~460. <b>Worst 20%</b> = the 460 stocks with the lowest 1-year returns. <b>Worst 20–40%</b> = the next 460. "
    "<b>Middle 20%</b>, <b>best 20–40%</b>, then <b>best 20%</b> = the 460 highest. "
    f"This year: worst 20% = below {pct(q[1]['max'], 1, True)}; worst 20–40% = {pct(q[2]['min'], 1, True)} to "
    f"{pct(q[2]['max'], 1, True)}; middle = {pct(q[3]['min'], 1, True)} to {pct(q[3]['max'], 1, True)}; "
    f"best 20–40% = {pct(q[4]['min'], 1, True)} to {pct(q[4]['max'], 1, True)}; best 20% = above "
    f"{pct(q[5]['min'], 1, True)}.<br/>"
    "<b>NN / NP / PN / PP</b> = 1Y negative or positive, then September negative or positive "
    "(NN = down on both, PP = up on both). <b>N−P</b> = negative group's return minus positive group's.")
keybox = Table([[P(key_txt, "body")]], colWidths=[W])
keybox.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(INK)),
                            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
story += [box, Spacer(1, 10), keybox, Spacer(1, 8), P("Findings", "h2")]
findings = [
    f"<b>1Y sort: no Q4 edge.</b> Sharpe {num(K['sh_n1y'])} vs {num(K['sh_p1y'])}; Q4 spread "
    f"{pct(K['q4_np'], 2, True)}, t {num(K['q4_np_t'], 1, True)}.",
    f"<b>1M sort: the clean Q4 edge.</b> September losers minus September winners {pct(K['q4_1m'], 2, True)} in Q4 "
    f"(t {num(K['q4_1m_t'], 1, True)}, positive {frac(K['q4_1m_hit'])}); Oct 15–Dec 31 {pct(K['w_1m31'], 2, True)} "
    f"(t {num(K['w_1m31_t'], 1, True)}, positive {frac(K['w_1m31_hit'])}, worst year {pct(K['w_1m31_worst'], 2, True)}).",
    f"<b>Oct 1–15 is the flush.</b> 1Y losers minus winners {pct(K['oct_np'], 2, True)} (t {num(K['oct_np_t'], 1, True)}), "
    f"1Y+1M {pct(K['oct_nn'], 2, True)}, worst vs best 20% on 1Y {pct(K['oct_q'], 2, True)}. October is also the worst "
    f"calendar month for the 1Y spread after SPY beta ({pct(oct_m.alpha, 2, True)}).",
    f"<b>Mid-October to mid-December is the recovery.</b> 1Y+1M neg minus pos, Oct 15–Dec 15: "
    f"{pct(K['w_nn'], 2, True)} (t {num(K['w_nn_t'], 1, True)}, positive {frac(K['w_nn_hit'])}, worst "
    f"{pct(K['w_nn_worst'], 2, True)}, ex-2020 {pct(K['w_nn_x'], 2, True)}). Basket alpha {pct(K['nn_al'], 1, True)} "
    f"annualized. Nov 1–15 ({pct(K['nov_nn'], 2, True)}) and Dec 1–15 ({pct(K['dec_nn'], 2, True)}, positive "
    f"{frac(K['dec_nn_hit'])}) carry it; Dec 16–31 is flat ({pct(K['dec2_np'], 2, True)} on the 1Y sort).",
    f"<b>The relative low prints in November.</b> The loser-vs-winner path bottomed in November in "
    f"{len(nov_troughs)} of {len(YEARS)} years ({', '.join(str(y) for y in nov_troughs)}).",
    f"<b>January is dead.</b> 1Y losers minus winners {pct(K['jan_np'], 2, True)}, 1Y+1M {pct(K['jan_nn'], 2, True)}. "
    f"Dec 15 re-sort to Jan 31: 1Y losers Sharpe {num(K['rb_neg_sh'])} vs winners {num(K['rb_pos_sh'])}; 1Y best 20% "
    f"{num(K['rb_q5_sh'])} beats −30%-or-worse losers {num(K['rb_deep_sh'])}.",
    f"<b>Mega-caps over $200B: the September bounce fails.</b> September losers lagged SPY from Oct 15 to Dec 15 in "
    f"{frac(1 - rk('long_sep', 'Mega', 'hit'))} years ({pct(rk('long_sep', 'Mega', 'mean'), 1, True)} a year, t "
    f"{num(rk('long_sep', 'Mega', 't'), 1, True)}); the long/short is flat. "
    f"In the top 50 names there is no loser effect at all (page 4).",
    f"<b>Long-only edge is mostly beta.</b> Basket A beat SPY by {pct(rk('long_a', 'All', 'mean'), 1, True)} a year "
    f"(t {num(rk('long_a', 'All', 't'), 1, True)}), basket B by {pct(rk('long_b', 'All', 'mean'), 1, True)} "
    f"(t {num(rk('long_b', 'All', 't'), 1, True)}). The spread trades are where the signal is.",
]
story += [Paragraph(f, ST["bullet"], bulletText="•") for f in findings]

na, ca = names("A")
nb, cb = names("B")
nc, cc = names("C")
story += [PageBreak(), P(f"Strategy ranking by market cap, {SPAN}", "h1"),
          P("Every strategy tested in each market-cap tier. Long/short strategies are scored on the yearly long-minus-"
            "short return; long-only on the yearly basket return minus SPY. t = average divided by its standard error "
            "across the 8 years; higher means a steadier, stronger signal. Market cap at each Sep 30 is estimated from "
            "today's market cap and the stock's price change since then.", "note"),
          P("Overall ranking, top 14", "h2"),
          P("Ranked on four factors weighted equally: signal strength (t), average return per year, positive years, "
            "and worst year.", "note"),
          rank_table(), Spacer(1, 10),
          P("Every strategy in every tier (average per year and t; blue = t 2.5 or higher, orange = t −2 or lower)",
            "h2"), grid_table(), Spacer(1, 8),
          P(f"<b>Best by tier.</b> All caps and mid: Sept losers − Sept winners, Oct 15 → Dec 31. Large: same pair over "
            f"the full Q4, positive {frac(rk('sep_ls_q4', 'Large', 'hit'))} years. Small: the same Oct 15 → Dec 31 pair, "
            f"{pct(rk('sep_ls_dec31', 'Small', 'mean'), 1, True)} a year; the Oct 1–15 trade has the highest t "
            f"({num(rk('oct_rev', 'Small', 't'), 1, True)}) but pays {pct(rk('oct_rev', 'Small', 'mean'), 1, True)} over "
            f"two weeks. Mega: no strategy works; September losers "
            f"lag SPY.", "body")]

lg = {t: tier_names(t, "long") for t in ["Mega", "Large", "Mid", "Small"]}
sh = {t: tier_names(t, "short") for t in ["Mega", "Large", "Mid", "Small"]}
story += [PageBreak(), P("Q4 2026: which names", "h1"),
          P("<b>Strategy #1 today: long September losers, short September winners</b> (Sep 30, 2026 sort, largest "
            "first).", "body"),
          P(f"<b>Large ($10–200B)</b>, long ({lg['Large'][1]}): {lg['Large'][0]}… "
            f"Short ({sh['Large'][1]}): {sh['Large'][0]}…", "body"),
          P(f"<b>Mid ($2–10B)</b>, long ({lg['Mid'][1]}): {lg['Mid'][0]}… Short ({sh['Mid'][1]}): {sh['Mid'][0]}…",
            "body"),
          P(f"<b>Small (&lt;$2B)</b>, long ({lg['Small'][1]}): {lg['Small'][0]}… "
            f"Short ({sh['Small'][1]}): {sh['Small'][0]}…", "body"),
          P(f"<b>Mega (&gt;$200B), excluded.</b> September losers ({lg['Mega'][1]}): {lg['Mega'][0]}… "
            f"September winners ({sh['Mega'][1]}): {sh['Mega'][0]}…", "body"),
          Spacer(1, 6),
          P("<b>Quintiles = 1-year return rank.</b> All ~2,300 stocks are lined up by their 1-year return to Sep 30 "
            "and split into five equal groups. The group tells you how bad or good a stock's last 12 months were "
            "compared with everything else. This year's groups:", "body"),
          fifths_table(), Spacer(1, 10),
          P("<b>The three best baskets for Oct 15 → Dec 15, tested 2018–2025.</b> A has the best odds (up 7 of 8 "
            "years, mildest bad year). B has the highest Sharpe and alpha with more swing. C held among the top 100 "
            "but not when mega-caps are defined by market cap (page 2), so treat it as weak. Every name carries its "
            "basket's record; the study tests baskets, not single stocks.", "body"),
          picks_table(), Spacer(1, 10),
          P(f"<b>A</b> ({ca} names, largest first): {na}…", "body"),
          P(f"<b>B</b> ({cb} names, largest first): {nb}…", "body"),
          P(f"<b>C</b> ({cc} names): {nc}.", "body"),
          P("172 names sit in both A and B; that overlap tested weaker than either basket alone. Full lists in the "
            "appendix.", "note")]
story += [PageBreak(), P(f"Mega-caps, {SPAN}", "h1"),
          P("The main study equal-weights roughly 2,000 names, so AAPL counts the same as a $2B stock. Here the same "
            "tests run only on the 100 (and 50) biggest names, re-picked at each Sep 30 from data available then (past "
            "market caps aren't in the data, so trading activity stands in for size), so the list never uses "
            "hindsight.", "note"),
          P(f"<b>Read:</b> the Oct 1–15 flush shows up in the top 100 ({pct(MK['oct'], 2, True)}, losers lag "
            f"{frac(1 - MK['oct_hit'])} years). After that the loser edge is directional but not significant: Q4 "
            f"Sharpe {num(MK['q4_sh_n'])} vs {num(MK['q4_sh_p'])}, Oct 15–Dec 15 spread {pct(MK['w_np'], 2, True)} "
            f"(t {num(MK['w_np_t'], 1, True)}). In the top 50 it disappears (Q4 {pct(MK['q4_50'], 2, True)}). The one "
            f"mega-cap bucket that stands out is 1Y negative with a positive September: about {MK['np_n']:.0f} names, "
            f"Oct 15–Dec 15 {pct(MK['np_ret'], 1, True)} vs SPY {pct(MK['spy_w'], 1, True)}, beat SPY "
            f"{int(round(MK['np_beat'] * MK['np_yrs']))} of {MK['np_yrs']} years.", "body"),
          P("Top 100: bucket stats", "h2"), mega_bucket_table("100"),
          P("Pair spreads, top 100 vs top 50", "h2"), mega_spread_table()]
story += [PageBreak(), P("The biggest names, year by year", "h1"),
          P("Each cell: bucket at Sep 30 (NN = 1Y neg + 1M neg, NP = 1Y neg + 1M pos, PN = 1Y pos + 1M neg, PP = 1Y pos + "
            "1M pos), then the stock's Q4 return minus SPY's, in points. Blue cells = 1Y negative at Sep 30.", "note"),
          mega_name_table(), Spacer(1, 8),
          P("Pooled across those names and years (excess return vs SPY)", "h2"), mega_agg_table(), Spacer(1, 8),
          P("Where they sit now (Sep 30, 2026)", "h2"),
          P(f"<b>NN</b> (1Y neg + 1M neg): {now_list('NN')}. <b>NP</b> (1Y neg + 1M pos): {now_list('NP')}. "
            f"<b>PN</b> (1Y pos + 1M neg): {now_list('PN')}. <b>PP</b> (1Y pos + 1M pos): {now_list('PP')}. "
            "Full top 100 at the end of the report.", "body")]

story += [PageBreak(), P(f"Q4 risk-adjusted returns by bucket, {SPAN}", "h1"),
          P("Buckets formed at the last September close, equal weight, buy-and-hold to Dec 31. Annualized stats pool "
            "every Q4 trading day across the years. Alpha and beta vs SPY; Sharpe and Sortino over 13-week T-bills. "
            "Avg Q4 / Median / Beat SPY / Worst are per-year figures. 1Y worst 20% … best 20% = the stock's 1-year "
            "return rank among all stocks that Sep 30 (see page 1).", "note"),
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
          P("N−P = negative minus positive; NN−PP = 1Y+1M negative minus 1Y+1M positive. Trough = day the 1Y loser-vs-winner "
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
          P("Oct 15 sort recomputes 1Y and 1M as of Oct 15 (point in time). Median = spread of per-year bucket "
            "medians. Ex-20 drops 2020.", "note"),
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
    f"<b>Universe.</b> The 2,367 tickers in the uploaded finviz list. Daily dividend- and split-adjusted price "
    f"data from Yahoo Finance via yfinance. Formation years {START}–{LAST} for the Q4 work; Q1 {START} through Q3 "
    f"2026 for the other-quarter comparison; Jan {START} through Sep 2026 for calendar months. "
    f"{B['n_stock_years']:,} stock-years pass the Sep 30 screen.",
    "<b>Signals.</b> Formed at the last trading day of September: 1Y = trailing 12-month total return, 1M = "
    "September return. Eligibility: a full 12 months of history; untradeable micro-shells are screened out. "
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

story += [PageBreak(), P("Top 100 names, Sep 30, 2026", "h1"),
          P("Sorted by market cap. Bucket codes as above; blue = 1Y negative. QTD through Oct 9, 2026.", "note"),
          top_now_table()]
story += [PageBreak(), P("Appendix: every name in baskets A, B and C, Sep 30, 2026", "h1"),
          P(f"All {len(CANDS):,} names, sorted by market cap. Basket A = worst 20–40% of 1-year returns and September down; "
            "B = worst 40% of 1-year returns and September down more than 10%; C = top-100 name with 1Y down and September up. "
            "QTD through Oct 9, 2026.", "note"), candidates_table()]


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
