"""Q4 2026 long-only buy list: small caps (< $2B) down more than 10% in September, with company, sector,
industry and market cap. Landscape, two tables side by side.

usage: python one_pager.py TIERS_CSV UNIVERSE_CSV LONG_SCAN_CSV CELLS_CSV OUT.pdf
"""
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TIERS, UNIV, SCAN, CELLS, OUT = (Path(a) for a in sys.argv[1:6])
F = Path("/usr/share/fonts/truetype/crosextra")
pdfmetrics.registerFont(TTFont("Sans", str(F / "Carlito-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(F / "Carlito-Bold.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
PAGE = landscape(letter)
M = 0.4 * inch


def pct(v, d=1):
    return f"{v * 100:+.{d}f}%".replace("-", "−")


def cap(m):
    return f"${m / 1e3:.2f}B" if m >= 1e3 else f"${m:.0f}M"


def cut(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


t = pd.read_csv(TIERS)
u = pd.read_csv(UNIV)
u["ticker"] = u.Ticker.str.replace(".", "-", regex=False)
buy = t[(t.tier == "Small") & (t.r1 < -0.10)].merge(u[["ticker", "Company", "Sector", "Industry", "Market Cap"]],
                                                     on="ticker", how="left")
buy = buy.sort_values(["Sector", "Market Cap"], ascending=[True, False]).reset_index(drop=True)
broad_n = int(((t.tier == "Small") & (t.r1 < 0)).sum())
scan = pd.read_csv(SCAN).set_index(["basket", "tier", "window"])
s = scan.loc[("Sept down >10%", "Small", "Oct 15-Dec 31")]
b = scan.loc[("Sept losers", "Small", "Oct 15-Dec 31")]
spy_sh = pd.read_csv(CELLS).query("cell == 'SPY' and window == 'Oct15_Dec31'").sharpe.iloc[0]
by_sector = buy.Sector.value_counts()

st = {k: ParagraphStyle(k, fontName="Sans", fontSize=sz, leading=sz * 1.3, textColor="#111111", spaceAfter=sp)
      for k, sz, sp in [("h", 15, 2), ("b", 9.5, 2), ("s", 8, 2)]}
story = [
    Paragraph("Q4 2026 long-only buy list", st["h"]),
    Paragraph(f"<b>Buy all {len(buy)} stocks below: market cap under $2B, down more than 10% in September 2026. Equal "
              f"weight. Entry Oct 15, 2026. Exit Dec 31, 2026. Invalidation: basket trails SPY by more than "
              f"{pct(s.worst_vs_spy)} at Dec 31.</b>", st["b"]),
    Paragraph(f"2018–2025: beat SPY by {pct(s.excess_mean)} a year (median {pct(s.excess_median)}), {int(s.beat)} of "
              f"{int(s.years)} years; alpha {pct(s.alpha_ann)} annualized, t {s.t_alpha:.1f}; Sharpe {s.sharpe:.2f} vs "
              f"SPY {spy_sh:.2f}; beta {s.beta:.1f}. Broader version: all {broad_n} small caps that fell in September, "
              f"{pct(b.excess_mean)} a year vs SPY, t {b.t_excess:.1f}.", st["s"]),
    Paragraph("By sector: " + ", ".join(f"{k} {v}" for k, v in by_sector.items()) + ". Sorted by sector, then market "
              "cap. Market cap and classifications from the uploaded finviz list; the under-$2B cut uses market cap at "
              "Sep 30, so a few names that have rallied since now show slightly above $2B.", st["s"]),
    Spacer(1, 4),
]

HDR = ["Ticker", "Company", "Sector", "Industry", "Mkt cap"]
W = [33, 113, 74, 109, 36]
RH = 7.0


def cells(r):
    return [r.ticker, cut(r.Company, 34), cut(r.Sector, 22), cut(r.Industry, 33), cap(r["Market Cap"])]


def page_table(chunk):
    n = -(-len(chunk) // 2)
    left, right = chunk.iloc[:n], chunk.iloc[n:]
    rows = [HDR + [""] + HDR]
    for i in range(n):
        line = cells(left.iloc[i]) + [""] + (cells(right.iloc[i]) if i < len(right) else [""] * 5)
        rows.append(line)
    tb = Table(rows, colWidths=W + [10] + W, rowHeights=RH)
    style = [("FONT", (0, 0), (-1, -1), "Sans", 6.6), ("FONT", (0, 0), (-1, 0), "Sans-Bold", 6.6),
             ("TEXTCOLOR", (0, 0), (-1, -1), "#111111"), ("TOPPADDING", (0, 0), (-1, -1), 0),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5), ("LEFTPADDING", (0, 0), (-1, -1), 2),
             ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("ALIGN", (4, 0), (4, -1), "RIGHT"),
             ("ALIGN", (10, 0), (10, -1), "RIGHT"), ("LINEBELOW", (0, 0), (4, 0), 0.6, colors.black),
             ("LINEBELOW", (6, 0), (10, 0), 0.6, colors.black)]
    for r in range(2, len(rows), 2):
        style += [("BACKGROUND", (0, r), (4, r), colors.HexColor("#f2f2f2")),
                  ("BACKGROUND", (6, r), (10, r), colors.HexColor("#f2f2f2"))]
    tb.setStyle(TableStyle(style))
    return tb


FIRST, REST = 62, 74          # rows per column: page 1 (below the header text), later pages
start, cap_rows = 0, FIRST
while start < len(buy):
    chunk = buy.iloc[start:start + 2 * cap_rows]
    if start:
        story.append(PageBreak())
    story.append(page_table(chunk))
    start += 2 * cap_rows
    cap_rows = REST


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7)
    canvas.drawRightString(PAGE[0] - M, 0.25 * inch, f"Q4 2026 long-only buy list · page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=PAGE, leftMargin=M, rightMargin=M, topMargin=0.35 * inch,
                        bottomMargin=0.4 * inch, title="Q4 2026 long-only buy list")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, len(buy))
