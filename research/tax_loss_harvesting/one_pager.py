"""One-page Q4 2026 long-only buy list: small caps (< $2B) down more than 10% in September.

usage: python one_pager.py TIERS_CSV LONG_SCAN_CSV CELLS_CSV OUT.pdf
"""
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TIERS, SCAN, CELLS, OUT = (Path(a) for a in sys.argv[1:5])
F = Path("/usr/share/fonts/truetype/crosextra")
pdfmetrics.registerFont(TTFont("Sans", str(F / "Carlito-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(F / "Carlito-Bold.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")


def pct(v, d=1):
    return f"{v * 100:+.{d}f}%".replace("-", "−")


t = pd.read_csv(TIERS)
buy = t[(t.tier == "Small") & (t.r1 < -0.10)].sort_values("ticker")
broad_n = int(((t.tier == "Small") & (t.r1 < 0)).sum())
scan = pd.read_csv(SCAN).set_index(["basket", "tier", "window"])
s = scan.loc[("Sept down >10%", "Small", "Oct 15-Dec 31")]
b = scan.loc[("Sept losers", "Small", "Oct 15-Dec 31")]
spy_sh = pd.read_csv(CELLS).query("cell == 'SPY' and window == 'Oct15_Dec31'").sharpe.iloc[0]

st = {k: ParagraphStyle(k, fontName="Sans", fontSize=sz, leading=sz * 1.35, textColor="#111111", spaceAfter=sp)
      for k, sz, sp in [("h", 18, 4), ("b", 10.5, 4), ("s", 8.5, 3)]}
story = [
    Paragraph("Q4 2026 long-only buy list", st["h"]),
    Paragraph(f"<b>Buy all {len(buy)} stocks below: market cap under $2B, down more than 10% in September 2026. "
              "Equal weight. Entry Oct 15, 2026. Exit Dec 31, 2026.</b>", st["b"]),
    Paragraph(f"Highest long-only signal strength in the 2018–2025 test. Beat SPY by {pct(s.excess_mean)} a year on "
              f"average (median {pct(s.excess_median)}), {int(s.beat)} of {int(s.years)} years. Alpha vs SPY "
              f"{pct(s.alpha_ann)} annualized, t {s.t_alpha:.1f}. Sharpe {s.sharpe:.2f} vs SPY {spy_sh:.2f} over the "
              f"same window. Beta {s.beta:.1f}.", st["b"]),
    Paragraph(f"<b>Invalidation:</b> basket trails SPY by more than {pct(s.worst_vs_spy)} at Dec 31, the worst year "
              "since 2018.", st["b"]),
    Paragraph(f"Broader version with the same signal strength: all {broad_n} small caps that fell in September, "
              f"{pct(b.excess_mean)} a year vs SPY, t {b.t_excess:.1f}. Over $200B market cap there is no long-only "
              "signal.", st["s"]),
    Spacer(1, 6),
]
cols = 8
rows_n = -(-len(buy) // cols)
tick = buy.ticker.tolist()
grid = [[tick[c * rows_n + r] if c * rows_n + r < len(tick) else "" for c in range(cols)] for r in range(rows_n)]
tbl = Table(grid, colWidths=[(letter[0] - 1.2 * inch) / cols] * cols, rowHeights=10.6)
tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Sans", 8.5), ("TEXTCOLOR", (0, 0), (-1, -1), "#111111"),
                         ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                         ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd"))]))
story += [tbl, Spacer(1, 6),
          Paragraph("Sorted A–Z. Sep 30, 2026 sort; market cap at Sep 30 estimated from the uploaded list. Prices from "
                    "Yahoo Finance via yfinance.", st["s"])]
doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.5 * inch, bottomMargin=0.45 * inch, title="Q4 2026 long-only buy list")
doc.build(story)
print("wrote", OUT, len(buy))
