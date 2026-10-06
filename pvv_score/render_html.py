"""Render results/REPORT.md to a styled HTML page (for publishing) and a print HTML (for the PDF)."""
import re
import sys
import pathlib
import markdown

from .config import RESULTS_DIR

STYLE = """<title>PVV Score Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: single reading column, wide tables scroll in their own frame; deutan-safe blue/orange accents */
:root{--bg:#f7f8fa;--fg:#1a1f2b;--muted:#5b6475;--rule:#d9dde6;--accent:#1f5fbf;--accent2:#d9731a;--code-bg:#eef1f6;--th-bg:#e8edf6;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#13161d;--fg:#e6e9ef;--muted:#9aa3b5;--rule:#2b3140;--accent:#6ea3f2;--accent2:#f0a35e;--code-bg:#1c212c;--th-bg:#1f2735;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#13161d;--fg:#e6e9ef;--muted:#9aa3b5;--rule:#2b3140;--accent:#6ea3f2;--accent2:#f0a35e;--code-bg:#1c212c;--th-bg:#1f2735;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.55;margin:0;padding-block:32px 64px;padding-inline:16px}
main{max-width:1100px;margin:0 auto;min-width:0}
h1{font-size:1.7rem;line-height:1.25;text-wrap:balance;margin:0 0 .6em;border-bottom:3px solid var(--accent);padding-bottom:.4em}
h2{font-size:1.25rem;margin:2.2em 0 .6em;color:var(--accent);text-wrap:balance}
h3{font-size:1.05rem;margin:1.6em 0 .5em;color:var(--accent2)}
p,li{max-width:80ch} ul{padding-left:1.2em} li{margin:.3em 0}
code{font-family:var(--mono);font-size:.86em;background:var(--code-bg);padding:.1em .35em;border-radius:3px}
pre{background:var(--code-bg);padding:12px 14px;overflow-x:auto;border-radius:6px;font-size:.85em} pre code{background:none;padding:0}
.tw{overflow-x:auto;margin:1em 0;border:1px solid var(--rule);border-radius:6px;max-height:80vh}
table{border-collapse:collapse;font-family:var(--mono);font-size:.76rem;font-variant-numeric:tabular-nums;white-space:nowrap}
th{background:var(--th-bg);text-align:left;padding:6px 10px;font-weight:600;position:sticky;top:0}
td{padding:4px 10px;border-top:1px solid var(--rule)} tr:nth-child(even) td{background:color-mix(in srgb,var(--code-bg) 50%,transparent)}
strong{font-weight:600} hr{border:0;border-top:1px solid var(--rule)} .meta{color:var(--muted);font-size:.9rem;margin-bottom:2em}
</style>
"""
PRINT = """
@page{size:A4 landscape;margin:12mm}
body{font-size:10.5px;padding:0;background:#fff;color:#1a1f2b} main{max-width:none}
h1{font-size:19px}h2{font-size:14px;page-break-after:avoid;page-break-before:auto}h3{font-size:12px;page-break-after:avoid}
.tw{overflow:visible;border:0;max-height:none} table{font-size:6.9px;white-space:normal;width:100%} th,td{padding:1.5px 3px} th{position:static}
tr{page-break-inside:avoid} pre{white-space:pre-wrap}
</style>"""


def main(out_dir: str):
    md = (RESULTS_DIR / "REPORT.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    meta = '<p class="meta">Generated from branch <code>claude/practical-hamilton-sspq2e</code>, file <code>pvv_score/results/REPORT.md</code>. Ranking as of the last completed session named in the tables.</p>'
    html = STYLE + "<main>\n" + meta + "\n" + body + "\n</main>\n"
    out = pathlib.Path(out_dir)
    (out / "pvv_report.html").write_text(html)
    pr = html.replace('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">', "")
    pr = pr.replace("</style>", PRINT)
    pr = '<!doctype html><html><head><meta charset="utf-8">' + pr.replace("<main>", "</head><body><main>") + "</body></html>"
    (out / "pvv_report_print.html").write_text(pr)
    print("wrote", out / "pvv_report.html", out / "pvv_report_print.html")


if __name__ == "__main__":
    main(sys.argv[1])
