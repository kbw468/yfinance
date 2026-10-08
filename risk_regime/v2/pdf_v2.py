#!/usr/bin/env python3
"""Print the v2 dashboard to a Letter PDF with Chromium. Usage: python3 pdf_v2.py dashboard_v2.html out.pdf"""
import sys, os
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]; html = open(src).read()
with sync_playwright() as p:
    try: b = p.chromium.launch()
    except Exception: b = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH', '/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell'))
    pg = b.new_page(viewport={'width': 1100, 'height': 900}, color_scheme='light')
    pg.set_content('<!doctype html><html><head><meta charset="utf-8"></head><body>' + html + '</body></html>', wait_until='load'); pg.wait_for_timeout(500); pg.emulate_media(media='print')
    pg.pdf(path=out, format='Letter', print_background=True, margin={'top': '12mm', 'bottom': '12mm', 'left': '10mm', 'right': '10mm'}); b.close()
print('wrote', out, os.path.getsize(out) // 1024, 'KB')
