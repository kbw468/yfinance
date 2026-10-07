"""Data-quality cleaning applied after download, before any factor is computed.

Every rule here was motivated by an observed problem in the raw yfinance panel:
  * Pre-US-listing zero-volume stubs (SW, FERG, AMCR): thousands of 0-volume, stale-price days
    inherited from a foreign listing. Masked up to and including the last long zero-volume run.
  * Isolated zero-volume days (HWM, CHTR, VRT ...): volume set NaN so volume ratios skip them.
  * A handful of single-day OHLC inconsistencies (High < Close etc.): High/Low repaired to envelope O/H/L/C.
  * Partial intraday row for the current session: dropped so every bar is a completed session.
  * Unadjusted corporate actions (CTVA spin-off 2026-10-01, no split/dividend recorded by Yahoo):
    the event day return is masked and the ticker is excluded from live ranking.
"""
import datetime as dt
import numpy as np
import pandas as pd

FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# ticker -> date of unadjusted corporate action (detected: |ret| > 50% with 10x volume and adj == raw)
KNOWN_UNADJUSTED_EVENTS = {"CTVA": "2026-10-01"}
ZERO_SHARE = 0.25  # share of zero-volume days (trailing 250) that marks pre-listing stub data


def _mask_zero_volume_stubs(panel: dict) -> tuple[dict, dict]:
    """Foreign-listing stubs show up as a long stretch where a large share of days print zero volume,
    ending abruptly at the US listing date. Cutoff = last zero-volume day, provided >= ZERO_SHARE of the
    250 sessions before it were zero-volume. Isolated zero prints (CHTR 2010, VRT 2019) do not qualify."""
    V = panel["Volume"]
    cutoffs = {}
    for t in V.columns:
        v = V[t].dropna()
        z = v == 0
        if not z.any():
            continue
        last_zero = z[z].index[-1]
        share = z.loc[:last_zero].iloc[-250:].mean()
        if share >= ZERO_SHARE:
            cutoffs[t] = last_zero
    for t, cutoff in cutoffs.items():
        for f in FIELDS:
            panel[f].loc[:cutoff, t] = np.nan
    return panel, cutoffs


def clean_panel(panel: dict, today: dt.date | None = None, verbose=True, keep_partial: bool = False) -> tuple[dict, dict]:
    """keep_partial=True keeps a current-session (intraday) bar: used only by the intraday projection."""
    panel = {k: v.copy() for k, v in panel.items()}
    report = {}

    # 1. drop a partial current-session row. The current date's bar is kept only once the US session is closed
    #    and Yahoo has finalised it (after 21:30 UTC); before that it is intraday and is dropped.
    now = dt.datetime.utcnow()
    today = today or now.date()
    last = panel["Close"].index[-1].date()
    session_final = now >= dt.datetime.combine(today, dt.time(21, 30))
    if not keep_partial and (last > today or (last == today and not session_final)):
        for k in panel:
            panel[k] = panel[k].iloc[:-1]
        report["dropped_partial_row"] = str(last)

    # 2. pre-listing zero-volume stubs
    panel, cutoffs = _mask_zero_volume_stubs(panel)
    report["zero_volume_stub_cutoffs"] = {k: str(v.date()) for k, v in cutoffs.items()}

    # 3a. trailing placeholder bars: zero volume AND flat OHLC AND the last bar of the series. A name that has stopped
    #     trading (cash acquisition, delisting) keeps printing a frozen bar on Yahoo; it is not a session. Masked in
    #     every field so the name drops out of eligibility on that date instead of being scored off a phantom bar.
    V = panel["Volume"]
    O, H, L, C = (panel[f] for f in ["Open", "High", "Low", "Close"])
    flat0 = (V == 0) & (O == H) & (H == L) & (L == C) & C.notna()
    trailing = {}
    for t in C.columns:
        s = C[t].dropna()
        while len(s) and bool(flat0.loc[s.index[-1], t]):
            li = s.index[-1]
            for f in panel:
                panel[f].loc[li, t] = np.nan
            trailing[t] = str(li.date())
            s = s.iloc[:-1]
    report["trailing_placeholder_bars"] = trailing
    if verbose and trailing:
        print(f"trailing placeholder bars masked: {trailing}")
    # 3b. isolated zero-volume days -> NaN volume (price kept)
    V = panel["Volume"]
    iso = (V == 0) & panel["Close"].notna()
    report["isolated_zero_volume_days"] = int(iso.sum().sum())
    panel["Volume"] = V.mask(iso)

    # 4. OHLC envelope repair
    O, H, L, C = (panel[f] for f in ["Open", "High", "Low", "Close"])
    hi = pd.concat([O, H, L, C], keys=list("OHLC")).groupby(level=1).max()
    lo = pd.concat([O, H, L, C], keys=list("OHLC")).groupby(level=1).min()
    viol = ((H < hi) | (L > lo)) & C.notna()
    report["ohlc_repairs"] = int(viol.sum().sum())
    panel["High"] = H.where(~viol, hi)
    panel["Low"] = L.where(~viol, lo)

    # 5. known unadjusted corporate actions: mask from the event forward (history before stays usable)
    for t, d in KNOWN_UNADJUSTED_EVENTS.items():
        if t in panel["Close"].columns:
            for f in FIELDS:
                panel[f].loc[d:, t] = np.nan
    report["unadjusted_events_masked"] = KNOWN_UNADJUSTED_EVENTS

    # 6. negative / zero prices
    for f in ["Open", "High", "Low", "Close"]:
        bad = panel[f] <= 0
        if bad.any().any():
            report[f"nonpositive_{f}"] = int(bad.sum().sum())
            panel[f] = panel[f].mask(bad)

    if verbose:
        for k, v in report.items():
            print(f"[clean] {k}: {v}")
    return panel, report
