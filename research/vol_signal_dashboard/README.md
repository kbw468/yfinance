# Drawdown Signal Board

Builds the SPY drawdown dashboard published at https://claude.ai/artifact/LNYtWCQKz4X9Fd6wNGfeuy.

```
python build_dashboard.py --out signal-board.html --json signal-board.json   # rebuild the page
python select_barometer.py                                                    # re-check barometer inputs (report)
python select_barometer.py --apply                                            # ... and switch if the rule says so
```

## Standing rules (from the owner — every scheduled run follows these)

1. **Highest-probability multifactor model, period.** The barometer uses whichever equal-weight set of
   3+ inputs scores best out of sample on SPY 5% and 8% drawdown odds. No single input may carry more
   than 50% of the score's movement.
2. **Only inputs that sharpen.** An input stays only if it improves out-of-sample results. Nothing is
   kept for curiosity or narrative, and no input is added just to make the model look diversified.
3. **Don't dilute.** Prefer fewer, stronger inputs and tiers over more, weaker ones.
4. **Probability first.** Every reading is shown with the odds of a 5% and an 8% drop that actually
   followed it out of sample. Direction (5%) and size (8%) both matter.
5. **Post-COVID weighting.** Judge models on mid-2020 onward first (0DTE, CTAs, vol-control funds, pod
   shops), with 2015–2020 as a check that the sign does not flip.
6. **No silent model changes.** Scheduled refreshes never change the model. Only the monthly re-check
   may switch barometer inputs, and only when `select_barometer.py` says switch (a different set beats
   the live one by at least 0.01 AUC on both 5% and 8% odds since mid-2020). Every switch is reported.
7. **Short status, led by the call.** Status replies are three lines max: (1) the call, BUY, HOLD or
   SELL, and what flips it at the next close; (2) what changed, then the barometer reading and the
   call's odds; (3) whether rate of change is setting up better or worse than the prior run, with the
   1/2/3-day moves of VIX, VXN, VVIX and VXN/VIX first (the owner reads short horizons first), then the
   10-session moves and credit. Longer analysis only when asked.
8. **Presentation.** Deutan-safe blue/orange only, no red/green. No risk warnings, no narrative, no
   hedging language.
9. **One call: BUY, HOLD or SELL. No other labels.** SELL is a Working or High conviction print in the
   last 10 sessions. Otherwise, with SPY within 1.5% of its high, it's HOLD at barometer 7+ and BUY at 6 or
   lower. With SPY more than 1.5% off its high, it's BUY down to −5% and HOLD below. The barometer feeds
   the call and is never the headline.

## Schedule (America/New_York, weekdays)

- On demand ("update", "run update", "rerun"): same steps and the same three-line reply as a
  scheduled run, using live prices if the session is open.
- 9:56 AM and 2:56 PM: intraday refresh and republish.
- 4:40 PM: after-close refresh and republish (DSPX may still show the prior day until Cboe posts).
- 1st of each month, 5:10 PM: run `select_barometer.py --apply`; if it switches, commit, push,
  rebuild and republish.

## Inputs

- Yahoo Finance via yfinance: SPY, QQQ, HYG, IEF, ^VIX, ^VVIX, ^VXN, DX-Y.NYB (plus XLU and ^MOVE
  when a barometer candidate needs them).
- Cboe daily history CSVs: DSPX, and COR1M when needed.
- `barometer.json`: the live barometer inputs, selection date and evidence.
