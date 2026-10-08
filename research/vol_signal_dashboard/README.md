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
7. **Presentation.** Deutan-safe blue/orange only, no red/green. Status lines lead with what changed,
   then the barometer reading, its band and its odds, then the tier flags and next-close triggers.
   No risk warnings, no narrative, no hedging language.

## Schedule (America/New_York, weekdays)

- 9:56 AM and 2:56 PM: intraday refresh and republish.
- 4:40 PM: after-close refresh and republish (DSPX may still show the prior day until Cboe posts).
- 1st of each month, 5:10 PM: run `select_barometer.py --apply`; if it switches, commit, push,
  rebuild and republish.

## Inputs

- Yahoo Finance via yfinance: SPY, QQQ, HYG, IEF, ^VIX, ^VVIX, ^VXN, DX-Y.NYB (plus XLU and ^MOVE
  when a barometer candidate needs them).
- Cboe daily history CSVs: DSPX, and COR1M when needed.
- `barometer.json`: the live barometer inputs, selection date and evidence.
