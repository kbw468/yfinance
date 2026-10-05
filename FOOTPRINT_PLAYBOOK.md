# Price / Volume / Volatility Footprint Playbook

Built from 12 case studies. Constraints: market cap >= $5B, 1-year return < +100% at entry.
Everything below was tested on a 1,300-name US universe across 2024-26 (28 monthly dates), 2018-20 (20 dates) and 2020-22 (28 dates), plus 195 ETFs 2017-22.
Files: `footprint_screen.py` (rerun any date), `footprint_*.csv` (latest output).

## Case library

| Case | Window | Move | Type | Caught at T0? |
|---|---|---|---|---|
| OKTA | May 2026 -> | +188% | Octane turn | B 80 (month +1) |
| CRWD | May 2026 -> | +139% | Octane turn | B 93 |
| TXG | Oct 2025 -> | +752% | Octane turn | B 96-100 (was $1.5B at ignition) |
| NEM | Jan 2025 -> | +265% | Octane turn -> steady | B 87 at month +1; STEADY rose 41 -> 83 during run |
| NVDA | Jun 2019 -> Feb 2020 | +136% | Octane turn | B 77-99 for 20 straight months incl. the crash |
| VLO | Sep 2025 -> | +178% | Octane turn | B 86 |
| HII | May 2025 -> Feb 2026 | +99% | Octane / earnings | B 79 |
| VIK | May 2025 -> Jun 2026 | +156% | Octane turn | B 50 (thin history) |
| TD | May 2025 -> Feb 2026 | +58% | Steady leader | STEADY 97 |
| CSX | May 2025 -> Jul 2026 | +93% | Steady leader | STEADY 65 |
| GWRE | Jun 2024 -> Feb 2025 | +99% | Steady leader / earnings gap | B 52, STEADY 53 (missed) |
| DBA | Jun 2020 -> May 2022 | +74% | Low-vol decoupled | STEADY 73-87 pre-low, 91-100 during |

## The three shapes

**Octane turn.** Laggard on the year (RS12 -20% to -50%), 20-55% below the 2-year high, relative vol 2-4x SPY, correlation to SPY 0.1-0.4 (NVDA 2019 the exception at 0.7). The turn is visible one month after the low: close in the top of the 20-day range, price above the anchored VWAP from the low, up-volume over down-volume, a +3% day within the last two weeks. Run: 55-60% up days, Sharpe 2.3-3.5, path drawdowns 15-20% that recover in 3-8 sessions, bought intraday (overnight share 7-35%).

**Steady leader.** At or within 12% of highs, flat-to-up on the year, low absolute vol, beta < 1. Run: 59-60% up days, 8 of 9 months positive, Sharpe 2.8-3.7, 70-100% of sessions within 5% of the high, max drawdown 4-19%, earnings contribute ~10 points, dips close in 1-3 weeks. The "no dip to buy" trait is real but appears WITH the run, not before it.

**Low-vol decoupled (DBA).** Relative vol < 0.5x SPY, beta ~0, multi-year decline, then 2% months stacked for two years with a worst month of -2%. The steady composite catches it. A dedicated rule has no backtested edge in ETFs (median +5% vs +4% base). Watchlist only.

## What predicts the big move (universe-validated)

Top decile of composite B, 250-day forward:

| Era | P(+50%) top decile | Base | P(-30%) top decile | Base |
|---|---|---|---|---|
| 2024-26 (22 dates) | 33% | 17% | 9% | 7% |
| 2018-20 (20 dates) | 17% | 9% | 19% | 11% |
| 2020-22 (28 dates) | 32% | 18% | 18% | 9% |

ENERGY alone (relative vol + beta + recent +3% day) has the highest upside hit rate (42% in 2024-26) and the same downside. Adding the $2-5B tier raises the downside tail by ~4 points (survivorship).

**Regime rule.** Dates 3-6 weeks after an index correction low: top-decile hit rate 55%, median +71%. All other dates: 36%, +26%. In late 2021-22 the top decile lost 30%+ 40-60% of the time. This is a post-correction tool.

## What predicts the steady move

STEADY composite top decile, 2024-25, 21 dates: P(+30% with path DD < 15%) 11.8% vs 4.9% base; bottom decile 0.3%. Median path drawdown -17% vs -28%; P(down 20%+) 7% vs 16%. In 2020-22 the hit-rate edge did not replicate; the drawdown benefit did.

## Near-term (60-day) within early-stage names

Only the ratio of +3% days to -3% days over 60 sessions separated winners: top third 34% vs bottom third 27% for +20% in 60 days. Modest. Footprint strength tells you who can run, not when.

## Things that looked like signal in the cases and failed in the universe

Accumulation/distribution line, up/down volume ratio, volume dry-up, Bollinger compression, quiet 52-week-high breakouts (loud ones win), the 13-rule binary screen, 3-month persistence of the near-term score, the DBA-type ETF rule, the steady composite inside ETFs, any predictor of forward Sharpe at the universe level.

## In-run confirmation (not prediction)

First-hour cumulative return flips from negative pre-ignition to the largest positive block in the run. Share of days closing above session VWAP rises. 60 sessions with 70%+ of days within 3% of the running high and no 5% dip: forward path drawdown -20% vs -28%, P(down 20%+) 9% vs 16%, median return unchanged.

## How to use

1. After an index correction low, run `footprint_screen.py` 3-6 weeks later. Bucket 1 is the list.
2. Any time: Bucket 2 for the steady type. Expect financials, energy, utilities, pipelines.
3. Watch the TURN flag on Bucket 3 names; when it flips, treat as Bucket 1 with lower vol.
4. Once in a run, add on strength. Dips in both shapes close in days.
