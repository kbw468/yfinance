# RA21: what precedes a superior 21-session risk-adjusted trade

No Gaussian statistics anywhere: no standard deviation, variance, z-score, Sharpe, Sortino, OLS, correlation, skew or
square-root-of-time scaling. Building blocks are returns, medians, empirical percentiles, drawdown paths, counts and shares,
L1 capture beta, trees, Mann-Whitney AUC and binned isotonic maps.

## The target (outcomes.py)
Signal at the close of t. Buy the open of t+1. Stop 8% below entry; a gap through the stop fills at that open. Otherwise
exit at the close of t+21. SPY is bought at the same open and sold at the same exit close.

* `xs_21`: trade return minus SPY's.
* `dd_21`: deepest drawdown of the position while held, on closes, with the entry as the first peak and the stop fill on a stop day.
* `mar_21 = xs_21 / max(|dd_21|, 2%)`: excess return per unit of drawdown suffered.
* **Superior** (`sup_21`): `mar_21` in the top 20% of the universe for that entry date, and `xs_21 > 0`.

The base rate is 20%. The target is flat across volatility: 19.8–20.2% in every typical-move quintile and 19.1–22.1% across
L1-beta quintiles, so low-volatility names cannot meet it by default. A trade counts only once its whole window has printed.
Otherwise the newest sessions would hold only the trades already stopped out.

Universe: S&P 500 + 400. Eligible names only: price at least $5, $10M median dollar volume, 252 sessions of history.
Sample: 2015 to date, 2.28M name-days, with the COVID window (Feb–Jun 2020) removed.

## Characteristics searched (174: 164 per name, 10 market state)

| Family | Features | Source |
|---|---|---|
| Base (steady/features.py, 105) | rates of change; excess vs SPY and sector; distance to highs and lows; drawdown paths; path efficiency; up-day, up-pair and new-high shares; gain-to-pain; higher lows; close location; overnight share; median-move volatility and its rates of change; median-volume rates of change; up-volume share; pullback volume; accumulation minus distribution; dollar volume; up/down capture; L1 beta; RS line; own-year percentiles of 30 of these | |
| Events | the largest volume shock in 63 sessions: size, age, the day's excess return and gap, whether price held it | extra.py |
| Earnings timing | report due from the quarterly cadence of volume spikes | extra.py |
| Seasonality | the same calendar window in prior years: median excess, share positive, risk-adjusted rank, share superior | extra.py |
| Industry | leave-one-out peers' 5/21/63-session return and breadth; peer gap | extra.py |
| Lottery and tails | high-volume premium inputs; largest and five largest daily gains; worst day; 95th vs 5th percentile tail ratio | extra.py |
| Momentum and path | 12-1 momentum and its information discreteness; excess-path efficiency vs SPY and sector; trailing excess per drawdown; leadership persistence; rank momentum | extra.py |
| Intraday structure | overnight vs intraday return; range position; bounce off lows; streaks; sign persistence; share price | extra.py |
| Own track record | each name's own record of superior trades, stop-outs and in-trade drawdowns; industry peers' recent superior rate | extra2.py |
| Short interest | FINRA days to cover and its changes | shortint.py |
| Market state | SPY vs its high; SPY 21/63-session return; VIX and MOVE vs their own year; breadth; cross-sectional spread | extra.py |

## Results
`results/ra21/`: characteristics_univariate.csv, characteristics_by_year.csv, characteristics_by_market_state.csv,
transparent_score.txt, rules_confirmed.csv, model_walkforward.txt, model_variants.txt, books.txt, profile_*.csv,
screen_latest.csv, screen_record.csv, today.csv.

1. **Single characteristics are weak.** The best per-date AUC is 0.517, where 0.5 means no information. Four hold in at
   least 10 of 12 years and inside beta and volatility terciles:
   * the name's own 3-year record of superior trades
   * a report due by volume cadence
   * same-calendar-window superior share
   * leadership persistence
2. **Combining them adds little.**
   * A transparent score, chosen on 2015–2020 only and tested on 2021 onward, reaches AUC 0.514. Its top decile is 21.5% superior.
   * Walk-forward trees on all features reach 0.51. The shuffled-label control is 0.50.
   * Short interest scores 0.496–0.507.
3. **Most multi-factor rules fail out of time.** 58 rules were discovered in 2015–2020 and confirmed in 2021–2023. In 2024
   onward their median lift is 0.65, and only 17% beat the base rate.
   * The large-sample confirmed rules are the 10 with at least 800 confirmation cases. Ranked by confirmation lift, the top
     three are all the same family. All three held in the untouched hold-out, with lifts of 1.80, 1.64 and 1.73.
   * The other large-sample rules all lost in the hold-out (0.59–0.70). Most are a pullback after a spike; one held at 1.24.
4. **The V-recovered leader** (screen.py) has three legs:
   * a deep drawdown inside the last 6 months, bottom 20%
   * a smooth climb over the last 3 months, top 20% for least depth below its running high
   * leadership: 6-month return vs sector, RS line at its 52-week high, or 1-year up-capture in the top 20%

   It has been superior on 30.2% of 21-session trades vs 20% for all names. The rate is 31.5% in 2015–2020, 26.7% in
   2021–2023 and 31.3% from 2024. Lift is above 1 in 10 of 12 years; it failed in 2022 and in 2026 so far. Matched
   on L1-beta or volatility quintile on the same date, the comparison rate is 20.9%, so the effect is not beta. Tighter
   cuts give higher rates: 37% at 10%, 34% at 15%, 30% at 20%, 25% at 25%. With SPY within 5% of its high it is 31.5%;
   with SPY more than 5% below, 19% (2021 onward).
5. **Sector refinement.** Energy, Basic Materials and Utilities fail in every window. There the setup is superior 16% of
   the time in 2015–2020, 10% in 2021–2023 and 2% from 2024, vs 33%, 31% and 33% elsewhere. The failure was already
   visible in the discovery window. Excluding those sectors:

   | | Superior 21d | 42d | 63d | Median excess 21d | Mean excess 21d | Stopped | Median L1 beta |
   |---|---|---|---|---|---|---|---|
   | V-recovered leader, ex commodity/utility | 32.4% | 30.3% | 28.4% | +0.9% | +2.7% | 32% | 1.49 |
   | Same universe | 20.0% | 20.0% | 20.0% | −0.8% | 0.0% | 28% | |

   It is 5,974 completed trades on 284 names over 1,873 sessions. Lift is above 1 in 10 of 12 years: 2016 2.26, 2024 2.07, 2022 0.84, 2026 so far 0.91.
6. **Books.** Rules: next open, 8% stop, 21 sessions, 10 bp round trip, 21 staggered sleeves, daily marks. MAR is CAGR / |deepest drawdown|.

   | Period | V-recovered ex commodity/utility | SPY held | Every eligible name | Model top 20 |
   |---|---|---|---|---|
   | 2017–2019, in sample | 12.9% / −15.6% / 0.83 | 14.8% / −19.3% / 0.77 | 11.1% / −15.8% / 0.71 | 17.1% / −10.8% / 1.59 |
   | 2021–2023, confirmation | 16.4% / −16.9% / 0.97 | 10.5% / −24.5% / 0.43 | 6.1% / −19.2% / 0.32 | 5.2% / −15.5% / 0.34 |
   | 2024 on, hold-out | 43.5% / −18.0% / 2.41 | 21.2% / −18.8% / 1.13 | 9.8% / −14.6% / 0.67 | 10.4% / −7.3% / 1.43 |

   * The setup book holds a median of 2 names when it fires. It has a name on 59–69% of rebalance days.
   * The model book reaches its MAR through low beta, with a median L1 beta of 0.49 from 2024. That is the low-beta
     answer, not alpha.
7. **Other learning objectives change nothing** (model_variants.txt, stock features, 2017+ walk-forward). Each objective's
   mean per-date AUC: median regression on the trade's MAR percentile 0.497; LambdaRank inside each date 0.507;
   trained on the 42-session label 0.510. The top decile is 20.6–21.5% superior.
8. **Gradient** (V-recovered ex commodity/utility, all three legs at the same cut): 39% superior at a 10% cut, 36% at 15%,
   32% at 20%, 27% at 25%, 25% at 30%.
9. **Model ranking.** Tonight's model probabilities span 16–24% (today.csv). That is too narrow to rank on.

## Commands
```
python -m pvv_score.steady.build          # base features (trailing only; outcome names can no longer collide)
python -m pvv_score.ra21.build            # trade outcomes + extra features -> CACHE/ra21/table.parquet
python -m pvv_score.ra21.extra2; python -m pvv_score.ra21.shortint
python -m pvv_score.ra21.characterize     # single characteristics
python -m pvv_score.ra21.transparent      # characteristics chosen on 2015-2020, tested 2021+
python -m pvv_score.ra21.rules            # conjunctions: discovery / confirmation / hold-out
python -m pvv_score.ra21.model            # walk-forward trees + shuffled control
python -m pvv_score.ra21.books            # portfolio tests
python -m pvv_score.ra21.screen           # tonight's V-recovered leaders, near misses, yearly record
```
