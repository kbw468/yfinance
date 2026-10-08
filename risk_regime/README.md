# Risk-On / Risk-Off Onset System: ROC-based read of 13 vol & rate indices vs 38 ETFs

Backtest window: 1990-01 to 2026-10-07 (series-dependent start), **Feb 1 2020 to Jul 31 2020 excluded** from every statistic (any observation whose date or forward window touches that range is dropped). All signals are **rates of change** normalized to each series' own trailing 252-day distribution ("ROC z"). Levels are used only as conditioning context.

Turning points were defined objectively on SPY so signals could be scored against cycle timing rather than against average days:

- **Risk-off onset (36 events):** SPY at a 63-day high, followed by a drop of 7% or more within 42 days. Anchor = peak day.
- **Risk-on onset (40 events):** SPY trough after a 7%+ drawdown, followed by +7% within 42 days with no lower low. Anchor = trough day.

Raw output for every test is in `results/`. Scripts in `scripts/` regenerate everything from Yahoo via this yfinance fork.

---

## 1. Bottom line

**Nothing here leads a top by itself.** Vol indices are contemporaneous with SPY (same-day correlation -0.58 to -0.80; one-day-ahead correlation of their changes with SPY is zero everywhere, including MOVE, VVIX, GVZ, SKEW). The edge is not in a single index moving. It is in **which dimensions move together, which refuse to, and what the ROC of one index is doing relative to the ROC of another at a specific point in the cycle.**

What actually discriminates (base rate of a 5%+ SPY drawdown in the next 21 days is 17.4%; base mean 21-day forward return is +0.86%):

| Configuration (all ROC z-scores) | Fires | P(5% DD in 21d) | Fwd 21d | Reads as |
|---|---|---|---|---|
| **SETUP: TNX 5d chg z > 1 while VIX 5d ROC z < -0.3** | 288 | 0.25 | +0.32% | Yields ripping, equity vol asleep. Fires in 56% of tops, median 18 days before the -5% break. The longest-lead configuration found. |
| **ONSET: VIX 5d ROC z > 1 while VVIX/VIX 5d ROC z < -1** | 180 | 0.27 | +0.78% | VIX spike that vol-of-vol does not confirm. Covers 47% of tops, median 11 days before the -5% break. |
| ONSET: VIX 21d ROC z < -0.3 and VIX 5d ROC z > 1 | 62 | 0.27 | +0.75% | First impulse after a month of compression. Fwd 5d and 10d are negative. |
| ONSET: above, and MOVE 5d ROC z < 0 | 14 | 0.43 | -0.46% | Equity vol breaking out while bond vol compresses. Rare, highest conviction. |
| ONSET: VIX 5d ROC z > 0.3 while OVX 5d ROC z < -1 | 39 | 0.31 | -0.15% | Equity vol up, oil vol down. |
| CONTINUATION: VIX 3d ROC z > 1 and VIX/VIX3M 3d ROC z > 1 with SPY already 7%+ off high | 64 | 0.39 | +1.07% | Second leg. Highest drawdown probability of any rule. |
| **FAILURE: VIX 10d ROC z < -1 and TNX 5d chg z > 0.5 with SPY 5%+ off high** | 56 | 0.39 | **-1.04%** | Bear-market rally tell. Vol fading while yields rise inside a drawdown. The only rule with negative 21d and 42d forward returns. |
| **CAPITULATION: VIX, GVZ 5d ROC z > 1, MOVE 5d ROC z > 0.5, TNX 5d chg z < -0.5** | 52 | 0.15 | **+2.50%** (63d: +4.22%) | Every dimension moving at once. Fires at median day -1 vs the trough. 77% of fires positive at 21d. |
| CAPITULATION: VIX and GVZ 5d ROC z both > 1 | 116 | 0.17 | +2.07% (63d: +4.22%) | Gold vol confirming equity vol. Covers 38% of troughs at median day -2. |
| CAPITULATION: VIX 5d ROC z > 1.5 with VVIX/VIX 5d ROC z > 0 | 9 | 0.00 | +3.99% (63d: +7.60%) | VVIX outrunning VIX in a spike. 9 for 9. |
| ON-CONFIRM: VIX 5d ROC z < -1, VVIX/VIX 5d ROC z > 0.3, VIX/VIX3M 5d ROC z < -0.5 | 224 | 0.19 | +1.16% | Confirms the flip at median day +4 after the trough. Confirmation, not edge. |

Benign configurations that look scary and are not:

| Configuration | Fires | P(5% DD) | Fwd 21d |
|---|---|---|---|
| VIX 5d ROC z > 1 **with** VVIX/VIX 5d ROC z > -0.3 (VVIX confirming) | 58 | 0.155 | +1.40% |
| MOVE 5d ROC z > 1 while VIX 5d ROC z < 0.3 (bond vol alone) | 177 | 0.113 | +0.99% |
| GVZ 5d ROC z > 1 while VIX 5d ROC z < 0.3 (gold vol alone) | 122 | 0.115 | +1.48% |
| OVX 5d ROC z > 1 while VIX 5d ROC z < 0.3 (oil vol alone) | 118 | 0.186 | +1.10% |

Coverage: 92% of the 36 risk-off onsets had at least one SETUP or ONSET rule fire before the -5% break; 72% had a fire at or before the peak day itself. See `results/scorecard_out.txt` for the per-event firing days.

---

## 2. Verdict on each index (ROC basis)

| Index | Verdict | What its ROC is for |
|---|---|---|
| **^VIX** | Core, but as the *reference* ROC, never alone | Every rule is "VIX ROC relative to X ROC". VIX 21d ROC z < -0.3 defines the compressed state tops come out of. VIX ROC z leads nothing on its own (1-day-ahead corr 0.03). |
| **^VVIX** | **Highest value.** The discriminator of a VIX spike | VVIX/VIX ROC spread decides whether a VIX spike is an onset (VVIX lagging, P(DD)=0.27) or a dip (VVIX confirming/outrunning, P(DD)=0.155, fwd21 +1.4% to +4%). In the risk-off peak study, VVIX/VIX 5d ROC z sits at +0.4 to +0.7 in the 3 days into the top, then flips to -0.8 to -1.4 through the break. At troughs it bottoms at -1.33 on the low day and flips to +0.87 by day +4: the cleanest risk-on confirm in the set. VVIX is the one series with a Granger-significant lag-2 effect on SPY (p=0.03). |
| **^MOVE** | High value, as a *qualifier* and *dimension tag*, zero value as a leader | MOVE spiking alone is benign for SPY (P(DD)=0.11). MOVE ROC z compresses to -0.4/-0.6 into equity tops and only turns up at day +2 to +6. Its job: (a) when VIX breaks out and MOVE does not confirm, odds of continuation go to 0.43; (b) when it confirms along with GVZ and yields falling, you are at a low (capitulation rule); (c) it tags the event as rate-led, which sets the ticker playbook (TLT, IEF, XHB, XLRE, XLP underperform in a rate-led onset; they outperform in a flight-to-quality onset). |
| **^TNX / ^TYX** | High value, as the SETUP dimension | TNX 5d chg z > 1 with VIX compressing is the longest-lead configuration (median 18d to the -5% break, 56% coverage). TYX ROC fires at or before the peak in 85% of tops, TNX in 76%; both fire before the trough too (yields lead price at both ends). Inside a drawdown, yields turning up while VIX fades is the rally-failure tell (fwd21 -1.04%). TYX and TNX ROCs are near-identical; TYX is marginally earlier. |
| **^GVZ** | Useful at the *bottom*, benign at the top | GVZ 5d ROC z peaks at +1.2 the day before troughs (gold getting liquidated). VIX+GVZ ROC both > 1 is the broadest capitulation detector (116 fires, fwd63 +4.2%). GVZ alone spiking means nothing for SPY. GVZ ROC z compresses to -0.6 three days before tops. |
| **^OVX** | Dimension tag and a divergence tell | OVX 5d ROC z fires early in 56% of tops (fired-by excess +0.2 at day -2) but on its own P(DD) is at base. VIX up while OVX collapsing (-1 z) is a 0.31 onset rule. OVX spike with VIX quiet = energy-specific, buy the rest. |
| **^VIX9D** | Fastest *confirmer*, weak leader | VIX9D ROC fires by day +4 in 55% more events than baseline (best post-peak confirmer in the set) and VIX9D/VIX ROC z is +0.3 above baseline in the 7 days before tops (n=13 only, 2011+). Front-end panic (VIX9D/VIX ROC z > 1 with VIX spiking) marks lows: fwd63 +4.3%. |
| **^VIX3M / ^VIX6M** | Term-structure denominator | VIX/VIX3M 3d ROC z > 1 together with VIX 3d ROC z > 1 is the confirmation of a break (fires day 0 to +2 at 92% of break days) and, once SPY is 7%+ down, the continuation rule (P=0.39). VIX6M ROC alone has no onset value; it is the slow leg. |
| **^VXN** | Redundant with VIX for SPY | Corr of daily ROC with VIX ROC is 0.89. VXN ROC z is slightly more extreme at tops (+1.0 at day +5 vs VIX +0.74), reflecting QQQ beta. Use VIX; watch VXN/VIX ROC only for a Nasdaq-specific read. |
| **^VIX1D** | Noise, with one exception | 868 observations. Quintiles of VIX1D ROC against forward returns are non-monotonic. The 0DTE panic flag (VIX1D 3d ROC z > 2 with VIX 3d ROC z > 1) fired 11 times, 10 positive at 21d, fwd63 +7.8%. Too few observations to weight; keep it as a capitulation accelerant only. |
| **^SKEW** | **No.** | SKEW 10d ROC z runs +0.4 to +0.6 in the 10 days before tops, then goes negative after. That looks like a lead, but SKEW ROC rising while VIX ROC is quiet fires 368 times with P(DD)=0.166, which is *below* base. It rises before tops and before nothing in equal measure. Confirmed as the weakest series in the set. Only use: SKEW 5d ROC turning positive 3-5 days after a trough is a mild risk-on confirm. |
| **^VIX1Y** | Unavailable | Yahoo serves one row. Dropped. |

---

## 3. The ROC tape at turning points

### 3a. Into a risk-off peak (36 events, median ROC5 z, offsets vs peak day)

| Offset | -10 | -5 | -3 | -1 | 0 | +1 | +2 | +3 | +5 |
|---|---|---|---|---|---|---|---|---|---|
| VIX | -0.21 | -0.18 | -0.33 | -0.24 | -0.20 | 0.05 | 0.43 | 0.46 | 0.74 |
| VVIX/VIX | 0.70 | 0.01 | 0.39 | 0.71 | 0.48 | 0.16 | -0.45 | -0.44 | -0.82 |
| MOVE | 0.15 | -0.32 | -0.42 | -0.40 | -0.55 | -0.23 | 0.14 | 0.11 | 0.31 |
| MOVE/VIX | 0.11 | -0.07 | 0.22 | 0.09 | -0.10 | -0.18 | -0.22 | -0.48 | -0.52 |
| TNX (chg z) | 0.25 | 0.18 | -0.05 | -0.07 | -0.13 | 0.06 | 0.09 | 0.28 | 0.26 |
| TYX (chg z) | 0.17 | 0.20 | 0.02 | 0.01 | -0.15 | -0.06 | 0.02 | 0.22 | 0.45 |
| SKEW (ROC10 z) | 0.46 | 0.39 | 0.08 | 0.10 | 0.09 | -0.01 | 0.04 | -0.23 | -0.16 |
| GVZ | -0.30 | -0.18 | -0.61 | -0.06 | -0.26 | 0.01 | 0.20 | 0.10 | 0.30 |
| OVX | 0.03 | -0.04 | -0.22 | -0.09 | 0.17 | 0.23 | 0.36 | 0.49 | 0.48 |
| VIX/VIX3M | 0.04 | -0.06 | -0.36 | -0.30 | -0.30 | -0.17 | 0.43 | 0.42 | 0.57 |

Read: tops form with **VIX, MOVE and GVZ ROC all compressing** and **VVIX/VIX ROC, SKEW ROC and yields ROC rising**. The vol complex is bid in its second derivative (VVIX) while the first derivative (VIX) is still falling. The break shows up as VIX/VIX3M ROC and VIX ROC turning positive together at day +1/+2, with VVIX/VIX collapsing (VVIX did not keep up with the spike). MOVE turns up last.

### 3b. Order of first fire (ROC3 z > 1, window -10 to +10, median day vs peak)

TYX -6, SKEW -5, TNX -5, OVX -4.5, MOVE -4, VVIX -3.5, VIX9D -3, GVZ -3, VXN 0, VIX6M +1, **VIX +1.5**, VIX3M +2.

Share firing at or before the peak: TYX 85%, TNX 76%, VVIX 72%, SKEW 72%, VIX9D 67%, MOVE 63%, GVZ 62%, OVX 56%, VXN 50%, **VIX 41%**.

Rates and vol-of-vol tick first. VIX itself is one of the last to move. SKEW is early but, as above, it is early everywhere.

### 3c. Into a risk-on trough (40 events, median ROC5 z, offsets vs trough)

| Offset | -5 | -3 | -1 | 0 | +1 | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|---|---|---|---|
| VIX | 0.01 | 0.48 | 0.83 | **1.39** | 0.49 | -0.18 | -0.73 | -1.16 | -1.35 |
| VVIX/VIX | -0.36 | -0.75 | -0.92 | **-1.33** | -0.51 | -0.19 | 0.35 | **0.87** | 1.10 |
| MOVE | 0.12 | 0.67 | **1.14** | 0.92 | 0.49 | 0.11 | -0.05 | -0.44 | -0.27 |
| MOVE/VIX | -0.19 | -0.28 | -0.58 | -0.73 | -0.19 | 0.10 | 0.65 | **1.22** | 1.11 |
| GVZ | -0.10 | 0.35 | **1.20** | 1.11 | 0.91 | 0.04 | -0.35 | -0.78 | -1.02 |
| TNX (chg z) | 0.04 | -0.39 | -0.52 | **-0.64** | -0.44 | -0.06 | 0.12 | 0.16 | 0.37 |
| SKEW | -0.12 | -0.16 | -0.46 | **-0.70** | -0.18 | -0.08 | 0.39 | 0.50 | 0.68 |
| VIX/VIX3M | 0.31 | 0.43 | 0.75 | 0.83 | 0.13 | -0.26 | -0.68 | -1.02 | -1.28 |

Read: the low is the day **everything** peaks together: VIX ROC, GVZ ROC (day -1), MOVE ROC (day -1), with yields ROC at their most negative and SKEW ROC at its most negative. That simultaneity *is* the capitulation rule. The confirm is VVIX/VIX ROC flipping from -1.3 to +0.9 within four sessions while MOVE/VIX ROC does the same: vol-of-vol and bond vol both stop falling slower than VIX. Yields turning up (TNX chg z +0.37 at day +5) is normal *after* a real low; yields turning up while VIX is fading *before* a real low is the failure rule.

---

## 4. Cross-dimension interaction grids (P of 5%+ SPY drawdown in 21d; base 0.17)

VIX 5d ROC z (rows) by MOVE 5d ROC z (columns):

| | MOVE < -1 | -1..-0.3 | -0.3..0.3 | 0.3..1 | > 1 |
|---|---|---|---|---|---|
| VIX > 1 | **0.31** | 0.20 | 0.20 | 0.17 | 0.22 |
| VIX 0.3..1 | 0.22 | 0.22 | 0.15 | 0.14 | 0.17 |
| VIX < -1 | 0.14 | 0.14 | 0.21 | 0.09 | 0.10 |

VIX 5d ROC z by VVIX/VIX 5d ROC z:

| | VVIX/VIX < -1 | -1..-0.3 | -0.3..0.3 | 0.3..1 | > 1 |
|---|---|---|---|---|---|
| VIX > 1 | **0.27** | 0.17 | 0.15 | 0.00 (n=18) | 0.00 (n=4) |
| VIX < -1 | 0.00 (n=3) | 0.11 | 0.22 | 0.16 | 0.16 |

Mean fwd21 in the VIX>1 row: -1 col +0.85%, 0.3..1 col **+3.30%**, >1 col **+5.64%**.

VIX 5d ROC z by TNX 5d chg z:

| | TNX < -1 | -1..-0.3 | -0.3..0.3 | 0.3..1 | > 1 |
|---|---|---|---|---|---|
| VIX > 1 | 0.20 (fwd21 +1.91%) | 0.24 | 0.22 | 0.21 | 0.15 (fwd21 +1.35%) |
| VIX -1..-0.3 | 0.15 | 0.16 | 0.20 | 0.17 | **0.24 (fwd21 +0.22%)** |
| VIX -0.3..0.3 | 0.16 | 0.18 | 0.17 | 0.14 | 0.16 (fwd21 +0.18%) |

VIX spiking with yields collapsing is a flight-to-quality dip (fwd21 +1.9%). Yields ripping with VIX asleep is the setup.

MOVE 5d ROC z by TNX 5d chg z: MOVE < 0 and TNX > 1 → P 0.21 to 0.26, fwd21 -0.33% to +0.19%. MOVE > 1 and TNX < -1 → fwd21 **+1.85%**. Bond vol confirming a yield collapse is a low; yields rising without bond vol is the slow bleed.

VIX 21d ROC z (trend) by VIX 5d ROC z (impulse): compression for a month followed by a first impulse (21d -1..-0.3, 5d > 1) → **P 0.32**, the single highest cell. Both deeply negative (both < -1) → P 0.25, fwd21 +0.25%: complacency. Both > 1 → 0.21 and fwd21 +1.56%: already-in-stress, mean-reverting.

Full grids including GVZ, OVX, SKEW, VIX9D in `results/interact_out.txt`.

---

## 5. Dimension of an onset and the ticker playbook

Each of the 36 risk-off onsets was tagged by which dimensions moved (max ROC10 z over the 10 days after the peak): RATEVOL (MOVE), YIELDS_UP (TNX), FTQ (TNX down), OIL (OVX), GOLDVOL (GVZ). Only one event (Jul 1998) was VOL_ONLY. 27 of 36 contained a rates dimension. The dimension sets the ticker response.

### Median 10-day return relative to SPY after the peak, by dimension

| | Rate-led (n=27) | Flight-to-quality (n=15) | Oil (n=9) |
|---|---|---|---|
| **TLT** | +3.8 | **+5.6** | +3.8 |
| **IEF** | +3.5 | +4.5 | +3.5 |
| XLU | +2.3 | +2.4 | +2.4 |
| XLP | +1.9 | +2.6 | +2.6 |
| GLD | +1.8 | +1.9 | +1.6 |
| XLV | +1.2 | +1.4 | +1.4 |
| KRE / KBE | -0.7 / -0.2 | **-1.2 / -1.0** | **+2.6 / +3.4** |
| XHB | **-2.3** | -0.1 | 0.0 |
| XSD | -2.1 | -1.3 | -2.1 |
| XME | **-3.1** | -4.4 | -3.7 |
| XOP / XES | -0.6 / +0.3 | **-4.5 / -3.4** | -2.8 / -4.5 |
| BNO | +2.6 | -3.1 | -1.4 |

Rate-led onsets hit housing, semis, metals; banks hold. Flight-to-quality onsets hit banks and energy hardest and TLT is the best hedge in the set. Oil onsets: banks outperform, energy complex is the epicenter, TLT still works.

### After a CAPITULATION fire (VIX+GVZ+MOVE up, yields down), median 10d relative to SPY

QQQ +0.44, XLK +0.29, XAR +0.25, XLY +0.20 | TLT **-2.32**, IEF -2.27, XES -2.35, XOP -1.83, XME -1.75, GLD -1.27.

Rotate out of the hedges the day the capitulation flag prints. Duration gives back 2 to 3% relative within 10 sessions.

### After the FAILURE tell (vol fading, yields up, inside a drawdown), median 10d relative to SPY

XES +2.4, BNO +2.3, XOP +2.2, XLE +2.1, XLU +1.9, XLRE +1.7, XHS +1.7, TLT +1.3, XLP +1.2 | **XHB -2.3**, XPH -1.5, KBE -1.3, KRE -1.1, XLF -0.8, QQQ -0.4.

### Tickers that lead price at the turn (ROC10 of ticker/SPY at the anchor day, median)

At peaks: IEF -3.3%, TLT -2.7%, HYG -2.2%, XLP -1.8% (defensives and duration were already underperforming into the high; the bond market was selling off into the equity top). XME +2.0%, XBI +1.6% (speculative beta leading into the top).

At troughs: TLT +8.0%, IEF +6.6%, GLD +6.1%, HYG +3.4%, XLP +2.8% (85% of troughs). XES -4.2%, XOP -3.7%, XME -2.8%, XSD -2.1%, KBE -1.6% (the sectors to buy at the capitulation print; see above).

### Regime-conditional behavior across the full ETF list

`results/tickers_out.txt` has the full table for all 38 tickers at 5/10/21 days after each anchor, with hit rates. Short version of the 21-day relative returns from the peak: TLT +5.2, IEF +3.9, XLP +3.2, XLU +3.0, GLD +2.3, XLV +2.2 at one end; XME -4.0, XOP -3.3, KCE -3.1, XSD -2.9, XHB -2.4, XTN -2.3 at the other. From the trough: XSD +4.3, XME +2.8, KCE +2.5, XSW +2.0, XLF +1.9, XHB +2.0 lead; TLT -10.5, IEF -8.1, GLD -7.1, BNO -5.2, HYG -4.0, XLP -3.3 lag.

---

## 6. Robustness, stated plainly

Sub-period check (P of 5% DD, base in brackets): the SETUP rule holds 1990-2007 (0.25 vs 0.17) and 2008-2015 (0.31 vs 0.22), and is flat 2016-2026 (0.16 vs 0.15). The VVIX-lag onset rule holds in both periods it exists (0.32 vs 0.22; 0.22 vs 0.15). The capitulation rules get *stronger* post-2016 (fwd21 +3.5% vs base +1.25%). The failure tell holds in all three periods with negative fwd21 in each. MOVE-alone is benign in all three.

Threshold sensitivity: moving the SETUP thresholds between 0.7 and 1.3 moves P between 0.21 and 0.25, never to base. The VVIX-lag rule moves between 0.23 and 0.29 across nine threshold pairs. These are plateaus, not spikes.

What did not work, so you do not have to re-test it: a binarized "pre-top setup score" summing the lifts (VVIX/VIX up, SKEW up, yields up, MOVE wake, VIX9D lead, commodity vol) does not separate the 36 peaks from ordinary near-high days (mean 2.5 vs 2.4). The medians in section 3a are real but faint; they only become usable as *pairs* (VIX vs VVIX, VIX vs TNX, VIX vs MOVE). A persistent OFF/ON state machine keyed off VIX spikes labels post-spike days as OFF, and those days return 14.8% annualized because most spikes are dips. The system is event-driven by construction.

---

## 7. Current read (2026-10-07 close)

SPY 777 at -0.24% from its 63-day high. VIX 15.08 (12th pctile), VIX 5d ROC -8%, 21d ROC -4%. VIX9D/VIX 0.78 (7th pctile), VVIX 83 (2nd pctile). MOVE 102.6 at the **95th percentile** with 21d ROC +30% (z +1.56). TNX 5.28% at the 99th percentile, 21d chg +47bp (z **+2.43**); TYX 5.66%, 21d z +2.48. TNX 5d chg z is -0.46 and MOVE 5d z is -0.81, so the SETUP rule is not live today; it requires a fresh 5-day yield impulse. The rate-led tag printed on Oct 2 and Oct 5. Configuration: yields and bond vol at 1-year extremes on a 21-day basis, equity vol compressed, VVIX at its lows. That is the section 3a shape on a slower clock. The trigger to watch is TNX/TYX 5d chg z crossing +1 while VIX 5d ROC z stays below -0.3 (SETUP), and then any VIX 5d ROC z > 1 print where VVIX/VIX 5d ROC z is below -1 (ONSET). If that VIX print comes with VVIX/VIX ROC z above -0.3, it is a dip.

The dashboard (`dashboard.html`) renders this state, every rule's live/not-live status, the last two years of fires on the SPY chart, and the per-rule ticker playbook. Regenerate with `python scripts/fetch.py && python scripts/rebuild.py && python scripts/features_roc.py && python scripts/events.py && python scripts/scorecard.py && python scripts/playbook.py && python scripts/snapshot.py && python scripts/tick_roc.py && python scripts/tick_snapshot.py && python scripts/rv.py && python scripts/rv_snapshot.py && python scripts/build_dashboard.py`.

---

## 8. Ticker-side ROC pass (the 38 as signals, not just as responders)

Each ticker's log ratio to SPY was differenced at 5/10/21 days and z-scored against its own trailing 252 days ("relative ROC z"). Full output in `results/tick_roc_out.txt`.

### 8a. Who is already moving before the top (median relative ROC5 z, offsets vs the 36 peaks)

| | -5 | -3 | -1 | 0 | +3 | +5 |
|---|---|---|---|---|---|---|
| HYG | -0.25 | -0.13 | -0.31 | **-0.94** | -0.15 | +0.68 |
| IEF | -0.36 | -0.25 | -0.58 | -0.65 | +0.29 | +0.99 |
| TLT | -0.48 | -0.32 | -0.40 | -0.35 | +0.22 | +0.73 |
| XLP | -0.14 | -0.21 | -0.57 | -0.55 | +0.38 | +1.02 |
| XLU | -0.09 | -0.27 | +0.10 | -0.03 | +0.32 | +0.54 |
| IWM | +0.21 | **+0.56** | +0.23 | +0.24 | -0.20 | -0.05 |
| KBE | -0.01 | +0.54 | +0.06 | +0.12 | 0.00 | +0.16 |
| KIE | +0.19 | +0.60 | +0.14 | +0.15 | -0.15 | +0.39 |
| XLI | +0.29 | +0.59 | +0.01 | +0.02 | +0.20 | +0.23 |
| XME | +0.30 | +0.20 | -0.07 | -0.05 | **-0.77** | -0.63 |
| XSD | -0.17 | +0.12 | +0.27 | +0.54 | -0.28 | -0.38 |
| QQQ | +0.08 | +0.34 | +0.25 | +0.31 | +0.18 | -0.40 |

Tops are a beta chase: small caps, banks, insurers, industrials are winning relative into the high while duration, credit and staples are losing relative. **HYG cracks on the peak day itself** (-0.94 z), the sharpest single-ticker tell in the set. The rotation flips within three sessions.

### 8b. Troughs (median relative ROC5 z vs the 40 troughs)

| | -3 | -1 | 0 | +2 | +3 | +5 |
|---|---|---|---|---|---|---|
| IEF | +0.94 | +1.42 | **+1.88** | +0.14 | -0.29 | **-1.76** |
| TLT | +1.10 | +1.40 | +1.87 | -0.19 | -0.43 | -1.72 |
| GLD | +0.77 | +1.36 | +1.74 | -0.05 | -0.06 | -1.53 |
| XLP | +0.80 | +1.10 | +1.45 | -0.13 | -0.31 | -1.09 |
| HYG | +0.65 | +1.21 | +1.43 | -0.08 | -0.28 | -1.48 |
| XLU | +0.64 | +0.90 | +1.21 | -0.25 | -0.41 | -0.87 |
| KCE | -0.43 | -0.57 | -0.65 | +0.08 | **+0.48** | +0.38 |
| XLF | -0.16 | -0.40 | -0.67 | +0.22 | +0.19 | +0.55 |
| XSD | -0.53 | -0.34 | -0.69 | -0.03 | +0.10 | +0.41 |
| QQQ | -0.74 | -0.55 | -0.43 | -0.02 | +0.27 | +0.39 |
| XHB | -0.16 | -0.10 | -0.11 | -0.22 | +0.26 | +0.74 |

The low is the day the safety trade climaxes: duration, gold, staples, credit all print +1.2 to +1.9 relative ROC z on day 0 and are at -0.9 to -1.8 by day +5. First bounce leaders in order: KCE, XLRE, XLY, XLK, QQQ, KBE, XHB.

### 8c. Single-ticker relative ROC as a SPY drawdown predictor (P of 5% DD in 21d, bottom vs top quintile; base 0.174)

| Ticker | ROC window | Bottom Q | Top Q | Direction |
|---|---|---|---|---|
| **XLU** | 21d | 0.119 | **0.269** | utilities outperforming = risk-off ahead |
| KBE | 21d | **0.244** | 0.119 | banks underperforming = risk-off ahead |
| GLD | 10d | 0.122 | 0.230 | gold outperforming = risk-off ahead |
| XRT | 10d | 0.238 | 0.132 | retail underperforming |
| KRE | 21d | 0.252 | 0.149 | regionals underperforming |
| HYG | 21d | 0.165 | 0.243 | credit outperforming (late-cycle grab) |
| XLV | 21d | 0.165 | 0.243 | health care outperforming |

### 8d. Ticker relative ROC × index ROC pair rules (first-fires, ex-2020)

| Pair | First-fires | P(5% DD/21d) | Fwd 21d | Note |
|---|---|---|---|---|
| **HYG rel ROC5 z < -1 and VIX 5d ROC z > 1** | 7 | **0.71** | **-10.6%** | credit breaking relative while vol spikes. 7 fires, all inside major breaks (last: 2018-01-16). Standout cell in the whole study. |
| KRE rel ROC5 z < -1 and TNX 5d chg z > 1 | 35 | 0.46 | -1.8% | regionals breaking while yields rip. **Removed from the dial and the pair board on 2026-10-08 at the user's direction** (regional banks are not a 2026 macro channel). Its edge was 2007-09: 14 of 35 fires; the 2018-26 fires were P 0.27 with fwd21 +1.45%. See section 16. |
| TLT rel ROC5 z > 1 and VIX 5d ROC z < -1 | 6 | 0.50 | -0.6% | duration bid while vol collapses; TLT itself then gives back 3.4% relative in 10d |
| XLU rel ROC5 z > 1 and TNX 5d chg z > 1 | 37 | 0.38 | -0.5% | utilities bid into rising yields. Last fire 2026-07-22. |
| XLRE rel ROC5 z > 1 and TNX 5d chg z > 1 | 13 | 0.31 | +0.1% | REITs bid into rising yields |
| IWM rel ROC5 z > 1 and VIX 5d ROC z > 1 | 53 | 0.30 | -0.4% | small-cap beta chase into a vol rise |
| XLU rel ROC21 z > 1 | 197 | 0.31 | +0.2% | utilities outperforming for a month |

These are small-n cells (the full 4×4 grids with counts are in `results/tick_roc_out.txt`). They are live on the dashboard as the "pair rules" board.

### 8e. Ticker × index ROC matrix (median 10d forward return relative to SPY, index 5d ROC z > 1)

Consistent across every index spike: **XSD, QQQ, XLK, XAR, XSW outperform SPY in the 10 days after any vol-complex impulse** (+0.2 to +0.6% rel); **TLT, IEF, XES, HYG, GLD, BNO underperform** (-0.5 to -1.0% rel). Dimension-specific: XME +0.60 after a MOVE spike but -0.55 after a GVZ spike; XOP -0.77 after GVZ up but +0.27 after GVZ down; XHB -0.53 after TYX up, +0.33 after GVZ down; XLU +0.37 after MOVE down, -0.42 after VIX9D/VIX up. Full 38×26 matrix and the stress-regime ROC beta matrix are on the dashboard.

Stress-regime ROC betas (per 100% move in VIX): TLT +21, IEF +19, GLD +16.5, HYG +11, XLP +8, XLU +7 on the hedge side; XES -6.3, XME -6.1, XSD -5.3, KCE -5.1, XOP -4.8, KBE -3.3 on the beta side. Per 1 pct-pt in TYX in stress: XES +10.2, XOP +10.1, XME +8.6, KBE +7.6, KCE +6.4 vs TLT -28, IEF -19, GLD -13, XLU -6.3, XLRE -5.5.

---

## 9. Realized-vol layer: ticker realized ratios × index ROC

Three realized-vol objects per ticker, each z-scored against its own trailing 252 days: **RV10/RV21** (realized term structure), **rel RV** (ticker 21d realized ÷ SPY 21d realized, and its 5d ROC), and the **native implied/realized pairs** (VIX÷SPY RV, VXN÷QQQ RV, MOVE÷TLT and IEF RV, GVZ÷GLD RV, OVX÷XOP, BNO and XLE RV, VIX÷IWM and HYG RV). Plus realized-vol breadth (share of the 38 with RV10/RV21 above 1.2). Full output in `results/rv_out.txt`.

### 9a. What realized vol does at the turns

Into tops, SPY's own realized term structure is **compressing** (RV10/RV21 z -0.13 to -0.16 from day -5 to -1) while realized is already **expanding in the cyclical and energy names**: XLY +0.51, XLB +0.49, XES +0.46, XLE +0.41, XOP +0.40 at day -3. Realized-vol breadth is at its floor at tops (3% of tickers expanding). After the peak, realized expands everywhere, led by QQQ/XLK/XAR (+0.6 to +0.7 by day +4). The IV/RV ROC for SPY, QQQ, IWM, HYG dips negative at days -3 to -1 (realized catching up to implied into the high) and MOVE÷TLT RV is -0.4 to -0.5 (bond realized expanding faster than MOVE into equity tops).

At troughs, every IV/RV ratio ROC flips from positive on the low day to -1.0 to -1.8 by day +4/+5 (implied collapses while realized stays elevated: the premium drains). IWM VIX/RV level z peaks at +1.28 on the trough day. Realized-vol breadth peaks at 16% on days +1 to +3 after the low, three to five times its baseline, and is back to 3% by day +9. **Breadth of realized expansion confirms a low; it never leads one.**

### 9b. IV/RV pairs as predictors (P of 5% SPY drawdown in 21d by quintile of level; base 0.174)

| Pair | Q1 (implied cheap vs realized) | Q5 (implied rich) | Fwd21 Q1 / Q5 |
|---|---|---|---|
| **VIX ÷ HYG RV** | **0.306** | 0.130 | **-0.05% / +1.39%** |
| **OVX ÷ XLE RV** | **0.272** | 0.134 | +0.50% / +1.36% |
| OVX ÷ XOP RV | 0.245 | 0.158 | +0.33% / +1.30% |
| VIX ÷ SPY RV | 0.220 | 0.147 | +0.61% / +1.08% |
| VXN ÷ QQQ RV | 0.215 | 0.117 | +0.30% / +1.23% |
| OVX ÷ BNO RV | 0.105 | **0.222** | +1.42% / +0.90% (opposite sign) |
| MOVE ÷ TLT RV, ROC5 z | 0.174 | **0.202** | MOVE rising faster than bond realized = risk-off |

Equity implied cheap relative to credit realized is the strongest single realized-vol tell in the set.

### 9c. Realized × implied ROC pair rules (first-fires, ex-2020)

| Rule | First-fires | P(5% DD/21d) | Fwd21 | Last fire | Read |
|---|---|---|---|---|---|
| SPY VIX/RV ROC5 z < -1 and VIX 5d ROC z > 1 | 48 | **0.29** | +0.81% | 2026-06-11 | realized outrunning implied in the spike: continuation |
| SPY VIX/RV ROC5 z > 1 and VIX 5d ROC z > 1 | 211 | 0.20 | **+1.60%** | 2026-07-29 | implied outrunning realized: fear premium, dip |
| QQQ VXN/RV ROC5 z < -1 and VXN 5d ROC z > 1 | 41 | 0.27 | **-0.49%** | 2026-06-11 | Nasdaq realized leading implied |
| **VIX ÷ HYG RV pct252 < 20** | 110 | 0.26 | +1.00% | **2026-06-18, LIVE** | equity implied cheap vs credit realized |
| OVX ÷ XLE RV pct252 < 20 | 100 | 0.23 | +0.59% | 2026-08-28 | oil implied cheap vs energy-equity realized |
| TLT RV10/RV21 z < -1 and MOVE 5d ROC z > 1 | 45 | 0.24 | -0.19% | 2026-07-22 | bond implied spiking off compressed bond realized |
| HYG RV10/RV21 z > 1 and VIX 5d ROC z > 1 | 61 | 0.25 | +0.79% | 2026-03-24 | credit realized expanding into the spike |
| GLD GVZ/RV ROC5 z > 1 and GVZ 5d ROC z < -1 | 13 | 0.46 | -2.04% | 2022-04-07 | small n |
| XOP OVX/RV ROC5 z > 1 and OVX 5d ROC z < -1 | 11 | 0.45 | -0.27% | 2025-05-13 | small n |

Cell-level (every day, not first-fire) versions of the same configurations are in `results/rv_out.txt` with counts; KRE RV10/RV21 z > 1 with TNX 5d chg z > 1 is P 0.24 / fwd21 -0.25% there and HYG RV10/RV21 z < -1 with VIX 5d ROC z < -1 (both compressed) is P 0.28.

The realized/implied ROC spread is the realized-vol version of the VVIX finding in section 2: a spike where implied outruns realized is a dip; a spike where realized outruns implied continues.

### 9d. Per-ticker: own realized expansion × VIX ROC (median 10d return relative to SPY)

| Ticker | RV expanding, VIX up | RV expanding, VIX down | RV compressed, VIX up | RV compressed, VIX down |
|---|---|---|---|---|
| XES | **-1.74** | -0.86 | +0.33 | -0.67 |
| XME | **-1.44** | -0.29 | +0.37 | -0.12 |
| IEF | -1.31 | -0.86 | -0.84 | -0.82 |
| XHE | -1.16 | +0.52 | -0.13 | +0.46 |
| XLRE | -0.94 | +0.46 | -0.93 | -0.79 |
| XLE | -0.89 | +0.25 | **+0.49** | -0.13 |
| XTN | -0.88 | **+1.50** | -0.07 | +0.20 |
| KRE | -0.53 | **+0.87** | -0.48 | -0.44 |
| XRT | -0.20 | **+0.94** | -0.58 | +0.03 |
| XLU | +0.11 | -0.60 | **+1.03** | +0.48 |
| XAR | **+0.98** | +0.58 | -0.27 | +0.02 |
| XBI | +0.78 | -0.38 | +0.20 | **+1.29** |
| XSW | +0.29 | +0.52 | +0.40 | +0.33 |

When a ticker's own realized is expanding and VIX is rising, energy services, metals and duration are the ones to be out of. When a ticker's realized expanded into a low and VIX then collapses, transports, retail and regional banks are the rip. Utilities are the hold when VIX rises and their realized has not moved. Full 38-row table on the dashboard.

Per-ticker RV10/RV21 top-vs-bottom quintile has no predictive power for SPY drawdowns on its own (XLC 0.21 vs 0.13 and XTN 0.15 vs 0.08 are the widest gaps; most are flat). Realized vol in a single ticker tells you about that ticker's forward relative return, not about the index.


### 9e. Current realized-vol read (2026-10-07)

VIX ÷ HYG 21d realized is at the **10th percentile** (level z -1.11, 5d ROC z -0.72): credit is realizing more vol than equity implied is pricing. That rule is live, the only live rule across all three layers. MOVE ÷ TLT realized is at the 93rd percentile (level z +1.46) while IEF's is at the 47th: MOVE is rich to long-bond realized, not to the belly. SPY RV10/RV21 is below 1 with breadth at 0% of tickers expanding (z -1.02): realized is compressed across the entire list, which is the tops configuration from 9a if the cyclical/energy names start expanding first. Watch XLY, XLB, XES, XLE, XOP RV10/RV21 z crossing +0.4 while SPY's stays negative.

---

## 10. Analogs to the 2026-10-07 reading

Nearest-neighbor search of today's 29-feature state (percentile levels, 5d and 21d ROC z of every index, term structure, cross-vol ratios, SPY position, realized vol, credit IV/RV) against every day since 2008, grouped into distinct episodes; plus a reduced 14-feature search back to 1991 and coarse rule-based matches. Output in `results/analog_out.txt` and `results/coarse_out.txt`.

Weighted toward today's extremes (rates and bond vol at 1-year highs, equity vol and VVIX at lows, SPY at its high): 2021-03-15, 2016-11-28, 2026-05-22, 2015-05-20, 2022-04-20, 2022-09-12, 2021-10-22, 2023-03-03, 2022-01-12, 2010-12-14, 2018-10-03, 2013-02-01. Median forward: 63d 0.0%, 126d -0.1%, max drawdown 63d -3.2%, 126d -8.1%. Share with a 5%+ drawdown within 63 days 0.33 (base 0.31); 10%+ within 126 days 0.42 (base 0.22). The set splits cleanly: the analogs that occurred with SPY at its high and VIX near its lows (Mar 2021, Nov 2016, Feb 2013, Dec 2010) went straight up; the analogs that occurred with SPY already 3-4% off and VIX mid-range (Apr 2022, Sep 2022, Jan 2022, Oct 2018) were the start of 12-21% drawdowns within 5 to 14 sessions. Today sits with the first group on SPY position and VIX, and with the second group on rates and bond vol.

Coarse match (TNX pct >= 90, TNX 21d chg z >= 1.5, VIX pct <= 30, SPY within 3% of high, 1991+): 13 episodes, median 63d max drawdown -3.9%, 38% had a 5%+ drawdown within 63 days, 15% a 10%+ within 126 days. Adding MOVE pct >= 80 narrows to 5 episodes (Jul 2013, Nov 2016, Feb 2021, Oct 2021, Jan 2022), four benign and one (Jan 2022) the start of the 2022 bear. Adding VVIX pct <= 25 leaves 4, all benign over 63 days, with the Oct 2021 one making a 7.6% drawdown 87 days later.

---

## 11. Environment: two states, tuned to stay in

Objective restated by the user: err on the side of being in; drawdowns are secondary. Tested in `scripts/stayin.py` (output in `results/stayin_out.txt`): event-driven OUT spells of 5 to 21 sessions after the high-conviction onset configurations (with and without price confirmation, with and without early re-entry), the previous drawdown-first state book, a price-only break rule, and the balanced dial with OUT at progressively deeper readings. The event-driven versions whipsaw (7 to 10 OUT spells a year, 7 to 9% annual return, max drawdown -29 to -37%) and the price-only rule is worthless (5% a year, -70% drawdown). The balanced dial with a deep OUT threshold dominates everything tested.

**The method.** The component score with balanced weights (risk-off and risk-on fires persist 10 to 42 sessions as in section 1; no at-the-highs softener), plus one realized-vol ROC rule adopted after the section 13 tests: the VIX-minus-realized premium's 5-day change z below -1 while VIX 21-day ROC z is above 1 (realized catching up to a month of rising implied), -8 for 15 sessions. The regional-banks-vs-yields pair rule was removed on 2026-10-08 (section 16); the figures below are the current dial without it. Two states:

- **OUT** when the dial reads 20 or below. Back **IN** only when it is above 35.
- **IN** otherwise.

No middle state. About 2.3 state changes a year, 21 OUT spells in 18 years. Definitions in `scripts/score.py`.

What each state meant, 2008 to date (Feb-Jul 2020 excluded):

| State | Time share | P(5% DD in 21d) | P(10% DD in 63d) | Mean fwd 21d | Mean fwd 63d | Mean 63d max DD | Worst 5% of 63d outcomes |
|---|---|---|---|---|---|---|---|
| IN | 91% | 0.14 | 0.12 | +1.2% | +3.1% | -4.3% | -8.7% |
| OUT | 9% | **0.52** | **0.48** | **-2.3%** | **-1.5%** | **-13.6%** | **-30.8%** |

Fully invested in SPY when IN, cash when OUT, lagged a day:

| | Ann. return | Vol | Sharpe | Max drawdown (log) | Ulcer | Time invested |
|---|---|---|---|---|---|---|
| **Stay-in book, current** (premium-collapse rule in, regional-bank rule out) | **14.7%** | 14.8% | 1.00 | **-20.9%** | 5.27 | 91% |
| Stay-in book with the regional-bank rule (to 2026-10-08) | 15.0% | 14.6% | 1.03 | -20.9% | 4.99 | 90% |
| Stay-in book, before the premium-collapse rule | 14.8% | 14.8% | 1.00 | -24.5% | 5.29 | 91% |
| Drawdown-first book (section 11, prior) | 12.0% | 11.5% | 1.05 | -20.0% | 4.01 | 66% |
| SPY buy and hold | 10.9% | 18.6% | 0.59 | -73.1% | 13.47 | 100% |

Yearly, book vs SPY: 2008 -5.3 vs -45.9; 2009 +40.7 vs +23.4; 2014 +14.4 vs +12.6; 2018 +1.5 vs -4.7; 2022 -11.8 vs -20.1; identical to SPY in 2012, 2016, 2017, 2019, 2021, 2023, 2024 and 2026; within 2 points everywhere else. The 2009 figure is the book being back IN from mid-March 2009 at full size.

The 21 OUT spells and what SPY did while the book was out: 2008 (six spells, SPY -7.0, -5.5, +1.5, +0.8, **-33.8**, -13.3); Apr 2010 -0.7; Jul 2010 +3.9; Aug 2011 +0.2; Feb 2013 +0.1; Jun 2013 -2.3; Sep 2014 -1.0; Sep 2015 +2.1 (one session); Jan-Mar 2018 -3.7; Oct 2018 -3.0; Sep 2020 -2.1; May-Jun 2022 -13.8; Aug-Sep 2022 -5.9; Nov 2022 **+8.1** (the one costly miss, 14 sessions); Mar 2025 -2.6; Apr 2025 **+11.0** (out for the rebound). Thirteen of 21 spells avoided a loss; the two costly ones were both rebounds the dial was slow to release. The March 2026 spell in the previous version (SPY -5.7% while out) was the regional-bank rule's doing and is gone with it.

Episodes (share of SPY's peak-to-trough loss the book took; share of the 42-day rebound it caught):

| Peak | SPY | Book | Loss captured | Rebound captured | Sessions to OUT |
|---|---|---|---|---|---|
| 2008-05 | -72.3% | -14.0% | 19% | 64% | 29 |
| 2010-04 | -17.1% | -14.3% | 84% | 13% | 0 |
| 2011-04 | -20.6% | -19.4% | 94% | 100% | 81 |
| 2015-07 | -12.7% | -12.7% | 100% | 92% | never |
| 2015-11 | -13.7% | -13.7% | 100% | 100% | never |
| 2018-01 | -10.6% | 0.0% | 0% | -100% | 0 |
| 2018-09 | -21.5% | -19.1% | 89% | 100% | 17 |
| 2022-01 | -28.1% | -13.1% | 47% | 41% | 85 |
| 2023-07 | -10.5% | -10.5% | 100% | 100% | never |
| 2024-07 | -8.8% | -8.8% | 100% | 100% | never |
| 2025-02 | -20.8% | -20.9% | 100% | 95% | 13 |
| 2026-01 | -9.3% | -9.3% | 100% | 100% | never |

Median loss captured 100%, median rebound captured 100%. This is the trade: it rides every ordinary correction in full, the 2025 Q1 one and the 2026 Q1 one included (in 2025 it went OUT on day 13, SPY fell 2.6% during the spell, then it came back IN on April 9 and missed the first +11.0% of the rebound before re-entering). What it does is sidestep the bulk of 2008 and 2022 and the February 2018 event, and catch the rebounds in full everywhere else.

**2026-10-07: dial 54, state IN** (363 sessions, since April 2025). Components: base 60, plus 10 flip confirmation (late September), minus 8 equity implied cheap vs credit realized (live), minus 5 rates pressure with VIX asleep, minus 3 VIX and VVIX at the floor. OUT needs a reading of 20 or below: from here that takes the credit crack (-20) plus a VVIX-lagging VIX spike (-15), or three onset-class fires stacked, with the flip confirmation expired.

## 12. Volume

Volume rate of change was tested as a dimension (SPY, QQQ, IWM, HYG, TLT, GLD, XLE, KRE: today's volume vs trailing month, 5d and 21d average-volume ROC, all z-scored, and a signed version by the direction of the 5-day price move). Output in `results/volume_out.txt`.

Signatures are clear: into the 36 tops, 21-day volume ROC is negative throughout (-0.4 to -0.5 z for SPY, -0.8 for IWM at day -3): volume dries up into highs. At the 40 lows, day-of volume vs the trailing month is +0.9 z on the trough day and the day before, 21-day volume ROC is +0.9 to +1.0, and signed volume is -0.8 (volume expanding on down moves). Volume then collapses to -1.2 z within six sessions of the low.

As a predictor, volume adds nothing on its own: every quintile of every volume feature sits within two points of the 17.4% base rate. Two interaction cells carried information: HYG 5d volume ROC z below -1 while VIX 5d ROC z is above 1 (credit volume drying up during a vol spike) had P 0.34 and fwd21 -0.16%; SPY 5d volume ROC z above 1 with SPY 5d return z above 1 (buying climax) had fwd21 +0.26% against +0.86% base. Adding the credit-volume rule and a volume-climax capitulation qualifier to the dial did not improve the state backtest (the climax add fired early in 2025 Q1 and raised the loss captured from 43% to 92%), so volume is reported here as context and is not in the dial. Price, volatility and the rates complex carry the timing; volume confirms the climax after the fact.

---

## 13. Realized-vol ratios at several windows, crosses, and the era test

Scripts `rvratio.py`, `cross.py`, `qual.py`; outputs `results/rvratio_out.txt`, `cross_out.txt`, `qual_out.txt`.

**Windows.** 5/21 and 10/21 are turn-timing ratios (peak +2 to +3 z within two days of a low, collapse within a week; flat into tops). 21/63 and 21/126 are regime ratios (SPY 21/63 runs -0.8 to -1.0 z for the five sessions into every top; +0.5 to +0.9 for weeks after a low). No ratio predicts on its own; every quintile sits near base rate.

**As qualifiers on the rules (first-fires):** the rally-failure tell with SPY 21/63 z below zero has P(5% DD) 0.57, fwd21 -2.3%, mean 63d max DD -12.4% (vs 0.35 / -0.7% / -8.6% when realized is expanding). First impulse after compression with 21/63 z below zero: 0.35 / -0.1% / -6.1% vs 0.19 / +1.6% / -4.5%. Full capitulation with 21/63 z below zero: 0.00 / +3.6% / -2.1% (n=10). Flip confirmation with 10/21 z below zero: 0.28 / -0.05% vs 0.15 / +1.7%: a vol-collapse confirmation that follows no realized expansion confirms nothing.

**Applied to the dial (tested, not adopted):** (1) flip confirmation counts only if SPY 10/21 z is at or above zero; (2) rally-failure and first-impulse rules weighted heavier when 21/63 z is below zero; (3) capitulation prints weighted heavier when 21/63 z is below zero. Each alone and all together: 14.4 to 14.8% a year vs 14.8% current, same -24.5% max drawdown, identical episode captures, one to three extra OUT spells. No improvement. The qualifiers describe which fires are the dangerous ones but the stay-in dial at a 20 threshold already waits for enough of them to stack; the information is already in the price of admission. Not adopted.

**Crosses, by era (2008-17 vs 2018-26 vs 2021-26).** SPY 10/21 crossing above 1.2: mean next 21 days +0.9% in 2008-17, **+2.0% in 2018-26, +2.3% in 2021-26**, with P(next 5 days negative) falling from 0.40 to 0.29. With SPY down over the prior week (the selling version): +0.8% then **+2.3% / +2.8%**, mean 21d max drawdown -3.1% then -2.5% / -1.8%. The 5-day mean absolute move after the cross is 0.8x the 5 days before in every era and the 10-day realized vol measured 10 days later is 0.75 to 0.9x: **a 10/21 cross above 1.2 marks the end of the acceleration, not the start, and it has become a sharper buy signal in the pod / vol-control era, not a sell signal.** Realized vol crosses revert; the selling climaxes into the cross.

The same cross with VIX 5d ROC z below zero (realized up, implied not chasing): +0.9% in 2008-17, **+2.8% in 2018-26**, P(5d negative) 0.23. With VVIX/VIX ROC z above zero: 0.0% then +2.8%. With VVIX/VIX ROC below -1: +1.3% then +1.1%, with the 2018-26 next-10-day mean at -0.7% and 21d max DD -4.3%. In the recent era, the cross is bought unless vol-of-vol is lagging.

The opposite is the compression cross. SPY 10/21 crossing below 0.8: next 21 days +1.5% in 2008-17, **+0.8% in 2018-26, +0.6% in 2021-26**, next 10 days -0.2% in 2021-26, and realized 10 days later is 1.25 to 1.3x the level at the cross. Compression crosses lead to expansion and have gone from benign to flat in the recent era. That is where a vol-control bid being withdrawn would show, and it does, modestly.

21d realized crossing above 20% (a common vol-target deleveraging zone): next 21 days +0.9% in 2008-17, **-1.3% in 2018-26, -3.5% in 2021-26**, mean 21d max DD -3.2% then -5.4% / -6.6%, with the selling version at -3.0% / -6.4% in 2021-26 (n=8). Crossing 25%: next 21 days +2.4% / +2.8% in the recent eras (n=11 / 8) with the first 3 days negative. **The 20% realized cross is the one level where the recent era shows acceleration of selling after the cross; 25% is where it exhausts.** Small samples.

Reading across: in the 2018-26 era the market has sold *into* realized-vol ratio crosses and bought *out of* them faster than before, consistent with systematic deleveraging completing at the cross. The one place selling continued after a cross is 21-day realized crossing 20%. None of this changes the dial; it sharpens what to do at the moment a cross prints.

**21/63 crosses specifically** (`scripts/cross2163.py`, `results/cross2163_out.txt`). SPY 21/63 crossing above 1.0: 107 events since 2008, next 21 days +1.3%, next quarter +2.7%, 14% chance of a 10% drawdown in the quarter, all at or slightly better than base. A fast cross (5-day ROC z above 1) is better than a slow one (+1.8% / +4.0% vs +0.9% / +1.7%), and in 2018-26 the fast cross has 20% odds of a negative month vs 42% for a slow one. Higher thresholds are not more dangerous: crossing 1.2 or 1.3 gives the same forward returns with a larger mean drawdown (-7%). The cross is a bounce point, not a break point. The one context that turns it: **SPY 21/63 crossing 1.0 while the 10-year yield's 21-day change z is above 1** (realized vol expanding into a yield rip): 14 events, next 10 days -2.0%, next month -0.6%, 21-day max drawdown -4.3%, and in 2018-26 a 43% chance of a 10% drawdown within the quarter. The second: crossing with VIX 21d ROC z at or below zero (realized up, implied not following), which in 2018-26 gives -0.3% over the next month and a -7.3% mean quarterly drawdown, versus +1.4% / -5.4% when implied is rising with it. Crossing with MOVE 21d ROC z above 1 is the opposite, +2.7% next month, +6.0% next quarter. QQQ shows the same shape with weaker numbers. Today SPY 21/63 is 0.93 (21-day low 0.64) and QQQ 0.82 (low 0.53): both are rising toward the line from compressed bases, and the yield context is the one that historically mattered.

**ROC-only conjunction** (`scripts/rvroc.py`, `results/rvroc_out.txt`): realized vol ROC at 5/10/21 days, realized acceleration, ratio ROC and the 5-day change in the VIX-realized premium, crossed with every index ROC, no level gates. Danger cells (n>=40, P(5% DD/21d) >= 0.30 vs 0.18 base): premium falling fast (VIX-realized 5d change z < -1) while VIX 21d ROC z > 1, P 0.41, next 10 days -1.1%, next month -1.3%; 21/63 ratio 5d ROC z > 1 with TNX 21d change z > 1, P 0.40, next month -0.6%; premium falling fast with OVX 5d ROC z > 1, P 0.36; realized acceleration z > 1 with TNX 21d change z > 1, P 0.34; realized 21d ROC z > 1 with SKEW 10d ROC z > 1, P 0.32, next month -0.6%. Safe cells (P <= 0.07, 40+ of them, every one with a positive forward month): almost all share realized vol compressing (5/10/21-day ROC z < -1) or the front end of the VIX curve falling (VIX9D/VIX 5d ROC z < -1); realized 5d ROC z < -1 with MOVE 5d ROC z > 1 is P 0.05 to 0.07 with +1.6% to +2.2% next month (bond vol spiking into compressed equity realized = buy). Three-way: realized expanding (5d ROC z > 1) with VIX rising and VVIX/VIX ROC between 0 and 1 is P 0.43 (n=7); realized expanding with VIX 5d ROC z between -1 and 0 and VVIX/VIX ROC z > 1 is P 0.08 with +2.0% (realized up, implied calm, vol-of-vol bid = benign, n=12). Month trend x week impulse of realized: both > 1 is the only cell above base (0.25, -4.2% mean 21d max DD); the first week-impulse out of a compressed month (21d ROC z < -1, 5d ROC z > 1) is P 0.11, +1.4%, which is the opposite of the VIX version in section 4. Today: every realized ROC is at or below zero (5d -0.36, 10d -0.21, 10-day realized 5d ROC -1.23); 21-day ROC +0.56 is the only positive reading.

**Adopted into the dial** (`scripts/addrv.py`): each of the seven strongest realized-ROC cells was added to the stay-in dial as a rule, alone and together. Six were neutral or worse (adding all seven: 13.5% a year, 40 OUT spells, diluted). One improved every statistic without adding a spell: the premium-collapse rule (VIX-minus-realized 5d change z < -1 with VIX 21d ROC z > 1) at -8 for 15 sessions takes the stay-in book from 14.8% / -24.5% / 23 spells to **15.0% / -20.9% / 23 spells**, cutting the 2011 loss captured from 119% to 94%, and is stable at -8, -10 and -12. It is in the dial and on the realized-vol board. The 21/63-into-yields rule, the realized-acceleration rule and the two risk-on realized rules were not adopted.

---

## 14. Breadth: equal weight vs cap weight, ratio rate of change

Scripts `fetch_ew.py`, `breadth.py`, `breadth_snapshot.py` (in the `run_all.sh` chain), `addbreadth.py`, `addbreadth2.py`, `breadth_check.py` (research); outputs `results/breadth_out.txt`, `addbreadth_out.txt`, `addbreadth2_out.txt`, `breadth_check_out.txt`, `breadth_data.json`.

**Construction.** RSP/SPY, QQQE/QQQ and eleven Invesco equal-weight sector ETFs against their SPDR cap-weight counterparts (RSPT/XLK, RSPS/XLP, RSPH/XLV, RSPF/XLF, RSPD/XLY, RSPG/XLE, RSPU/XLU, RSPM/XLB, RSPN/XLI, RSPR/XLRE, RSPC/XLC). Ratio = log(equal weight / cap weight); ROC over 5, 10, 21 and 63 sessions and a 5-day acceleration (change in the 5-day ROC), each z-scored against its own trailing 252 days. Composites: the median of the nine core sector ratio ROC z, and the share of the nine sectors where equal weight beat cap weight over 21 days. History starts 2003 (RSP), 2006-07 (sector suite), 2012 (QQQE), so this layer is a 2005+ study.

**Signatures.** At the week and month horizon breadth does not lead tops: RSP/SPY 21d ratio ROC z runs -0.1 to -0.3 for ten sessions either side of the 36 peaks, the sector composite sits at zero, the share of sectors with equal weight winning is flat. At the quarter horizon the narrowing is visible: RSP/SPY 63d ratio ROC z is -0.4 to -0.7 every session from ten before to ten after the top, QQQE/QQQ -0.4 to -0.9. Into the 40 lows equal weight loses faster, not slower: RSP/SPY 21d ROC z is -0.6 on the trough day, share z -0.7, the sector composite -0.5, and all three stay negative for eight sessions after the low before turning. Breadth confirms a low about two weeks late; it does not time one.

**Prediction.** Every week- and month-horizon breadth feature sits within three points of the 17.4% base rate in every quintile, with two exceptions: QQQE/QQQ 21d ROC z top quintile (equal-weight Nasdaq outperforming hard) P 0.24 with fwd21 +0.42% and bottom quintile 0.09; RSPU/XLU 21d top quintile 0.23. The 63d horizon is the one standalone read that is monotone: RSP/SPY 63d ROC z bottom quintile P 0.23 and fwd21 +0.38%, top quintile 0.14 and +1.25%.

**Interactions (all days, P of a 5% SPY drawdown within 21d, base 0.17).** RSP/SPY 21d ROC z below -1 with TNX 21d change z above 1: 0.33, fwd21 -0.74% (n=112). The same with the 63d ROC: 0.41, fwd21 -1.76% (n=97), 0.46 in 2018-26. QQQE/QQQ 63d below -1 with TNX 21d z above 1: 0.41, -1.61%. Equal weight outperforming (RSP/SPY 21d ROC z above 0) while SPY's 21d return z is below -1: 0.33 to 0.39, fwd21 -0.2 to -0.55%, which is defensive rotation inside a decline and not a low. Sector composite 21d below -1 with VIX 5d ROC z between -1 and 0: 0.35 (n=74). The safe side: RSP/SPY 63d above 1 with VIX 21d ROC z below -1 (broadening while vol compresses): 0.06, fwd21 +1.80%. First-fire era split for quarter-narrowing-into-yields: 2008-17 n=10, P 0.30, fwd21 -3.9%; 2018-26 n=9, P 0.56, fwd21 -2.5%. Fires cluster in Oct-Nov 2005, Oct 2007, Oct 2008, Sep-Nov 2014, Jun and Nov 2015, Feb and Sep-Oct 2018, Dec 2024, Nov 2025 and May 2026.

**Added to the dial (fourteen variants, none adopted).** Current stay-in book: 15.03% a year, max drawdown -20.9%, 23 OUT spells. The best breadth add, month-narrowing-into-yields at -8 for 15 sessions: 15.34%, Sharpe 1.06 vs 1.03, same max drawdown, 26 spells, 2018-09 loss captured 94% vs 89%, 2022 49% vs 47%. The extra return is 2013 +1.4, 2015 +1.0, 2020 +3.6 and 2024 +2.6 points against 2018 -1.0, 2022 -0.5 and 2025 -1.1. It adds three OUT spells (Mar 2015; Mar 2022 with SPY +1.8% while out; Dec 2024 to Jan 2025 with SPY -3.2%) and moves three exits a few days earlier (Jun 2013, Sep 2014, Sep 2020). That is a coin-flip gain bought with more switching, and it fails the standard every other addition was held to: better return and drawdown together with no new spells. The quarter version: 15.13%, -21.2%, 25 spells. Equal-weight rotation inside a decline as a rule: 14.15%; as a -5 or -8 context: 15.10%, -21.2%, 24 spells. Risk-on breadth adds (equal weight accelerating after a drawdown at +8; equal weight strong with VIX collapsing at +6): 14.59% / -24.5% and 14.97% / -23.0%. The QQQE/QQQ 21d-ROC-below-minus-1 rule prints 15.22% but its first-fire P is 0.04: a benign condition wearing a negative weight, improving the book by accident. Nothing from breadth goes into the dial.

**On the dashboard** as a context board, not a rule: RSP/SPY and QQQE/QQQ 21d and 63d ratio change and ROC z, sector share and composite, the five cells above with live / last-10-day status and their all-days and 2018+ odds, the full pair table, and a z-chart set of RSP/SPY 21d and 63d ratio ROC against TNX 21d change. `run_all.sh` fetches the fourteen equal-weight tickers.

**2026-10-07.** RSP/SPY -4.2% over 21 sessions (ratio ROC z -1.58) and -4.6% over 63 (z -1.05); QQQE/QQQ -3.6% / -3.5% (z -1.45 / -0.87); RSPC/XLC -4.8% (z -2.04), RSPM/XLB z -2.22, RSPS/XLP z -1.63; three of nine sectors have equal weight ahead over the month (share z -0.79). TNX 21d change z is +2.43. Both narrowing-into-yields cells are live: the month version since September 18 (13 sessions), the quarter version as of today. Historically that pairing carried 0.33 to 0.41 odds of a 5% drawdown inside a month and a mean next month of -0.7% to -1.8%. The dial does not count it. The dial reads 44, IN.

---

## 15. Operations: the daily refresh and its guards

**Schedule.** A routine fires into the build session every weekday at 5:41pm New York, runs `scripts/run_all.sh`, commits `results/` and `dashboard.html` as `Daily refresh: vol tape regime <date>`, pushes, republishes the dashboard to the same artifact link, posts the spoken summary in the session and sends a one-line phone notification. The time follows the data: Yahoo posts the equity and bond closes by 4:00pm, the CBOE vol indices by 4:15pm, MOVE at 4:17pm and SKEW at 5:00pm New York, so 5:41pm leaves SKEW a 40-minute margin. A routine that spawned a fresh session was tried first and abandoned: a fresh session starts without the repository, credentials or packages, and its permission layer refused the pipeline script and the package install.

**Guards in the chain.** `fetch.py` exits non-zero if any required series comes back empty after three attempts, and prints which series end before the latest session; `^VIX1Y` (one row) and `^RVX` (refused) are the known exceptions and are excluded downstream. `build.py` exits non-zero unless the ticker universe is exactly the 38 ETFs plus LQD and SHY and there are 15 index series; this is what caught the equal-weight files leaking into the universe on the second run of the chain. `snapshot.py` exports a `stale` list (any index series, before its 3-day forward-fill, or any ticker whose last close is before the as-of session); the dashboard header shows it in orange and the routine's summary reads it out. Index series are forward-filled up to three sessions in `rebuild.py`; tickers are not filled.

**Regeneration.** `run_all.sh` rebuilds everything from an empty `work/` directory (equal-weight data goes to `work/data_ew/`, everything else to `work/data/`), installing the yfinance fork and scikit-learn, statsmodels and scipy if they are missing. Verified: a from-scratch run reproduces the committed result files to two decimals; a second run in the same directory reproduces them again. Day-to-day differences at the third decimal come from Yahoo's adjusted-price revisions.

**Not verified yet.** An unattended scheduled fire while the session is idle, and a fire after the session's container has been reclaimed (the repository is re-cloned and the routine re-fetches and reinstalls; the path is the same as the from-scratch run above, but it has not been observed under the scheduler).

---

## 16. Change log

**2026-10-08: regional-banks-vs-yields rule removed.** The user's call: regional banks are not a channel of the 2026 economy, the largest money-center bank alone outweighs the whole KRE index, and a rule built on them would not survive a room of allocators. The data agrees more than it disagrees. Of the rule's 35 first-fires since 2005, 14 came in 2007-09, where it was right (2008-17: P 0.61, fwd21 -4.5%); the 15 fires of 2018-26 carried P 0.27 with a mean next month of +1.45%, which is no edge. In the dial it only ever pushed OUT; the sessions it changed were Feb, Apr and Jun 2008, Sep 2013, Aug-Sep 2015, Mar 2020, Oct-Nov 2022, Apr 2025 and Mar 2026. Without it the stay-in book goes from 15.03% to 14.70% a year at the same -20.9% max drawdown, 23 to 21 OUT spells, the Oct-Nov 2022 miss shrinks from +9.6% to +8.1% of SPY sat out, and the book rides the Jan-Mar 2026 drawdown in full (-9.3%) instead of 44% of it. Today's dial moves from 44 to 54, state IN either way. Removed from `scripts/score.py`, the pair-rule board in `scripts/tick_snapshot.py`, the dashboard and the PDF; the research scripts that reference the earlier baseline (`score2.py`, `addrv.py`, `addbreadth*.py`) are left as the record of what was tested against. KRE stays in the 38-ticker tape.
