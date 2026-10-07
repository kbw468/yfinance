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
| **KRE rel ROC5 z < -1 and TNX 5d chg z > 1** | 35 | **0.46** | -1.8% | regionals breaking while yields rip. Last fire 2026-05-15. |
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

## 11. Exposure dial (0 = no beta, 100 = max long high beta), tuned for drawdown first

Objective set by the user: minimize drawdown, accept muted upside in V-shaped recoveries. The dial is built from every rule in sections 1, 8 and 9 plus slow context flags. Base 60. Risk-off rules subtract for a fixed number of sessions after they fire; risk-on rules add. Weights: rally-failure tell and credit crack -20, VVIX-lagging onset and second-leg continuation -15, impulse-after-compression, bank/yield break, realized-outruns-implied -10, rates setup -8, full capitulation print +25, gold-vol/VVIX-outrunning capitulation +12, flip confirmation +10, VVIX-confirming spike +8. Context: rates pressure with VIX asleep -5, complacency -5, in a 7%+ drawdown with no capitulation print -10, vol collapsing from a high +10, at the highs +5, VIX and VVIX both at the floor -3.

Drawdown-first asymmetry (the difference from a balanced dial): risk-off fires persist 1.5x longer (15 to 32 sessions), risk-on adds persist half as long (10 to 21 sessions), the dial is capped at 30 whenever SPY is 5%+ off its 63-day high without a flip confirmation or capitulation print in the last 10 sessions, and after any reading at or below 35 the dial may rise by at most 4 points per session until it is back above 60. Clamped to 0-100. Definitions in `scripts/score.py`; the variants tested and rejected are in `scripts/score2.py`, `score3.py`, `score4.py`.

Backtest 2008 to date, Feb-Jul 2020 excluded, holding SPY at dial/100 lagged a day with the rest in cash:

| | Ann. return | Vol | Sharpe | Max drawdown (log) | Ulcer |
|---|---|---|---|---|---|
| Drawdown-first dial | 7.2% | 6.7% | 1.07 | **-15.3%** | 2.54 |
| Balanced dial (rejected) | 9.5% | 9.6% | 0.98 | -21.5% | 4.25 |
| SPY buy and hold | 10.9% | 18.6% | 0.59 | -73.1% | 13.47 |

Average exposure 44%; the dial sits at 30 or below 28% of the time, which is the price of the slow re-entry. By band, 21-day forward: 0-30 has 33% odds of a 5% drop and a mean +0.5% (the band now includes the slow climb out of every low, which dilutes it versus the balanced dial's 50%); 31-45 is 13% odds and +1.2%; 46-55 is 10% and +1.2%; every band above 55 is 10-13% with positive forward returns. Separation is in the bottom band; above 45 the dial tells you to be in, not how much more.

Episode table, share of SPY's peak-to-trough loss the dial-weighted book took, and share of SPY's 42-day rebound it captured:

| Peak | Trough | SPY | Dial book | Loss captured | Rebound captured | Dial at peak / trough / +21d |
|---|---|---|---|---|---|---|
| 2008-05 | 2009-03 | -72.3% | -3.0% | 4% | 15% | 47 / 0 / 21 |
| 2010-04 | 2010-07 | -17.1% | -4.4% | 26% | -6% | 17 / 0 / 0 |
| 2011-04 | 2011-10 | -20.6% | -14.8% | **72%** | 49% | 67 / 39 / 15 |
| 2015-07 | 2015-08 | -12.7% | -5.2% | 41% | 29% | 89 / 47 / 0 |
| 2015-11 | 2016-02 | -13.7% | -7.1% | 52% | 57% | 21 / 80 / 11 |
| 2018-01 | 2018-02 | -10.6% | -0.3% | 3% | -8% | 10 / 0 / 0 |
| 2018-09 | 2018-12 | -21.5% | -4.7% | 22% | 31% | 67 / 18 / 0 |
| 2022-01 | 2022-10 | -28.1% | +0.2% | 0% | 13% | 20 / 18 / 0 |
| 2023-07 | 2023-10 | -10.5% | -4.9% | 46% | 55% | 42 / 51 / 67 |
| 2024-07 | 2024-08 | -8.8% | -3.4% | 39% | 49% | 57 / 43 / 79 |
| **2025-02** | **2025-04** | **-20.8%** | **-4.2%** | **20%** | **9%** | 61 / 13 / 0 |
| **2026-01** | **2026-03** | **-9.3%** | **-2.5%** | **27%** | **28%** | 48 / 7 / 50 |

Median loss captured 27%, median rebound captured 29%. The two V-shaped episodes the user named: Q1 2025 took 4.2% of SPY's 20.8% and caught 9% of the 18.9% rebound; Q1 2026 took 2.5% of 9.3% and caught 28% of the 18% rebound. The known weak spot is 2011: the April top came with the dial at 67 after the spring capitulation prints and the August crash was a continuation the rules only caught in part.

Yearly, dial book vs SPY: 2008 -2.5 vs -45.9; 2011 -2.2 vs +1.9; 2018 +5.6 vs -4.7; 2022 -0.1 vs -20.1; 2013 +13.3 vs +28.0; 2019 +14.1 vs +27.2; 2021 +10.0 vs +25.3; 2023 +15.6 vs +23.3; 2024 +8.0 vs +22.2; 2025 +5.5 vs +16.3. Every down year for SPY was flat to positive for the dial book; every up year gave back roughly half.

**2026-10-07: 25.** Base 60, minus 8 for the rates setup (yields ripped while VIX slept on September 23; with the longer window it persists 15 sessions), minus 10 for regional banks breaking relative while yields ripped (same week, persists 22 sessions), minus 8 for equity implied vol cheap against credit realized vol (live), minus 6 for bond implied vol spiking off compressed bond realized (fired within the last three weeks), minus 5 for rates pressure with the VIX asleep, minus 3 for VIX and VVIX both at the floor, plus 5 for being at the highs. The late-September flip confirmation no longer counts because risk-on adds now expire after 10 sessions. Bottom band. The dial fell from the 60s to the 20s on September 23 and the slow re-entry rule means it can climb at most 4 points a session once the September fires roll off in mid-October, so a clean tape takes it back to the 50s by late October, not before.
