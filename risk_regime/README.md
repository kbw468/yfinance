# Vol Tape Regime, version 2

A from-scratch rebuild of the risk-on / risk-off onset system: 15 volatility and rate indices (^VIX ^VVIX ^MOVE ^VXN ^GVZ ^OVX ^SKEW ^VIX1D ^VIX9D ^VIX3M ^VIX6M ^TNX ^TYX ^FVX ^IRX) against 38 ETFs, with full OHLCV, multi-estimator realized vol, volume and the volatility of volume, range compression, cross-sectional dispersion and correlation, equal-weight breadth, and multifractal spectra. Every result below is **walk-forward out of sample**: a model or threshold used in year Y was fit only on data whose labels ended before year Y. Feb 1 to Jul 31 2020 is excluded from every statistic. Scripts in `v2/`, outputs in `results/v2_*_out.txt`, run order in `v2/run_v2.sh`.

---

## 0. Why version 1 was retired

Version 1 scored rules and contexts on the full sample and adopted the ones that looked best. That is selection bias, and the audit (`v2/audit_v1.py`, `results/audit_v1_out.txt`) shows what it bought. Of 31 components in the v1 dial, 11 hold in both halves of the sample (2008-17 and 2018-26). Three of the five "context" adjustments had the wrong sign: *rates pressure with the VIX asleep* and *VIX and VVIX on the floor* are benign conditions (5 and 8 percent odds of a 5% drawdown within a month, against an 18 percent base) that were counting against the dial, and *vol collapsing from a high* carried 43 percent odds in 2008-17 while counting for it. Six rules reverse in 2018-26: the small-cap beta chase (P 0.36 then 0.16), realized-outruns-implied for SPY and QQQ (fwd21 -2.2% then +2.5%), credit realized expanding into a spike (0.33 then 0.14), the rates setup (0.26 then 0.20, no better than base) and the first-impulse rule. The regional-banks rule the user objected to was one of them (0.61 then 0.27, fwd21 -4.5% then +1.5%). The in-sample stay-in backtest (15% a year, -21% drawdown) was never an out-of-sample number. Nothing from v1 carries into v2 except the objective turning points and the exclusion window.

## 1. Data and guards

`fetch.py` pulls daily OHLCV for the 15 indices (CBOE indices have open-high-low), the 38 ETFs plus LQD and SHY, and `fetch_ew.py` the 14 Invesco equal-weight ETFs into a separate folder. The fetch fails if any required series is empty; the build fails if the ticker universe is not exactly the expected set; the snapshot reports any series that has not posted the as-of session. Index series are forward-filled up to three sessions, tickers never.

## 2. Feature store (`v2/features.py`, 3,950 features, `results/manifest.json`)

No z-scores and no moving averages of price. Every feature is a rate of change, a ratio, or a **trailing percentile rank over 504 sessions** of a raw quantity against its own history, which assumes nothing about the distribution.

| Family | Count | What |
|---|---|---|
| Index level ranks | 44 | each index vs its own 252 / 504 / 1260-session history |
| Index ROC and ranks | 225 | log change over 1, 2, 3, 5, 10, 21, 42, 63 sessions (yield changes for the rate series), ranked; 5-day acceleration |
| Index daily range | 18 | CBOE indices' own high-low range, ranked |
| Term structure and cross-asset ratios | 132 | VIX/VIX3M, VIX9D/VIX, VIX1D/VIX9D, VIX3M/VIX6M, VIX/VIX6M, VVIX/VIX, VVIX/VIX3M, MOVE/VIX, VXN/VIX, GVZ/VIX, OVX/VIX, SKEW/VIX: level rank and ROC over 1 to 21 sessions |
| Curve | 9 | 30y-10y, 10y-5y, 10y-3m: level rank and change ranks |
| ETF ROC, relative ROC, drawdown | 1,028 | own and relative-to-SPY log ROC over 1 to 63 sessions, ranked; distance from 63- and 252-session highs |
| ETF realized vol | 1,240 | close-to-close at 5/10/21/63/126, Parkinson at 5/10/21, Garman-Klass at 5/10/21, level ranks, ratios (5/21, 10/21, 21/63, 21/126, Parkinson/close-to-close), ROC of vol, vol of vol |
| ETF volume | 434 | volume rank vs 63 and 252 sessions, 5- and 21-day volume ROC, **volatility of volume** (21-session std of daily log volume changes) and its ROC, Amihud illiquidity, signed volume |
| ETF range | 240 | daily range rank, 5-vs-63-session range compression, narrow-range counts, close location, gaps |
| Implied vs realized | 106 | VIX/VXN/MOVE/GVZ/OVX over the native ETF's realized vol (three estimators), level rank, ROC, premium change |
| Cross-sectional | 30 | dispersion of 1/5/21-day returns across the 38, average pairwise correlation (21, 63), realized-vol breadth, volume breadth, compression breadth, ROC breadth |
| Equal weight vs cap weight | 6 | RSP/SPY and QQQE/QQQ ratio ROC ranks at 5, 21, 63 |

## 3. Multifractal and long-memory layer (`v2/fractal.py`, 176 features)

Rolling 504-session multifractal detrended fluctuation analysis on the daily log returns of SPY, QQQ, IWM, HYG, TLT, GLD, XLE and on VIX changes: generalized Hurst exponents h(q) for q in {-4, -2, 2, 4}, the spectrum width h(-4) - h(4), the DFA scaling exponent of absolute returns (volatility persistence), and Lo-MacKinlay variance ratios at 5 and 21 days on 252 sessions. Each as a raw value, a trailing rank, and 21- and 63-session change ranks. SPY's h(2) averages 0.45 (mildly anti-persistent), its width 0.21, its absolute-return exponent 0.76 (long memory in volatility).

What they are worth: in the era-consistent screen (section 5) the persistence of credit returns, `MF:HYG:h2_rank`, is the one multifractal feature in the top tier (consistent IC +0.148 with shallower drawdowns: when high-yield returns are trending rather than mean-reverting the next month is safer). Eleven multifractal features pass the 0.05 bar; none of the width or variance-ratio features of SPY itself does. The multifractal spectrum is a regime descriptor more than a timing signal at this horizon.

## 4. Labels and objective vol episodes (`v2/labels.py`)

Targets on SPY, each masked where its window touches Feb-Jul 2020: forward return, maximum drawdown and maximum run-up over 5, 10, 21, 42 and 63 sessions; `riskoff21` = 5% drawdown within 21 sessions (base 0.169); `riskoff63` = 10% within 63 (0.135); `riskon21` = 5% rally within 21 (0.186); `volexp21` = 21-session realized vol 21 sessions ahead at least 1.5x today's (0.143); `epstart21` = a vol episode starts within 21 sessions (0.115).

Vol episodes are objective: a realized episode starts when SPY's 21-day realized vol crosses above its trailing-504 80th percentile and ends below the 50th; an implied episode likewise on the VIX level rank. 43 realized and 62 implied episodes since 1995, median length 29 sessions, median SPY drawdown inside an episode -1.9% (most episodes are not crashes).

What the tape does into an episode start (median trailing rank, 76 starts): VIX 5-day ROC rank goes 0.61 at -5 sessions to 0.84 at -1 and 0.95 on the start day while the VVIX/VIX 5-day ROC rank falls from 0.53 to 0.22 and 0.09, which is the v1 discriminator confirmed on an objective anchor; SPY volume rank rises from 0.65 to 0.83 to 0.94; the range-compression ratio from 0.61 to 0.77 to 0.83; the 5-day change in average pairwise correlation from 0.56 to 0.69 to 0.84; ROC breadth falls from 0.55 to 0.32 to 0.24; VIX/VIX3M 5-day ROC rank from 0.61 to 0.86 to 0.96. The ramp is visible two to three sessions ahead of the threshold cross, because the cross is itself the tail of a continuous move. After the start, HYG, TLT, IEF, GLD, XLP and XLU sit at relative-ROC ranks of 0.8 to 0.9 (flight to quality is contemporaneous), energy services, oil and brokers at 0.3.

## 5. The era-consistent screen (`v2/screen.py`)

Every feature's Spearman rank correlation with the forward 21-session drawdown depth and run-up, in three eras (to 2012, 2013-19, 2020-26), with top- and bottom-decile hit rates for the 5% drawdown label. Overlapping labels make the effective sample about n/21, so the screen asks for the same sign in all three eras and reports the weakest era's |IC| as the score. 423 of 4,126 features clear 0.05 in every era; 130 clear 0.08; 62 clear 0.10.

**High value precedes deeper drawdowns in every era:** the level ranks of VIX3M, VIX6M, VXN, VIX9D, VIX and MOVE (-0.10 to -0.14); the Parkinson and close-to-close vol ranks of HYG and LQD (credit range vol, -0.11 to -0.13) and of XME, XBI, XLK, XHE; Amihud illiquidity ranks of IWM and HYG (-0.13, -0.11); the Parkinson-over-close-to-close ratio of KRE, KBE, XSW, XPH, BNO (wide intraday ranges relative to close-to-close moves, -0.10 to -0.14); healthcare's 63-day relative ROC (-0.15, a quarter of defensive leadership); the 42-day ROC of VIX3M and VIX6M (-0.10).

**High value precedes a safer tape in every era:** HYG 21-day signed volume (+0.17), credit momentum and distance from highs (HYG roc63, dd63, dd252: +0.14 to +0.15), credit return persistence (`MF:HYG:h2_rank` +0.15), the VVIX/VIX level rank and its 21-day ROC (+0.10 to +0.12: vol-of-vol high relative to vol), GVZ/VIX level rank (+0.11), cyclical momentum (transports, small caps, retail roc63 ranks +0.11 to +0.14).

**High value precedes bigger rallies in every era:** the same vol levels (VIX9D rank +0.30, VIX rank +0.28), realized vol levels of SPY, IWM, XLF, KIE (+0.27 to +0.31). High vol is a two-tailed state: it precedes both the deeper drawdowns and the bigger rallies, and the rally side is the stronger of the two.

Rate-of-change features of the vol indices are weaker than their level ranks for the drawdown target at this horizon: VIX 21-day ROC rank has consistent IC -0.03 to -0.09 against -0.07 to -0.15 for the level rank. ROC features matter more for the vol-expansion and rally targets (section 7).

## 6. The ticker layer on its own terms (`v2/tickers.py`)

For each ETF, 39 of its own features (realized-vol ratios, volume rank and ROC, volatility of volume, range compression, liquidity, relative ROC, drawdown) were screened against its own forward 21-session drawdown depth and return, in the same three eras. The result is mostly null: 20 of 38 ETFs have at most one own-feature that is era-consistent at |IC| 0.05, and no feature family is consistent across the universe (the median consistent IC of every own-feature across tickers is 0.00). The exceptions: HYG, whose Parkinson vol rank (-0.19) and distance from the 252-day high (+0.18) are strong; the daily range rank and the Parkinson vol rank, which precede deeper drawdowns for about a quarter of the ETFs (QQQ, XLK, SPY, TLT, XLF, XLY among them); distance from the 252-day high, safer for 16 percent; narrow-range counts, safer for 11 percent. The volatility of volume, which the rebuild computed for every ETF, is consistent for no more than 7 percent of them in either direction. An ETF's own volume and vol structure does not time its own drawdowns at a monthly horizon; the market-level vol state does.

Around the 76 vol-episode starts, the forward 10-session return relative to SPY from the start day is negative for energy services (-1.8%), communications (-1.1%), oil (-1.0%), brokers (-1.0%), exploration (-1.0%), transports (-0.9%) and bonds (IEF -0.8%, which has already rallied into the start), and positive for aerospace (+0.7%), semis (+0.6%), pharma (+0.4%), homebuilders (+0.4%), telecom (+0.3%) and utilities (+0.3%). When cross-sectional dispersion is in its top quintile, oil, bonds, banks and energy services lag SPY over the next ten sessions by 0.4 to 0.5 percent at the median.

## 7. The walk-forward model (`v2/wf.py`, `v2/model2.py`, `results/v2_model2_out.txt`)

**Protocol.** For each test year 2005 to 2026 the model is fit on every session whose label window ends at least 100 calendar days before January 1 of that year (so no training label overlaps the test year), and predicts the test year. Feature selection happens inside the training window: a feature is eligible only if its rank correlation with the target has the same sign in both halves of the window, and the 15 with the largest weaker-half correlation are used. The learner is a gradient-boosted ensemble of depth-2 trees, 200 rounds, constrained to be **monotone** in each feature in the direction the training window gave it, with no tuning of anything. Inputs are ranks; the learner is a sum of step functions. Nothing assumes a distribution. Features built on KRE are excluded from the candidate set by the user's direction. The configuration was fixed before the capacity variants below were compared.

**Out-of-sample results, 2005 to date, Feb-Jul 2020 excluded** (AUC; hit rate by predicted quintile, lowest to highest; base rate; 95% interval from a bootstrap over calendar years):

| Target | AUC 2005-12 | 2013-19 | 2020-26 | pooled | 95% CI | hit rate by quintile | base |
|---|---|---|---|---|---|---|---|
| 5% rally within 21 sessions (P_on) | 0.80 | 0.79 | 0.75 | **0.79** | 0.74-0.83 | 0.03, 0.04, 0.14, 0.22, **0.41** | 0.17 |
| realized vol 1.5x within 21 (P_vol) | 0.65 | 0.80 | 0.82 | **0.75** | 0.68-0.80 | 0.04, 0.08, 0.11, 0.19, **0.35** | 0.15 |
| 5% drawdown within 21 (P_off) | 0.62 | 0.59 | 0.72 | **0.64** | 0.54-0.72 | 0.10, 0.16, 0.10, 0.15, **0.31** | 0.16 |
| 10% drawdown within 63 (P_off63) | 0.60 | 0.59 | 0.69 | 0.61 | 0.45-0.76 | 0.08, 0.13, 0.08, 0.14, 0.22 | 0.13 |

The rally and vol-expansion sides are strongly predictable out of sample. The drawdown side is weakly predictable: the top quintile of P_off draws down 5% within a month 31 percent of the time against 16 percent, but the year-bootstrap interval reaches down to 0.54, and the 10%-in-a-quarter target is not distinguishable from chance by that interval.

**What the models use.** P_vol: the implied-over-realized ratios (VIX over SPY and IWM close-to-close and Parkinson vol, selected in every one of the 22 windows), the Parkinson-over-close-to-close ratio, the 21-day change in the VIX premium, the 21/63 realized ratio and the 21-day ROC of realized vol (negative). P_on: realized vol levels of SPY under all three estimators, the VIX level ranks, the SKEW/VIX level rank (negative). P_off: the Parkinson, Garman-Klass and close-to-close vol levels of QQQ, XLK and XLY, which is to say the level of technology realized vol. The drawdown model is a vol-level model; the rate-of-change features the brief put first do not survive selection for this target.

**Capacity, and the single features (all walk-forward):**

| P_off model | AUC 2005-12 | 2013-19 | 2020-26 | pooled |
|---|---|---|---|---|
| 150 features, depth 3, unconstrained (the overfit reference) | 0.49 | 0.49 | 0.63 | 0.53 |
| 30 features, depth 2, monotone | 0.65 | 0.52 | 0.67 | 0.63 |
| **15 features, depth 2, monotone (production)** | 0.62 | 0.59 | 0.72 | **0.64** |
| 8 features, depth 1, monotone | 0.61 | 0.56 | 0.70 | 0.63 |
| raw VIX level rank (504) | 0.68 | 0.67 | 0.60 | 0.66 |
| raw VIX3M level rank (252) | 0.65 | 0.65 | 0.66 | 0.66 |
| raw SPY realized-vol level rank | 0.68 | 0.59 | 0.67 | 0.66 |
| raw HYG Parkinson vol rank | 0.63 | 0.60 | 0.65 | 0.63 |
| raw XLV 63-day relative ROC | 0.59 | 0.65 | 0.66 | 0.62 |
| raw HYG signed volume (negative) | 0.58 | 0.66 | 0.66 | 0.62 |
| raw average pairwise correlation rank | 0.55 | 0.57 | 0.60 | 0.58 |
| raw VIX 21-day ROC rank | 0.51 | 0.56 | 0.55 | 0.54 |
| raw VIX 5-day ROC rank | 0.51 | 0.55 | 0.52 | 0.52 |

Three things to take from this table. Capacity destroys the drawdown signal: the 150-feature model that a careless build would ship scores at chance. The monotone small models land at 0.63 to 0.64 whatever their size. And the best single predictor of a 5% drawdown within a month, out of sample, is the level of implied or realized vol against its own two-year history, at 0.66, which the fitted model does not beat. The VIX's rate of change on its own is worth 0.52 to 0.54 for this target: level, not ROC, carries the drawdown information at a monthly horizon. ROC earns its place on the vol-expansion side, where the premium's 21-day change and the realized-vol ROC are among the selected features in every window.

**Permutation test of the whole pipeline (`v2/nulltest.py`).** The label series is shifted by a random offset of at least a year, which keeps its base rate and autocorrelation and breaks its link to the features, and the complete pipeline, selection included, is rerun 20 times per target. For P_off the null pipelines score a mean AUC of 0.497 with a standard deviation of 0.048 and a maximum of 0.612; the real 0.636 is 2.9 standard deviations above the null mean and above every null draw. For P_vol the null mean is 0.500 with a standard deviation of 0.025 and a maximum of 0.533; the real 0.748 is 10.0 standard deviations above the null mean and above every null draw. For P_on the null mean is 0.508 with a standard deviation of 0.056 and a maximum of 0.622; the real 0.789 is 5.0 standard deviations above the null mean and above every null draw. The drawdown model clears its null by a comfortable but not enormous margin; the other two are nowhere near chance.

**What the probabilities meant out of sample** (`results/v2_decision_out.txt`, decile tables on the dashboard). The top decile of P_off: 35 percent drew down 5% within a month, mean maximum drawdown -4.5%, and yet mean forward return +1.8% and a 52 percent chance of a 5% rally, because a high drawdown probability is a high-vol state and high vol is two-tailed. The top decile of P_on: 51 percent rallied 5%, 35 percent drew down 5%. The top decile of P_vol: realized vol expanded 1.5x 43 percent of the time, with a mean forward return of +0.8% and drawdown odds at base. The joint cells make the point: when P_vol is above its 90th percentile and P_on below its median, the next month returns +0.9% with 7 percent drawdown odds. An expected vol expansion is not an expected selloff; it is usually realized vol normalizing from a compressed base.

## 8. The decision layer, out of sample (`v2/decision.py`)

Two-state rules on trailing-756-session quantiles of the out-of-sample probabilities (past values only; nothing is fit), fully invested in SPY when IN and in cash when OUT, lagged one session, 2006 to date, Feb-Jul 2020 excluded, drawdowns in log units (-80 log is -55% in price). The grid was specified before it was run and is reported whole.

**The pre-registered rule failed.** Before the grid was run, the headline was fixed as rule B: OUT when P_vol is above its 90th percentile while P_on is below its median, back IN when P_vol falls below its 70th percentile or P_on rises above its 80th. Out of sample it returns 8.8% a year against 10.8% for buy-and-hold with the same -80 maximum drawdown and 54 spells. Every vol-expansion-based rule in the grid (12 variants) is below buy-and-hold with no drawdown benefit, for the reason in section 7: the state it keys on is benign.

**The drawdown-probability family is the only one that does anything.**

| Rule | ann % | vol % | Sharpe | max DD (log) | ulcer | exposure | spells | ann 05-12 / 13-19 / 20-26 | max DD 05-12 / 13-19 / 20-26 |
|---|---|---|---|---|---|---|---|---|---|
| SPY buy and hold | 10.8 | 18.1 | 0.60 | -80.3 | 15.2 | 1.00 | 0 | 4.0 / 13.6 / 15.2 | -80.3 / -21.5 / -28.1 |
| **C: OUT P_off > 95th pct, IN < 80th (reference)** | **11.2** | 13.6 | **0.82** | **-25.0** | 7.3 | 0.87 | 14 | 9.3 / 11.0 / 13.6 | -20.6 / -22.5 / -25.0 |
| C: OUT > 90th, IN < 80th | 11.5 | 13.2 | 0.87 | -23.8 | 6.2 | 0.84 | 18 | 10.8 / 11.9 / 11.6 | -20.6 / -15.7 / -23.8 |
| C: OUT > 98th, IN < 80th | 11.4 | 14.4 | 0.79 | -28.1 | 7.8 | 0.89 | 9 | 8.5 / 11.2 / 14.8 | -20.9 / -22.5 / -28.1 |
| C: OUT > 95th, IN < 60th (slow re-entry) | 9.4 | 12.8 | 0.73 | -27.4 | 8.2 | 0.79 | 13 | 6.6 / 9.7 / 12.0 | -20.6 / -17.7 / -27.4 |
| F: OUT P_off > 95th and P_off63 > 80th, IN both < 60th | 10.4 | 12.2 | 0.85 | -21.8 | 5.8 | 0.75 | 8 | 7.5 / 10.2 / 13.7 | -20.6 / -13.0 / -21.8 |
| D: OUT P_off > 95th and P_on < median | 10.8 | 18.1 | 0.60 | -80.3 | 15.2 | 1.00 | 0 | never triggers | |

Rule D never triggers: there is no session where the drawdown probability is extreme and the rally probability is low, because they are the same state. The reference rule was chosen from this pre-specified grid after the grid was run, and its re-entry threshold was moved from the 60th to the 80th percentile after the slow version was seen to sit out 2009 and 2016 (the 60th-percentile versions are in the table too). That is one round of adaptation and it is disclosed; the whole C family at the 80th-percentile re-entry beats buy-and-hold on return and cuts the maximum drawdown by two thirds, so the result does not hang on one cell.

**What the reference rule actually is.** Yearly, book vs SPY: 2008 -8.8 vs -45.9; 2009 +19.0 vs +23.4; 2015 -7.3 vs +1.2; 2016 +7.6 vs +11.3; 2018 -5.9 vs -4.7; 2019 +21.9 vs +27.2; 2022 -17.0 vs -20.1; 2025 +5.1 vs +16.3; 2026 +11.7 vs +13.9; identical to SPY in 2010-14, 2017, 2021, 2023 and 2024. Its 14 OUT spells (SPY while out): Jul-Aug 2006 +1.5%; three in Oct-Nov 2007 (-0.1, +2.1, -1.7); **Jan 23 2008 to Jun 2 2009, 343 sessions, -29.0%**; Aug-Nov 2015 +11.3%; Mar 2016 +6.1%; Feb-Mar 2018 +7.8%; Mar-May 2018 0.0%; Oct 2018 to Jan 2019 -3.1%; Feb-Apr 2022 +8.2%; May-Jul 2022 -8.2%; Apr-May 2025 +11.2%; Jun-Jul 2026 +3.5%. Nine of the fourteen spells sat out a rally. The value of the rule is 2008 and the two 2022 legs; in the V-shaped corrections of 2015, 2018, 2024 and 2025 it sells into the low and misses the rebound, because the drawdown probability rises with realized vol, and realized vol peaks at the low. Ex-2008 the rule gives up 1.6 to 3.6 points a year against buy-and-hold and does not reduce the 2013-19 drawdown at all. A reader whose objective is to stay in should treat this as a crash-regime filter with a known cost in ordinary corrections, not as a timing model.

**The dial.** 100 minus the trailing percentile of P_off. OUT at 5 or below, IN again above 20. It is the same object as the reference rule, on the 0-100 scale the user asked for.

**ADD as its own book (`v2/addbook.py`, `results/v2_addbook_out.txt`).** The action page says ADD when the rule is IN and the rally probability is at or above its trailing 80th percentile. Scored as a book, 2006 to date, lagged daily rebalance, everything out of sample: as a sleeve that is long SPY only on ADD sessions and flat otherwise, the instantaneous condition is long 14% of IN sessions and earns 9.4 basis points a session against 5.1 for all IN sessions; held until the rally probability falls below the 60th percentile it is long 22% of IN sessions at 8.3 basis points. As a no-leverage book, 75% when IN and 100% only on ADD (held) makes 9.39% a year against 8.40% for the reserve never deployed and 11.21% for always full, with the same Sharpe (0.80 to 0.82); the 50% base gives 7.58 against 5.60 and 11.21. The ADD sessions pay about 1.7 times the average IN session, so ADD tells where a marginal dollar is best put to work; it does not beat being fully invested, and no version of it is a reason to hold a reserve. On that result ADD was taken off the instruction page, which says only HOLD or REDUCE; the rally condition stays on the evidence page as a note for new money. A version that goes above 100% on the rally condition was not run: it is a leveraged bet, not a test of this one. The 2020-included path moves every number by less than 0.2 points.

**The 2020 path (`v2/path2020.py`, `results/v2_path2020_out.txt`).** Every statistic above excludes Feb to Jul 2020 by design. Scoring the frozen rules, fits and thresholds unchanged, with the window included: the reference rule went OUT on 2020-03-02 with SPY 8.6% off its high, took 0.22 of the Feb 19 to Mar 23 decline, came back IN on 2020-06-02, and finished 2020 at +16.6% against SPY +16.8%. Its maximum drawdown is -25.0 log (-22.2% in price) on both paths; the annualized return including the window is 11.04% against 10.66% for buy and hold. The C(0.90, 0.8) variant: same exit, +10.1% in 2020. F(0.95): same exit, back IN only on 2020-08-14, +7.2% in 2020. Buy and hold: -55.2% price maximum drawdown on both paths.

## 9. Today, 2026-10-07 close

State **IN** since 2026-07-14, dial **90**. Not inside a vol episode (the last realized episode ended July 16; implied and realized episodes in 2026 ran Feb 5 to Apr 30 and Mar 31 to Apr 30).

| | today | trailing pct | what the current decile meant out of sample |
|---|---|---|---|
| P(5% drawdown in 21 sessions) | 0.083 | 10th | the bottom half of the distribution ran 6 to 25 percent, about 12 pooled, mean next month +0.5 to +1.4% |
| P(5% rally in 21 sessions) | 0.024 | 14th | 2 percent rallied 5%, 10 percent drew down 5%, mean next month +0.8% |
| P(realized vol 1.5x in 21 sessions) | 0.041 | 42nd | 10 percent expanded, mean next month +0.6% |
| P(15% decline in 63 sessions), reading only | 0.009 | 67th | 3 percent saw one (base 5), mean worst close within 63 sessions -4.3% |

Why the drawdown probability is low: the Parkinson and Garman-Klass vol of XLK, XLY and QQQ sit between the 5th and 16th percentiles of their two-year ranges, and those are the model's features. Why the rally probability is low: SPY realized vol under every estimator is at the 11th to 25th percentile and the VIX level rank at the 14th, with only healthcare's realized vol (86th) pushing the other way. Why the vol-expansion probability is middling: implied is rich against realized for IWM (91st percentile) and against SPY's Parkinson vol (80th), which pushes it up, while the Parkinson-over-close-to-close ratio (24th) and the 21-day realized ROC (72nd, counted negatively) push it down.

The tape in ranks: VIX level 0.14, VIX 5-day ROC 0.28, VVIX/VIX 5-day ROC 0.53, VIX/VIX3M 5-day ROC 0.24, MOVE level 0.85 with its 5-day ROC 0.18, ten-year yield 21-day change 0.96, SPY realized level 0.25, VIX over realized 0.53, SPY volume 0.06, volatility of volume 0.36, range compression 0.58, HYG Parkinson vol 0.84 with HYG signed volume -0.35, dispersion 0.39, average correlation 0.30 with its 5-day change 0.65, realized-vol breadth 0.07, ROC breadth 0.24, RSP/SPY 21-day 0.06 and 63-day 0.12. Multifractal: SPY h2 0.42 (18th percentile, more anti-persistent than usual), spectrum width 0.30 (75th), absolute-return exponent 0.77 (38th); HYG h2 0.39 (6th percentile, credit returns unusually mean-reverting; the screen found high HYG persistence to be the safe side), width 0.36 (92nd); VIX h2 0.33 (10th), VIX variance ratios at the 3rd and 4th percentiles (the VIX is mean-reverting hard at the 5- and 21-day scale).

## 10. Operations

`v2/run_v2.sh` runs fetch, the equal-weight fetch, the v1 build step (kept for its universe guard), then features, fractal, labels, model2, decision, tickers, big (the 15% layer, with its permutation test not rerun daily), onset, snapshot, the dashboard and the PDF; about eleven minutes when nothing else is running (662 seconds on the final verification rerun with the 15% layer; 428 without it), so the routine runs it in the background with a forty-minute cap. That verification rerun refetched every series and rebuilt everything from scratch: the out-of-sample AUCs reproduced to three decimals (P_on 0.789 became 0.788), today's reading reproduced exactly, and the reference rule's row of the grid did not move. Yahoo's adjusted closes shift at the sixth decimal between fetches, and that is enough to move the P_vol rules, which trade 40 to 116 times, by up to 0.3 points of annualized return; those rows are not stable to the data and nothing is built on them. The daily routine fires into the build session at 5:41pm New York on weekdays, runs that chain, commits `results/` and `dashboard_v2.html`, republishes the dashboard to the same artifact link, posts the spoken summary in the session and sends a one-line phone notification. The audit, the screen and the permutation test are research steps and are not rerun daily. The data guards from v1 (empty-fetch failure, universe check, stale-series report) remain.

## 11. What is not claimed

The 15% declines are the loss that matters, and section 12 shows that the ones starting from a quiet tape are not caught before they are 5 to 10% in; the dedicated 15% model does not clear its permutation null. The drawdown probability is weak: a pooled AUC of 0.64 with a year-bootstrap interval that reaches 0.54, no better than the raw VIX level rank, and its calibration in the middle deciles is not monotone. The stay-in rule's advantage over buy-and-hold is 2008 and, on the included path, March 2020; everywhere else it costs return. The equal-weight breadth, volume, volatility-of-volume, range-compression and multifractal features were all built and screened, and none of them earns a place in the drawdown model; they are on the page as tape. The rally and vol-expansion probabilities are the robust outputs of this work. The research scripts in `v2/` reproduce every number here from the raw downloads, and `results/v2_*_out.txt` hold the full tables. Nothing in v1 (`README_v1.md`) should be relied on.

## 13. VIX outrunning VVIX: the second REDUCE leg (`v2/vvsig.py`, `v2/vix_vvix.py`, `results/v2_vvsig_out.txt`)

Every time VIX's 21-day rate of change reached the top 10% of its trailing two years, split by whether the 21-day rate of change of the VIX/VVIX ratio was also in its top 10% (VIX climbing faster than the vol of vol):

| VIX 21-day ROC fires | fires | 5% SPY drop within 21 sessions | 5% rally |
|---|---|---|---|
| any day, since 2006 | | 17% | 17% |
| VIX outrunning VVIX, since 2006 | 27 | 44% | 22% |
| VVIX keeping up, since 2006 | 39 | 13% | 23% |
| VIX outrunning VVIX, since 2023 | 4 | 50% | 25% |
| VVIX keeping up, since 2023 | 8 | 12% | 12% |

The 12 drops among the 27 come from 2008, 2011, 2014, 2015 (two), 2016, 2018 (three), 2022 (two), Feb 2025 and Mar 2026. As an exit, live until VIX's 21-day rate of change is back below its median, 2020 included:

| book | 2010-26 ann / worst DD | 2018-26 | 2023-26 |
|---|---|---|---|
| SPY buy and hold | 13.4% / -33.7% | 13.7% / -33.7% | 20.2% / -18.8% |
| reference rule alone | 11.6% / -22.2% | 11.8% / -22.2% | 16.6% / -17.5% |
| VIX outrunning VVIX alone | 13.8% / -17.4% | 14.8% / -17.4% | 16.8% / -13.2% |
| instruction: reference OR VIX outrunning VVIX | 12.1% / -16.8% | 12.8% / -16.8% | 16.2% / -12.5% |

On the user's decision this became the second REDUCE leg on the action page: REDUCE when the reference rule is OUT or VIX is outrunning VVIX. The reference leg stays because the signal does not exist before 2009 and the reference rule's value is the 2008 kind of tape. Limits: the signal was found in this project's research and adopted after its record was seen, one of many cuts tried, so the split and the books are in-sample for that choice; the 90th-percentile thresholds and the median re-entry were picked from four re-entry variants (all four are in the research log); August 2024 is a miss (VVIX kept pace and SPY still fell 6%); it has been live 42 times since 2009, about two and a half a year.

## 12. The 15% question (`v2/big.py`, `v2/onset.py`, `results/v2_big_out.txt`, `results/v2_big_null_out.txt`, `results/v2_onset_out.txt`)

The loss that matters is 15%, not 5%. Stage 8 rebuilt the target around it and scored everything above on it.

**Events.** Zigzag on SPY closes: a 15% fall from the running high opens an event, a 10% rebound off the low or a new high closes it. Seventeen since 1993; eleven since 2005 excluding 2020, of which five are legs of 2007-09 (Oct 2007 -17.8, May 2008 -37.5, Oct 2008 -17.2, Nov 2008 -24.9, Jan 2009 -27.1) and six stand alone (Apr 2010 -15.7, Apr 2011 -18.6, Sep 2018 -19.3, Jan 2022 -23.0, Aug 2022 -16.7, Feb 2025 -18.8). Peak-to-trough took 10 to 114 sessions.

**Labels.** A 15% decline from today's close within 63 sessions (base rate 5.1% since 2005, 267 positive sessions spread over eight calendar years) and within 126 sessions (11.2%); sessions whose window touches Feb to Jul 2020 are masked. The existing probabilities against the 63-session label, out of sample: P_off 0.675 (0.68 / 0.71 / 0.67 by era), P_off63 0.689 but 0.51 in 2020-26, P_on 0.762 (a high rally probability is a high-vol state, which is also where 15% declines live; 0.81 in 2005-12, 0.55 to 0.58 after), P_vol inverted 0.635.

**Dedicated models**, same protocol, purge longer than the label window:

| model | AUC 05-12 | 13-19 | 20-26 | all | year-bootstrap CI |
|---|---|---|---|---|---|
| 15%/63, K15 depth-2 monotone | 0.75 | 0.82 | 0.55 | 0.724 | 0.587-0.830 |
| 15%/63, K30 depth-2 monotone | 0.82 | 0.71 | 0.56 | 0.774 | 0.594-0.887 |
| 15%/63, K8 depth-1 monotone | 0.77 | 0.77 | 0.50 | 0.742 | 0.606-0.854 |
| 15%/63, K150 depth-3 free | 0.72 | 0.22 | 0.23 | 0.610 | 0.310-0.779 |
| 15%/126, K15 depth-2 monotone | 0.57 | 0.40 | 0.45 | 0.563 | 0.358-0.734 |
| raw XLV relative strength, 63d | 0.92 | 0.93 | 0.54 | 0.818 | |
| raw SPY 63d ROC rank, inverted | 0.83 | 0.44 | 0.74 | 0.771 | |
| raw SPY realized 21d rank | 0.76 | 0.44 | 0.71 | 0.719 | |
| raw VIX level rank | 0.73 | 0.71 | 0.53 | 0.689 | |

The features the 63-session model selects (healthcare relative strength, XLK realized vol, XLK 63-day momentum and drawdown, sector drawdowns) describe a decline already underway with defensives leading. Its permutation test: null mean 0.530, standard deviation 0.115, maximum 0.780; the real 0.724 is 1.7 standard deviations above the null mean and inside the null's range. With eight calendar years of positives the walk-forward cannot establish the 15% model, and the 126-session one is at chance. The 15%/63 probability is on the page as a reading, not as a decision input.

**When each probability crossed its trailing 95th percentile**, by event (sessions after the peak, SPY decline already in place):

| peak | depth | P_off (5%/21) | 15%/63 model | P_vol |
|---|---|---|---|---|
| 2007-10-09 | -17.8 | +6s, -1.4% | +58s, -6.9% | +6s, -1.4% |
| 2008-05-19 | -37.5 | +85s, -15.6% | +79s, -13.1% | never |
| 2008-10-13, 11-04, 2009-01-06 | -17 to -27 | at the peak, already extreme | at the peak | never |
| 2010-04-23 | -15.7 | never | never | +9s, -7.3% |
| 2011-04-29 | -18.6 | never | +78s, -17.0% | +4s, -2.1% |
| 2018-09-20 | -19.3 | +24s, -9.2% | +5s, -0.5% | never |
| 2022-01-03 | -23.0 | +36s, -10.3% | +43s, -12.2% | +15s, -9.1% |
| 2022-08-16 | -16.7 | never | never | never |
| 2025-02-19 | -18.8 | +33s, -17.5% | never | +5s, -3.0% |

Two onset types. From a stressed tape (the 2008-09 legs, 2018 Q4, Aug 2022) the drawdown probabilities are already at or near their ceiling at the peak; from a quiet tape (2007, 2010, 2011, Jan 2022, 2025) they sleep at the peak (P_off at the 5th percentile of its trailing three years at the Feb 2025 peak, the dedicated model at the 26th) and only the vol-expansion probability moves early, in five of those eight cases, three of them before SPY was 5% down. P_vol alone failed as a stay-in rule (section 8) because it fires 40 to 116 times and its OUT state ends as soon as vol has expanded.

**The decision grid re-scored on the 15% events** ("taken" = share of each peak-to-trough decline the book took, 1.0 = all of it; "false" = OUT spells not followed by any 10% decline; totals are log-return points, 2006 to date):

| rule | ann % | Sharpe | max DD | spells | false | mean taken | avoided inside | given up outside |
|---|---|---|---|---|---|---|---|---|
| SPY buy and hold | 10.8 | 0.60 | -80.3 | 0 | 0 | 1.00 | 0 | 0 |
| **C: OUT P_off > 95th, IN < 80th (reference)** | 11.2 | 0.82 | -25.0 | 14 | 5 | 0.50 | 142 | 134 |
| C: OUT > 90th, IN < 80th | 11.5 | 0.87 | -23.8 | 18 | 8 | 0.43 | 157 | 144 |
| J: OUT P_off > 90th and 15%/63 > 80th, IN P_off < 80th | 11.7 | 0.88 | -23.8 | 13 | 4 | 0.44 | 155 | 136 |
| F: OUT P_off > 95th and P_off63 > 80th, IN both < 60th | 10.4 | 0.85 | -21.8 | 8 | 3 | 0.36 | 171 | 181 |
| F: OUT > 90th and P_off63 > 80th | 9.5 | 0.80 | -21.8 | 11 | 6 | 0.35 | 174 | 200 |
| E: OUT P_off63 > 90th, IN < 60th | 9.4 | 0.77 | -20.6 | 21 | 14 | 0.35 | 171 | 199 |
| G: OUT 15%/63 > 95th, IN < 80th | 11.2 | 0.80 | -35.5 | 14 | 5 | 0.53 | 127 | 118 |
| H: OUT 15%/126 > 95th, IN < 80th | 10.8 | 0.73 | -30.7 | 14 | 8 | 0.66 | 107 | 108 |

The reference rule took none of the four 2008-09 legs after the first, 0.47 of Oct 2007, 0.45 of 2018, 0.61 of Jan 2022, 0.92 of 2025, and all of 2010, 2011 and Aug 2022. Every rule that takes less of the declines gives up more outside them. The best row by return and Sharpe is J, which differs from the reference in three events (0.18 vs 0.47 of Oct 2007, 0.28 vs 0.45 of 2018, 0.49 vs 0.61 of Jan 2022) and has four fewer false spells; on eleven events that is not separable from noise, and the reference rule stays.

**Onset rules.** A vol-expansion alert (P_vol above its trailing 90th percentile within the last 42 sessions) that becomes OUT only once SPY is a given percent below its 63-session closing high, against the same price trigger with no vol filter, 133 rules in all (`results/v2_onset_out.txt`):

| price trigger, re-entry | control: ann / spells / false / taken | with vol alert: ann / spells / false / taken |
|---|---|---|
| 5%, calmed probabilities | 7.35 / 197 / 111 / 0.38 | 7.49 / 124 / 67 / 0.52 |
| 5%, new 42d high | 5.33 / 43 / 27 / 0.25 | 5.77 / 30 / 18 / 0.42 |
| 7%, calmed probabilities | 8.30 / 95 / 37 / 0.42 | 8.73 / 57 / 21 / 0.54 |
| 7%, new 42d high | 6.84 / 28 / 14 / 0.30 | 8.13 / 17 / 8 / 0.44 |
| 10%, calmed probabilities | 9.36 / 43 / 9 / 0.48 | 10.07 / 25 / 4 / 0.62 |
| 10%, new 42d high | 10.05 / 13 / 1 / 0.42 | 10.68 / 7 / 0 / 0.58 |

At a 5% trigger every rule, filtered or not, loses 3 to 5.5 points a year to buy-and-hold with 30 to 200 spells. The vol filter cuts the spell count by a third to a half and adds 0.1 to 1.3 points a year at every trigger depth, which is the one place the vol complex adds information beyond price: it separates, modestly, the dips that become 15% declines from the ones that do not. Nothing in the grid beats the reference rule, and the only rules that match buy-and-hold are the 10% triggers, which by construction are out after the first 10%.

**What this means.** In this data a 15% decline that starts from a quiet tape is not distinguishable at its peak, or at 5% down, from the five-to-seven percent dips since 2006 that recovered. The vol complex confirms at 1 to 9% down in five of eight such cases, and it also confirms most dips that end there. The reference rule's record on 15% declines is all of 2008-09 after the first leg, about half of Oct 2007, 2018 Q4 and Jan 2022, and none of 2010, 2011, Aug 2022 or 2025. No construct tested here improves on that without paying more outside the events than it saves inside.
