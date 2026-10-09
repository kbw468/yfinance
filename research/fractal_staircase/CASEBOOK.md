# Casebook

Runs you flagged by name, what each looked like going in, and whether that profile
earns a place in the scan. A profile is added only if it beats what the scan already
uses (the quiet base: lift 1.33 all names, 1.31 at $50B+), out of sample, in both
halves of the history. Test new cases with `case_study.py TICKER` and `archetypes.py`.

Lift = regime-neutral lift on the 63-session staircase hit (1.00 = same-week universe).
"$50B+" uses today's market cap, so it leans toward names that went on to win.

| Case | Window | The run | Study signals going in | Archetype tested | Lift all names (2006–16 / 2017–26) | Lift $50B+ (cohort 1.23) | Verdict |
|---|---|---|---|---|---|---|---|
| TD | Dec 2024 – Jul 2026 | +155% in 390 sessions, worst pullback 8.9% | Vol bottom 0–5% all along; quiet base 2/4 rungs (6-month return too strong); legs launched from 63-day ranges in the tightest 0.4–4.5% near highs | Low-vol leader re-basing: tight range, top-quintile 6-month return, near high, low vol | 0.98 (0.96 / 1.00) | 1.06 | Not added |
| VLO | Apr – Oct 2026 | +87% in 127 sessions, worst pullback 9.6%, median daily move 1.6% | Vol mid-to-upper; quiet base 0/4; 6-month return top decile | Mid/high-vol momentum at 52-week highs | 0.85 (0.76 / 0.93) | 1.07 | Not added |
| JNJ | Aug 2025 – Feb 2026 | +51% in 144 sessions, worst pullback 4.6% | 63-day range tightest 1–7%, vol bottom 2–7%, 6-month drawdown bottom 5–8%, 6–9% below high; quiet base 2/4 (6-month return upper half) | Tight, very quiet base near highs, 6-month return upper half | 0.98 (1.00 / 0.97) | 1.11 | Not added |
| MSI (prototype; lifts on the 126-session target) | Oct 2023 – Nov 2024 | +87% in 260 sessions, worst pullback 5.9%; best 126 sessions (Apr 18 → Oct 16 2024) +41% with a 3.8% worst pullback, top 0.1% of all stock-weeks by gain over worst pullback | Base Jun–Oct 2023 ($263–$288); vol bottom 1–5%, 6-month drawdown bottom 1–4%, 63-day range tightest 0.5–3%; quiet base 2/4 (6-month return 50th–70th pct) | Scored on its own target: +25% over 126 sessions with worst pullback ≤ 6% (0.72% of stock-weeks) | Quiet base 2.70 (1.99 / 3.87); JNJ/MSI base 2.51 (1.94 / 3.42); TD re-base 3.05 (2.60 / 3.58) | Watch 2.85 (2.11 / 4.74) | Added as the prototype watch |
| TGT | Mar – Sep 2026 | +41% (+53% to the Aug high) in 147 sessions, but worst pullback 10.7% and ten dips over 5%; 0 of 91 63-session windows met the hit definition | V-recovery from −44% (Nov 2025 low); old highs rolled out of the 52-week window; vol mid; quiet base 0/4 | Bounce of 35%+ off a 35%+ fall within 126 sessions, back near the 52-week high | 0.93 (0.75 / 1.01) | 1.44 (0.95 / 1.56) | Not added: fails all names, unstable at $50B+, and the today's-cap cohort favours recoveries that worked |

Prototype watch (added after MSI)
- Target: +25% or more over 126 sessions with no pullback deeper than 6%. MSI-grade runs are rare
  (0.72% of stock-weeks) and almost never come from loud stocks: the quietest volatility decile carries
  2.6× the odds, the loudest 0.01×.
- On this longer target, the TD and JNJ/MSI profiles work as well as the quiet base, even though they
  add nothing to the 63-session hit. Watch = quiet base or JNJ/MSI base or TD re-base, $2B+, no
  regional banks, not pinned (volatility and 63-day range both in the bottom 1%, typical of a pending
  deal). Lift 2.85 (2006–16: 2.11, 2017–26: 4.74), hit 2.3% vs 0.72%, above 1 in 17 of 20 years.
- Each filter was checked in both halves: below $2B 1.03 / 0.49; regional banks 1.02 / 0.73;
  pinned 1.30 / 1.12.

Notes for recognising these later
- On the 63-session hit, TD-type and JNJ-type are the quiet base without the lagging 6-month
  return, and dropping that rung costs most of the edge (1.33 → 1.10–1.13). On the MSI-grade
  126-session target they hold up (2.5–3.1), which is why they are in the prototype watch.
- VLO-type is the profile the study scores lowest: strong momentum with ordinary volatility.
- TGT-type recoveries trend hard but rarely hold drawdowns under a third of the gain.
