# Steady-alpha research track (MSI-2024 type)

Separate from THE LIST; nothing here feeds the production model.

- Outcome, per horizon h (21/42/63): next-h-session Sharpe in the universe's top 20% AND max drawdown in the shallowest 30% AND return beats SPY.
- Population: low- and mid-beta terciles only.
- Conditions: production quintile conditions (universe-ranked) plus the same factors ranked within the low+mid-beta group (`w_` prefix). 213 in all.
- Windows: discovery 2015-2020 (COVID Feb-Jun 2020 excluded), confirmation 2021-2023, untouched holdout 2024 onward.
- Rules as production: discovery n >= 300 and lift >= 1.30, triples must add +0.10 lift; confirmation n >= 150, lift >= 1.20, p_conf >= 0.8 x p_disc.
- Control: a within-date shuffled target run through the identical pipeline discovered 0 signatures (real target: 5,097 discovered, 2,887 confirmed).

Scripts were run from a scratch directory; paths at the top of each file point there and need adjusting to rerun.
