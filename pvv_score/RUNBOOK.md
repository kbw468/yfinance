# THE LIST: runbook

## What runs when

| | Script | Schedule | What it does | Time |
|---|---|---|---|---|
| Nightly | `pvv_score/bin/nightly.sh` | weekdays after the close (22:35 UTC) | download prices and vol indices, build the recent-window factor rows, score every name from the **frozen** model, update history / movers / scorecard, render page and PDFs, run the hard checks, commit and push | ~15 min |
| Weekly refit | `pvv_score/bin/refit.sh` | Saturday | rebuild the research table from 2010, re-estimate both composites walk-forward, rediscover (2015–2020) and reconfirm (2021+) signatures, recalibrate every curve, **freeze** the model, score from it, verify the frozen scorer reproduces the full chain's list exactly, history, page, PDFs, checks, commit and push | ~1.5 h |

The model is frozen between refits on purpose: a name moving up or down the list between two sessions is then tape, not refit noise. When a refit lands, `MOVERS.txt` says so on the first line and the comparison against the previous session carries that caveat.

A routine fires a fresh container with no checkout and only the base image: the routine first attaches the repository with the
session's `add_repo` tool (plain `git clone` is refused by the git proxy until then), clones, checks out the branch, and the
script installs pyarrow, scikit-learn, scipy, lightgbm and yfinance's own dependencies if they are missing (about two
minutes). Pre-installing them in the environment's setup script removes that step.

Cache lives in `PVV_CACHE` (default `pvv_score/.cache`, not committed). A fresh container rebuilds it from the download in a few minutes.

## Reproducibility and the one source of jitter

On a fixed price panel the frozen scorer reproduces the full chain's list exactly (probabilities, tiers, basis, signature
counts, rank order) and a fresh container reproduces it from an empty cache. Across two downloads, Yahoo's **adjusted**
closes can differ in the fourth decimal (max 0.0005, relative 2e-6; raw closes are identical). Continuous factors absorb
that at 1e-5. Sign-based factors (up-day shares, streaks, higher-low cadence, on-balance volume) can flip on a handful
of knife-edge days, which moves their three-year norms by a few hundredths and the composite percentile by up to about
one point on a few names. Measured on 2026-10-06 between a 04:15 and a 05:05 download: 0 tier changes, 0 signature-count
changes, 241 rank changes inside tiers, largest probability change 0.4 points. The gates do not treat this as a failure;
it is the data source, not the code, and it is an order of magnitude below a normal session's tape move.

## Schedules

| Routine | Cron (UTC) | Fires |
|---|---|---|
| THE LIST nightly | `35 22 * * 1-5` | a fresh session runs `nightly.sh`, publishes the page, sends the three PDFs, posts a summary |
| THE LIST weekly refit | `48 12 * * 6` | a fresh session runs `refit.sh`, publishes, sends, summarises the model change |

## The frozen model (`results/model/`)

| File | Contents |
|---|---|
| `weights.json` | current test fold's beta-bucket weights, state layer and level layer |
| `signatures.csv` | confirmed three-condition signatures with confirmation probability and lift |
| `calib_composite.json` | composite probability curves per beta bucket (every step ≥ 500 cases) |
| `calib_signature.json` | signature probability curves per composite quintile (every step ≥ 500 cases) |
| `model.json` | provenance: fitted-through date, fold year, counts, sha256 id; the id is printed on the list and stored with every snapshot |

## Outputs (`results/`)

* `THE_LIST.csv / .txt / .html / .pdf`: every ranked name, probability descending. `Δrk 1d`, `ΔP 1d`, `ΔP 5d` are the change in rank and probability since the previous session and five sessions back.
* `THE_LIST_by_mktcap.pdf / .csv`: same list sorted by market cap with cap weight across the universe and inside each tier.
* `MOVERS.txt / .pdf`: entries and exits of Tiers 1–2, biggest probability gains and drops over one and five sessions, list-level change (tier counts, breadth, SPY, VIX, VXN, MOVE, IWM realised vol), and the realised scorecard.
* `history/snapshots/THE_LIST_<date>.csv`: the list as published each session. `history/list_history.csv` is the same as one long table; `history/list_metrics.csv` is one row per session.
* `history/scorecard.csv`: for every snapshot with 42 / 63 sessions elapsed, realised top-quartile rate by printed tier and probability band against what was printed. Starts reporting once the first snapshot is 42 sessions old.
* `signature_by_composite_oos.csv`, `signature_depth_oos.csv`, `buylist_probability_table.csv`, `falsification_summary.csv`: the realised tables behind the probabilities.

## The gate (`python -m pvv_score.checks [--frozen]`)

Exits non-zero, and the chain stops before committing, if any of these fail: every ranked name has a composite on the calibration scale; tonight's signature firing recomputed from the raw factor rows matches the list for every name; P equals the max of its three routes; tiers follow the P bands; rank order is P descending; beta terciles follow the historical rule; every ranked name closed on the as-of date; no stub, zero-volume or glitch bars; vol indices are fresh; page, PDFs and CSV were built from the same list; every frozen curve rests on ≥ 500 cases per step; the list carries the frozen model's id.

## One name: `python -m pvv_score.why TICKER`

Rank, tier, probabilities, composite percentile, which quintile of the universe each factor sits in, which ROC thresholds are met, the confirmed signatures firing and the strongest ones one condition away.

## Manual run

```
export PVV_TARGET=smooth_42
bash pvv_score/bin/nightly.sh        # score tonight from the frozen model
bash pvv_score/bin/refit.sh          # full refit (weekly)
```
