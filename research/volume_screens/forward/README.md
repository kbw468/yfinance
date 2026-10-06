# Forward score (price, volume, volatility only)

`python fwd_rate.py TICKER` returns two 1-10 scores. **forward63** is the primary score and uses a ridge + LightGBM blend.
**forward21** uses ridge only. A score is the percentile of the model's predicted forward risk-adjusted return within
the reference universe (2,084 tradable US names, price >= $3, ADV >= $1M), mapped to 1-10.

## Training
- Universe: both finviz lists pooled (431 large caps + ~1,880 small/mid caps), weekly dates 2020-10 to 2026-07.
- Features: 70 price/volume/volatility measures (`fwdfeat.py`), cross-sectionally ranked each date.
- Target: rank of forward 21d / 63d return divided by forward realised vol.
- Walk-forward: expanding window, refit every 6 weeks, training labels embargoed past the scoring date. OOS from 2022-05.

## Out-of-sample results (`oos_evaluation.csv`)
| horizon | model | universe | IC | NW t | top decile Sharpe | universe Sharpe |
|---|---|---|---|---|---|---|
| 63d | blend | all | 0.107 | 5.5 | 0.234 | 0.099 |
| 63d | blend | small/mid | 0.092 | 4.2 | 0.171 | 0.059 |
| 63d | blend | large | 0.034 | 1.8 | 0.275 | 0.227 |
| 21d | ridge | all | 0.062 | 3.9 | 0.124 | 0.051 |
| 21d | ridge | large | 0.001 | 0.0 | 0.121 | 0.139 |

IC was positive in every calendar year. Most of the spread comes from the bottom decile: high-vol, wide-range,
lottery-type names. The top decile beats the universe by about 7%/yr in raw return.
The 21d score carries no large-cap signal, and the 63d score is weak for large caps.

## Heaviest weights (ridge, 63d)
Negative: range-based spread, 21d/63d/252d realised vol, idiosyncratic vol, price level, overnight return drift, ADV, worst-day size.
Positive: overnight variance share, smallest worst day, distance above 52w low, proximity to 52w high, cc/Parkinson, beta, turnover.

## Maintenance
`refresh_ref.py` rebuilds the reference cross-section from Nasdaq. Refit the model by rerunning `extra.py`, `model.py`, `final_fit.py`
against fresh panels.
