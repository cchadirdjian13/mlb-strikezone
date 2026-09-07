# mlb-strikezone

How the called strike zone actually moves — measured from 3.7 million taken
pitches, 2015–2025.

Every pitch a batter does not swing at is a decision by the home-plate umpire,
recorded to the inch by Statcast. That makes the strike zone one of the few
places in sport where you can measure the gap between a written rule and how it
is enforced. This repo measures that gap.

## Finding

**The zone an umpire calls on 3-0 is 36% larger than the one they call on 0-2**
— 3.78 sq ft against 2.78 sq ft.

The effect is not symmetric. Starting from 0-0 (3.52 sq ft):

| moving from 0-0 to | zone area | change |
| --- | --- | --- |
| 3-0 (three balls) | 3.78 sq ft | +7.4% |
| 0-2 (two strikes) | 2.78 sq ft | −21.0% |

Two strikes take away nearly three times as much zone as three balls add. The
umpire is not simply "evening up the count" — the pull toward not ringing a
hitter up on a borderline pitch is much stronger than the pull toward not
walking them.

![Zone area by count](figures/zone_area_by_count.png)

Plotted as contours, the two extreme counts barely overlap at the edges. The
0-2 zone sits inside the rulebook box on every side; the 3-0 zone sits outside
it on every side.

![3-0 vs 0-2 zone](figures/zone_contour_3-0_vs_0-2.png)

Two secondary results:

- **Handedness.** The zone is 3.36 sq ft for right-handed batters against
  3.27 for left-handed ones, and both extend past the outside edge of the plate
  while stopping short on the inside.

  ![Called-strike rate by handedness](figures/called_strike_rate_by_handedness.png)

- **Drift.** The zone has shrunk 5.0% since 2015, from 3.41 to 3.24 sq ft, with
  2025 the smallest of the eleven seasons. Note the y-axis on this chart spans
  about a tenth of a square foot — the trend is consistent, but small next to
  the count effect above.

  ![Zone area by season](figures/zone_area_by_season.png)

## How the zone is measured

There is no single "the strike zone", so this uses the standard empirical one:
lay a 0.1 ft grid over the plate, and call a cell part of the zone when at
least half the pitches taken there were called strikes. Zone area is the number
of qualifying cells. Everything is computed on taken pitches only
(`called_strike` and `ball`); swings, hit-by-pitches and pitchouts are dropped.

Two choices worth stating plainly:

- **Heights are absolute, not per-batter.** Area is measured in real feet above
  the plate, pooling all batters, so a tall hitter's zone and a short hitter's
  zone are averaged together. The dashed box on the charts is the *mean*
  rulebook zone (2.57 sq ft, from a mean `sz_bot` of 1.60 ft and `sz_top` of
  3.41 ft) and is a visual reference only — the measured zone is wider than it
  partly because pooling batter heights stretches the vertical extent. Compare
  counts to each other, not to the dashed box.
- **Cells are pooled with their neighbours** over a 0.3 ft window before the
  rate is taken. This matters more than it sounds: hitters swing at anything
  near the middle on two strikes, so the heart of the 0-2 zone holds only a
  handful of *taken* pitches per cell. Without pooling, those cells fall under
  the sample floor and blow a hole in the middle of the measured zone, which
  understates two-strike area and inflates the headline number — this analysis
  reported 42.7% before the fix and 36.0% after.

## Modelling the call

Ranking umpires means asking whether a call differs from what the pitch itself
predicted, so the zone has to be modelled before anyone is blamed for it. Two
models, both cross-validated in five folds **grouped on `game_pk`**: pitches
within a game share an umpire, a park and a day's conditions, so splitting on
pitches would leak all of that across the fold boundary and flatter the result.
Every pitch gets a probability from a model that never saw its game.

| | log loss | Brier | AUC |
| --- | --- | --- | --- |
| always predict the base rate | 0.6350 | — | — |
| logistic, location + count + handedness | 0.1869 | 0.0581 | 0.9761 |
| gradient boosting, + pitch type and velocity | 0.1716 | 0.0532 | 0.9798 |

Those headline numbers flatter both models, because two thirds of taken pitches
are obvious. Restricting to the **contested band** — pitches the model itself
puts between 0.1 and 0.9 — is the honest test, and there the two are close:
0.5587 against 0.5503 log loss, over roughly 900,000 pitches, about a quarter
of the data. Neither model discriminates well on a coin flip, which is the
point: those are the calls that are genuinely up to the umpire.

Where they differ is calibration, and it is not a small difference.

![Calibration error](figures/calibration.png)

The logistic baseline is systematically overconfident straight through the
contested band, peaking at **3.6 percentage points** too high around a
predicted 0.62. The gradient booster stays within 1.2 points across the same
range. That gap is the whole reason attribution uses the boosted model's
residuals: a 3.6-point bias sitting in the middle of the contested band is
larger than the umpire effects being measured, and would be silently
redistributed onto individual umpires as if it were their behaviour.

Note the chart plots *error*, observed minus predicted, rather than observed
against predicted. On a conventional diagonal calibration plot both models look
perfect — the first version of this chart used equal-count bins and put six of
ten points below 0.03, which hid the bow entirely.

## Reproducing

Python 3.12 via [uv](https://docs.astral.sh/uv/). Data is pulled from Baseball
Savant and is not committed.

```bash
uv sync
uv run python src/mlb_strikezone/ingest.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
uv run python src/mlb_strikezone/features.py
uv run python src/mlb_strikezone/umpires.py
uv run python src/mlb_strikezone/analysis.py
uv run python src/mlb_strikezone/model.py
```

The ingest step is the slow one — it pulls a month at a time and caches one
parquet per season under `data/raw/`, skipping any season already present.
`features.py` builds `data/processed/called_pitches.parquet` (3.7M rows).
`umpires.py` is quick — one MLB Stats API call per season gets the whole
umpiring crew, so 11 requests cover 25,193 games. `analysis.py` prints the
tables above and writes the four zone figures.

`model.py` is the other slow one — ten model fits over 3.7M pitches. It writes
`data/processed/predictions.parquet` and then reports from that file, so a
rerun reuses the saved predictions and only re-derives the scores and the
calibration chart. Pass `--refit` to actually fit again.

Each module self-tests without touching the data:

```bash
uv run python src/mlb_strikezone/ingest.py --check
uv run python src/mlb_strikezone/features.py --check
uv run python src/mlb_strikezone/umpires.py --check
uv run python src/mlb_strikezone/analysis.py --check
uv run python src/mlb_strikezone/model.py --check
```

## Limitations

- The 50% grid zone is descriptive, not a model. It cannot separate the count
  effect from anything correlated with count — pitch type, velocity, and where
  catchers set up all shift with the count, and none of that is controlled for
  here.
- 2020 is a 60-game season and is roughly a third the sample of the others.
- No umpire attribution yet. The plate umpire for all 25,193 games is joined,
  the model is fitted, and the out-of-fold residuals exist — but nobody is
  ranked here.
- The model knows where the pitch was and what it was, not who was behind the
  plate or who caught it. That is deliberate: those are the effects to be
  measured, so they must stay out of the prediction.

## Next

Attribute the residual (actual call − predicted strike probability) to
individual umpires and catchers, **fitting both jointly**. A given catcher's
framing shows up in the residual of every umpire he works with, and vice versa;
ranking them from separate fits would count the same edge calls twice and
credit them to both. One fit with both sets of effects competing for the same
variance is the only way the numbers mean what they claim to.

129 of 138 umpires clear the 2,000 called-pitch minimum, and the contested band
alone holds roughly 890,000 pitches, so the sample is there.
