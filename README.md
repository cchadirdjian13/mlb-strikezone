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

## Who moves the zone

With a calibrated probability per pitch, the residual — actual call minus
predicted probability — is what location cannot explain. Umpires and catchers
are fitted to that residual **together**, in one ridge regression with both sets
of effects side by side. Fitting them separately would double-count: a
catcher's framing sits in the residual of every umpire he works with, and every
umpire's tendency sits in the residual of every catcher. The ridge penalty is
set at 2,000, which in this design reads in pitches — someone with 2,000 called
pitches keeps about half their raw residual, which is also the sample floor
below which a number isn't worth believing.

Effects are in called strikes per 100 taken pitches, net of the other party.

| umpire | pitches | per 100 | SE | extra strikes |
| --- | --- | --- | --- | --- |
| Doug Eddings | 45,238 | **+3.06** | 0.16 | +1,385 |
| Bill Miller | 47,280 | +2.42 | 0.15 | +1,144 |
| Lance Barrett | 45,452 | +1.88 | 0.17 | +855 |
| … | | | | |
| Mark Wegner | 41,910 | −1.64 | 0.14 | −689 |
| Alfonso Márquez | 46,604 | −1.60 | 0.12 | −747 |
| Tom Woodring | 13,354 | **−1.78** | 0.22 | −237 |

| catcher | pitches | per 100 | SE | extra strikes |
| --- | --- | --- | --- | --- |
| Tyler Flowers | 31,597 | **+2.59** | 0.16 | +820 |
| Yasmani Grandal | 62,725 | +2.08 | 0.10 | +1,307 |
| Austin Hedges | 49,954 | +1.98 | 0.11 | +987 |
| … | | | | |
| Isiah Kiner-Falefa | 5,065 | −2.04 | 0.28 | −103 |
| Edgar Quero | 4,807 | −2.14 | 0.28 | −103 |
| Ramón Cabrera | 3,670 | **−2.46** | 0.32 | −90 |

**Catchers move the zone about as much as umpires do.** The umpire spread across
129 qualifiers is 4.8 calls per 100; the catcher spread across 205 is 5.1. Who
is catching is worth roughly as much as who is calling.

The catcher list is also the closest thing here to external validation. Nothing
in this pipeline knows what pitch framing is — the model sees only location,
count, handedness, pitch type and velocity, and the residual is attributed
blind. It independently returns Flowers, Grandal, Hedges, Mathis, Barnes,
Trevino and Posey at the top, which is essentially the framing leaderboard the
public metrics have been publishing for a decade.

### How much of this ranking is real

Standard errors come from a bootstrap that resamples **whole games**, 200
replicates. The unit matters: within a game the umpire is fixed and the calls
share a park, a day and a zone, so resampling individual pitches would treat
thousands of correlated calls as independent draws and report errors several
times too small.

The median standard error is 0.17 calls per 100 for umpires and 0.20 for
catchers. That is small against the extremes and not against the middle.

![Leaderboard with two-standard-error bars](figures/attribution_caterpillar.png)

Two things follow, and both are constraints on how the table above should be
read:

- **Only 80 of 129 umpires, and 127 of 205 catchers, are more than two standard
  errors from zero.** The rest of each list is indistinguishable from having no
  effect at all. Every name shown in the tables above clears that bar
  comfortably; the middle of the leaderboard does not.
- **Adjacent ranks are not real.** Separating two people needs roughly 0.5 calls
  per 100 between them. Doug Eddings really is above Bill Miller, but Nick
  Mahrley at +1.40 and Mike Estabrook at +1.38 are one person in two rows, and
  no amount of ranking them will change that.

These are the standard errors of the shrunken estimate — they describe how much
the ridge coefficient would move under resampling, not the full uncertainty
about a person's true effect, which also carries the shrinkage bias.

### The joint fit changed less than expected

Worth stating plainly, because it argues against the design decision that
produced it. Comparing the joint estimates against fitting each role *alone
with identical shrinkage* — which isolates the partner control from the
regularisation — the difference is about **0.05 calls per 100**, against effects
that range over ±2.5.

![Joint against solo attribution](figures/joint_vs_marginal_attribution.png)

Both roles land on the diagonal. The confounding is real but small in this
sample, because over eleven seasons umpires work with many catchers and
catchers with many umpires, so partners largely average out. Ridge shrinkage
moves the numbers three to seven times further than the joint fit does (0.16
per 100 for umpires, 0.36 for catchers).

That is a reason to trust the leaderboard, not a reason to skip the joint fit:
the correction being small is a finding, and it isn't one you can assert
without doing the fit. It would not stay small for a single season, or for a
catcher who caught for one crew.

### Catchers converged. Umpires did not.

The leaderboard above pools eleven seasons into one number per person, which
hides whether the pool is changing. Refitting one season at a time answers that,
with two adjustments the result depends on entirely:

- **A far lighter ridge penalty, 100 rather than 2,000.** Shrinkage is roughly
  `n / (n + alpha)`, and a season holds an eighth of an umpire's pooled
  workload — 2020 a twenty-fifth. The pooled penalty would shrink short seasons
  hardest and manufacture a convergence trend out of nothing but sample size.
  At 100 the shrinkage stays under 5% in every season.
- **The spread is corrected for estimation noise.** The observed standard
  deviation of a season's estimates carries their own standard errors on top of
  the real variation, and carries more of them when there is less data. The
  reported spread subtracts the mean squared error. Without this 2020 is the
  *most* variable umpire season in the sample (raw 1.29, corrected 1.11) purely
  because it is the shortest.

![Spread between people by season](figures/spread_by_season.png)

Each season's spread carries a nested bootstrap interval, drawn as a band above:
the outer loop resamples games, and because the spread is a variance with the
estimation noise subtracted, every outer replicate has to re-estimate that noise
from its own resampled games. Seasons are then weighted by their precision when
the trend is fitted, so 2020's wide band does not pull the line like a full
season's narrow one.

| | 2015 | 2025 | slope per season |
| --- | --- | --- | --- |
| catchers | 1.27 | 0.80 | **−0.052 ± 0.010** |
| umpires | 1.16 | 0.85 | −0.014 ± 0.008 |

**The catcher spread has collapsed by about a third, and the trend is five
standard errors from flat.** In 2017 the gap between a good and a bad framer was
half again what it is now. That is consistent with framing becoming a known and
priced skill over this period — teams that can measure it stop employing
catchers who are bad at it — though this data can show only the compression, not
the cause.

**The umpire spread is not established as moving.** The end points tempt a
different story: 1.16 down to 0.85 reads as a 27% decline. But the slope is
−0.014 ± 0.008, and the middle of the series wobbles between 0.94 and 1.06 with
no direction — 2023 and 2024 are both *higher* than 2017, and their bands
overlap almost everything.

That is 1.9 standard errors, which is short of the conventional bar but close
enough that "umpires are not converging" would be overclaiming in the other
direction. The honest reading is that eleven seasons cannot separate a slow
umpire convergence from none at all, while the same eleven seasons settle the
catcher question five times over. Weighting the seasons by their bootstrap
precision moved the slope from −0.0134 to −0.0142 and changed no conclusion,
which is its own small result: the season estimates are precise enough that it
did not matter.

## Poking at it yourself

Everything above is a fixed view of the data. `app.py` is the interactive one —
filter the leaderboard by role and sample floor, pull up any individual's season
by season history, and redraw the zone for any count and batter side.

![The Streamlit view](figures/streamlit_app.png)

```bash
uv run streamlit run src/mlb_strikezone/app.py
```

It reads only the generated parquet files, and says which pipeline step to run
if one is missing rather than failing on an empty directory.

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
uv run python src/mlb_strikezone/attribution.py
uv run python src/mlb_strikezone/drift.py
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

`attribution.py` reads those predictions back, fits the joint leaderboard and
writes `data/processed/attribution.parquet`. The ridge itself takes seconds —
it is sparse, two non-zeros per row — and the 200 bootstrap replicates behind
the standard errors take about three minutes.

Each module self-tests without touching the data:

```bash
uv run python src/mlb_strikezone/ingest.py --check
uv run python src/mlb_strikezone/features.py --check
uv run python src/mlb_strikezone/umpires.py --check
uv run python src/mlb_strikezone/analysis.py --check
uv run python src/mlb_strikezone/model.py --check
uv run python src/mlb_strikezone/attribution.py --check
uv run python src/mlb_strikezone/drift.py --check
uv run python src/mlb_strikezone/app.py --check
```

## Limitations

- The 50% grid zone is descriptive, not a model. It cannot separate the count
  effect from anything correlated with count — pitch type, velocity, and where
  catchers set up all shift with the count, and none of that is controlled for
  here.
- 2020 is a 60-game season and is roughly a third the sample of the others.
- The model knows where the pitch was and what it was, not who was behind the
  plate or who caught it. That is deliberate: those are the effects to be
  measured, so they must stay out of the prediction.
- Attribution is a linear fit on residuals, not a logistic one with an offset.
  The residual is heteroscedastic, so the estimates are interpretable but not
  efficient. The standard errors are bootstrapped rather than read off the fit,
  so they do not inherit that inefficiency, but they describe the shrunken
  estimate and not the shrinkage bias.
- A catcher's effect absorbs anything correlated with him that the model does
  not see — his pitching staff's command, his team's park, the pitch mix he
  calls. It is a catcher-shaped residual, not a measurement of framing skill in
  isolation.
- Umpire and catcher are separable only because crews and catchers cross over
  across eleven seasons. Over one season, or for a catcher who caught for a
  single crew, they would not be.

## Next

The pipeline is complete end to end: ingest, called-pitch table, umpires,
descriptive zone, model, attribution. What would sharpen it, roughly in order
of value per unit of work:

- **An unbiased interval on the season spreads.** The nested bootstrap's draws
  sit about 0.11 above the statistic — an artefact of resampling twice, since an
  inner bootstrap drawn from already-resampled games understates that
  replicate's noise. Only the width is used, so the intervals are symmetric by
  construction. A bias-corrected accelerated bootstrap, or an analytic
  cluster-robust standard error in place of the inner loop, would give a
  properly asymmetric one.
