# Power BI model

A star schema over the analysis output, built by `export.py`. The tables in
`tables/` are CSV, small enough to import directly — no gateway, no connector,
no scheduled refresh. They are named `tables/` rather than `data/` because
`.gitignore` ignores `data/` at any depth, and these have to be committed for
the model to be openable from a clone.

Rebuild them with:

```bash
uv run python src/mlb_strikezone/export.py
```

## Schema

Three dimensions and six facts. Everything joins on a single column.

| table | grain | rows |
| --- | --- | --- |
| `dim_person` | one umpire, catcher or pitcher | 3,038 |
| `dim_season` | one season, with a short-season flag | 11 |
| `dim_count` | one count, plus an `Any` member | 13 |
| `fact_effects` | pooled effect per person | 3,038 |
| `fact_season_effects` | person × season | 11,183 |
| `fact_spread` | season × role, with interval | 33 |
| `fact_zone_area` | count × batter side | 39 |
| `fact_zone_grid` | count × side × grid cell | 54,424 |
| `fact_calibration` | model × probability band | 40 |

`person_key` is `role-id` rather than `id`, because an umpire and a catcher can
share an id and joining on the raw id would fan out every measure built on it.

`dim_count` includes an `Any` row on purpose. The zone facts hold all-counts
rows, and without a matching dimension member they would join to a blank and
silently drop out of any count slicer. Filter with `specific_count = TRUE` when
a visual should show the twelve real counts only.

## Loading

1. **Home → Get data → Text/CSV**, and load all nine files from `tables/`.
2. In each query, check Power Query typed the columns sensibly — `season` as
   Whole Number, the `_per_100` and rate columns as Decimal Number, the
   `clears_zero` / `qualified` / `short_season` / `specific_count` columns as
   True/False.
3. **Close & Apply**.

## Relationships

Create these in **Model view**. All are one-to-many, single direction, from the
dimension to the fact.

| from | to |
| --- | --- |
| `dim_person[person_key]` | `fact_effects[person_key]` |
| `dim_person[person_key]` | `fact_season_effects[person_key]` |
| `dim_season[season]` | `fact_season_effects[season]` |
| `dim_season[season]` | `fact_spread[season]` |
| `dim_count[count]` | `fact_zone_area[count]` |
| `dim_count[count]` | `fact_zone_grid[count]` |

Power BI will offer to autodetect. Check what it made rather than trusting it —
it sometimes guesses a relationship on `role`, which is not unique anywhere and
will quietly multiply your measures.

## Measures

Put these in `fact_effects` unless noted.

```dax
Effect per 100 = AVERAGE ( fact_effects[effect_per_100] )

Extra strikes = SUM ( fact_effects[extra_strikes] )

Qualifiers = CALCULATE ( COUNTROWS ( fact_effects ), fact_effects[qualified] = TRUE () )

Clear of zero =
CALCULATE (
    COUNTROWS ( fact_effects ),
    fact_effects[qualified] = TRUE (),
    fact_effects[clears_zero] = TRUE ()
)

Share clear of zero = DIVIDE ( [Clear of zero], [Qualifiers] )

Spread of effects =
CALCULATE (
    STDEV.S ( fact_effects[effect_per_100] ),
    fact_effects[qualified] = TRUE ()
)
```

In `fact_zone_area`:

```dax
Zone area = AVERAGE ( fact_zone_area[area_sq_ft] )

Zone area vs 0-0 =
VAR Baseline =
    CALCULATE ( [Zone area], ALL ( dim_count ), dim_count[count] = "0-0" )
RETURN
    DIVIDE ( [Zone area] - Baseline, Baseline )
```

`Zone area vs 0-0` is the headline finding as a measure: it reads +7.4% on 3-0
and −21.0% on 0-2. Format it as a percentage.

In `fact_spread` and `fact_calibration`:

```dax
Spread = AVERAGE ( fact_spread[spread] )
Spread low = AVERAGE ( fact_spread[lo] )
Spread high = AVERAGE ( fact_spread[hi] )

Calibration gap = AVERAGE ( fact_calibration[gap] )
```

## Pages

**1 — The count effect.** Clustered column chart, axis `dim_count[count]`
filtered to `specific_count = TRUE`, value `Zone area vs 0-0`, sorted by
`dim_count[balls]` then `[strikes]`. Add a card for `Zone area` and a slicer on
`fact_zone_area[stand]`. This is the finding, so it goes first.

**2 — Leaderboard.** Matrix with `dim_person[name]` on rows and
`Effect per 100`, `se_per_100`, `Extra strikes` as values. Slicer on
`dim_person[role]`, and a numeric range slicer on `dim_person[career_pitches]`
defaulted to 2,000. Conditional-format the effect column on a diverging colour
scale centred at zero. Add cards for `Qualifiers` and `Share clear of zero` —
that second number is the honest one, and it is the reason the matrix should not
be read top to bottom as a ranking.

**3 — Convergence.** Line chart, axis `dim_season[season]`, values `Spread`,
`Spread low`, `Spread high`, legend `fact_spread[role]`. Filter out
`short_season = TRUE`, or leave it in and mark it — 2020 is a 60-game season and
its interval is roughly twice as wide.

**4 — The zone.** Matrix with `fact_zone_grid[plate_z]` on rows descending,
`[plate_x]` on columns, and `AVERAGE(called_strike_rate)` as the value.
Conditional-format the background on a diverging scale, then shrink the font and
column widths until the cells read as a heatmap. Slicers on `dim_count[count]`
and `[stand]`.

Page 4 is the one that will look worse than the matplotlib version. Power BI has
no continuous 2D heatmap, so a conditionally formatted matrix is the closest
thing, and it renders as blocks rather than a smooth field. It is honest, just
coarser.

## Publishing

**Publish** in Power BI Desktop pushes to the Power BI Service. To get a public
link like the Streamlit app, use **File → Embed report → Publish to web** in the
Service. That makes the report readable by anyone with the URL and indexable by
search engines, so only do it with data you are happy to have public — which
this is, being aggregates of public Statcast data.
