# CLAUDE.md — mlb-strikezone

## What this project is

A data-analytics portfolio project analyzing MLB umpire strike-zone calls using
Statcast pitch-level data. Core question: how does the called strike zone vary by
count, batter handedness, pitch type, umpire, and catcher, and how has it changed
2015–2025?

Deliverables: a reproducible pipeline (ingest → processed table → analysis),
a README with a stated finding and 3–4 key charts, and optionally a Streamlit app.
This is a CV piece — clarity and reproducibility matter as much as the result.

## Stack

- Python 3.12 managed by **uv**. Always run things with `uv run python ...`.
  Never `pip install` directly; use `uv add <pkg>`.
- Data: pybaseball (Statcast via Baseball Savant), MLB Stats API for umpires.
- Storage: parquet files under `data/`. Use DuckDB for multi-season queries.
- Analysis: pandas, scikit-learn, matplotlib. LightGBM later for the model.

## Layout

```
data/raw/         statcast_{year}.parquet, one file per season (gitignored)
data/processed/   called_pitches.parquet, umpires.parquet (gitignored)
notebooks/        exploration only; anything reused goes into src/
src/mlb_strikezone/    package root; uv_build requires this path
  ingest.py            pulls Statcast one month at a time, caches per season
  features.py          builds the called-pitch table
  umpires.py           home-plate umpire per game_pk from MLB Stats API
```

## Rules for data pulls

- **Never re-download a season that already exists in data/raw/.** Check for the
  file first. pybaseball cache is enabled; leave it on.
- Pull in one-month chunks. Savant times out on long ranges.
- Regular season only: filter `game_type == "R"`.
- 2020 is a 60-game season; keep it but treat it separately in analysis.
- Do not commit anything under `data/`.

## Data conventions

- Target: `description in {"called_strike", "ball"}` only. Drop swings, HBP,
  pitchouts, etc.
- Location: `plate_x`, `plate_z` in feet, catcher's view, measured at the front
  of the plate.
- Rulebook zone: `abs(plate_x) <= 0.708` and `sz_bot <= plate_z <= sz_top`.
  `sz_top`/`sz_bot` are per-pitch estimates and noisy; if a fixed vertical zone
  is used instead (1.5–3.5 ft), say so explicitly in the README.
- `stand` = batter side, `p_throws` = pitcher hand. Flip `plate_x` sign for LHB
  only when an inside/outside frame is wanted, and name the column differently
  (`plate_x_adj`) so the two are never confused.
- `fielder_2` is the catcher. `pitcher`/`batter` are MLBAM IDs; use
  `pybaseball.playerid_reverse_lookup` for names.
- Umpire join is on `game_pk`. Minimum sample for umpire/catcher leaderboards:
  ~2,000 called pitches.
- Count column format: `"balls-strikes"` string, e.g. `"3-0"`.

## Modeling conventions

- Baseline: logistic regression on location (with polynomial terms) + count +
  handedness. Then gradient boosting on the same plus pitch type and velocity.
- Evaluate with log-loss and calibration, not accuracy alone.
- Residual (actual call − predicted strike probability) is what gets attributed
  to umpire or catcher. Always control for location before ranking anyone.

## Working style

- Prefer small, composable functions in `src/` over notebook cells.
- Every script runnable from the repo root: `uv run python src/mlb_strikezone/<file>.py`.
- When adding a chart, save it to `figures/` with a descriptive filename and
  reference it from the README.
- Keep the README's stated finding quantified ("the 3-0 zone is X% larger than
  the 0-2 zone"), not qualitative.
- Ask before changing the schema of `called_pitches.parquet`; downstream
  notebooks depend on it.
