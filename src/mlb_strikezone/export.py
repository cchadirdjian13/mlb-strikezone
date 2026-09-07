# ABOUTME: Builds the small parquet files the Streamlit app reads, so the app can
# ABOUTME: be deployed from the repo without the pitch tables it was measured from.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_strikezone.analysis import (
    CELL_AREA,
    COUNTS,
    X_CENTERS,
    Z_CENTERS,
    strike_rate_grid,
)
from mlb_strikezone.attribution import ATTRIBUTION, MIN_PITCHES
from mlb_strikezone.drift import (
    SEASON_EFFECTS,
    SEASON_SPREADS,
    SHORT_SEASON,
    spread_by_season,
)
from mlb_strikezone.model import MODELS, PREDICTIONS, calibration_table

CALLED_PITCHES = Path("data/processed/called_pitches.parquet")

# Committed, unlike data/. These are derived aggregates measuring a few hundred
# kilobytes, not the Statcast pulls, and the app is useless without them.
APP_DATA = Path("app_data")
ZONE_GRIDS = APP_DATA / "zone_grids.parquet"
APP_ATTRIBUTION = APP_DATA / "attribution.parquet"
APP_SEASONS = APP_DATA / "season_effects.parquet"

# Power BI reads CSV without a connector or a gateway, and these are small.
# Named tables/ rather than data/, because .gitignore ignores data/ at any depth
# and these have to be committed for the model to be openable from a clone.
POWERBI = Path("powerbi/tables")

ANY = "Any"
STANDS = [ANY, "R", "L"]
GRID_COLUMNS = ["plate_x", "plate_z", "is_strike", "count", "stand", "sz_top", "sz_bot"]

ATTRIBUTION_COLUMNS = ["role", "id", "name", "pitches", "effect", "se"]
SEASON_COLUMNS = ["role", "id", "season", "pitches", "effect", "se"]


def select(pitches, count, stand):
    """Filter to a count and batter side. 'Any' keeps everything on that axis."""
    if count != ANY:
        pitches = pitches[pitches["count"] == count]
    if stand != ANY:
        pitches = pitches[pitches["stand"] == stand]
    return pitches


def zone_grids(pitches):
    """One called-strike grid per count and batter side.

    The app used to read all 3.7M pitches to draw these, which is a 150MB file it
    cannot carry to a host. The grid it actually renders is 40 by 40, so the
    thirty-nine combinations together are a few hundred kilobytes."""
    ix, iz = np.meshgrid(
        np.arange(len(X_CENTERS)), np.arange(len(Z_CENTERS)), indexing="ij"
    )
    frames = []
    for count in [ANY] + COUNTS:
        for stand in STANDS:
            subset = select(pitches, count, stand)
            rate = strike_rate_grid(subset)
            frames.append(
                pd.DataFrame(
                    {
                        "count": count,
                        "stand": stand,
                        "ix": ix.ravel().astype("int16"),
                        "iz": iz.ravel().astype("int16"),
                        "rate": rate.ravel().astype("float32"),
                        "pitches": np.int32(len(subset)),
                        "sz_bot": np.float32(subset["sz_bot"].mean()),
                        "sz_top": np.float32(subset["sz_top"].mean()),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def person_key(frame):
    """Power BI relates tables on one column, and neither role nor id is unique
    on its own — an umpire and a catcher can share an id."""
    return frame["role"].astype(str) + "-" + frame["id"].astype(str)


def build_tables(attribution, seasons, spreads, grids, calibration):
    """A star schema: three dimensions, five facts, every fact joined on a key.

    Shaped for a data model rather than for charts. The point of doing this in
    Power BI at all is relationships, measures and slicers; handing it one wide
    flat table would throw that away."""
    dim_person = attribution.assign(person_key=person_key(attribution))[
        ["person_key", "role", "id", "name", "pitches"]
    ].rename(columns={"id": "person_id", "pitches": "career_pitches"})

    dim_season = pd.DataFrame({"season": sorted(seasons["season"].unique())})
    dim_season["short_season"] = dim_season["season"] == SHORT_SEASON

    # "Any" is a real member, not a missing one. Left out, every all-counts row
    # in the zone facts joins to a blank dimension member and disappears from a
    # count slicer without saying so.
    dim_count = pd.DataFrame({"count": [ANY] + COUNTS})
    specific = dim_count["count"] != ANY
    dim_count["balls"] = dim_count["count"].str[0].where(specific).astype("Int64")
    dim_count["strikes"] = dim_count["count"].str[-1].where(specific).astype("Int64")
    dim_count["specific_count"] = specific

    fact_effects = attribution.assign(
        person_key=person_key(attribution),
        effect_per_100=attribution["effect"] * 100,
        se_per_100=attribution["se"] * 100,
        extra_strikes=attribution["effect"] * attribution["pitches"],
        clears_zero=attribution["effect"].abs() > 2 * attribution["se"],
        qualified=attribution["pitches"] >= MIN_PITCHES,
    )[["person_key", "effect_per_100", "se_per_100", "extra_strikes", "clears_zero",
       "qualified"]]

    fact_season_effects = seasons.assign(
        person_key=person_key(seasons),
        effect_per_100=seasons["effect"] * 100,
        se_per_100=seasons["se"] * 100,
    )[["person_key", "season", "effect_per_100", "se_per_100", "pitches"]]

    fact_spread = spreads[["season", "role", "spread", "lo", "hi", "spread_se", "people"]]

    # Area is the count of grid cells called a strike at least half the time, so
    # it comes straight off the stored grid without touching a pitch table.
    in_zone = grids.assign(in_zone=grids["rate"] >= 0.5)
    fact_zone_area = (
        in_zone.groupby(["count", "stand"], observed=True)
        .agg(area_sq_ft=("in_zone", lambda cells: cells.sum() * CELL_AREA),
             pitches=("pitches", "first"))
        .reset_index()
    )

    fact_zone_grid = grids.assign(
        plate_x=X_CENTERS[grids["ix"].to_numpy()],
        plate_z=Z_CENTERS[grids["iz"].to_numpy()],
    ).rename(columns={"rate": "called_strike_rate"})[
        ["count", "stand", "plate_x", "plate_z", "called_strike_rate"]
    ].dropna(subset=["called_strike_rate"])

    return {
        "dim_person": dim_person,
        "dim_season": dim_season,
        "dim_count": dim_count,
        "fact_effects": fact_effects,
        "fact_season_effects": fact_season_effects,
        "fact_spread": fact_spread,
        "fact_zone_area": fact_zone_area,
        "fact_zone_grid": fact_zone_grid,
        "fact_calibration": calibration,
    }


def build_calibration(predictions):
    """Calibration of both models, long by model so one slicer switches between."""
    frames = []
    for name, column in MODELS.items():
        table = calibration_table(predictions["is_strike"], predictions[column])
        frames.append(table.assign(model=name))
    return pd.concat(frames, ignore_index=True)[
        ["model", "band", "pitches", "predicted", "observed", "gap"]
    ]


def write_tables(tables, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = destination / f"{name}.csv"
        table.round(5).to_csv(path, index=False, encoding="utf-8")
        print(f"  {path.name}: {len(table):,} rows, {path.stat().st_size / 1024:.0f} KB")


def main():
    APP_DATA.mkdir(exist_ok=True)

    for source, destination, columns in [
        (ATTRIBUTION, APP_ATTRIBUTION, ATTRIBUTION_COLUMNS),
        (SEASON_EFFECTS, APP_SEASONS, SEASON_COLUMNS),
    ]:
        table = pd.read_parquet(source)[columns]
        table.to_parquet(destination, index=False)
        print(f"{destination}: {len(table):,} rows, {destination.stat().st_size / 1024:.0f} KB")

    pitches = pd.read_parquet(CALLED_PITCHES, columns=GRID_COLUMNS)
    grids = zone_grids(pitches)
    grids.to_parquet(ZONE_GRIDS, index=False)
    print(f"{ZONE_GRIDS}: {len(grids):,} rows, {ZONE_GRIDS.stat().st_size / 1024:.0f} KB "
          f"(from {CALLED_PITCHES.stat().st_size / 1024 / 1024:.0f} MB of pitches)")

    total = sum(path.stat().st_size for path in APP_DATA.glob("*.parquet"))
    print(f"\n{APP_DATA}/ totals {total / 1024 / 1024:.2f} MB")

    print(f"\n{POWERBI}/")
    seasons = pd.read_parquet(SEASON_EFFECTS)
    spreads = spread_by_season(seasons).merge(
        pd.read_parquet(SEASON_SPREADS), on=["season", "role"], how="left"
    )
    tables = build_tables(
        attribution=pd.read_parquet(ATTRIBUTION),
        seasons=seasons,
        spreads=spreads,
        grids=grids,
        calibration=build_calibration(pd.read_parquet(PREDICTIONS)),
    )
    write_tables(tables, POWERBI)
    total = sum(path.stat().st_size for path in POWERBI.glob("*.csv"))
    print(f"\n{POWERBI}/ totals {total / 1024 / 1024:.2f} MB")


def _self_check():
    rng = np.random.default_rng(0)
    size = 30_000
    pitches = pd.DataFrame(
        {
            "plate_x": rng.uniform(-1.5, 1.5, size),
            "plate_z": rng.uniform(1.0, 4.0, size),
            "is_strike": rng.random(size) < 0.3,
            "count": rng.choice(COUNTS, size),
            "stand": rng.choice(["R", "L"], size),
            "sz_top": np.full(size, 3.4),
            "sz_bot": np.full(size, 1.6),
        }
    )

    # "Any" is a no-op on both axes; naming either narrows.
    assert len(select(pitches, ANY, ANY)) == size
    assert len(select(pitches, "0-0", ANY)) < size
    assert len(select(pitches, "0-0", "R")) < len(select(pitches, "0-0", ANY))

    grids = zone_grids(pitches)
    cells = len(X_CENTERS) * len(Z_CENTERS)
    combinations = (len(COUNTS) + 1) * len(STANDS)

    # Every combination is stored whole, NaNs included, so the app can rebuild the
    # grid by position without reindexing anything.
    assert len(grids) == cells * combinations, len(grids)
    assert grids.groupby(["count", "stand"], observed=True).size().eq(cells).all()

    # The "Any" grid holds every pitch; a named count holds fewer.
    everything = grids[(grids["count"] == ANY) & (grids["stand"] == ANY)]
    assert everything["pitches"].iloc[0] == size
    named = grids[(grids["count"] == "0-0") & (grids["stand"] == "R")]
    assert 0 < named["pitches"].iloc[0] < size

    # Indices address the grid exactly once each.
    assert set(everything["ix"]) == set(range(len(X_CENTERS)))
    assert set(everything["iz"]) == set(range(len(Z_CENTERS)))
    assert not everything.duplicated(subset=["ix", "iz"]).any()

    # An umpire and a catcher can share an id, so role alone or id alone would
    # collide and silently fan out every measure built on the relationship.
    clash = pd.DataFrame({"role": ["umpire", "catcher"], "id": [7, 7]})
    assert person_key(clash).nunique() == 2, "person_key collides across roles"

    attribution = pd.DataFrame(
        {
            "role": ["umpire", "catcher", "pitcher"],
            "id": [1, 1, 2],
            "name": ["An Ump", "A Catcher", "A Pitcher"],
            "pitches": [30_000, 20_000, 900],
            "effect": [0.02, -0.01, 0.005],
            "se": [0.004, 0.004, 0.02],
        }
    )
    seasons_in = pd.DataFrame(
        {
            "role": ["umpire", "umpire"],
            "id": [1, 1],
            "season": [2015, SHORT_SEASON],
            "pitches": [3_000, 1_200],
            "effect": [0.02, 0.01],
            "se": [0.005, 0.008],
        }
    )
    spreads_in = pd.DataFrame(
        {"season": [2015], "role": ["umpire"], "spread": [1.1], "lo": [1.0],
         "hi": [1.2], "spread_se": [0.05], "people": [80]}
    )
    calibration_in = pd.DataFrame(
        {"model": ["baseline"], "band": ["0.00-0.05"], "pitches": [10],
         "predicted": [0.02], "observed": [0.03], "gap": [0.01]}
    )
    tables = build_tables(attribution, seasons_in, spreads_in, grids, calibration_in)

    # Referential integrity. A fact key with no dimension row is the failure that
    # shows up in Power BI as a blank slicer value rather than as an error.
    people = set(tables["dim_person"]["person_key"])
    assert set(tables["fact_effects"]["person_key"]) <= people
    assert set(tables["fact_season_effects"]["person_key"]) <= people
    assert set(tables["fact_season_effects"]["season"]) <= set(tables["dim_season"]["season"])
    # Including "Any", which is a member of the dimension rather than a gap in it.
    assert set(tables["fact_zone_area"]["count"]) <= set(tables["dim_count"]["count"])
    assert set(tables["fact_zone_grid"]["count"]) <= set(tables["dim_count"]["count"])
    assert ANY in set(tables["dim_count"]["count"])

    # Dimensions must be unique or every measure joined to them double counts.
    assert not tables["dim_person"]["person_key"].duplicated().any()
    assert not tables["dim_season"]["season"].duplicated().any()
    assert not tables["dim_count"]["count"].duplicated().any()

    # The short season is flagged, since it needs excluding in most visuals.
    assert tables["dim_season"].set_index("season").loc[SHORT_SEASON, "short_season"]

    # Area comes off the grid, and the "Any" grid must be the widest zone there is.
    areas = tables["fact_zone_area"].set_index(["count", "stand"])["area_sq_ft"]
    assert (areas >= 0).all() and areas.max() < 16, areas.max()
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _self_check()
    else:
        main()
