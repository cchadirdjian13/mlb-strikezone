# ABOUTME: Builds the small parquet files the Streamlit app reads, so the app can
# ABOUTME: be deployed from the repo without the pitch tables it was measured from.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_strikezone.analysis import (
    COUNTS,
    X_CENTERS,
    Z_CENTERS,
    strike_rate_grid,
)
from mlb_strikezone.attribution import ATTRIBUTION
from mlb_strikezone.drift import SEASON_EFFECTS

CALLED_PITCHES = Path("data/processed/called_pitches.parquet")

# Committed, unlike data/. These are derived aggregates measuring a few hundred
# kilobytes, not the Statcast pulls, and the app is useless without them.
APP_DATA = Path("app_data")
ZONE_GRIDS = APP_DATA / "zone_grids.parquet"
APP_ATTRIBUTION = APP_DATA / "attribution.parquet"
APP_SEASONS = APP_DATA / "season_effects.parquet"

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
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _self_check()
    else:
        main()
