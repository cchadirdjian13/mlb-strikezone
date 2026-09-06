# ABOUTME: Builds the called-pitch analysis table from the raw per-season Statcast
# ABOUTME: parquets, writing data/processed/called_pitches.parquet.
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Half the plate (8.5 in) plus a ball radius (1.45 in), in feet. See CLAUDE.md.
ZONE_HALF_WIDTH = 0.708

RAW_COLUMNS = [
    "game_pk",
    "game_date",
    "game_year",
    "pitcher",
    "batter",
    "fielder_2",
    "stand",
    "p_throws",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "balls",
    "strikes",
    "description",
]

ID_COLUMNS = ["game_pk", "pitcher", "batter", "fielder_2"]

LOCATION_COLUMNS = ["plate_x", "plate_z", "sz_top", "sz_bot"]


def build_called_pitches(raw):
    """Filter a raw Statcast frame to taken pitches and add the derived columns."""
    df = raw[raw["description"].isin(["called_strike", "ball"])].copy()
    df = df.dropna(subset=LOCATION_COLUMNS + ["stand", "balls", "strikes"])

    df["is_strike"] = df["description"] == "called_strike"
    df["count"] = (
        df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    )
    # Positive plate_x_adj is inside to the batter, for either handedness.
    df["plate_x_adj"] = np.where(df["stand"] == "L", -df["plate_x"], df["plate_x"])
    df["in_zone"] = (
        (df["plate_x"].abs() <= ZONE_HALF_WIDTH)
        & (df["plate_z"] >= df["sz_bot"])
        & (df["plate_z"] <= df["sz_top"])
    )

    for col in ID_COLUMNS:
        df[col] = df[col].astype("int64")
    for col in ["balls", "strikes"]:
        df[col] = df[col].astype("int8")

    return df.drop(columns=["description"]).reset_index(drop=True)


def build_all():
    paths = sorted(RAW.glob("statcast_*.parquet"))
    if not paths:
        raise RuntimeError(f"no raw seasons in {RAW}; run ingest.py first")
    frames = []
    for path in paths:
        season = build_called_pitches(pd.read_parquet(path, columns=RAW_COLUMNS))
        print(f"{path.name}: {len(season):,} called pitches")
        frames.append(season)
    out = pd.concat(frames, ignore_index=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED / "called_pitches.parquet"
    out.to_parquet(dest, index=False)
    print(f"wrote {dest}: {len(out):,} rows, {out['game_year'].nunique()} seasons")


def _self_check():
    raw = pd.DataFrame(
        {
            "game_pk": [1, 1, 1, 1, 1],
            "game_date": ["2024-04-01"] * 5,
            "game_year": [2024] * 5,
            "pitcher": [100.0] * 5,
            "batter": [200.0] * 5,
            "fielder_2": [300.0] * 5,
            "stand": ["R", "L", "R", "R", "R"],
            "p_throws": ["R"] * 5,
            "pitch_type": ["FF"] * 5,
            "release_speed": [95.0] * 5,
            "plate_x": [0.5, 0.5, 1.5, 0.0, None],
            "plate_z": [2.5, 2.5, 2.5, 5.0, 2.5],
            "sz_top": [3.4] * 5,
            "sz_bot": [1.6] * 5,
            "balls": [3, 0, 1, 2, 0],
            "strikes": [0, 2, 1, 1, 0],
            "description": ["called_strike", "ball", "ball", "ball", "ball"],
        }
    )
    out = build_called_pitches(raw)

    # The null-location row is dropped; the swing-free rows all survive.
    assert len(out) == 4, len(out)
    assert "description" not in out.columns
    assert out["is_strike"].tolist() == [True, False, False, False]
    assert out["count"].tolist() == ["3-0", "0-2", "1-1", "2-1"]
    # Sign flips for the lefty only.
    assert out["plate_x_adj"].tolist() == [0.5, -0.5, 1.5, 0.0]
    # Inside; inside; wide of the plate; above sz_top.
    assert out["in_zone"].tolist() == [True, True, False, False]
    assert all(out[c].dtype == "int64" for c in ID_COLUMNS)
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _self_check()
    else:
        build_all()
