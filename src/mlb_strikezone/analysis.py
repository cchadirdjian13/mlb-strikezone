# ABOUTME: Measures the empirical 50% called-strike zone from called_pitches.parquet,
# ABOUTME: broken out by count, handedness and season, and writes the README figures.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

PROCESSED = Path("data/processed/called_pitches.parquet")
FIGURES = Path("figures")

# The zone is measured on a fixed grid in feet rather than modelled: a cell is
# "in the zone" when at least half the pitches taken there were called strikes.
STEP = 0.1
CELL_AREA = STEP**2
X_EDGES = np.arange(-2.0, 2.0 + STEP / 2, STEP)
Z_EDGES = np.arange(0.5, 4.5 + STEP / 2, STEP)
X_CENTERS = X_EDGES[:-1] + STEP / 2
Z_CENTERS = Z_EDGES[:-1] + STEP / 2
# Applies to the pooled count, so ~11 taken pitches per raw cell. Thinner than
# this is noise, not a zone edge; those cells are dropped, not called balls.
MIN_POOLED = 100

ZONE_HALF_WIDTH = 0.708
COUNTS = [f"{b}-{s}" for b in range(4) for s in range(3)]

SMOOTH_CELLS = 3  # pool each cell with its neighbours over a 0.3 ft window

COLUMNS = ["game_year", "stand", "plate_x", "plate_z", "sz_top", "sz_bot", "is_strike", "count"]


def _pool(counts):
    """Sum each cell together with its neighbours over a SMOOTH_CELLS window."""
    return uniform_filter(counts, SMOOTH_CELLS, mode="constant") * SMOOTH_CELLS**2


def strike_rate_grid(df):
    """Called-strike rate per grid cell, NaN where the cell is under-sampled.

    Rates come from pooled neighbourhoods, not raw cells. Hitters swing at
    anything near the middle on two strikes, so the heart of the 0-2 zone holds
    only a handful of taken pitches per cell; without pooling those cells fall
    under the sample floor and punch a hole in the middle of the measured zone."""
    taken, _, _ = np.histogram2d(df["plate_x"], df["plate_z"], bins=[X_EDGES, Z_EDGES])
    called = df[df["is_strike"]]
    strikes, _, _ = np.histogram2d(
        called["plate_x"], called["plate_z"], bins=[X_EDGES, Z_EDGES]
    )
    taken, strikes = _pool(taken), _pool(strikes)
    with np.errstate(invalid="ignore", divide="ignore"):  # empty cells; masked below
        rate = strikes / taken
    rate[taken < MIN_POOLED] = np.nan
    return rate


def zone_area(rate):
    """Square feet of grid where a taken pitch is called a strike at least half the time."""
    return float((rate >= 0.5).sum() * CELL_AREA)  # NaN cells compare False


def area_by(df, column, keys):
    """Zone area and sample size for each key of `column`, in the given key order."""
    rows = []
    for key in keys:
        subset = df[df[column] == key]
        rows.append({column: key, "pitches": len(subset), "area": zone_area(strike_rate_grid(subset))})
    return pd.DataFrame(rows)


def _draw_rulebook_zone(ax, df, **kwargs):
    style = {"color": "black", "lw": 1, "ls": "--", **kwargs}
    bot, top = df["sz_bot"].mean(), df["sz_top"].mean()
    ax.plot(
        [-ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH],
        [bot, bot, top, top, bot],
        **style,
    )


def _contour(ax, rate, **kwargs):
    """0.5 contour of a rate grid. Under-sampled cells sit outside the zone, so
    filling them with 0 keeps the contour closed instead of punching holes in it."""
    ax.contour(X_CENTERS, Z_CENTERS, np.nan_to_num(rate).T, levels=[0.5], **kwargs)


def _square_zone_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.0, 4.0)
    ax.set_xlabel("horizontal location (ft, catcher's view)")
    ax.set_ylabel("height (ft)")


def figure_area_by_count(counts, path):
    """Balls on the x-axis, one line per strike count: both effects at once, and
    the fact that a strike costs more area than a ball gains."""
    area = counts.set_index("count")["area"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strikes, color in zip(range(3), ["#c1442f", "#7d7d7d", "#3b6ea5"]):
        ax.plot(
            range(4),
            [area[f"{b}-{strikes}"] for b in range(4)],
            marker="o",
            color=color,
            label=f"{strikes} strike" + ("s" if strikes != 1 else ""),
        )
    ax.set_xticks(range(4))
    ax.set_xlabel("balls")
    ax.set_ylabel("50% zone area (sq ft)")
    ax.set_title("Umpires shrink the zone with two strikes, widen it with three balls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_count_contours(df, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for count, color in [("3-0", "#c1442f"), ("0-2", "#3b6ea5")]:
        _contour(ax, strike_rate_grid(df[df["count"] == count]), colors=color)
        ax.plot([], [], color=color, label=f"{count} count")
    _draw_rulebook_zone(ax, df, label="rulebook zone (mean)")
    _square_zone_axes(ax)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("50% strike zone: 3-0 vs 0-2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_handedness_heatmaps(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, stand, title in zip(axes, ["R", "L"], ["Right-handed batters", "Left-handed batters"]):
        subset = df[df["stand"] == stand]
        mesh = ax.pcolormesh(
            X_EDGES, Z_EDGES, strike_rate_grid(subset).T, cmap="RdBu_r", vmin=0, vmax=1
        )
        _contour(ax, strike_rate_grid(subset), colors="black", linewidths=1.5)
        _draw_rulebook_zone(ax, subset)
        _square_zone_axes(ax)
        ax.set_title(title)
    axes[1].set_ylabel("")
    fig.colorbar(mesh, ax=axes, label="called-strike rate")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_area_by_season(seasons, path):
    """The y-axis is zoomed to a range of about a tenth of a square foot, so the
    title carries the size of the move to stop the slope overselling it."""
    first, last = seasons["area"].iloc[0], seasons["area"].iloc[-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(seasons["game_year"], seasons["area"], marker="o", color="#3b6ea5")

    short = seasons[seasons["game_year"] == 2020]
    ax.scatter(
        short["game_year"], short["area"], s=90, facecolor="white",
        edgecolor="#3b6ea5", zorder=3, label="2020 (60-game season)",
    )
    ax.set_ylabel("50% zone area (sq ft)")
    ax.set_xlabel("season")
    ax.set_title(f"Zone area has drifted down {1 - last / first:.1%} since 2015")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_parquet(PROCESSED, columns=COLUMNS)
    print(f"{len(df):,} called pitches, {df['game_year'].nunique()} seasons")

    counts = area_by(df, "count", COUNTS)
    seasons = area_by(df, "game_year", sorted(df["game_year"].unique()))
    hands = area_by(df, "stand", ["R", "L"])

    print("\nzone area by count (sq ft)")
    print(counts.to_string(index=False))
    print("\nzone area by season (sq ft)")
    print(seasons.to_string(index=False))
    print("\nzone area by batter side (sq ft)")
    print(hands.to_string(index=False))

    wide = counts.set_index("count").loc["3-0", "area"]
    tight = counts.set_index("count").loc["0-2", "area"]
    print(f"\n3-0 zone is {wide / tight - 1:.1%} larger than the 0-2 zone")

    FIGURES.mkdir(exist_ok=True)
    figure_area_by_count(counts, FIGURES / "zone_area_by_count.png")
    figure_count_contours(df, FIGURES / "zone_contour_3-0_vs_0-2.png")
    figure_handedness_heatmaps(df, FIGURES / "called_strike_rate_by_handedness.png")
    figure_area_by_season(seasons, FIGURES / "zone_area_by_season.png")
    print(f"wrote 4 figures to {FIGURES}/")


def _self_check():
    # Every pitch inside a known 1.0 x 2.0 ft box is a strike, everything else a ball,
    # so the recovered 50% zone should be that box's 2.0 sq ft.
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, 200_000)
    z = rng.uniform(1.0, 4.0, 200_000)
    inside = (np.abs(x) <= 0.5) & (z >= 2.0) & (z <= 4.0)
    df = pd.DataFrame({"plate_x": x, "plate_z": z, "is_strike": inside})

    area = zone_area(strike_rate_grid(df))
    assert abs(area - 2.0) < 0.1, area

    # Thinning the middle of the box the way a two-strike count does must not
    # hollow out the measured zone: the area should survive nearly intact.
    core = (np.abs(x) <= 0.2) & (z >= 2.4) & (z <= 3.2)
    thinned = df[~core | (rng.random(len(df)) < 0.02)]
    assert abs(zone_area(strike_rate_grid(thinned)) - 2.0) < 0.1, zone_area(
        strike_rate_grid(thinned)
    )

    # Under-sampled cells drop out rather than counting as zone.
    assert np.isnan(strike_rate_grid(df.head(50))).all()

    # A cell called a strike exactly half the time is in; just under is out.
    rate = np.full((len(X_CENTERS), len(Z_CENTERS)), np.nan)
    rate[0, 0], rate[0, 1] = 0.5, 0.499
    assert zone_area(rate) == CELL_AREA, zone_area(rate)
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _self_check()
    else:
        main()
