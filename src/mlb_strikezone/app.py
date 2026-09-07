# ABOUTME: Streamlit view over the attribution leaderboard and the called zone,
# ABOUTME: so both can be filtered and drilled into rather than read off a README.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mlb_strikezone.analysis import (
    COUNTS,
    X_CENTERS,
    X_EDGES,
    Z_CENTERS,
    Z_EDGES,
    ZONE_HALF_WIDTH,
    strike_rate_grid,
    zone_area,
)
from mlb_strikezone.attribution import ATTRIBUTION, MIN_PITCHES, ROLES
from mlb_strikezone.drift import SEASON_EFFECTS

CALLED_PITCHES = Path("data/processed/called_pitches.parquet")
ZONE_COLUMNS = ["plate_x", "plate_z", "is_strike", "count", "stand", "sz_top", "sz_bot"]

REQUIRED = {
    ATTRIBUTION: "attribution.py",
    SEASON_EFFECTS: "drift.py",
    CALLED_PITCHES: "features.py",
}


def load_attribution():
    return pd.read_parquet(ATTRIBUTION)


def load_season_effects(names):
    """Per-season effects carry ids but no names; the pooled table has both."""
    seasons = pd.read_parquet(SEASON_EFFECTS)
    return seasons.merge(names[["role", "id", "name"]], on=["role", "id"], how="left")


def load_pitches(columns):
    """Columns are an argument, not a global read inside the body, so that the
    Streamlit cache key changes when they do. Read as a global, adding a column
    leaves the old frame cached and the new column missing at runtime."""
    return pd.read_parquet(CALLED_PITCHES, columns=list(columns))


def leaderboard(effects, role, min_pitches):
    """Qualifying people in one role, ranked, in readable units."""
    board = effects[(effects["role"] == role) & (effects["pitches"] >= min_pitches)]
    board = board.assign(
        per_100=board["effect"] * 100,
        se_per_100=board["se"] * 100,
        extra_strikes=board["effect"] * board["pitches"],
        clear=board["effect"].abs() > 2 * board["se"],
    )
    return board.sort_values("per_100", ascending=False)[
        ["name", "pitches", "per_100", "se_per_100", "clear", "extra_strikes"]
    ].reset_index(drop=True)


def trajectory(seasons, role, name):
    """One person's effect season by season, with two-standard-error bounds."""
    person = seasons[(seasons["role"] == role) & (seasons["name"] == name)].sort_values("season")
    return person.assign(
        per_100=person["effect"] * 100,
        lo=(person["effect"] - 2 * person["se"]) * 100,
        hi=(person["effect"] + 2 * person["se"]) * 100,
    )[["season", "pitches", "per_100", "lo", "hi"]].set_index("season")


def zone_frame(pitches, count, stand):
    """Filter to a count and batter side. 'Any' keeps everything."""
    if count != "Any":
        pitches = pitches[pitches["count"] == count]
    if stand != "Any":
        pitches = pitches[pitches["stand"] == stand]
    return pitches


def zone_figure(pitches, rate):
    """Heatmap, 50% contour and the mean rulebook box, drawn the way the README
    figures are. Streamlit's built-in scatter leaves gaps between grid cells and
    cannot draw the box, which would make the caption describe a chart that is
    not there."""
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    mesh = ax.pcolormesh(X_EDGES, Z_EDGES, rate.T, cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(X_CENTERS, Z_CENTERS, np.nan_to_num(rate).T, levels=[0.5],
               colors="black", linewidths=1.5)

    bottom, top = pitches["sz_bot"].mean(), pitches["sz_top"].mean()
    ax.plot(
        [-ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH, -ZONE_HALF_WIDTH],
        [bottom, bottom, top, top, bottom],
        color="black", ls="--", lw=1,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.0, 4.0)
    ax.set_xlabel("horizontal location (ft, catcher's view)")
    ax.set_ylabel("height (ft)")
    fig.colorbar(mesh, ax=ax, label="called-strike rate")
    return fig


def trajectory_figure(person, name):
    """A shaded band, not three lines. Streamlit's line chart plots the bounds as
    their own series, which reads as three findings instead of one estimate."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(person.index, person["lo"], person["hi"], color="#3b6ea5", alpha=0.2, lw=0)
    ax.plot(person.index, person["per_100"], marker="o", color="#3b6ea5")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("season")
    ax.set_ylabel("per 100 called pitches")
    ax.set_title(f"{name}, season by season (±2 SE)")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="MLB strike zone", layout="wide")
    st.title("The called strike zone, 2015-2025")

    missing = [path for path in REQUIRED if not path.exists()]
    if missing:
        st.error(
            "Missing generated data. Run the pipeline first:\n\n"
            + "\n".join(f"- `{path}` — build with `{REQUIRED[path]}`" for path in missing)
        )
        return

    effects = st.cache_data(load_attribution)()
    seasons = st.cache_data(load_season_effects)(effects)

    board_tab, zone_tab = st.tabs(["Who moves the zone", "The zone itself"])

    with board_tab:
        left, right = st.columns([1, 3])
        with left:
            role = st.radio("role", list(ROLES.values()))
            floor = st.slider(
                "minimum called pitches", 0, 20_000, MIN_PITCHES, step=500,
                help="Below about 2,000 an effect is not worth reading on its own.",
            )
        board = leaderboard(effects, role, floor)

        with right:
            st.caption(
                f"{len(board)} {role}s qualify. Effects are called strikes per 100 taken "
                "pitches, net of the other party. `clear` marks two standard errors from zero."
            )
            st.dataframe(
                board,
                width="stretch",
                height=380,
                column_config={
                    "pitches": st.column_config.NumberColumn("called pitches", format="%d"),
                    "per_100": st.column_config.NumberColumn("per 100", format="%.2f"),
                    "se_per_100": st.column_config.NumberColumn("SE", format="%.2f"),
                    "extra_strikes": st.column_config.NumberColumn(
                        "extra strikes", format="%d"
                    ),
                },
            )

        if len(board):
            st.subheader("One person, season by season")
            who = st.selectbox(f"{role}", board["name"].tolist())
            person = trajectory(seasons, role, who)
            if person.empty:
                st.info("No seasons clear the per-season sample floor for this person.")
            else:
                st.pyplot(trajectory_figure(person, who))
                st.caption(
                    "Per-season effects are far noisier than the pooled number — a season "
                    "holds an eighth of a career here, so the band is wide by design."
                )

    with zone_tab:
        st.caption(
            "Share of taken pitches called a strike, on a 0.1 ft grid pooled with "
            "neighbours. The dashed rulebook width is +/- 0.708 ft."
        )
        picker, chart = st.columns([1, 3])
        with picker:
            count = st.selectbox("count", ["Any"] + COUNTS)
            stand = st.radio("batter side", ["Any", "R", "L"])

        pitches = st.cache_data(load_pitches)(tuple(ZONE_COLUMNS))
        selected = zone_frame(pitches, count, stand)

        with chart:
            if len(selected) < 5_000:
                st.warning(f"Only {len(selected):,} pitches match; the grid would be noise.")
            else:
                rate = strike_rate_grid(selected)
                st.pyplot(zone_figure(selected, rate))
                st.metric(
                    "50% zone area",
                    f"{zone_area(rate):.2f} sq ft",
                    help=f"Measured over {len(selected):,} taken pitches.",
                )


def _self_check():
    effects = pd.DataFrame(
        {
            "role": ["umpire", "umpire", "catcher"],
            "id": [1, 2, 3],
            "name": ["Big Zone", "Small Sample", "A Catcher"],
            "effect": [0.02, -0.03, 0.01],
            "se": [0.005, 0.02, 0.004],
            "pitches": [30_000, 500, 20_000],
        }
    )

    # The floor keeps the well-sampled umpire and drops the other; the catcher is
    # never in an umpire board regardless of how many pitches he caught.
    board = leaderboard(effects, "umpire", 2_000)
    assert board["name"].tolist() == ["Big Zone"], board["name"].tolist()
    assert abs(board.loc[0, "per_100"] - 2.0) < 1e-9, board.loc[0, "per_100"]
    assert abs(board.loc[0, "extra_strikes"] - 600) < 1e-9
    assert bool(board.loc[0, "clear"]) is True

    # Dropping the floor lets the thin one back in, and it must not read as clear.
    loose = leaderboard(effects, "umpire", 0)
    assert len(loose) == 2
    assert not bool(loose.set_index("name").loc["Small Sample", "clear"])

    seasons = pd.DataFrame(
        {
            "role": ["umpire"] * 2,
            "id": [1, 1],
            "name": ["Big Zone"] * 2,
            "season": [2016, 2015],
            "effect": [0.02, 0.01],
            "se": [0.005, 0.005],
            "pitches": [3_000, 3_100],
        }
    )
    line = trajectory(seasons, "umpire", "Big Zone")
    assert line.index.tolist() == [2015, 2016], line.index.tolist()
    assert abs(line.loc[2015, "lo"] - 0.0) < 1e-9, line.loc[2015, "lo"]
    assert trajectory(seasons, "umpire", "Nobody").empty

    # Built with exactly the columns the loader reads, so the zone path is
    # exercised against its real contract. Adding a column to a figure without
    # adding it to ZONE_COLUMNS fails here rather than in the running app.
    rng = np.random.default_rng(0)
    size = 4_000
    pitches = pd.DataFrame(
        {
            "plate_x": rng.uniform(-1.5, 1.5, size),
            "plate_z": rng.uniform(1.0, 4.0, size),
            "is_strike": rng.random(size) < 0.3,
            "count": rng.choice(["0-0", "3-0"], size),
            "stand": rng.choice(["R", "L"], size),
            "sz_top": np.full(size, 3.4),
            "sz_bot": np.full(size, 1.6),
        }
    )[ZONE_COLUMNS]

    # "Any" is a genuine no-op on both axes; naming either one narrows.
    assert len(zone_frame(pitches, "Any", "Any")) == size
    assert len(zone_frame(pitches, "0-0", "Any")) < size
    narrowed = zone_frame(pitches, "0-0", "R")
    assert 0 < len(narrowed) < len(zone_frame(pitches, "0-0", "Any"))

    # Both figures build end to end. They are presentation, but a broken axis or
    # a missing column would only show up by clicking through the running app.
    zone = zone_figure(narrowed, strike_rate_grid(narrowed))
    assert zone.axes, "zone figure drew nothing"
    plt.close(zone)

    line = trajectory_figure(trajectory(seasons, "umpire", "Big Zone"), "Big Zone")
    assert line.axes, "trajectory figure drew nothing"
    plt.close(line)
    print("ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
    else:
        main()
