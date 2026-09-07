# ABOUTME: Fetches the home-plate umpire for every regular-season game from the MLB
# ABOUTME: Stats API and writes data/processed/umpires.parquet, keyed on game_pk.
import sys
from pathlib import Path

import pandas as pd
import requests

PROCESSED = Path("data/processed")
CALLED_PITCHES = PROCESSED / "called_pitches.parquet"

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
SEASONS = range(2015, 2026)

COLUMNS = ["game_pk", "game_date", "ump_id", "ump_name"]


def home_plate_umpire(game):
    """(id, name) for the game's home-plate umpire, or None when no crew is listed.

    Postponed and forfeited games still appear on the schedule with an empty
    officials list, so a missing crew is expected rather than an error."""
    for official in game.get("officials", []):
        if official.get("officialType") == "Home Plate":
            person = official["official"]
            return person["id"], person["fullName"]
    return None


def parse_schedule(payload):
    """Flatten a schedule response into one row per game that has a plate umpire.

    The API lists a few hundred games twice per season, identically. Left in,
    those rows multiply pitches in any join on game_pk, so they are collapsed
    here; a repeat that genuinely disagrees is a different problem and raises."""
    rows = []
    for date in payload.get("dates", []):
        for game in date.get("games", []):
            umpire = home_plate_umpire(game)
            if umpire is None:
                continue
            rows.append(
                {
                    "game_pk": game["gamePk"],
                    "game_date": game["officialDate"],
                    "ump_id": umpire[0],
                    "ump_name": umpire[1],
                }
            )
    games = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(ignore_index=True)

    conflicting = games[games["game_pk"].duplicated(keep=False)]
    if len(conflicting):
        raise RuntimeError(f"game listed with more than one plate umpire:\n{conflicting}")
    return games


def fetch_season(year):
    """One request per season; the schedule endpoint hydrates the whole crew."""
    response = requests.get(
        SCHEDULE,
        params={"sportId": 1, "season": year, "gameType": "R", "hydrate": "officials"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    umpires = parse_schedule(payload)
    # totalGames counts the repeated listings too, so it reads higher than the
    # number of distinct games; it is an entry count, not a game count.
    entries = payload.get("totalGames", 0)
    print(f"{year}: {len(umpires):,} games with a plate umpire, from {entries:,} schedule entries")
    return umpires


def report_coverage(umpires):
    """Say how much of the called-pitch table this table can actually be joined to.

    A silent drop in coverage is the failure mode that matters here: the join
    would still succeed and simply attribute fewer pitches."""
    if not CALLED_PITCHES.exists():
        print(f"{CALLED_PITCHES} not built yet, skipping coverage check")
        return
    played = set(pd.read_parquet(CALLED_PITCHES, columns=["game_pk"])["game_pk"].unique())
    matched = played & set(umpires["game_pk"])
    print(
        f"coverage: {len(matched):,} of {len(played):,} games with called pitches "
        f"have an umpire ({len(matched) / len(played):.2%})"
    )


def build_all(years):
    umpires = pd.concat([fetch_season(y) for y in years], ignore_index=True)
    umpires["game_pk"] = umpires["game_pk"].astype("int64")
    umpires["ump_id"] = umpires["ump_id"].astype("int64")

    # One row per game is what makes this table safe to join on; anything else
    # silently multiplies pitches downstream rather than failing.
    if umpires["game_pk"].duplicated().any():
        raise RuntimeError("umpires table has more than one row for some game_pk")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED / "umpires.parquet"
    umpires.to_parquet(dest, index=False)
    print(f"wrote {dest}: {len(umpires):,} games, {umpires['ump_id'].nunique()} umpires")
    report_coverage(umpires)


def _self_check():
    payload = {
        "totalGames": 3,
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 1,
                        "officialDate": "2024-04-05",
                        "officials": [
                            {"official": {"id": 10, "fullName": "First Base Ump"},
                             "officialType": "First Base"},
                            {"official": {"id": 20, "fullName": "Plate Ump"},
                             "officialType": "Home Plate"},
                        ],
                    },
                    # Postponed: on the schedule, no crew assigned.
                    {"gamePk": 2, "officialDate": "2024-04-05", "officials": []},
                ]
            },
            {"games": [{"gamePk": 3, "officialDate": "2024-04-06", "officials": [
                {"official": {"id": 30, "fullName": "Other Plate Ump"},
                 "officialType": "Home Plate"}]}]},
        ],
    }
    # The API repeats some games verbatim; one game must yield one row.
    payload["dates"].append(payload["dates"][0])
    out = parse_schedule(payload)

    # The crewless game is dropped; the plate umpire is picked out of the crew
    # regardless of where in the list it sits.
    assert out["game_pk"].tolist() == [1, 3], out["game_pk"].tolist()
    assert out["ump_name"].tolist() == ["Plate Ump", "Other Plate Ump"]
    assert out["ump_id"].tolist() == [20, 30]

    # A repeat that disagrees is not a duplicate listing and must not be guessed at.
    conflicted = {"dates": [{"games": [
        {"gamePk": 1, "officialDate": "2024-04-05", "officials": [
            {"official": {"id": 20, "fullName": "Plate Ump"}, "officialType": "Home Plate"}]},
        {"gamePk": 1, "officialDate": "2024-04-05", "officials": [
            {"official": {"id": 99, "fullName": "Someone Else"}, "officialType": "Home Plate"}]},
    ]}]}
    try:
        parse_schedule(conflicted)
        raise AssertionError("expected a conflicting plate umpire to raise")
    except RuntimeError:
        pass

    # An empty schedule still yields the right shape, so a concat cannot fail.
    assert list(parse_schedule({}).columns) == COLUMNS
    assert len(parse_schedule({"dates": []})) == 0
    print("ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--check"]:
        _self_check()
    else:
        build_all([int(y) for y in args] or SEASONS)
