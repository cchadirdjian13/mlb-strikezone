# ABOUTME: Pulls Statcast pitch-level data one month at a time and caches one
# ABOUTME: parquet file per season under data/raw/. Never re-pulls an existing season.
import sys
import time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from pybaseball import statcast, cache

cache.enable()
RAW = Path("data/raw")


def month_ranges(year):
    """Yield (start, end) for each month Mar-Oct, covering the regular season window."""
    for m in range(3, 11):
        yield date(year, m, 1), date(year, m + 1, 1) - timedelta(days=1)


def _pull_month(year, start, end):
    """Pull one month. Savant intermittently returns a body pybaseball cannot
    parse; that is transient, so retry once and then fail loudly rather than
    write a season quietly missing a month of games."""
    for attempt in (1, 2):
        try:
            return statcast(start_dt=str(start), end_dt=str(end))
        except pd.errors.ParserError:
            if attempt == 2:
                raise
            print(f"{year}: unparseable response for {start} -> {end}, retrying")
            time.sleep(5)


def pull_season(year):
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"statcast_{year}.parquet"
    if out.exists():
        print(f"{year}: already cached")
        return
    frames = []
    for start, end in month_ranges(year):
        print(f"{year}: pulling {start} -> {end}")
        df = _pull_month(year, start, end)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError(f"{year}: Statcast returned no rows for any month")
    season = pd.concat(frames, ignore_index=True)
    season = season[season["game_type"] == "R"]
    season.to_parquet(out, index=False)
    print(f"{year}: saved {len(season):,} pitches")


def _self_check():
    r = list(month_ranges(2020))
    assert len(r) == 8, r
    assert r[0] == (date(2020, 3, 1), date(2020, 3, 31)), r[0]
    assert r[-1] == (date(2020, 10, 1), date(2020, 10, 31)), r[-1]
    assert all(s <= e for s, e in r)
    print("ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--check"]:
        _self_check()
    else:
        for y in [int(y) for y in args] or [2025]:
            pull_season(y)
