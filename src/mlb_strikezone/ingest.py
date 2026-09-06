import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from pybaseball import statcast, cache

cache.enable()
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def month_ranges(year):
    """Yield (start, end) for each month Mar–Oct, regular season window."""
    for m in range(3, 11):
        start = date(year, m, 1)
        end = date(year, m + 1, 1) - timedelta(days=1) if m < 12 else date(year, 12, 31)
        yield start, end

def pull_season(year):
    out = RAW / f"statcast_{year}.parquet"
    if out.exists():
        print(f"{year}: already cached")
        return
    frames = []
    for start, end in month_ranges(year):
        print(f"{year}: pulling {start} → {end}")
        df = statcast(start_dt=str(start), end_dt=str(end))
        if len(df):
            frames.append(df)
    season = pd.concat(frames, ignore_index=True)
    season = season[season["game_type"] == "R"]
    season.to_parquet(out, index=False)
    print(f"{year}: saved {len(season):,} pitches")

if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2025]
    for y in years:
        pull_season(y)