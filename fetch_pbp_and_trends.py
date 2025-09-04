from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from touchdown.src.pbp_td_trends import last_n_seasons, build_and_cache_trends


DATA_DIR = Path(__file__).resolve().parent / "data"


def fetch_pbp(seasons: List[int]) -> None:
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception as e:
        print(f"nfl_data_py not available: {e}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Download per-season and write pbp_<year>.csv
    for y in seasons:
        try:
            print(f"Fetching PBP for {y}…")
            df = nfl.import_pbp_data([int(y)])
            if df is None or df.empty:
                print(f"No PBP rows for {y}.")
                continue
            out_fp = DATA_DIR / f"pbp_{y}.csv"
            df.to_csv(out_fp, index=False)
            print(f"Wrote {out_fp}")
        except Exception as e:
            print(f"Failed PBP {y}: {e}")


def fetch_rosters(seasons: List[int]) -> None:
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception as e:
        print(f"nfl_data_py not available: {e}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Fetching rosters for {seasons}…")
        ros = nfl.import_seasonal_rosters(seasons)
        if ros is None or ros.empty:
            print("No rosters returned.")
            return
        out_fp = DATA_DIR / f"seasonal_rosters_{seasons[0]}_{seasons[-1]}.csv"
        ros.to_csv(out_fp, index=False)
        print(f"Wrote {out_fp}")
    except Exception as e:
        print(f"Failed roster fetch: {e}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch nflfastR PBP and rosters for last N seasons, then build trends cache.")
    ap.add_argument("--season", type=int, required=True, help="Current season (e.g., 2025)")
    ap.add_argument("--years", type=int, default=5, help="Number of seasons back to include (default 5)")
    args = ap.parse_args(argv)

    seasons = last_n_seasons(args.season, args.years)
    fetch_pbp(seasons)
    fetch_rosters(seasons)

    # Build cache using the newly downloaded local CSVs
    paths = build_and_cache_trends(args.season, seasons)
    print(f"Trend cache: {paths.team_pos_shares}")
    print(f"Trend players: {paths.player_td_counts}")


if __name__ == "__main__":
    main()
