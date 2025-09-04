from __future__ import annotations

import argparse
from pathlib import Path

from touchdown.src.pbp_td_trends import build_and_cache_trends, last_n_seasons


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build and cache last-N-years touchdown trends by team/position and player counts.")
    ap.add_argument("--season", type=int, required=True, help="Current season (e.g., 2025); trends use prior N seasons.")
    ap.add_argument("--years", type=int, default=5, help="Number of seasons back (default 5).")
    args = ap.parse_args(argv)

    seasons = last_n_seasons(args.season, args.years)
    paths = build_and_cache_trends(args.season, seasons)
    print(f"Wrote team position shares to: {paths.team_pos_shares}")
    print(f"Wrote player TD counts to: {paths.player_td_counts}")


if __name__ == "__main__":
    main()
