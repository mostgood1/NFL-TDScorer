from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(BASE))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rebuild team and player TD likelihoods for a given season/week.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args(argv)

    data_dir = BASE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ensure trends cache exists (optional)
    run([PY, str(BASE / "build_td_trends.py"), "--season", str(args.season), "--years", "5"])  # no-op if already exists

    # 2) Team TDs
    run([PY, str(BASE / "td_likelihood_week.py"), "--season", str(args.season), "--week", str(args.week), "--out", str(data_dir / f"td_likelihood_{args.season}_wk{args.week}.csv")])

    # 3) Player anytime TDs
    run([PY, str(BASE / "player_td_likelihood_week.py"), "--season", str(args.season), "--week", str(args.week), "--out", str(data_dir / f"player_td_likelihood_{args.season}_wk{args.week}.csv")])

    print("Rebuild complete.")


if __name__ == "__main__":
    main()
