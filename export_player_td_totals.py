from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _load_pbp_for_season(season: int) -> pd.DataFrame:
    # Prefer local CSV
    local = DATA_DIR / f"pbp_{season}.csv"
    if local.exists():
        try:
            return pd.read_csv(local)
        except Exception:
            pass
    # Fallback: nfl_data_py
    try:
        import nfl_data_py as nfl  # type: ignore
        return nfl.import_pbp_data([int(season)])
    except Exception:
        return pd.DataFrame()


def compute_player_td_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["player", "rush_td", "rec_td", "total_td"])

    # Ensure required columns
    for c in [
        "rush_touchdown",
        "pass_touchdown",
        "rusher_player_name",
        "receiver_player_name",
    ]:
        if c not in df.columns:
            df[c] = 0

    # Booleans to ints
    df["is_rush_td"] = pd.to_numeric(df["rush_touchdown"], errors="coerce").fillna(0).astype(int)
    df["is_rec_td"] = pd.to_numeric(df["pass_touchdown"], errors="coerce").fillna(0).astype(int)

    rush_names = (
        df.loc[df["is_rush_td"] == 1, "rusher_player_name"].dropna().astype(str).str.strip()
    )
    rec_names = (
        df.loc[df["is_rec_td"] == 1, "receiver_player_name"].dropna().astype(str).str.strip()
    )

    rush_counts = rush_names.value_counts().rename_axis("player").reset_index(name="rush_td")
    rec_counts = rec_names.value_counts().rename_axis("player").reset_index(name="rec_td")

    out = pd.merge(rush_counts, rec_counts, on="player", how="outer").fillna(0)
    out["rush_td"] = out["rush_td"].astype(int)
    out["rec_td"] = out["rec_td"].astype(int)
    out["total_td"] = (out["rush_td"] + out["rec_td"]).astype(int)
    out = out.sort_values(["total_td", "rush_td", "rec_td", "player"], ascending=[False, False, False, True])
    return out


def main(season: int, out_path: Optional[Path] = None) -> Path:
    df = _load_pbp_for_season(season)
    totals = compute_player_td_totals(df)
    out = out_path or (DATA_DIR / f"player_td_totals_{season}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    totals.to_csv(out, index=False)
    print(f"Wrote player TD totals to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export player TD totals for a given season")
    parser.add_argument("--season", type=int, default=2024, help="Season year, e.g., 2024")
    parser.add_argument("--out", type=str, default="", help="Output CSV path (optional)")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else None
    main(args.season, out_path)
