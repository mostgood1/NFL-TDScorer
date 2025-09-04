from __future__ import annotations

"""
Analyze historical weekly TD distributions (rush/pass and by position) and
compare them with the current week's predicted player TD CSV.

Outputs a short textual summary to stdout.
"""

from pathlib import Path
import sys
from typing import Iterable, Dict

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so local package imports work when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse helpers and data locations from the project
from touchdown.src import pbp_td_trends as trends


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _canonical_pos_keep_fb(pos: str | None) -> str:
    s = (pos or "").strip().upper()
    if s in {"HB", "RB"}:
        return "RB"
    if s == "FB":
        return "FB"
    if s == "WR":
        return "WR"
    if s == "TE":
        return "TE"
    if s == "QB":
        return "QB"
    return "OTHER"


def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    # Prefer local files via internal helper; fall back to nfl_data_py if needed
    try:
        df = trends._load_local_pbp(list(seasons))  # type: ignore[attr-defined]
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        try:
            df = trends._safe_import_pbp(list(seasons))  # type: ignore[attr-defined]
        except Exception:
            df = pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    try:
        rosters = trends._safe_import_rosters(list(seasons))  # type: ignore[attr-defined]
    except Exception:
        rosters = pd.DataFrame()
    return rosters if rosters is not None else pd.DataFrame()


def id_to_pos_map(rosters: pd.DataFrame) -> Dict[str, str]:
    if rosters is None or rosters.empty:
        return {}
    out: Dict[str, str] = {}
    # Try common id columns
    id_cols = [c for c in ["gsis_id", "player_id", "nfl_id", "pfr_id"] if c in rosters.columns]
    pos_col = "position" if "position" in rosters.columns else None
    if not id_cols or not pos_col:
        return {}
    for _, r in rosters.iterrows():
        pid = None
        for c in id_cols:
            v = r.get(c)
            if pd.notna(v) and str(v).strip():
                pid = str(v).strip()
                break
        if not pid:
            continue
        pos = _canonical_pos_keep_fb(str(r.get(pos_col) or ""))
        out[pid] = pos
    return out


def summarize_weekly_distributions(current_season: int, week: int, seasons: Iterable[int]) -> None:
    pbp = load_pbp(seasons)
    if pbp is None or pbp.empty:
        print("No PBP data available to compute historical baselines.")
        return

    # Ensure minimal columns
    for c in ["season", "week", "rush_touchdown", "pass_touchdown", "rusher_player_id", "receiver_player_id"]:
        if c not in pbp.columns:
            pbp[c] = pd.NA

    pbp["is_rush_td"] = pd.to_numeric(pbp["rush_touchdown"], errors="coerce").fillna(0).astype(int)
    pbp["is_rec_td"] = pd.to_numeric(pbp["pass_touchdown"], errors="coerce").fillna(0).astype(int)

    # League-wide weekly totals across seasons
    wk = pbp.groupby(["season", "week"]).agg(
        rush_td=("is_rush_td", "sum"),
        pass_td=("is_rec_td", "sum"),
    ).reset_index()
    wk["total_td"] = wk["rush_td"] + wk["pass_td"]

    # Baseline for this week index (e.g., Week 1) across selected seasons
    wkN = wk[wk["week"] == int(week)].copy()
    baseline = {
        "rush_td_avg": float(wkN["rush_td"].mean()),
        "pass_td_avg": float(wkN["pass_td"].mean()),
        "total_td_avg": float(wkN["total_td"].mean()),
        "total_td_p25": float(wkN["total_td"].quantile(0.25)),
        "total_td_p75": float(wkN["total_td"].quantile(0.75)),
    }

    # Position shares for pass and rush TDs (for this week index across years)
    rosters = load_rosters(seasons)
    pos_map = id_to_pos_map(rosters)

    def map_pos(col: str) -> pd.Series:
        return pbp[col].astype(str).map(lambda x: pos_map.get(x, ""))

    pbp_week = pbp[pbp["week"] == int(week)].copy()
    pbp_week["rush_pos"] = map_pos("rusher_player_id")
    pbp_week["rec_pos"] = map_pos("receiver_player_id")

    rush_counts = (
        pbp_week[pbp_week["is_rush_td"] == 1]["rush_pos"].apply(_canonical_pos_keep_fb)
        .value_counts()
    )
    pass_counts = (
        pbp_week[pbp_week["is_rec_td"] == 1]["rec_pos"].apply(_canonical_pos_keep_fb)
        .value_counts()
    )

    def shares(s: pd.Series) -> pd.Series:
        t = float(s.sum())
        return (s / t) if t > 0 else s * 0

    rush_shares = shares(rush_counts)
    pass_shares = shares(pass_counts)

    # Load our current predicted CSV and compute analogous metrics
    pred_fp = DATA_DIR / f"player_td_likelihood_{current_season}_wk{week}.csv"
    pred = pd.read_csv(pred_fp) if pred_fp.exists() else pd.DataFrame()
    if pred.empty:
        print(f"Predictions file not found: {pred_fp}")
        return

    pred_tot = float(pred["expected_td"].sum())
    pred_rush = pred.groupby("position")["exp_rush_td"].sum()
    pred_pass = pred.groupby("position")["exp_rec_td"].sum()
    pred_rush_share = (pred_rush / pred_rush.sum()).fillna(0.0)
    pred_pass_share = (pred_pass / pred_pass.sum()).fillna(0.0)

    # Print concise comparison
    def fmt_pct(x: float) -> str:
        return f"{x*100:.2f}%"

    print("Historical baselines (" + ", ".join(str(s) for s in seasons) + f") for Week {week}:")
    print(f"  Avg total TDs: {baseline['total_td_avg']:.1f}  (P25 {baseline['total_td_p25']:.1f}, P75 {baseline['total_td_p75']:.1f})")
    print(f"  Avg pass TDs:  {baseline['pass_td_avg']:.1f}")
    print(f"  Avg rush TDs:  {baseline['rush_td_avg']:.1f}")

    # Normalize series to include common keys
    def norm_keys(series: pd.Series, keys: list[str]) -> pd.Series:
        out = pd.Series({k: 0.0 for k in keys}, dtype=float)
        for k, v in series.items():
            out[k] = float(v)
        return out

    pass_keys = ["WR", "TE", "RB", "FB", "QB", "OTHER"]
    rush_keys = ["RB", "QB", "FB", "WR", "TE", "OTHER"]

    hs_pass = norm_keys(pass_shares, pass_keys)
    hs_rush = norm_keys(rush_shares, rush_keys)
    ps_pass = norm_keys(pred_pass_share, pass_keys)
    ps_rush = norm_keys(pred_rush_share, rush_keys)

    print("\nReceiving TD share by position (historical vs. predicted):")
    for k in pass_keys:
        print(f"  {k:>5}: hist {fmt_pct(hs_pass.get(k, 0.0))} | pred {fmt_pct(ps_pass.get(k, 0.0))}")

    print("\nRushing TD share by position (historical vs. predicted):")
    for k in rush_keys:
        print(f"  {k:>5}: hist {fmt_pct(hs_rush.get(k, 0.0))} | pred {fmt_pct(ps_rush.get(k, 0.0))}")

    # Predicted total vs baseline
    print(f"\nPredicted total expected TDs (week {week}): {pred_tot:.1f}")


if __name__ == "__main__":
    # Compare Week 1 vs last 5 completed seasons (exclude current)
    CURRENT_SEASON = 2025
    WEEK = 1
    seasons = list(range(2020, 2025))
    summarize_weekly_distributions(CURRENT_SEASON, WEEK, seasons)
