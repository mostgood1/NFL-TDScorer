from __future__ import annotations

from pathlib import Path
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data_sources import load_games, load_team_stats, load_lines
from .features import merge_features
from .weather import load_weather_for_games


# Point to NFL-Touchdown/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TD_RATE = float(os.environ.get("TD_RATE", "0.74"))  # fraction of points/7 that become offensive TDs


def _implied_points(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    total = row.get("total")
    spread_home = row.get("spread_home")
    if (pd.isna(total) or total is None) and ("close_total" in row):
        total = row.get("close_total")
    if (pd.isna(spread_home) or spread_home is None) and ("close_spread_home" in row):
        spread_home = row.get("close_spread_home")
    try:
        total_f = float(total)
        spread_f = float(spread_home)
    except Exception:
        return None, None
    margin = -spread_f
    home_pts = (total_f + margin) / 2.0
    away_pts = total_f - home_pts
    return home_pts, away_pts


def _weather_factor(row: pd.Series) -> float:
    is_dome = 0.0
    if "is_dome" in row and pd.notna(row["is_dome"]):
        try:
            is_dome = float(row["is_dome"]) or 0.0
        except Exception:
            is_dome = 0.0
    if is_dome >= 0.5:
        return 1.0
    wind = row.get("wx_wind_mph")
    precip = row.get("wx_precip_pct")
    factor = 1.0
    try:
        w = float(wind)
        if w > 15.0:
            over = max(0.0, w - 15.0)
            factor *= max(0.85, 1.0 - 0.025 * (over / 5.0))
    except Exception:
        pass
    try:
        p = float(precip)
        if p >= 60.0:
            factor *= 0.925
    except Exception:
        pass
    return float(factor)


def _pace_factor(pace_secs_play_prior: Optional[float], league_avg_pace: float) -> float:
    try:
        if pace_secs_play_prior is None or pd.isna(pace_secs_play_prior):
            return 1.0
        f = float(league_avg_pace) / float(pace_secs_play_prior)
        return float(np.clip(f, 0.9, 1.1))
    except Exception:
        return 1.0


def _epa_factor(off_prior: Optional[float], opp_def_prior: Optional[float], beta: float = 2.0) -> float:
    try:
        o = float(off_prior) if off_prior is not None else 0.0
        d = float(opp_def_prior) if opp_def_prior is not None else 0.0
        diff = o - d
        val = float(np.exp(beta * diff))
        return float(np.clip(val, 0.7, 1.3))
    except Exception:
        return 1.0


def _team_rows_from_game(row: pd.Series, league_avg_pace: float) -> list[dict]:
    # Baseline points from market implied total, split by spread
    home_pts, away_pts = _implied_points(row)
    if home_pts is None or away_pts is None:
        total = 44.0
        home_pts = total / 2.0
        away_pts = total / 2.0
    # Global game weather impact (both teams share environment)
    wf = _weather_factor(row)
    # Team-specific relative factors (don’t inflate game total): EPA and pace
    home_epa_prior = row.get("home_off_epa_prior")
    away_epa_prior = row.get("away_off_epa_prior")
    home_opp_def_prior = row.get("away_def_epa_prior")
    away_opp_def_prior = row.get("home_def_epa_prior")
    home_pace_prior = row.get("home_pace_prior")
    away_pace_prior = row.get("away_pace_prior")
    f_home = _epa_factor(home_epa_prior, home_opp_def_prior) * _pace_factor(home_pace_prior, league_avg_pace)
    f_away = _epa_factor(away_epa_prior, away_opp_def_prior) * _pace_factor(away_pace_prior, league_avg_pace)
    # Conserve baseline total TDs: distribute base_game across teams by relative strength
    # Scale points/7 by league TD rate to better match historical weekly totals
    base_game = max(0.05, float(home_pts + away_pts) / 7.0 * TD_RATE)
    denom = float(f_home + f_away) if (f_home is not None and f_away is not None) else 0.0
    if denom <= 0:
        w_home = 0.5
        w_away = 0.5
    else:
        w_home = float(f_home) / denom
        w_away = float(f_away) / denom
    # Apply weather to the game total, not individually per team
    total_lambda = base_game * float(wf)
    home_lambda = max(0.0, total_lambda * w_home)
    away_lambda = max(0.0, total_lambda * w_away)
    home_p_td = float(1.0 - np.exp(-home_lambda))
    away_p_td = float(1.0 - np.exp(-away_lambda))
    return [
        {
            "season": row.get("season"),
            "week": row.get("week"),
            "game_id": row.get("game_id"),
            "date": row.get("date"),
            "team": row.get("home_team"),
            "opponent": row.get("away_team"),
            "is_home": 1,
            "implied_points": float(home_pts),
            "expected_tds": float(home_lambda),
            "td_likelihood": home_p_td,
            "off_epa_prior": home_epa_prior,
            "opp_def_epa_prior": home_opp_def_prior,
            "pace_prior": home_pace_prior,
        },
        {
            "season": row.get("season"),
            "week": row.get("week"),
            "game_id": row.get("game_id"),
            "date": row.get("date"),
            "team": row.get("away_team"),
            "opponent": row.get("home_team"),
            "is_home": 0,
            "implied_points": float(away_pts),
            "expected_tds": float(away_lambda),
            "td_likelihood": away_p_td,
            "off_epa_prior": away_epa_prior,
            "opp_def_epa_prior": away_opp_def_prior,
            "pace_prior": away_pace_prior,
        },
    ]


def compute_td_likelihood(season: Optional[int] = None, week: Optional[int] = None) -> pd.DataFrame:
    games = load_games()
    team_stats = load_team_stats()
    lines = load_lines()

    if games is None:
        games = pd.DataFrame()
    if season is not None:
        games = games[games.get("season").astype("Int64") == int(season)].copy() if not games.empty else games
    if week is not None:
        games = games[games.get("week").astype("Int64") == int(week)].copy() if not games.empty else games

    def _synth_from_lines(_lines: pd.DataFrame, _season: Optional[int], _week: Optional[int]) -> pd.DataFrame:
        if _lines is None or _lines.empty:
            return pd.DataFrame()
        df = _lines.copy()
        if _season is not None:
            df = df[df.get("season").astype("Int64") == int(_season)]
        if _week is not None:
            df = df[df.get("week").astype("Int64") == int(_week)]
        if df.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "season": df.get("season"),
            "week": df.get("week"),
            "game_id": df.get("game_id", pd.Series([pd.NA]*len(df))),
            "date": df.get("date", pd.Series([pd.NA]*len(df))),
            "home_team": df.get("home_team"),
            "away_team": df.get("away_team"),
            "home_score": pd.NA,
            "away_score": pd.NA,
        })
        out = out.dropna(subset=["home_team","away_team"])
        def _mk_gid(r):
            if pd.notna(r.get("game_id")) and str(r.get("game_id")).strip() != "":
                return r["game_id"]
            s = str(r.get("season")) if pd.notna(r.get("season")) else ""
            w = str(r.get("week")) if pd.notna(r.get("week")) else ""
            ht = str(r.get("home_team", "")).split()[-1][:3].upper()
            at = str(r.get("away_team", "")).split()[-1][:3].upper()
            return f"{s}-{w:0>2}-{ht}-{at}"
        out["game_id"] = out.apply(_mk_gid, axis=1)
        return out

    need_upcoming = (season is None and week is None)
    if games.empty:
        synth = _synth_from_lines(lines, season, week)
        if synth is not None and not synth.empty:
            games = synth
        else:
            return pd.DataFrame(columns=["season","week","game_id","team","opponent","is_home","implied_points","expected_tds","td_likelihood"]) 

    try:
        wx = load_weather_for_games(games)
    except Exception:
        wx = None

    feat = merge_features(games, team_stats, lines, wx)

    if need_upcoming:
        target = feat[feat["home_score"].isna() | feat["away_score"].isna()].copy()
    else:
        target = feat.copy()

    if target.empty:
        return pd.DataFrame(columns=["season","week","game_id","team","opponent","is_home","implied_points","expected_tds","td_likelihood"]) 

    pace_cols = []
    for c in ["home_pace_prior","away_pace_prior","home_pace_secs_play","away_pace_secs_play"]:
        if c in target.columns:
            pace_cols.append(c)
    if pace_cols:
        league_avg_pace = pd.to_numeric(pd.concat([target[c] for c in pace_cols], axis=0), errors="coerce").dropna().mean()
    else:
        league_avg_pace = 27.5
    if pd.isna(league_avg_pace):
        league_avg_pace = 27.5

    rows: list[dict] = []
    for _, r in target.iterrows():
        rows.extend(_team_rows_from_game(r, float(league_avg_pace)))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    try:
        ranks = out["td_likelihood"].rank(method="min").astype(float)
        out["td_score"] = ((ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9) * 100.0).round(2)
    except Exception:
        out["td_score"] = (out["td_likelihood"].fillna(0.0) * 100.0).round(2)

    cols = [
        "season","week","date","game_id","team","opponent","is_home",
        "implied_points","expected_tds","td_likelihood","td_score",
        "off_epa_prior","opp_def_epa_prior","pace_prior",
        "home_pass_rate_prior","away_pass_rate_prior","home_rush_rate_prior","away_rush_rate_prior",
    ]
    out = out.reindex(columns=[c for c in cols if c in out.columns])
    return out.sort_values(["season","week","td_score"], ascending=[True, True, False])
