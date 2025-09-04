from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List
from pathlib import Path

import pandas as pd
import numpy as np

from .team_normalizer import normalize_team_name


# Local data dir
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class TrendsPaths:
    team_pos_shares: Path
    player_td_counts: Path
    def_pos_allowed_shares: Path


def _paths(start_season: int, end_season: int) -> TrendsPaths:
    return TrendsPaths(
        team_pos_shares=DATA_DIR / f"team_pos_td_shares_{start_season}_{end_season}.csv",
        player_td_counts=DATA_DIR / f"player_td_counts_{start_season}_{end_season}.csv",
        def_pos_allowed_shares=DATA_DIR / f"def_pos_td_allowed_shares_{start_season}_{end_season}.csv",
    )


def last_n_seasons(current_season: int, n: int = 5) -> list[int]:
    end_season = int(current_season) - 1
    start_season = max(1999, end_season - (n - 1))
    return list(range(start_season, end_season + 1))


def _read_pbp_csvs(files: List[Path], seasons: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if "season" in df.columns:
                df = df[df["season"].isin(list(seasons))]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    try:
        out = pd.concat(frames, ignore_index=True)
    except Exception:
        out = frames[0]
    return out


def _load_local_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    # 1) Env var override
    env_dir = Path(str((Path.cwd() / "").joinpath("")))  # no-op, just to keep Path in scope
    try:
        import os
        env_val = os.environ.get("NFL_PBP_DIR")
        if env_val:
            p = Path(env_val)
            if p.exists() and p.is_dir():
                pats = [f"*{y}*.csv" for y in seasons]
                files = []
                for pat in pats:
                    files.extend(list(p.glob(pat)))
                if files:
                    return _read_pbp_csvs(files, seasons)
    except Exception:
        pass
    # 2) Look in local data dir with common patterns
    patterns = []
    for y in seasons:
        y = int(y)
        patterns.extend([
            f"pbp_{y}.csv",
            f"nflfastr_pbp_{y}.csv",
            f"nflfastR_pbp_{y}.csv",
            f"play_by_play_{y}.csv",
            f"pbp_{y}_regular.csv",
        ])
    files: List[Path] = []
    for pat in patterns:
        files.extend(list(DATA_DIR.glob(pat)))
    if files:
        return _read_pbp_csvs(files, seasons)
    return pd.DataFrame()


def _safe_import_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    # Prefer local files if present
    local = _load_local_pbp(seasons)
    if local is not None and not local.empty:
        return local
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return pd.DataFrame()
    try:
        return nfl.import_pbp_data(list(seasons))
    except Exception:
        # Some versions expose a different helper
        try:
            return nfl.import_pbp(list(seasons))  # type: ignore
        except Exception:
            return pd.DataFrame()


def _safe_import_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return pd.DataFrame()
    for fn in ("import_seasonal_rosters", "import_rosters"):
        try:
            rosters = getattr(nfl, fn)(list(seasons))  # type: ignore
            if rosters is not None and not rosters.empty:
                return rosters
        except Exception:
            continue
    return pd.DataFrame()


def _position_for_player_id(pid: str, rosters: pd.DataFrame) -> Optional[str]:
    if not pid or rosters is None or rosters.empty:
        return None
    for id_col in ["gsis_id", "player_id", "nfl_id", "pfr_id"]:
        if id_col in rosters.columns:
            m = rosters[rosters[id_col].astype(str) == str(pid)]
            if not m.empty:
                pos_col = "position" if "position" in m.columns else None
                if pos_col:
                    v = str(m.iloc[0][pos_col] or "").strip().upper()
                    return v if v else None
    return None


def _canonical_pos(pos: Optional[str]) -> str:
    s = (pos or "").strip().upper()
    if s in {"HB", "FB", "RB"}:
        return "RB"
    if s in {"WR"}:
        return "WR"
    if s in {"TE"}:
        return "TE"
    if s in {"QB"}:
        return "QB"
    return "OTHER"


def compute_td_trends(current_season: int, seasons: Optional[Iterable[int]] = None) -> dict:
    """Compute last-N seasons team position TD shares and per-player TD counts.

    Returns a dict with keys: team_pos_shares (DataFrame), player_td_counts (DataFrame),
    start_season, end_season.
    """
    if seasons is None:
        seasons = last_n_seasons(current_season, 5)
    seasons = sorted(int(s) for s in seasons)
    if not seasons:
        return {"team_pos_shares": pd.DataFrame(), "player_td_counts": pd.DataFrame(), "start_season": None, "end_season": None}

    start_season, end_season = seasons[0], seasons[-1]
    pbp = _safe_import_pbp(seasons)
    rosters = _safe_import_rosters(seasons)

    if pbp is None or pbp.empty:
        return {"team_pos_shares": pd.DataFrame(), "player_td_counts": pd.DataFrame(), "start_season": start_season, "end_season": end_season}

    df = pbp.copy()
    # Minimal columns
    for c in ["season", "posteam", "defteam", "rush_touchdown", "pass_touchdown", "rusher_player_id", "receiver_player_id", "rusher_player_name", "receiver_player_name"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["team"] = df["posteam"].astype(str).apply(normalize_team_name)
    df["def_team"] = df["defteam"].astype(str).apply(normalize_team_name)
    df["is_rush_td"] = pd.to_numeric(df["rush_touchdown"], errors="coerce").fillna(0).astype(int)
    df["is_rec_td"] = pd.to_numeric(df["pass_touchdown"], errors="coerce").fillna(0).astype(int)

    # Build per-play player and position
    def _row_pos_player(r: pd.Series) -> tuple[str, Optional[str], Optional[str]]:
        team = str(r.get("team") or "")
        if r.get("is_rush_td") == 1:
            pid = r.get("rusher_player_id")
            pos = _position_for_player_id(pid, rosters)
            if not pos and r.get("rusher_player_name"):
                pos = None
            return team, _canonical_pos(pos), r.get("rusher_player_name")
        if r.get("is_rec_td") == 1:
            pid = r.get("receiver_player_id")
            pos = _position_for_player_id(pid, rosters)
            if not pos and r.get("receiver_player_name"):
                pos = None
            return team, _canonical_pos(pos), r.get("receiver_player_name")
        return team, None, None

    pos_rows = []
    player_rows = []
    def_pos_rows = []
    for _, r in df.iterrows():
        if int(r.get("is_rush_td") or 0) == 1 or int(r.get("is_rec_td") or 0) == 1:
            team, pos, name = _row_pos_player(r)
            kind = "rush" if int(r.get("is_rush_td") or 0) == 1 else "pass"
            if pos:
                pos_rows.append({"team": team, "kind": kind, "position": pos, "count": 1})
                # Defensive allowed (use position of scorer)
                def_team = str(r.get("def_team") or "")
                if def_team:
                    def_pos_rows.append({"team": def_team, "kind": kind, "position": pos, "count": 1})
            if name:
                player_rows.append({"team": team, "kind": kind, "player": str(name), "count": 1})

    team_pos = pd.DataFrame(pos_rows)
    player_counts = pd.DataFrame(player_rows)
    def_pos = pd.DataFrame(def_pos_rows)

    if team_pos.empty:
        team_pos_shares = pd.DataFrame()
    else:
        # Aggregate and pivot to shares
        agg = team_pos.groupby(["team", "kind", "position"]).agg(td_count=("count", "sum")).reset_index()
        piv = agg.pivot_table(index=["team", "kind"], columns="position", values="td_count", aggfunc="sum", fill_value=0).reset_index()
        for col in ["WR", "TE", "RB", "QB", "OTHER"]:
            if col not in piv.columns:
                piv[col] = 0
        piv["total"] = piv[["WR", "TE", "RB", "QB", "OTHER"]].sum(axis=1).replace(0, np.nan)
        for col in ["WR", "TE", "RB", "QB", "OTHER"]:
            piv[col] = (piv[col] / piv["total"]).fillna(0.0)
        team_pos_shares = piv

    if not player_counts.empty:
        pc = player_counts.groupby(["team", "kind", "player"]).agg(td_count=("count", "sum")).reset_index()
    else:
        pc = pd.DataFrame(columns=["team", "kind", "player", "td_count"])

    if def_pos.empty:
        def_pos_shares = pd.DataFrame()
    else:
        dagg = def_pos.groupby(["team", "kind", "position"]).agg(td_count=("count", "sum")).reset_index()
        dpiv = dagg.pivot_table(index=["team", "kind"], columns="position", values="td_count", aggfunc="sum", fill_value=0).reset_index()
        for col in ["WR", "TE", "RB", "QB", "OTHER"]:
            if col not in dpiv.columns:
                dpiv[col] = 0
        dpiv["total"] = dpiv[["WR", "TE", "RB", "QB", "OTHER"]].sum(axis=1).replace(0, np.nan)
        for col in ["WR", "TE", "RB", "QB", "OTHER"]:
            dpiv[col] = (dpiv[col] / dpiv["total"]).fillna(0.0)
        def_pos_shares = dpiv

    return {
    "team_pos_shares": team_pos_shares,
    "player_td_counts": pc,
    "def_pos_allowed_shares": def_pos_shares,
        "start_season": start_season,
        "end_season": end_season,
    }


def build_and_cache_trends(current_season: int, seasons: Optional[Iterable[int]] = None) -> TrendsPaths:
    seasons_list = sorted(int(s) for s in (seasons or last_n_seasons(current_season, 5)))
    start_season, end_season = seasons_list[0], seasons_list[-1]
    res = compute_td_trends(current_season, seasons_list)
    paths = _paths(start_season, end_season)
    if res["team_pos_shares"] is not None and not res["team_pos_shares"].empty:
        paths.team_pos_shares.parent.mkdir(parents=True, exist_ok=True)
        res["team_pos_shares"].to_csv(paths.team_pos_shares, index=False)
    if res["player_td_counts"] is not None and not res["player_td_counts"].empty:
        paths.player_td_counts.parent.mkdir(parents=True, exist_ok=True)
        res["player_td_counts"].to_csv(paths.player_td_counts, index=False)
    if res.get("def_pos_allowed_shares") is not None and not res.get("def_pos_allowed_shares").empty:
        paths.def_pos_allowed_shares.parent.mkdir(parents=True, exist_ok=True)
        res["def_pos_allowed_shares"].to_csv(paths.def_pos_allowed_shares, index=False)
    return paths


def load_team_pos_shares(current_season: int, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
    seasons_list = sorted(int(s) for s in (seasons or last_n_seasons(current_season, 5)))
    start_season, end_season = seasons_list[0], seasons_list[-1]
    paths = _paths(start_season, end_season)
    if paths.team_pos_shares.exists():
        try:
            return pd.read_csv(paths.team_pos_shares)
        except Exception:
            pass
    # Compute on the fly if not cached
    res = compute_td_trends(current_season, seasons_list)
    tps = res.get("team_pos_shares")
    return tps if tps is not None else pd.DataFrame()


def load_def_pos_allowed_shares(current_season: int, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
    seasons_list = sorted(int(s) for s in (seasons or last_n_seasons(current_season, 5)))
    if not seasons_list:
        return pd.DataFrame()
    start_season, end_season = seasons_list[0], seasons_list[-1]
    paths = _paths(start_season, end_season)
    if paths.def_pos_allowed_shares.exists():
        try:
            return pd.read_csv(paths.def_pos_allowed_shares)
        except Exception:
            pass
    res = compute_td_trends(current_season, seasons_list)
    dps = res.get("def_pos_allowed_shares")
    return dps if dps is not None else pd.DataFrame()


def load_player_td_counts(current_season: int, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Load cached player TD counts for last-N seasons, or compute if absent.

    Returns a DataFrame with columns: team, kind ("rush"|"pass"), player, td_count.
    """
    seasons_list = sorted(int(s) for s in (seasons or last_n_seasons(current_season, 5)))
    if not seasons_list:
        return pd.DataFrame(columns=["team","kind","player","td_count"])
    start_season, end_season = seasons_list[0], seasons_list[-1]
    paths = _paths(start_season, end_season)
    if paths.player_td_counts.exists():
        try:
            return pd.read_csv(paths.player_td_counts)
        except Exception:
            pass
    res = compute_td_trends(current_season, seasons_list)
    pc = res.get("player_td_counts")
    return pc if pc is not None else pd.DataFrame(columns=["team","kind","player","td_count"])
