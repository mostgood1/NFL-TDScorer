from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from .td_likelihood import compute_td_likelihood
from .team_normalizer import normalize_team_name
from .data_sources import DATA_DIR as TEAM_DATA_DIR
from .pbp_td_trends import load_team_pos_shares, load_player_td_counts, load_def_pos_allowed_shares


DATA_DIR = TEAM_DATA_DIR

# Known fullbacks to reclassify regardless of roster position labeling
FULLBACK_OVERRIDES = {
    name.lower(): 'FB' for name in [
        'Reggie Gilliam', 'Kyle Juszczyk', 'Alec Ingold', 'Andrew Beck',
        'Khari Blasingame', 'C.J. Ham', 'CJ Ham', 'Jakob Johnson',
        'Patrick Ricard', 'Zander Horvath', 'Nick Bellore', 'Ben Mason',
        'Gabe Nabers', 'Michael Burton', 'Connor Heyward'
    ]
}


def _abbr_to_name_map() -> dict[str, str]:
    # Minimal static map to translate abbreviations from PBP to full team names used elsewhere
    return {
        "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
        "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
        "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
        "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
        "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
        "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    }


def _team_td_per_point(seasons: list[int]) -> tuple[dict[str, float], dict[str, float], float]:
    # Points for/against from games.csv keyed by full team names
    pts_for: dict[str, float] = {}
    pts_against: dict[str, float] = {}
    gfp = DATA_DIR / "games.csv"
    if gfp.exists():
        try:
            gdf = pd.read_csv(gfp)
            if "season" in gdf.columns:
                gdf = gdf[gdf["season"].astype(str).isin([str(s) for s in seasons])]
            if not gdf.empty:
                for _, r in gdf.iterrows():
                    h = str(r.get("home_team") or "").strip()
                    a = str(r.get("away_team") or "").strip()
                    hp = float(pd.to_numeric(r.get("home_score"), errors="coerce") or 0.0)
                    ap = float(pd.to_numeric(r.get("away_score"), errors="coerce") or 0.0)
                    if h:
                        pts_for[h] = pts_for.get(h, 0.0) + hp
                        pts_against[h] = pts_against.get(h, 0.0) + ap
                    if a:
                        pts_for[a] = pts_for.get(a, 0.0) + ap
                        pts_against[a] = pts_against.get(a, 0.0) + hp
        except Exception:
            pass
    # Offensive TDs (by posteam) and allowed (by defteam) from PBP
    abmap = _abbr_to_name_map()
    off_td: dict[str, int] = {}
    def_td_allowed: dict[str, int] = {}
    for fp in sorted(DATA_DIR.glob("pbp_*.csv")):
        try:
            import re as _re
            m = _re.search(r"pbp_(\d{4})", fp.name)
            if m and int(m.group(1)) not in seasons:
                continue
        except Exception:
            pass
        try:
            dfp = pd.read_csv(fp, usecols=lambda c: c in {
                "rush_touchdown","pass_touchdown","posteam","offense_team","defteam","defense_team"
            })
        except Exception:
            try:
                dfp = pd.read_csv(fp)
            except Exception:
                continue
        for col in ("rush_touchdown","pass_touchdown"):
            if col in dfp.columns:
                dfp[col] = pd.to_numeric(dfp[col], errors="coerce").fillna(0).astype(int)
            else:
                dfp[col] = 0
        dfp["is_off_td"] = ((dfp["rush_touchdown"] == 1) | (dfp["pass_touchdown"] == 1)).astype(int)
        off_col = "posteam" if "posteam" in dfp.columns else ("offense_team" if "offense_team" in dfp.columns else None)
        def_col = "defteam" if "defteam" in dfp.columns else ("defense_team" if "defense_team" in dfp.columns else None)
        if off_col is not None:
            for _, r in dfp[dfp["is_off_td"] == 1].iterrows():
                ab = str(r.get(off_col) or "").strip().upper()
                name = abmap.get(ab, ab)
                off_td[name] = off_td.get(name, 0) + 1
        if def_col is not None:
            for _, r in dfp[dfp["is_off_td"] == 1].iterrows():
                ab = str(r.get(def_col) or "").strip().upper()
                name = abmap.get(ab, ab)
                def_td_allowed[name] = def_td_allowed.get(name, 0) + 1
    offense_tpp: dict[str, float] = {}
    defense_tpp: dict[str, float] = {}
    for team, td in off_td.items():
        pts = float(pts_for.get(team, 0.0))
        if pts > 0:
            offense_tpp[team] = float(td) / pts
    for team, td in def_td_allowed.items():
        pts = float(pts_against.get(team, 0.0))
        if pts > 0:
            defense_tpp[team] = float(td) / pts
    league_avg = (sum(offense_tpp.values()) / max(len(offense_tpp), 1)) if offense_tpp else 0.06
    if league_avg <= 0:
        league_avg = 0.06
    return offense_tpp, defense_tpp, float(league_avg)


def _apply_tpp_to_teams(teams_df: pd.DataFrame, season: int) -> dict[tuple[object, str], float]:
    # Use prior season where possible
    seasons = [season - 1] if season and season >= 2025 else [season]
    off_tpp, def_tpp, league = _team_td_per_point(seasons)
    # Prepare adjustment per team row, then renormalize per game to conserve the game sum
    base_sum_by_game = teams_df.groupby("game_id")["expected_tds"].sum().to_dict()
    raw_adj: dict[tuple[object, str], float] = {}
    for _, r in teams_df.iterrows():
        gid = r.get("game_id")
        team = str(r.get("team"))
        opp = str(r.get("opponent"))
        base = float(r.get("expected_tds") or 0.0)
        off = off_tpp.get(team)
        deff = def_tpp.get(opp)
        if off is None and deff is None:
            target = base
        else:
            blended = (0.6 * off if off is not None else 0.0) + (0.4 * deff if deff is not None else 0.0)
            scale = (blended / league) if (blended and league) else 1.0
            target = base * float(scale)
        raw_adj[(gid, team)] = target
    # Renormalize per game to conserve original game sum
    adj: dict[tuple[object, str], float] = {}
    from collections import defaultdict
    sum_adj_by_game: dict[object, float] = defaultdict(float)
    for (gid, team), val in raw_adj.items():
        sum_adj_by_game[gid] += float(val)
    for (gid, team), val in raw_adj.items():
        base_total = float(base_sum_by_game.get(gid, 0.0))
        adj_total = float(sum_adj_by_game.get(gid, 0.0))
        if adj_total > 0:
            factor = base_total / adj_total if adj_total > 0 else 1.0
        else:
            factor = 1.0
        adj[(gid, team)] = float(val) * float(factor)
    return adj


def _load_player_usage() -> pd.DataFrame:
    fp = DATA_DIR / "player_usage_priors.csv"
    if not fp.exists():
        return pd.DataFrame(columns=[
            "season","team","player","position","rush_share","target_share","rz_rush_share","rz_target_share"
        ])
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame(columns=[
            "season","team","player","position","rush_share","target_share","rz_rush_share","rz_target_share"
        ])
    return df


def _enrich_player_names(players_df: pd.DataFrame, season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return players_df
    try:
        ros = nfl.import_seasonal_rosters([season])
    except Exception:
        return players_df
    if ros is None or ros.empty:
        return players_df
    df = players_df.copy()
    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    if team_src is None:
        return df
    ros = ros.copy()
    ros['team_norm'] = ros[team_src].astype(str).apply(normalize_team_name)
    def best_name(r: pd.Series) -> str:
        for k in ['display_name','player_display_name','full_name','player_name','football_name']:
            v = r.get(k)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        fn = str(r.get('first_name') or '').strip()
        ln = str(r.get('last_name') or '').strip()
        nm = f"{fn} {ln}".strip()
        return nm if nm else str(r.get('gsis_id') or '').strip()
    ros['best_name'] = ros.apply(best_name, axis=1)
    pos_col = 'position' if 'position' in ros.columns else None
    def first_token(s: str) -> str:
        parts = s.split()
        return parts[0] if parts else s
    def norm_token(s: str) -> str:
        return ''.join(ch for ch in s if ch.isalnum()).lower()
    lut_pos: dict[tuple[str, str, str], set[str]] = {}
    lut_any: dict[tuple[str, str], set[str]] = {}
    for _, rr in ros.iterrows():
        team_norm = str(rr['team_norm'])
        pos = str(rr.get(pos_col) or '') if pos_col else ''
        bn = best_name(rr)
        tok = norm_token(first_token(bn))
        fn = str(rr.get('first_name') or '')
        fn_tok = norm_token(first_token(fn)) if fn else ''
        for key_tok in filter(None, [tok, fn_tok]):
            kpos = (team_norm, pos, key_tok)
            lut_pos.setdefault(kpos, set()).add(rr['best_name'])
            kany = (team_norm, key_tok)
            lut_any.setdefault(kany, set()).add(rr['best_name'])
    def enrich_row(r: pd.Series) -> str:
        name = str(r.get('player') or '').strip()
        if not name:
            return name
        if ' ' in name:
            return name
        team_norm = normalize_team_name(str(r.get('team') or ''))
        pos = str(r.get('position') or '')
        tok = norm_token(name)
        cands = lut_pos.get((team_norm, pos, tok))
        if not cands:
            cands = lut_any.get((team_norm, tok))
        if not cands:
            return name
        if len(cands) == 1:
            return next(iter(cands))
        return name
    df['player'] = df.apply(enrich_row, axis=1)
    return df


def _load_rosters(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return pd.DataFrame()
    try:
        ros = nfl.import_seasonal_rosters([season])
        return ros if ros is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _apply_roster_positions(depth: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    """Use seasonal rosters to correct positions and cap FB influence.

    - Updates position from roster for matching names when available.
    - Flags FBs and caps their rushing share to <= 12% and receiving share near 0.
    - Redistributes any excess back to non-FB players within the same position group.
    """
    ros = _load_rosters(season)
    if ros is None or ros.empty:
        return depth

    team_src = 'team' if 'team' in ros.columns else ('recent_team' if 'recent_team' in ros.columns else ('team_abbr' if 'team_abbr' in ros.columns else None))
    name_cols = [c for c in ['display_name','player_display_name','full_name','player_name','football_name'] if c in ros.columns]
    pos_col = 'position' if 'position' in ros.columns else None
    if not team_src or not name_cols or not pos_col:
        return depth

    rteam = ros.copy()
    # Build a best_name similar to _enrich_player_names
    def best_name(r: pd.Series) -> str:
        for k in ['display_name','player_display_name','full_name','player_name','football_name']:
            v = r.get(k)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        fn = str(r.get('first_name') or '').strip()
        ln = str(r.get('last_name') or '').strip()
        nm = f"{fn} {ln}".strip()
        return nm if nm else str(r.get('gsis_id') or '').strip()
    rteam['best_name'] = rteam.apply(best_name, axis=1)
    rteam['team_norm'] = rteam[team_src].astype(str).apply(normalize_team_name)
    rteam = rteam[rteam['team_norm'] == team]
    if rteam.empty:
        return depth

    # Build a simple name index
    def name_set(r: pd.Series) -> set[str]:
        out = set()
        for c in name_cols:
            v = str(r.get(c) or '').strip()
            if v:
                out.add(v)
        fn = str(r.get('first_name') or '').strip()
        ln = str(r.get('last_name') or '').strip()
        if fn or ln:
            out.add(f"{fn} {ln}".strip())
            out.add(fn)
            out.add(ln)
        return out

    roster_entries = []
    roster_name_map = {}
    # Build unique-first-name mapping within team to canonical best_name
    if 'first_name' in rteam.columns:
        fn_counts = rteam['first_name'].astype(str).str.strip().str.lower().value_counts()
        unique_firsts = set(fn_counts[fn_counts == 1].index)
        unique_first_map = {
            str(row.get('first_name') or '').strip().lower(): row.get('best_name')
            for _, row in rteam.iterrows()
            if str(row.get('first_name') or '').strip().lower() in unique_firsts
        }
    else:
        unique_first_map = {}
    for _, rr in rteam.iterrows():
        nm_set = name_set(rr)
        for nm in nm_set:
            key = nm.lower()
            roster_entries.append((key, str(rr.get(pos_col) or '').strip().upper()))
            # Store best display name for this key
            roster_name_map[key] = str(rr.get('best_name') or nm)
    roster_pos = dict(roster_entries)

    out = depth.copy()
    # Update positions and canonical names when a roster match exists
    if 'player' in out.columns and 'position' in out.columns:
        valid_pos = {"QB","RB","WR","TE","FB"}
        def upd_both(row: pd.Series) -> tuple[str, str]:
            nm_raw = str(row.get('player') or '').strip()
            nm = nm_raw.lower()
            if not nm:
                return nm_raw, row.get('position')
            # First, if we can map to a canonical roster name, update the player name
            canon = None
            if ' ' in nm_raw:
                canon = roster_name_map.get(nm)
            else:
                # Try unique-first-name canonicalization when unambiguous on this team
                cand = unique_first_map.get(nm)
                # Only accept if roster position matches prior group (to avoid e.g., Josh -> Josh Palmer WR)
                prior_pos = str(row.get('position') or '').upper()
                cand_pos = roster_pos.get(str(cand or '').lower())
                def _pos_compatible(prior: str, candp: str) -> bool:
                    if not candp:
                        return False
                    if prior == '' or prior is None:
                        return True
                    if prior == candp:
                        return True
                    # Allow RB<->FB mapping only
                    if {prior, candp} <= {"RB","FB"}:
                        return True
                    return False
                canon = cand if _pos_compatible(prior_pos, str(cand_pos or '').upper()) else None
            if canon:
                nm_out = canon
                nm = canon.lower()
            else:
                nm_out = nm_raw
            # Apply known fullback overrides by canonical full name
            if nm in FULLBACK_OVERRIDES:
                return nm_out, 'FB'
            # Fallback to roster position mapping
            cand_pos = roster_pos.get(nm)
            prior_pos = str(row.get('position') or '').upper()
            if cand_pos and cand_pos in valid_pos and (prior_pos in ("",) or cand_pos == prior_pos or (prior_pos == 'RB' and cand_pos == 'FB')):
                return nm_out, cand_pos
            return nm_out, row.get('position')
        out[['player','position']] = out.apply(lambda r: pd.Series(upd_both(r)), axis=1)

    # Cap FB impact: treat FB as RB but cap their shares
    def cap_fb_shares(df: pd.DataFrame, share_col: str, cap: float) -> pd.DataFrame:
        d = df.copy()
        if 'position' not in d.columns or share_col not in d.columns:
            return d
        is_fb = d['position'].astype(str).str.upper() == 'FB'
        if not is_fb.any():
            return d
        total = d[share_col].sum()
        if total <= 0:
            return d
        # Compute excess above cap across all FB rows
        fb_vals = d.loc[is_fb, share_col].clip(lower=0.0)
        excess = float(max(0.0, fb_vals.sum() - cap))
        if excess <= 1e-12:
            return d
        d.loc[is_fb, share_col] = fb_vals * max(0.0, (cap / (fb_vals.sum() + 1e-12)))
        # Redistribute excess proportionally to non-FB same-pos group (RB)
        rb_mask = d['position'].astype(str).str.upper().isin(['RB']) & (~is_fb)
        mass = float(d.loc[rb_mask, share_col].sum())
        if rb_mask.any():
            if mass > 0:
                d.loc[rb_mask, share_col] = d.loc[rb_mask, share_col] + (d.loc[rb_mask, share_col] / mass) * excess
            else:
                # Distribute equally among RBs if they exist but currently have 0 share
                cnt = int(rb_mask.sum())
                if cnt > 0:
                    d.loc[rb_mask, share_col] = d.loc[rb_mask, share_col] + (excess / cnt)
        else:
            # If no RBs present, distribute to WR/TE fairly
            other_mask = d['position'].astype(str).str.upper().isin(['WR','TE']) & (~is_fb)
            mass2 = float(d.loc[other_mask, share_col].sum())
            if mass2 > 0:
                d.loc[other_mask, share_col] = d.loc[other_mask, share_col] + (d.loc[other_mask, share_col] / mass2) * excess
        # Renormalize to 1 within that column
        s = float(d[share_col].sum())
        if s > 0:
            d[share_col] = d[share_col] / s
        return d

    # Apply caps: rushing <= 12% to FB combined, passing nearly zero
    for col, cap in [("rz_rush_share", 0.12), ("rush_share", 0.12)]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
            out = cap_fb_shares(out, col, cap)
    for col in ("rz_target_share","target_share"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
            # Zero out FB target shares and renormalize
            is_fb = out['position'].astype(str).str.upper() == 'FB'
            if is_fb.any():
                removed = float(out.loc[is_fb, col].sum())
                out.loc[is_fb, col] = 0.0
                mass = float(out.loc[~is_fb, col].sum())
                if mass > 0 and removed > 0:
                    out.loc[~is_fb, col] = out.loc[~is_fb, col] + (out.loc[~is_fb, col] / mass) * removed
                s = float(out[col].sum())
                if s > 0:
                    out[col] = out[col] / s
    # Collapse duplicate player rows and renormalize shares
    share_cols = [c for c in ["rush_share","target_share","rz_rush_share","rz_target_share"] if c in out.columns]
    if share_cols:
        out = out.groupby(["player","position"], as_index=False)[share_cols].sum()
        for c in share_cols:
            s = float(out[c].sum())
            if s > 0:
                out[c] = out[c] / s
    return out


def _default_team_depth(team: str) -> pd.DataFrame:
    rows = [
        {"player": f"{team} QB1", "position": "QB", "rush_share": 0.10, "target_share": 0.00, "rz_rush_share": 0.10, "rz_target_share": 0.00},
        {"player": f"{team} RB1", "position": "RB", "rush_share": 0.45, "target_share": 0.10, "rz_rush_share": 0.50, "rz_target_share": 0.08},
        {"player": f"{team} RB2", "position": "RB", "rush_share": 0.25, "target_share": 0.05, "rz_rush_share": 0.25, "rz_target_share": 0.05},
        {"player": f"{team} WR1", "position": "WR", "rush_share": 0.03, "target_share": 0.25, "rz_rush_share": 0.02, "rz_target_share": 0.25},
        {"player": f"{team} WR2", "position": "WR", "rush_share": 0.02, "target_share": 0.20, "rz_rush_share": 0.01, "rz_target_share": 0.20},
        {"player": f"{team} WR3", "position": "WR", "rush_share": 0.01, "target_share": 0.12, "rz_rush_share": 0.01, "rz_target_share": 0.12},
        {"player": f"{team} TE1", "position": "TE", "rush_share": 0.00, "target_share": 0.15, "rz_rush_share": 0.00, "rz_target_share": 0.20},
        {"player": f"{team} TE2", "position": "TE", "rush_share": 0.00, "target_share": 0.05, "rz_rush_share": 0.00, "rz_target_share": 0.10},
    ]
    return pd.DataFrame(rows)


def _normalize_shares(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        s = out[c].sum()
        if s > 0:
            out[c] = out[c] / s
    return out


def _team_player_shares(usage: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    if usage is None or usage.empty:
        base = _default_team_depth(team)
    else:
        u = usage.copy()
        u = u[(pd.to_numeric(u.get("season"), errors='coerce') == season) & (u.get("team").astype(str) == team)]
        base = u if not u.empty else _default_team_depth(team)
    base = _normalize_shares(base, ["rush_share","target_share","rz_rush_share","rz_target_share"])
    base["team"] = team
    return base


def _split_team_tds(row: pd.Series, exp_tds_override: Optional[float] = None) -> Dict[str, float]:
    exp_tds = float(exp_tds_override if exp_tds_override is not None else (row.get("expected_tds") or 0.0))
    is_home = int(row.get("is_home") or 0)
    if is_home:
        pr = row.get("home_pass_rate_prior")
        rr = row.get("home_rush_rate_prior")
    else:
        pr = row.get("away_pass_rate_prior")
        rr = row.get("away_rush_rate_prior")
    try:
        pr = float(pr) if pr is not None else np.nan
    except Exception:
        pr = np.nan
    try:
        rr = float(rr) if rr is not None else np.nan
    except Exception:
        rr = np.nan
    if not np.isfinite(pr) and not np.isfinite(rr):
        pr, rr = 0.58, 0.42
    elif not np.isfinite(pr) and np.isfinite(rr):
        pr = max(0.0, min(1.0, 1.0 - rr))
    elif not np.isfinite(rr) and np.isfinite(pr):
        rr = max(0.0, min(1.0, 1.0 - pr))
    w_rush = 0.35 * rr
    w_pass = 0.65 * pr
    s = w_rush + w_pass
    if s <= 0:
        w_rush, w_pass = 0.42, 0.58
        s = 1.0
    rush_tds = exp_tds * (w_rush / s)
    pass_tds = exp_tds * (w_pass / s)
    return {"rush_tds": float(rush_tds), "pass_tds": float(pass_tds)}


def compute_player_td_likelihood(season: int, week: int) -> pd.DataFrame:
    teams = compute_td_likelihood(season=season, week=week)
    if teams is None or teams.empty:
        return pd.DataFrame(columns=["season","week","team","player","position","anytime_td_prob","expected_td"])

    usage = _load_player_usage()
    # Load last-5-years team position TD shares (cached or computed)
    try:
        tps = load_team_pos_shares(season)
    except Exception:
        tps = pd.DataFrame()
    # Load last-5-years player TD counts
    try:
        ptc = load_player_td_counts(season)
    except Exception:
        ptc = pd.DataFrame(columns=["team","kind","player","td_count"])
    # Load opponent defensive position TD shares allowed
    try:
        dps = load_def_pos_allowed_shares(season)
    except Exception:
        dps = pd.DataFrame()

    # Load 2024 per-player TD totals (rush/rec/total) as stronger priors, team-agnostic.
    # This complements the 5y team-specific counts above.
    totals_2024_path = DATA_DIR / "player_td_totals_2024.csv"
    rush24_map: dict[str, float] = {}
    rec24_map: dict[str, float] = {}
    if totals_2024_path.exists():
        try:
            tdf = pd.read_csv(totals_2024_path)
            for _, rr in tdf.iterrows():
                nm = str(rr.get("player") or "").strip().lower()
                if not nm:
                    continue
                try:
                    rsh = float(pd.to_numeric(rr.get("rush_td"), errors="coerce"))
                except Exception:
                    rsh = 0.0
                try:
                    rcv = float(pd.to_numeric(rr.get("rec_td"), errors="coerce"))
                except Exception:
                    rcv = 0.0
                if rsh > 0:
                    rush24_map[nm] = rsh
                if rcv > 0:
                    rec24_map[nm] = rcv
        except Exception:
            pass

    # Apply TPP adjustment with per-game conservation to team expected TDs
    adj_map = _apply_tpp_to_teams(teams, season)

    rows: list[dict] = []
    for _, r in teams.iterrows():
        team = str(r.get("team"))
        opp = str(r.get("opponent"))
        exp_tds = float(adj_map.get((r.get("game_id"), team), float(r.get("expected_tds") or 0.0)))
        split = _split_team_tds(r, exp_tds)
        rush_tds = split["rush_tds"]
        pass_tds = split["pass_tds"]

        depth = _team_player_shares(usage, int(r.get("season")), team)
        depth = _apply_roster_positions(depth, int(r.get("season")), team)
        rush_col = "rz_rush_share" if "rz_rush_share" in depth.columns else "rush_share"
        pass_col = "rz_target_share" if "rz_target_share" in depth.columns else "target_share"
        depth[rush_col] = pd.to_numeric(depth[rush_col], errors='coerce').fillna(0.0)
        depth[pass_col] = pd.to_numeric(depth[pass_col], errors='coerce').fillna(0.0)
        if depth[rush_col].sum() > 0:
            depth[rush_col] = depth[rush_col] / depth[rush_col].sum()
        if depth[pass_col].sum() > 0:
            depth[pass_col] = depth[pass_col] / depth[pass_col].sum()

        # Apply team-specific pass TD position shares if available
        # Defaults tilted closer to Week 1 historical receiving TD shares (WR-heavy)
        desired_default = {"WR": 0.50, "TE": 0.30, "RB": 0.20}
        desired_default_rz = {"WR": 0.46, "TE": 0.36, "RB": 0.18}
        desired = desired_default_rz if pass_col == "rz_target_share" else desired_default
        if tps is not None and not tps.empty:
            try:
                rowp = tps[(tps.get("team").astype(str) == team) & (tps.get("kind").astype(str) == "pass")]  # type: ignore
                if not rowp.empty:
                    wr = float(rowp.iloc[0].get("WR") or 0.0)
                    te = float(rowp.iloc[0].get("TE") or 0.0)
                    rb = float(rowp.iloc[0].get("RB") or 0.0)
                    s = wr + te + rb
                    if s > 0:
                        team_share = {"WR": wr / s, "TE": te / s, "RB": rb / s}
                        # Blend team shares with defaults; weight team more to reflect identity
                        alpha = 0.5
                        desired = {
                            k: max(0.0, (1 - alpha) * desired.get(k, 0.0) + alpha * team_share.get(k, 0.0))
                            for k in {"WR","TE","RB"}
                        }
                        # Normalize
                        ss = sum(desired.values())
                        if ss > 0:
                            desired = {k: v/ss for k, v in desired.items()}
            except Exception:
                pass

        # Soft-scale pass shares by position toward desired proportions (offense identity)
        def _group_sum(pos: str) -> float:
            m = depth["position"].astype(str).str.upper() == pos
            return float(depth.loc[m, pass_col].sum())

        beta_pass = 0.60 if pass_col == "target_share" else 0.70
        for pos in ("WR", "TE", "RB"):
            cur = _group_sum(pos)
            tgt = desired.get(pos, 0.0)
            blend = max(0.0, (1 - beta_pass) * cur + beta_pass * tgt)
            if cur > 0:
                factor = blend / cur
                m = depth["position"].astype(str).str.upper() == pos
                depth.loc[m, pass_col] = depth.loc[m, pass_col] * factor
        s = depth[pass_col].sum()
        if s > 0:
            depth[pass_col] = depth[pass_col] / s

        # Overlay opponent defensive allowed shares by position (softer than team identity)
        if dps is not None and not dps.empty:
            try:
                drow = dps[(dps.get("team").astype(str) == opp) & (dps.get("kind").astype(str) == "pass")]  # type: ignore
                if not drow.empty:
                    opp_shares = {k: float(drow.iloc[0].get(k) or 0.0) for k in ["WR","TE","RB"]}
                    # Normalize subset
                    tot = sum(opp_shares.values())
                    if tot > 0:
                        opp_shares = {k: v / tot for k, v in opp_shares.items()}
                        beta_def = 0.25  # small nudge
                        for pos in ("WR","TE","RB"):
                            cur = float(depth.loc[depth["position"].astype(str).str.upper() == pos, pass_col].sum())
                            tgt = opp_shares.get(pos, 0.0)
                            if cur > 0:
                                blend = (1 - beta_def) * cur + beta_def * tgt
                                factor = blend / cur
                                m = depth["position"].astype(str).str.upper() == pos
                                depth.loc[m, pass_col] = depth.loc[m, pass_col] * factor
                        s2 = float(depth[pass_col].sum())
                        if s2 > 0:
                            depth[pass_col] = depth[pass_col] / s2
            except Exception:
                pass

    # Similarly, softly bias rushing TD distribution by team rush TD position shares if present
        if tps is not None and not tps.empty:
            try:
                rowr = tps[(tps.get("team").astype(str) == team) & (tps.get("kind").astype(str) == "rush")]  # type: ignore
                if not rowr.empty and depth[rush_col].sum() > 0:
                    # Team share from trends
                    t_rush = {
                        "RB": float(rowr.iloc[0].get("RB") or 0.0),
                        "QB": float(rowr.iloc[0].get("QB") or 0.0),
                        "WR": float(rowr.iloc[0].get("WR") or 0.0),
                        "TE": float(rowr.iloc[0].get("TE") or 0.0),
                    }
                    sdr = sum(t_rush.values())
                    if sdr > 0:
                        # Blend with a default rush distribution closer to historical: lower QB, small WR/TE residuals
                        base_rush = {"RB": 0.74, "QB": 0.21, "WR": 0.04, "TE": 0.01}
                        alpha_r = 0.6
                        desired_rush = {k: max(0.0, (1 - alpha_r) * base_rush.get(k, 0.0) + alpha_r * (t_rush.get(k, 0.0) / sdr)) for k in base_rush.keys()}
                        # Normalize
                        sr0 = sum(desired_rush.values())
                        if sr0 > 0:
                            desired_rush = {k: v/sr0 for k, v in desired_rush.items()}
                        beta_rush = 0.5
                        for pos in ("RB", "QB", "WR", "TE"):
                            cur = float(depth.loc[depth["position"].astype(str).str.upper() == pos, rush_col].sum())
                            tgt = desired_rush.get(pos, 0.0)
                            blend = max(0.0, (1 - beta_rush) * cur + beta_rush * tgt)
                            if cur > 0:
                                factor = blend / cur
                                m = depth["position"].astype(str).str.upper() == pos
                                depth.loc[m, rush_col] = depth.loc[m, rush_col] * factor
                        sr = depth[rush_col].sum()
                        if sr > 0:
                            depth[rush_col] = depth[rush_col] / sr
            except Exception:
                pass

        # Defensive overlay for rushing shares by position
        if dps is not None and not dps.empty:
            try:
                drow = dps[(dps.get("team").astype(str) == opp) & (dps.get("kind").astype(str) == "rush")]  # type: ignore
                if not drow.empty and depth[rush_col].sum() > 0:
                    opp_rush = {k: float(drow.iloc[0].get(k) or 0.0) for k in ["RB","QB","WR","TE"]}
                    tot = sum(opp_rush.values())
                    if tot > 0:
                        opp_rush = {k: v/tot for k, v in opp_rush.items()}
                        beta_def_r = 0.25
                        for pos in ("RB","QB","WR","TE"):
                            cur = float(depth.loc[depth["position"].astype(str).str.upper() == pos, rush_col].sum())
                            tgt = opp_rush.get(pos, 0.0)
                            if cur > 0:
                                blend = (1 - beta_def_r) * cur + beta_def_r * tgt
                                factor = blend / cur
                                m = depth["position"].astype(str).str.upper() == pos
                                depth.loc[m, rush_col] = depth.loc[m, rush_col] * factor
                        sr2 = float(depth[rush_col].sum())
                        if sr2 > 0:
                            depth[rush_col] = depth[rush_col] / sr2
            except Exception:
                pass

        # Within-position adjustments using historic player TD counts (conservative blend)
        # Goal: nudge target/rush shares within each position toward players with more historic TDs on this team.
        def _apply_within_pos_counts(kind: str, col: str, weight: float) -> None:
            if ptc is None or ptc.empty or col not in depth.columns:
                return
            # Slice team/kind counts
            sub = ptc[(ptc.get("team").astype(str) == team) & (ptc.get("kind").astype(str) == kind)]
            if sub.empty:
                return
            # Build quick map: lower(player_name) -> count
            cnt_map = {str(n).strip().lower(): float(c) for n, c in zip(sub.get("player", []), sub.get("td_count", []))}
            if not cnt_map:
                return
            # For each position group, compute normalized counts for players present
            for pos in depth["position"].astype(str).str.upper().unique():
                m = depth["position"].astype(str).str.upper() == pos
                if not m.any():
                    continue
                cur_sum = float(depth.loc[m, col].sum())
                if cur_sum <= 0:
                    continue
                # Build weights from counts; fallback small epsilon to avoid zeroing newcomers
                names = depth.loc[m, "player"].astype(str)
                counts = np.array([cnt_map.get(str(x).strip().lower(), 0.0) for x in names], dtype=float)
                if counts.sum() <= 0:
                    continue
                w = counts / counts.sum()
                # Blend current within-group distribution with counts-based weights
                # Maintain group mass (cur_sum)
                current = depth.loc[m, col].to_numpy(dtype=float)
                target = w * cur_sum
                blended = (1.0 - weight) * current + weight * target
                # Write back
                depth.loc[m, col] = blended
            # Renormalize the entire column to sum to 1
            s2 = float(depth[col].sum())
            if s2 > 0:
                depth[col] = depth[col] / s2

        # Apply within-position nudges: pass TDs affect pass_col among WR/TE/RB; rush TDs affect rush_col among RB/QB/WR/TE
        # Use small weights to avoid overpowering priors and roster-based usage.
        _apply_within_pos_counts(kind="pass", col=pass_col, weight=0.15)
        _apply_within_pos_counts(kind="rush", col=rush_col, weight=0.20)

        # Stronger within-position priors from 2024 totals (team-agnostic) if available.
        def _apply_counts_from_map(counts_map: dict[str, float], col: str, weight: float) -> None:
            if not counts_map or col not in depth.columns:
                return
            for pos in depth["position"].astype(str).str.upper().unique():
                m = depth["position"].astype(str).str.upper() == pos
                if not m.any():
                    continue
                cur_sum = float(depth.loc[m, col].sum())
                if cur_sum <= 0:
                    continue
                names = depth.loc[m, "player"].astype(str)
                counts = np.array([float(counts_map.get(str(x).strip().lower(), 0.0)) for x in names], dtype=float)
                if counts.sum() <= 0:
                    continue
                w = counts / counts.sum()
                current = depth.loc[m, col].to_numpy(dtype=float)
                target = w * cur_sum
                blended = (1.0 - weight) * current + weight * target
                depth.loc[m, col] = blended
            s2 = float(depth[col].sum())
            if s2 > 0:
                depth[col] = depth[col] / s2

        # Apply with a moderately higher weight than 5y, as requested (e.g., 0.30)
        if rec24_map:
            _apply_counts_from_map(rec24_map, pass_col, weight=0.25)
        if rush24_map:
            _apply_counts_from_map(rush24_map, rush_col, weight=0.30)

        for _, p in depth.iterrows():
            exp_rush_td = float(rush_tds) * float(p[rush_col])
            rec_weight = float(p[pass_col])
            exp_rec_td = 0.0 if str(p.get("position")).upper() == "QB" else float(pass_tds) * rec_weight
            exp_any_td = exp_rush_td + exp_rec_td
            prob_any = float(1.0 - np.exp(-max(0.0, exp_any_td)))
            rows.append({
                "season": int(r.get("season")),
                "week": int(r.get("week")),
                "date": r.get("date"),
                "game_id": r.get("game_id"),
                "team": team,
                "opponent": opp,
                "player": p.get("player"),
                "position": p.get("position"),
                "is_home": int(r.get("is_home") or 0),
                "expected_td": exp_any_td,
                "anytime_td_prob": prob_any,
                "exp_rush_td": exp_rush_td,
                "exp_rec_td": exp_rec_td,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _enrich_player_names(out, season)
    out = out.sort_values(["season","week","game_id","team","anytime_td_prob"], ascending=[True, True, True, True, False])
    return out
