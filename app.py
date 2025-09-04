from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import json

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

    # Lightweight CSV cache (path -> (mtime, DataFrame)) to speed repeated loads
    _CSV_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
    def _read_csv_cached(path: Path, **kwargs) -> pd.DataFrame:
        try:
            mtime = path.stat().st_mtime
        except Exception:
            return pd.DataFrame()
        key = str(path)
        cached = _CSV_CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception:
            return pd.DataFrame()
        _CSV_CACHE[key] = (mtime, df)
        return df

    # Jinja2 filter: format a number with up to 2 decimals (trim trailing zeros)
    @app.template_filter("fmt2")
    def _fmt2_filter(val: object) -> str:
        try:
            f = float(val)
        except Exception:
            return str(val)
        s = f"{f:.2f}"
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    # Jinja filter: format numeric values with up to 2 decimals (trim trailing zeros)

    # Optional: admin-only route to rebuild player_meta.csv on-demand
    @app.route("/admin/rebuild-meta", methods=["POST"])
    def admin_rebuild_meta():
        if os.environ.get("TD_ADMIN", "").lower() not in {"1","true","yes"}:
            return {"ok": False, "error": "unauthorized"}, 403
        try:
            import subprocess, sys
            cmd = [sys.executable, str(BASE_DIR / "build_player_meta.py")]
            subprocess.check_call(cmd, cwd=str(BASE_DIR))
            # Clear in-process meta cache so new file is used immediately
            nonlocal _PLAYER_META
            _PLAYER_META = None
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}, 500

    def _latest_csv(prefix: str) -> Optional[Path]:
        if not DATA_DIR.exists():
            return None
        files = sorted(DATA_DIR.glob(f"{prefix}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    # Simple team abbrev from name and a text-avatar color
    def _team_abbrev(name: str) -> str:
        parts = str(name).split()
        if len(parts) == 0:
            return "NA"
        if len(parts) == 1:
            return parts[0][:3].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    def _team_avatar_style(name: str) -> str:
        # deterministic color from name
        h = abs(hash(name)) % 360
        return f"background: hsl({h},60%,40%); color: #fff;"

    # Class-based avatar color (12 buckets) to avoid inline styles
    def _team_avatar_class(name: str) -> str:
        bucket = abs(hash(name)) % 12
        return f"hue-{bucket}"

    # Load optional rosters for headshots
    _HEADSHOT_MAP: dict[str, str] = {}
    def _load_headshots(season: int) -> None:
        nonlocal _HEADSHOT_MAP
        if _HEADSHOT_MAP:
            return
        try:
            import nfl_data_py as nfl  # type: ignore
            ros = nfl.import_seasonal_rosters([season])
        except Exception:
            ros = None
        if ros is None or len(getattr(ros, 'columns', [])) == 0:
            # Try local cached rosters csv if present
            p = DATA_DIR / "seasonal_rosters_2020_2024.csv"
            if p.exists():
                try:
                    ros = pd.read_csv(p)
                except Exception:
                    ros = None
        if ros is None or ros.empty:
            return
        name_cols = [c for c in ["display_name","player_display_name","full_name","player_name","football_name"] if c in ros.columns]
        url_col = None
        for c in ["headshot_url","headshot","espn_headshot_url"]:
            if c in ros.columns:
                url_col = c
                break
        if not name_cols or not url_col:
            return
        m: dict[str, str] = {}
        for _, r in ros.iterrows():
            nm = None
            for c in name_cols:
                v = str(r.get(c) or '').strip()
                if v:
                    nm = v
                    break
            if not nm:
                continue
            url = str(r.get(url_col) or '').strip()
            if url:
                m[nm.lower()] = url
        _HEADSHOT_MAP = m

    def _headshot_for(player: str, season: int) -> Optional[str]:
        if not player:
            return None
        if not _HEADSHOT_MAP:
            _load_headshots(season)
        return _HEADSHOT_MAP.get(str(player).lower())

    # Player name matching helpers: generate multiple key variants to match PBP-style abbreviations
    def _name_key_variants(full_name: str) -> list[str]:
        s = str(full_name or "").strip()
        if not s:
            return []
        low = s.lower()
        # Split and strip punctuation; remove common suffixes if present (last token)
        import re as _re
        raw_tokens = [t for t in s.split() if t]
        tokens = [
            _re.sub(r"[.,]$", "", t)  # strip trailing punctuation
            for t in raw_tokens
            if t
        ]
        # Suffixes to ignore for matching
        suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "vi"}
        if len(tokens) >= 2 and tokens[-1].lower() in suffixes:
            tokens = tokens[:-1]
        out = [low]
        if tokens:
            first = tokens[0]
            finit = (first[0] + ".").lower() if first else ""
            last = tokens[-1]
            # default: initial + last word
            if finit and last:
                out.append(f"{finit}{last.lower()}")
                out.append(f"{finit}{' '}{last.lower()}")
            # handle common multi-word last-name particles (e.g., "St. Brown", "Van Noy", "De Vries")
            particles = {"st.", "st", "van", "von", "de", "de.", "da", "del", "della", "la", "le", "mac", "mc", "o'"}
            if len(tokens) >= 3:
                last2 = tokens[-2].lower()
                if last2 in particles:
                    combo = f"{tokens[-2]} {tokens[-1]}".lower()
                    out.append(f"{finit}{combo}")
            # also try hyphenated last names keeping hyphen
            if "-" in last:
                out.append(f"{finit}{last.lower()}")
        # dedupe preserve order
        seen = set()
        uniq = []
        for k in out:
            if k and k not in seen:
                seen.add(k)
                uniq.append(k)
        return uniq

    def _map_lookup_with_variants(m: dict[str, int | float], name: str) -> Optional[int | float]:
        for k in _name_key_variants(name):
            if k in m:
                return m[k]
        return None

    # Cached career TD maps built from local pbp_*.csv files
    _CAREER_TD_CACHE: Optional[tuple[dict[str,int], dict[str,int], dict[str,int]]] = None
    def _player_td_career_counts() -> tuple[dict[str,int], dict[str,int], dict[str,int]]:
        nonlocal _CAREER_TD_CACHE
        if _CAREER_TD_CACHE is not None:
            return _CAREER_TD_CACHE
        rush_map: dict[str,int] = {}
        rec_map: dict[str,int] = {}
        total_map: dict[str,int] = {}
        # Find all pbp_*.csv in data directory
        files = sorted(DATA_DIR.glob("pbp_*.csv"))
        for fp in files:
            try:
                df = pd.read_csv(fp, usecols=[
                    "rush_touchdown","pass_touchdown","rusher_player_name","receiver_player_name"
                ])
            except Exception:
                continue
            df = df.fillna(0)
            df["is_rush_td"] = pd.to_numeric(df["rush_touchdown"], errors="coerce").fillna(0).astype(int)
            df["is_rec_td"] = pd.to_numeric(df["pass_touchdown"], errors="coerce").fillna(0).astype(int)
            for _, r in df.iterrows():
                if int(r.get("is_rush_td") or 0) == 1:
                    n = str(r.get("rusher_player_name") or "").strip().lower()
                    if n:
                        rush_map[n] = rush_map.get(n, 0) + 1
                        total_map[n] = total_map.get(n, 0) + 1
                if int(r.get("is_rec_td") or 0) == 1:
                    n = str(r.get("receiver_player_name") or "").strip().lower()
                    if n:
                        rec_map[n] = rec_map.get(n, 0) + 1
                        total_map[n] = total_map.get(n, 0) + 1
        _CAREER_TD_CACHE = (rush_map, rec_map, total_map)
        return _CAREER_TD_CACHE

    # Optional: load precomputed player meta (fast path)
    _PLAYER_META: Optional[dict[str, dict[str, int]]] = None
    def _load_player_meta() -> Optional[dict[str, dict[str, int]]]:
        nonlocal _PLAYER_META
        if _PLAYER_META is not None:
            return _PLAYER_META
        fp = DATA_DIR / "player_meta.csv"
        if not fp.exists():
            _PLAYER_META = None
            return None
        df = _read_csv_cached(fp)
        if df is None or df.empty:
            _PLAYER_META = None
            return None
        req = ["player","td24_rush","td24_rec","td24_total","career_rush","career_rec","career_total"]
        for c in req:
            if c not in df.columns:
                _PLAYER_META = None
                return None
        meta: dict[str, dict[str,int]] = {}
        def _safe_int(val: object) -> int:
            try:
                v = pd.to_numeric(val, errors="coerce")
                return 0 if pd.isna(v) else int(v)
            except Exception:
                try:
                    return int(val or 0)
                except Exception:
                    return 0
        for _, r in df.iterrows():
            name = str(r.get("player") or "").strip().lower()
            if not name:
                continue
            meta[name] = {
                "td24_rush": _safe_int(r.get("td24_rush")),
                "td24_rec": _safe_int(r.get("td24_rec")),
                "td24_total": _safe_int(r.get("td24_total")),
                "career_rush": _safe_int(r.get("career_rush")),
                "career_rec": _safe_int(r.get("career_rec")),
                "career_total": _safe_int(r.get("career_total")),
            }
        _PLAYER_META = meta
        return _PLAYER_META

    # Depth charts: prefer nfl_data_py if available; fallback to expected_td heuristic
    def _build_depth_map(df: pd.DataFrame, season: int, week: int) -> dict[tuple[str,str,str], tuple[int,int]]:
        # Normalize helper
        def norm(s: object) -> str:
            return str(s or "").strip().lower()
        depth_map: dict[tuple[str,str,str], tuple[int,int]] = {}
        dch = None
        try:
            import nfl_data_py as nfl  # type: ignore
            try:
                dch = nfl.import_depth_charts([season])
            except TypeError:
                dch = nfl.import_depth_charts(seasons=[season])
        except Exception:
            dch = None
        # Try to use depth charts if present
        if dch is not None and getattr(dch, 'empty', True) is False:
            dc = dch.copy()
            # Best-effort column normalization
            # candidate columns for names/teams/pos/order
            name_cols = [c for c in ["player", "player_name", "full_name", "player_display_name"] if c in dc.columns]
            team_col = "team" if "team" in dc.columns else ("recent_team" if "recent_team" in dc.columns else None)
            pos_col = None
            for c in ["position", "pos", "position_group", "depth_chart_position"]:
                if c in dc.columns:
                    pos_col = c
                    break
            order_col = None
            for c in ["depth_chart_order", "order", "depth", "depth_team_rank"]:
                if c in dc.columns:
                    order_col = c
                    break
            # If essentials are missing, skip to heuristic
            if name_cols and team_col and pos_col:
                # filter to week if available
                if "week" in dc.columns:
                    try:
                        dc = dc[dc["week"].astype(int) == int(week)]
                    except Exception:
                        pass
                # reduce to core and coerce order
                dc = dc.copy()
                dc[team_col] = dc[team_col].astype(str)
                dc[pos_col] = dc[pos_col].astype(str)
                # Map some position variants
                def map_pos(p: str) -> str:
                    p = str(p or "").upper()
                    if p in {"HB","FB"}:
                        return "RB"
                    if p.startswith("WR"):
                        return "WR"
                    if p.startswith("TE"):
                        return "TE"
                    if p.startswith("RB"):
                        return "RB"
                    if p.startswith("QB"):
                        return "QB"
                    return p
                dc[pos_col] = dc[pos_col].map(map_pos)
                # build per team/pos groups
                # If order column missing, construct by grouping
                if order_col is None:
                    # Construct a simple rank by alphabetical name within team/pos as last resort
                    name_col = name_cols[0]
                    dc = dc[[team_col, pos_col, name_col]].copy()
                    dc.sort_values([team_col, pos_col, name_col], inplace=True)
                    dc['__rk'] = dc.groupby([team_col, pos_col]).cumcount() + 1
                    order_col = '__rk'
                else:
                    # ensure numeric
                    dc[order_col] = pd.to_numeric(dc[order_col], errors="coerce").fillna(9999).astype(int)
                # Use the first suitable name column to build mapping
                name_col = name_cols[0]
                # Build map
                for (t, p), grp in dc.groupby([team_col, pos_col], dropna=False):
                    g = grp.sort_values(order_col, ascending=True)
                    total = len(g)
                    for _, rr in g.iterrows():
                        nm = norm(rr.get(name_col))
                        if not nm:
                            continue
                        rk = int(rr.get(order_col) or 9999)
                        key = (norm(t), str(p), nm)
                        # keep best (lowest rank)
                        prev = depth_map.get(key)
                        if prev is None or rk < prev[0]:
                            depth_map[key] = (rk, total)
        # Fallback heuristic using expected_td if a player isn't in map
        if not depth_map:
            try:
                df_depth = df.copy()
                df_depth["expected_td"] = pd.to_numeric(df_depth.get("expected_td"), errors="coerce").fillna(0.0)
                df_depth["team"] = df_depth["team"].astype(str)
                df_depth["position"] = df_depth["position"].astype(str)
                df_depth["player"] = df_depth["player"].astype(str)
                for (team_name, pos_name), grp in df_depth.groupby(["team", "position"], dropna=False):
                    g = grp.sort_values("expected_td", ascending=False).reset_index(drop=True)
                    total = len(g)
                    for idx, rr in g.iterrows():
                        depth_map[(norm(team_name), str(pos_name), norm(rr.get("player")))] = (int(idx+1), int(total))
            except Exception:
                pass
        return depth_map

    # Team logos loader (tries NFL-Betting repo or local data) and accessor
    _TEAM_LOGO_MAP: dict[str, str] = {}
    def _load_team_logos() -> None:
        nonlocal _TEAM_LOGO_MAP
        if _TEAM_LOGO_MAP:
            return
        candidates = [
            DATA_DIR / "nfl_team_assets.json",
            (BASE_DIR.parent / "NFL-Betting" / "nfl_compare" / "data" / "nfl_team_assets.json"),
        ]
        data = None
        for p in candidates:
            try:
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        break
            except Exception:
                continue
        if not data:
            _TEAM_LOGO_MAP = {}
            return
        m: dict[str, str] = {}
        # fallback builder using ESPN CDN
        def cdn_url_for_abbr(abbr: Optional[str]) -> Optional[str]:
            if not abbr:
                return None
            code = str(abbr).strip().lower()
            # ESPN uses 'wsh' for Washington; normalize common edge cases
            overrides = {"was": "wsh"}
            code = overrides.get(code, code)
            return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"
        def add_entry(key: Optional[str], val: Optional[str], abbr: Optional[str] = None):
            if not key or not val:
                # try fallback from abbr
                if abbr:
                    fv = cdn_url_for_abbr(abbr)
                    if fv:
                        m[str(key).strip().lower()] = fv
                return
            m[str(key).strip().lower()] = str(val).strip()
        # Accept multiple shapes: {team: {logo_url}} or list of objects
        try:
            if isinstance(data, dict):
                # common patterns
                items = []
                if "teams" in data and isinstance(data["teams"], list):
                    items = data["teams"]
                elif "data" in data and isinstance(data["data"], list):
                    items = data["data"]
                else:
                    # maybe a direct mapping
                    for k, v in data.items():
                        url = None
                        abbr = None
                        if isinstance(v, dict):
                            url = v.get("logo_url") or v.get("logo") or v.get("url")
                            abbr = v.get("abbr") or v.get("abbreviation")
                            logos = v.get("logos")
                            if not url and isinstance(logos, dict):
                                url = logos.get("light") or logos.get("dark") or logos.get("primary")
                        elif isinstance(v, str):
                            url = v
                        add_entry(k, url, abbr)
                # parse list items
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    name = it.get("display_name") or it.get("full_name") or it.get("team") or it.get("name")
                    abbr = it.get("abbr") or it.get("abbreviation")
                    url = it.get("logo_url") or it.get("logo") or it.get("url")
                    logos = it.get("logos")
                    if not url and isinstance(logos, dict):
                        url = logos.get("light") or logos.get("dark") or logos.get("primary")
                    add_entry(name, url, abbr)
                    add_entry(abbr, url, abbr)
        except Exception:
            pass
        _TEAM_LOGO_MAP = m

    def _logo_for_team(team_name: str) -> Optional[str]:
        if not team_name:
            return None
        if not _TEAM_LOGO_MAP:
            _load_team_logos()
        if not _TEAM_LOGO_MAP:
            return None
        key_full = str(team_name).lower()
        key_abbr = _team_abbrev(team_name).lower()
        return _TEAM_LOGO_MAP.get(key_full) or _TEAM_LOGO_MAP.get(key_abbr)

    # 2024 team TD distribution caches
    _TEAM_TD24_KIND: Optional[dict[str, dict[str, float]]] = None  # team -> {pass, rush}
    _TEAM_TD24_POS: Optional[dict[str, dict[str, float]]] = None   # team -> {RB,WR,TE,QB} shares
    _TEAM_TD24_POS_COUNTS: Optional[dict[str, dict[str, int]]] = None  # team -> {RB,WR,TE,QB} counts
    def _get_team_2024_distributions() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
        """Compute team-level 2024 TD distributions.

        - Kind shares: pass vs rush
        - Position shares: WR/TE/RB/QB, mapped via seasonal_rosters positions
        Results are normalized to 1.0 when possible.
        """
        nonlocal _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS
        if _TEAM_TD24_KIND is not None and _TEAM_TD24_POS is not None and _TEAM_TD24_POS_COUNTS is not None:
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS
        kind: dict[str, dict[str, int]] = {}
        pos: dict[str, dict[str, int]] = {}
        # Load PBP 2024
        pbp_fp = DATA_DIR / "pbp_2024.csv"
        if not pbp_fp.exists():
            _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS = {}, {}, {}
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS
        try:
            pbp = _read_csv_cached(
                pbp_fp,
                usecols=[
                    "posteam", "rush_touchdown", "pass_touchdown",
                    "rusher_player_name", "receiver_player_name"
                ],
            )
        except Exception:
            # Fallback to full read
            try:
                pbp = _read_csv_cached(pbp_fp)
            except Exception:
                _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS = {}, {}, {}
                return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS
        if pbp is None or pbp.empty:
            _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS = {}, {}, {}
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS
        # Identify team column
        team_col = None
        for c in ["posteam", "offense_team", "team"]:
            if c in pbp.columns:
                team_col = c
                break
        if team_col is None:
            _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS = {}, {}, {}
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS
        # Coerce flags
        pbp = pbp.copy()
        for c in ["rush_touchdown", "pass_touchdown"]:
            if c not in pbp.columns:
                pbp[c] = 0
            pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
        # Player position map from seasonal_rosters 2024
        roster_fp = DATA_DIR / "seasonal_rosters_2020_2024.csv"
        pos_map: dict[str, str] = {}
        if roster_fp.exists():
            try:
                ros = _read_csv_cached(roster_fp, usecols=["season", "player_name", "position", "team"])  # type: ignore
                if ros is not None and not ros.empty:
                    try:
                        ros = ros[ros["season"].astype(int) == 2024]
                    except Exception:
                        pass
                    for _, r in ros.iterrows():
                        nm = str(r.get("player_name") or "").strip().lower()
                        p = str(r.get("position") or "").strip().upper()
                        if nm and p:
                            # Map variants to core groups
                            if p in {"HB", "FB"}:
                                p = "RB"
                            elif p.startswith("WR"):
                                p = "WR"
                            elif p.startswith("TE"):
                                p = "TE"
                            elif p.startswith("RB"):
                                p = "RB"
                            elif p.startswith("QB"):
                                p = "QB"
                            pos_map[nm] = p
            except Exception:
                pos_map = {}
        # Iterate plays and build counts
        for _, r in pbp.iterrows():
            tm = str(r.get(team_col) or "").strip()
            if not tm:
                continue
            # kind shares
            if int(r.get("rush_touchdown") or 0) == 1:
                kind.setdefault(tm, {"pass": 0, "rush": 0})["rush"] += 1
                n = str(r.get("rusher_player_name") or "").strip().lower()
                p = pos_map.get(n)
                if p in {"RB", "WR", "TE", "QB"}:
                    pos.setdefault(tm, {}).setdefault(p, 0)
                    pos[tm][p] += 1
            if int(r.get("pass_touchdown") or 0) == 1:
                kind.setdefault(tm, {"pass": 0, "rush": 0})["pass"] += 1
                n = str(r.get("receiver_player_name") or "").strip().lower()
                p = pos_map.get(n)
                if p in {"RB", "WR", "TE", "QB"}:
                    pos.setdefault(tm, {}).setdefault(p, 0)
                    pos[tm][p] += 1
        # Normalize to shares
        kind_out: dict[str, dict[str, float]] = {}
        pos_out: dict[str, dict[str, float]] = {}
        pos_counts_out: dict[str, dict[str, int]] = {}
        for tm, kv in kind.items():
            tot = float(kv.get("pass", 0) + kv.get("rush", 0))
            if tot > 0:
                kind_out[tm] = {"pass": kv.get("pass", 0) / tot, "rush": kv.get("rush", 0) / tot}
        for tm, kv in pos.items():
            tot = float(sum(kv.values()))
            if tot > 0:
                pos_out[tm] = {k: v / tot for k, v in kv.items() if k in {"RB", "WR", "TE", "QB"}}
                pos_counts_out[tm] = {k: int(v) for k, v in kv.items() if k in {"RB", "WR", "TE", "QB"}}
                # ensure all keys present
                for k in ["RB", "WR", "TE", "QB"]:
                    pos_out[tm][k] = float(pos_out[tm].get(k, 0.0))
                    pos_counts_out[tm][k] = int(pos_counts_out[tm].get(k, 0))
        _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS = kind_out, pos_out, pos_counts_out
        return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS

    # Quick per-player TD counts for a given season from local pbp csv if present
    _TD_SEASON_CACHE: dict[int, dict[str,int]] = {}
    def _player_td_counts_for_season(season: int) -> dict[str, int]:
        if season in _TD_SEASON_CACHE:
                return _TD_SEASON_CACHE[season] or {}
        fp = DATA_DIR / f"pbp_{season}.csv"
        if not fp.exists():
            _TD_SEASON_CACHE[season] = {}
            return _TD_SEASON_CACHE[season]
        # Read only needed columns when available; fallback to broader read
        allowed = {
            "rush_touchdown","pass_touchdown","rusher_player_name","receiver_player_name",
            "rush_attempt","complete_pass","touchdown"
        }
        try:
            df = _read_csv_cached(fp, usecols=lambda c: c in allowed).fillna(0)
        except Exception:
            df = _read_csv_cached(fp).fillna(0)
        if df is None or df.empty:
            _TD_SEASON_CACHE[season] = {}
            return _TD_SEASON_CACHE[season]
        # Build rush/pass TD flags robustly
        def _num(col: str) -> pd.Series:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            return pd.Series([0]*len(df), index=df.index, dtype=int)
        # Direct flags if present
        rush_flag = None
        for c in ["rush_touchdown","rushing_touchdown","rush_td","rusher_touchdown"]:
            if c in df.columns:
                rush_flag = _num(c)
                break
        pass_flag = None
        for c in ["pass_touchdown","passing_touchdown","pass_td","receiver_touchdown"]:
            if c in df.columns:
                pass_flag = _num(c)
                break
        # Derive from attempt/completion + touchdown if needed
        td_any = _num("touchdown")
        if rush_flag is None and ("rush_attempt" in df.columns and "touchdown" in df.columns):
            rush_flag = (_num("rush_attempt") & td_any).astype(int)
        if pass_flag is None and ("complete_pass" in df.columns and "touchdown" in df.columns):
            pass_flag = (_num("complete_pass") & td_any).astype(int)
        # Fallback to zeros if still None
        if rush_flag is None:
            rush_flag = pd.Series([0]*len(df), index=df.index, dtype=int)
        if pass_flag is None:
            pass_flag = pd.Series([0]*len(df), index=df.index, dtype=int)
        df["is_rush_td"] = rush_flag
        df["is_rec_td"] = pass_flag
        rows = []
        for _, r in df.iterrows():
            if r["is_rush_td"] == 1:
                n = str(r.get("rusher_player_name") or "").strip()
                if n:
                    rows.append(n)
            if r["is_rec_td"] == 1:
                n = str(r.get("receiver_player_name") or "").strip()
                if n:
                    rows.append(n)
        out: dict[str, int] = {}
        for n in rows:
            out[n.lower()] = out.get(n.lower(), 0) + 1
        _TD_SEASON_CACHE[season] = out
        return _TD_SEASON_CACHE[season]

    # Per-season passer TDs thrown
    _PASS_TD_SEASON_CACHE: dict[int, dict[str,int]] = {}
    def _player_pass_td_counts_for_season(season: int) -> dict[str, int]:
        if season in _PASS_TD_SEASON_CACHE:
                return _PASS_TD_SEASON_CACHE[season] or {}
        fp = DATA_DIR / f"pbp_{season}.csv"
        if not fp.exists():
            _PASS_TD_SEASON_CACHE[season] = {}
            return _PASS_TD_SEASON_CACHE[season]
        try:
            df = _read_csv_cached(fp, usecols=["pass_touchdown","passer_player_name"]).fillna(0)
        except Exception:
            try:
                df = _read_csv_cached(fp)
            except Exception:
                _PASS_TD_SEASON_CACHE[season] = {}
                return _PASS_TD_SEASON_CACHE[season]
        if df is None or df.empty:
            _PASS_TD_SEASON_CACHE[season] = {}
            return _PASS_TD_SEASON_CACHE[season]
        if "pass_touchdown" not in df.columns or "passer_player_name" not in df.columns:
            _PASS_TD_SEASON_CACHE[season] = {}
            return _PASS_TD_SEASON_CACHE[season]
        df["pass_touchdown"] = pd.to_numeric(df["pass_touchdown"], errors="coerce").fillna(0).astype(int)
        out: dict[str,int] = {}
        for _, r in df[df["pass_touchdown"] == 1].iterrows():
            n = str(r.get("passer_player_name") or "").strip().lower()
            if n:
                out[n] = out.get(n, 0) + 1
        _PASS_TD_SEASON_CACHE[season] = out
        return _PASS_TD_SEASON_CACHE[season]

    # Per-season rushing TDs (rusher only) for precise QB rush counts
    _RUSH_TD_SEASON_CACHE: dict[int, dict[str,int]] = {}
    def _player_rush_td_counts_for_season(season: int) -> dict[str, int]:
        if season in _RUSH_TD_SEASON_CACHE:
                return _RUSH_TD_SEASON_CACHE[season] or {}
        fp = DATA_DIR / f"pbp_{season}.csv"
        if not fp.exists():
            _RUSH_TD_SEASON_CACHE[season] = {}
            return _RUSH_TD_SEASON_CACHE[season]
        try:
            df = _read_csv_cached(fp, usecols=["rush_touchdown","rusher_player_name"]).fillna(0)
        except Exception:
            try:
                df = _read_csv_cached(fp)
            except Exception:
                _RUSH_TD_SEASON_CACHE[season] = {}
                return _RUSH_TD_SEASON_CACHE[season]
        if df is None or df.empty:
            _RUSH_TD_SEASON_CACHE[season] = {}
            return _RUSH_TD_SEASON_CACHE[season]
        if "rush_touchdown" not in df.columns or "rusher_player_name" not in df.columns:
            _RUSH_TD_SEASON_CACHE[season] = {}
            return _RUSH_TD_SEASON_CACHE[season]
        df["rush_touchdown"] = pd.to_numeric(df["rush_touchdown"], errors="coerce").fillna(0).astype(int)
        out: dict[str,int] = {}
        for _, r in df[df["rush_touchdown"] == 1].iterrows():
            n = str(r.get("rusher_player_name") or "").strip().lower()
            if n:
                out[n] = out.get(n, 0) + 1
        _RUSH_TD_SEASON_CACHE[season] = out
        return _RUSH_TD_SEASON_CACHE[season]

    # Career passer TDs thrown across available pbp_*.csv
    _PASS_TD_CAREER_CACHE: Optional[dict[str,int]] = None
    def _player_pass_td_career_counts() -> dict[str,int]:
        nonlocal _PASS_TD_CAREER_CACHE
        if _PASS_TD_CAREER_CACHE is not None:
                return _PASS_TD_CAREER_CACHE or {}
        out: dict[str,int] = {}
        files = sorted(DATA_DIR.glob("pbp_*.csv"))
        for fp in files:
            try:
                df = pd.read_csv(fp, usecols=["pass_touchdown","passer_player_name"])  # type: ignore
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df["pass_touchdown"] = pd.to_numeric(df["pass_touchdown"], errors="coerce").fillna(0).astype(int)
            for _, r in df[df["pass_touchdown"] == 1].iterrows():
                n = str(r.get("passer_player_name") or "").strip().lower()
                if n:
                    out[n] = out.get(n, 0) + 1
        _PASS_TD_CAREER_CACHE = out
        return _PASS_TD_CAREER_CACHE

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # Compute historical Touchdowns-per-point (TPP) by offense/defense; fallback to league avg when unavailable
    def _team_td_per_point(seasons: list[int]) -> tuple[dict[str,float], dict[str,float], float]:
        # Try loading a cached CSV if present; otherwise return conservative defaults
        fp = DATA_DIR / "team_tpp_cache.csv"
        offense: dict[str,float] = {}
        defense: dict[str,float] = {}
        league_avg = 0.06  # ~1 TD per ~16.7 points
        try:
            if fp.exists():
                df = pd.read_csv(fp)
                if not df.empty:
                    if "offense_tpp" in df.columns:
                        for _, r in df.iterrows():
                            tm = str(r.get("team") or "").strip()
                            if tm:
                                offense[tm] = float(r.get("offense_tpp") or league_avg)
                                defense[tm] = float(r.get("defense_tpp") or league_avg)
                    if "league_avg_tpp" in df.columns:
                        v = pd.to_numeric(df.get("league_avg_tpp").iloc[0], errors="coerce")
                        if not pd.isna(v):
                            league_avg = float(v)
        except Exception:
            pass
        return offense, defense, league_avg

    def _sort_view(view: pd.DataFrame, sort: str, default_sort: str, order: str) -> tuple[pd.DataFrame, str]:
        """Sort a DataFrame by a column name, trying numeric sort first without using deprecated options.

        Returns the sorted DataFrame and the (possibly adjusted) sort column name.
        """
        ascending = order.lower() == "asc"
        if sort not in view.columns:
            sort = default_sort if default_sort in view.columns else view.columns[0]

        col = view[sort]
        # If already numeric, sort directly
        if pd.api.types.is_numeric_dtype(col):
            return view.sort_values(by=[sort], ascending=ascending), sort

        # Try strict numeric conversion (will raise if any non-numeric)
        try:
            converted = pd.to_numeric(col)
            temp = view.assign(__sort_key=converted)
            temp = temp.sort_values(by=["__sort_key"], ascending=ascending).drop(columns=["__sort_key"])
            return temp, sort
        except Exception:
            pass

        # Graceful: map each value to float when possible, NaN otherwise
        def _to_float_or_nan(v):
            try:
                return float(v)
            except Exception:
                return float("nan")

        sort_key = col.map(_to_float_or_nan)
        if sort_key.notna().any():
            temp = view.assign(__sort_key=sort_key)
            temp = temp.sort_values(by=["__sort_key"], ascending=ascending).drop(columns=["__sort_key"])
            return temp, sort

        # Fallback: lexicographic sort on the original column
        return view.sort_values(by=[sort], ascending=ascending), sort

    @app.route("/teams")
    def teams():
        season = request.args.get("season")
        week = request.args.get("week")
        sort = request.args.get("sort", "td_score")
        order = request.args.get("order", "desc")
        team_filter = request.args.get("team")

        fp: Optional[Path] = None
        if season and week:
            fp = DATA_DIR / f"td_likelihood_{season}_wk{week}.csv"
        if fp is None or not fp.exists():
            fp = _latest_csv("td_likelihood")

        if fp is None or not fp.exists():
            return render_template("teams.html", rows=[], columns=[], file_name=None, sort=sort, order=order, team_filter=team_filter, season=season, week=week, message="No team likelihood CSV found in ./data")

        df = _read_csv_cached(fp)
        # Choose a minimal set of columns if available
        desired = [
            "season", "week", "date", "game_id",
            "team", "opponent", "is_home",
            "expected_tds", "td_score", "td_likelihood",
            "implied_points", "spread", "total"
        ]
        cols = [c for c in desired if c in df.columns]
        if not cols:
            cols = list(df.columns)
        view = df[cols].copy()

        # Filter by team
        if team_filter:
            view = view[view["team"].astype(str).str.contains(team_filter, case=False, na=False)]

        # Sorting (numeric-aware without deprecated options)
        view, sort = _sort_view(view, sort, default_sort="td_score", order=order)

        rows = view.to_dict(orient="records")
        return render_template("teams.html", rows=rows, columns=view.columns, file_name=fp.name, sort=sort, order=order, team_filter=team_filter, season=season, week=week, message=None)

    @app.route("/players")
    def players():
        season = request.args.get("season")
        week = request.args.get("week")
        sort = request.args.get("sort", "anytime_td_prob")
        order = request.args.get("order", "desc")
        team_filter = request.args.get("team")
        pos_filter = request.args.get("position")

        fp: Optional[Path] = None
        if season and week:
            fp = DATA_DIR / f"player_td_likelihood_{season}_wk{week}.csv"
        if fp is None or not fp.exists():
            fp = _latest_csv("player_td_likelihood")

        if fp is None or not fp.exists():
            return render_template("players.html", rows=[], columns=[], file_name=None, sort=sort, order=order, team_filter=team_filter, pos_filter=pos_filter, season=season, week=week, message="No player likelihood CSV found in ./data")

        df = pd.read_csv(fp)
        desired = [
            "season", "week", "date", "game_id",
            "team", "opponent", "is_home",
            "player", "position",
            "anytime_td_prob", "expected_td", "exp_rush_td", "exp_rec_td"
        ]
        cols = [c for c in desired if c in df.columns]
        if not cols:
            cols = list(df.columns)
        view = df[cols].copy()

        # Filters
        if team_filter:
            view = view[view["team"].astype(str).str.contains(team_filter, case=False, na=False)]
        if pos_filter:
            view = view[view["position"].astype(str).str.upper() == pos_filter.upper()]

        # Sorting (numeric-aware without deprecated options)
        view, sort = _sort_view(view, sort, default_sort="anytime_td_prob", order=order)

        rows = view.to_dict(orient="records")
        return render_template("players.html", rows=rows, columns=view.columns, file_name=fp.name, sort=sort, order=order, team_filter=team_filter, pos_filter=pos_filter, season=season, week=week, message=None)

    # New UI with cards and modals
    @app.route("/ui")
    def ui_home():
        # default to players tab; redirect so each page can manage its own context
        from flask import redirect, url_for
        season = request.args.get("season", "2025")
        week = request.args.get("week", "1")
        tab = request.args.get("tab", "players")
        if tab == "teams":
            return redirect(url_for("ui_teams", season=season, week=week))
        if tab == "qbs":
            return redirect(url_for("ui_qbs", season=season, week=week))
        return redirect(url_for("ui_players", season=season, week=week))

    @app.route("/ui/players")
    def ui_players():
        season = int(request.args.get("season", "2025"))
        week = int(request.args.get("week", "1"))
        pos_filter = str(request.args.get("pos", "ALL")).upper()
        sort_by = str(request.args.get("sort", "atd")).lower()
        team_filter = str(request.args.get("team", "ALL"))
        game_filter = str(request.args.get("game", "ALL"))
        # Load player CSV (latest fallback)
        fp: Optional[Path] = DATA_DIR / f"player_td_likelihood_{season}_wk{week}.csv"
        if not fp.exists():
            fp = _latest_csv("player_td_likelihood")
        if fp is None or not fp.exists():
            return render_template("ui.html", tab="players", cards=[], season=season, week=week, pos=pos_filter, sort=sort_by, team=team_filter, game=game_filter, teams=[], games=[], env=os.environ)
        df = pd.read_csv(fp)
        # supplement with per-season TD counts
        td_2024 = _player_td_counts_for_season(2024)
        td_2025 = _player_td_counts_for_season(2025)
        # Build options for Team and Game filters
        try:
            teams = sorted({str(x) for x in df["team"].dropna().astype(str).unique().tolist()})
        except Exception:
            teams = []
        games = []
        try:
            if "game_id" in df.columns:
                for gid, grp in df.groupby("game_id"):
                    if grp.empty:
                        continue
                    row = grp.iloc[0]
                    t = str(row.get("team") or "").strip()
                    o = str(row.get("opponent") or "").strip()
                    # Use full team names in the label
                    label = f"{t} vs {o}" if t and o else str(gid)
                    games.append({"value": str(gid), "label": label})
                games.sort(key=lambda x: x["label"])
        except Exception:
            games = []
        # Prefer cached depth chart file if present for reproducibility and injury overlays
        depth_fp = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
        depth_map: dict[tuple[str,str,str], tuple[int,int]] = {}
        if depth_fp.exists():
            try:
                ddf = pd.read_csv(depth_fp)
                if all(c in ddf.columns for c in ["team","position","player","depth_rank","depth_size"]):
                    ddf = ddf.copy()
                    ddf["team"] = ddf["team"].astype(str).str.lower().str.strip()
                    ddf["position"] = ddf["position"].astype(str)
                    ddf["player"] = ddf["player"].astype(str).str.lower().str.strip()
                    ddf["depth_rank"] = pd.to_numeric(ddf["depth_rank"], errors="coerce").fillna(9999).astype(int)
                    ddf["depth_size"] = pd.to_numeric(ddf["depth_size"], errors="coerce").fillna(0).astype(int)
                    for _, rr in ddf.iterrows():
                        key = (rr.get("team"), rr.get("position"), rr.get("player"))
                        depth_map[key] = (int(rr.get("depth_rank")), int(rr.get("depth_size")))
            except Exception:
                depth_map = {}
        if not depth_map:
            depth_map = _build_depth_map(df, season, week)
        # Attach normalized keys and depth ranks to df
        df = df.copy()
        df["_team_norm"] = df["team"].astype(str).str.strip().str.lower()
        df["_player_norm"] = df["player"].astype(str).str.strip().str.lower()
        ranks: list[Optional[int]] = []
        sizes: list[Optional[int]] = []
        for _, rr in df.iterrows():
            key = (rr.get("_team_norm"), str(rr.get("position")), rr.get("_player_norm"))
            rk, sz = depth_map.get(key, (None, None))
            ranks.append(rk)
            sizes.append(sz)
        df["_depth_rank"] = ranks
        df["_depth_size"] = sizes
        # Prepare raw and adjusted values
        df["_raw_exp_td"] = pd.to_numeric(df.get("expected_td"), errors="coerce").fillna(0.0)
        df["_raw_rush"] = pd.to_numeric(df.get("exp_rush_td"), errors="coerce").fillna(0.0)
        df["_raw_rec"] = pd.to_numeric(df.get("exp_rec_td"), errors="coerce").fillna(0.0)
        df["_adj_exp_td"] = 0.0
        df["_adj_rush"] = 0.0
        df["_adj_rec"] = 0.0
        for (tm, pos), grp in df.groupby(["team", "position"], dropna=False):
            if grp.empty:
                continue
            slots = grp.sort_values(["_raw_exp_td", "_raw_rec", "_raw_rush"], ascending=[False, False, False])
            slot_exp = slots["_raw_exp_td"].tolist()
            slot_rush = slots["_raw_rush"].tolist()
            slot_rec = slots["_raw_rec"].tolist()
            depth_sorted = grp.copy()
            depth_sorted["__rk"] = depth_sorted["_depth_rank"].apply(lambda v: 10**6 if pd.isna(v) else int(v))
            depth_sorted = depth_sorted.sort_values(["__rk", "_raw_exp_td"], ascending=[True, False])
            for j, (idx, _) in enumerate(depth_sorted.iterrows()):
                if j < len(slot_exp):
                    df.at[idx, "_adj_exp_td"] = float(slot_exp[j])
                    df.at[idx, "_adj_rush"] = float(slot_rush[j])
                    df.at[idx, "_adj_rec"] = float(slot_rec[j])
                else:
                    df.at[idx, "_adj_exp_td"] = 0.0
                    df.at[idx, "_adj_rush"] = 0.0
                    df.at[idx, "_adj_rec"] = 0.0
        # 2024 totals (rush/rec/total)
        rush24_map: dict[str, int] = {}
        rec24_map: dict[str, int] = {}
        total24_map: dict[str, int] = {}
        totals_fp = DATA_DIR / "player_td_totals_2024.csv"
        if totals_fp.exists():
            try:
                tdf = _read_csv_cached(totals_fp)
                for _, rr in tdf.iterrows():
                    name = str(rr.get("player") or "").strip().lower()
                    if name:
                        rv = pd.to_numeric(rr.get("rush_td"), errors="coerce")
                        ev = pd.to_numeric(rr.get("rec_td"), errors="coerce")
                        tv = pd.to_numeric(rr.get("total_td"), errors="coerce")
                        rush24_map[name] = 0 if pd.isna(rv) else int(rv)
                        rec24_map[name] = 0 if pd.isna(ev) else int(ev)
                        total24_map[name] = 0 if pd.isna(tv) else int(tv)
            except Exception:
                pass
        # Team expected tds to show matchup context
        team_fp = DATA_DIR / f"td_likelihood_{season}_wk{week}.csv"
        team_df = _read_csv_cached(team_fp) if team_fp.exists() else pd.DataFrame()
        team_exp: dict[tuple[object, str], float] = {}
        adj_team_exp: dict[tuple[object, str], float] = {}
        if not team_df.empty:
            for _, r in team_df.iterrows():
                gid = r.get("game_id"); tm = str(r.get("team"))
                team_exp[(gid, tm)] = float(r.get("expected_tds") or 0.0)
            for k, v in team_exp.items():
                adj_team_exp[k] = v
        # Load precomputed meta if present, else build runtime maps
        meta = _load_player_meta()
        if meta is None:
            career_rush_map, career_rec_map, career_total_map = _player_td_career_counts()
        else:
            career_rush_map = {k: v.get("career_rush", 0) for k, v in meta.items()}
            career_rec_map = {k: v.get("career_rec", 0) for k, v in meta.items()}
            career_total_map = {k: v.get("career_total", 0) for k, v in meta.items()}
        cards: list[dict] = []
        for _, r in df.iterrows():
            p = str(r.get("player"))
            team = str(r.get("team"))
            opp = str(r.get("opponent"))
            game_id = r.get("game_id")
            headshot = _headshot_for(p, season)
            logo_url = _logo_for_team(team)
            mrow = meta.get(p.lower()) if meta is not None else None
            try:
                d_rank = None if pd.isna(r.get("_depth_rank")) else int(r.get("_depth_rank"))
            except Exception:
                d_rank = r.get("_depth_rank")
            try:
                d_total = None if pd.isna(r.get("_depth_size")) else int(r.get("_depth_size"))
            except Exception:
                d_total = r.get("_depth_size")
            depth_label = f"{str(r.get('position'))}{d_rank}" if d_rank is not None else None
            raw_exp_td = float(r.get("_raw_exp_td") or 0.0)
            raw_rush = float(r.get("_raw_rush") or 0.0)
            raw_rec = float(r.get("_raw_rec") or 0.0)
            adj_exp_td = float(r.get("_adj_exp_td") or 0.0)
            adj_rush = float(r.get("_adj_rush") or 0.0)
            adj_rec = float(r.get("_adj_rec") or 0.0)
            import math
            raw_atd = 0.0 if raw_exp_td <= 0 else (1.0 - math.exp(-raw_exp_td))
            adj_atd = 0.0 if adj_exp_td <= 0 else (1.0 - math.exp(-adj_exp_td))
            card = {
                "player": p,
                "team": team,
                "opp": opp,
                "pos": r.get("position"),
                "depth_rank": d_rank if (d_rank is None or isinstance(d_rank, int)) else int(d_rank),
                "depth_size": d_total if (d_total is None or isinstance(d_total, int)) else int(d_total),
                "depth_label": depth_label,
                "atd": adj_atd,
                "adj_exp_td": adj_exp_td,
                "adj_exp_rush_td": adj_rush,
                "adj_exp_rec_td": adj_rec,
                "raw_atd": raw_atd,
                "exp_td": raw_exp_td,
                "exp_rush_td": raw_rush,
                "exp_rec_td": raw_rec,
                "td_2024": int(_map_lookup_with_variants(td_2024, p) or 0),
                "td_2025": int(_map_lookup_with_variants(td_2025, p) or 0),
                "td24_rush": (mrow.get("td24_rush") if mrow else _map_lookup_with_variants(rush24_map, p)) or 0,
                "td24_rec": (mrow.get("td24_rec") if mrow else _map_lookup_with_variants(rec24_map, p)) or 0,
                "td24_total": (mrow.get("td24_total") if mrow else _map_lookup_with_variants(total24_map, p)) or 0,
                "career_rush": (mrow.get("career_rush") if mrow else _map_lookup_with_variants(career_rush_map, p)) or 0,
                "career_rec": (mrow.get("career_rec") if mrow else _map_lookup_with_variants(career_rec_map, p)) or 0,
                "career_total": (mrow.get("career_total") if mrow else _map_lookup_with_variants(career_total_map, p)) or 0,
                "is_home": int(r.get("is_home") or 0) == 1,
                "game_exp_tds": team_exp.get((game_id, team), None),
                "game_exp_tds_adj": adj_team_exp.get((game_id, team), None),
                "team_abbr": _team_abbrev(team),
                "team_class": _team_avatar_class(team),
                "headshot": headshot,
                "logo_url": logo_url,
                "game_id": game_id,
            }
            cards.append(card)
        # Apply filters
        if pos_filter and pos_filter != "ALL":
            cards = [c for c in cards if str(c.get("pos", "")).upper() == pos_filter]
        if team_filter and team_filter != "ALL":
            cards = [c for c in cards if str(c.get("team", "")) == team_filter]
        if game_filter and game_filter != "ALL":
            cards = [c for c in cards if str(c.get("game_id", "")) == str(game_filter)]
        # Sort mapping
        def sort_key(c: dict):
            if sort_by == "atd":
                return float(c.get("atd", 0.0))
            if sort_by in ("exp", "expected", "adj_exp_td"):
                return float(c.get("adj_exp_td", 0.0))
            if sort_by in ("rush", "adj_exp_rush_td"):
                return float(c.get("adj_exp_rush_td", 0.0))
            if sort_by in ("rec", "adj_exp_rec_td"):
                return float(c.get("adj_exp_rec_td", 0.0))
            if sort_by in ("td24", "2024"):
                return float(c.get("td24_total", 0.0))
            if sort_by in ("career", "career_total"):
                return float(c.get("career_total", 0.0))
            if sort_by in ("team",):
                return str(c.get("team", "")).lower()
            if sort_by in ("player", "name"):
                return str(c.get("player", "")).lower()
            if sort_by in ("depth", "depth_rank"):
                dr = c.get("depth_rank")
                try:
                    return int(dr) if dr is not None else 1_000_000
                except Exception:
                    return 1_000_000
            return float(c.get("atd", 0.0))
        reverse = True
        if sort_by in ("team", "player", "name", "depth", "depth_rank"):
            reverse = False
        cards.sort(key=sort_key, reverse=reverse)
        return render_template("ui.html", tab="players", cards=cards, season=season, week=week, pos=pos_filter, sort=sort_by, team=team_filter, game=game_filter, teams=teams, games=games, env=os.environ)

    @app.route("/ui/teams")
    def ui_teams():
        season = int(request.args.get("season", "2025"))
        week = int(request.args.get("week", "1"))
        fp: Optional[Path] = DATA_DIR / f"td_likelihood_{season}_wk{week}.csv"
        if not fp.exists():
            fp = _latest_csv("td_likelihood")
        if fp is None or not fp.exists():
            return render_template("ui.html", tab="teams", cards=[], season=season, week=week, env=os.environ)
        df = pd.read_csv(fp)
        # Build adjusted team expected TDs using scoring-mix TPP
        offense_tpp, defense_tpp, league_avg_tpp = _team_td_per_point([2024])
        def _adj_team_tds(team: str, opp: Optional[str], base: float) -> float:
            off = offense_tpp.get(team)
            deff = defense_tpp.get(opp) if opp else None
            if off is None and deff is None:
                return float(base)
            blended = (0.6 * off if off is not None else 0.0) + (0.4 * deff if deff is not None else 0.0)
            denom = league_avg_tpp if league_avg_tpp > 0 else 0.06
            scale = blended / denom if blended and denom else 1.0
            return float(base) * float(scale)
        # Ensure required columns as strings
        df = df.copy()
        if "expected_tds" not in df.columns:
            df["expected_tds"] = 0.0
        if "opponent" not in df.columns:
            df["opponent"] = None
        df["__adj_expected_tds"] = df.apply(lambda r: _adj_team_tds(str(r.get("team")), str(r.get("opponent")) if r.get("opponent") is not None else None, float(r.get("expected_tds") or 0.0)), axis=1)
        # Load team position shares (last N seasons)
        tps_files = sorted(DATA_DIR.glob("team_pos_td_shares_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        tps = pd.read_csv(tps_files[0]) if tps_files else pd.DataFrame()
        def get_pos_shares(team: str, kind: str) -> dict[str, float]:
            if tps is None or tps.empty:
                return {}
            m = tps[(tps.get("team").astype(str) == team) & (tps.get("kind").astype(str) == kind)]
            if m.empty:
                return {}
            row = m.iloc[0]
            keys = ["WR","TE","RB","QB"]
            out = {k: float(row.get(k) or 0.0) for k in keys}
            # normalize selected keys
            s = sum(out.values())
            if s > 0:
                out = {k: v/s for k, v in out.items()}
            return out
        # default pass/rush split when not present in CSV
        PASS_FRAC = 0.58
        RUSH_FRAC = 1.0 - PASS_FRAC
        # 2024 historical shares
        kind24, pos24, pos24_counts = _get_team_2024_distributions()
        # Build cards
        # First, group by game to compute projected game TDs (adjusted)
        game_sum = df.groupby("game_id")["__adj_expected_tds"].sum().to_dict()
        cards: list[dict] = []
        for _, r in df.iterrows():
            team = str(r.get("team"))
            opp = str(r.get("opponent"))
            # Use adjusted team TDs only
            exp_tds = float(r.get("__adj_expected_tds") or 0.0)
            gid = r.get("game_id")
            logo_url = _logo_for_team(team)
            # position projections
            pass_sh = get_pos_shares(team, "pass")
            rush_sh = get_pos_shares(team, "rush")
            # expected by position
            by_pos = {"RB":0.0,"WR":0.0,"TE":0.0,"QB":0.0}
            for pos in ["WR","TE","RB"]:
                by_pos[pos] += PASS_FRAC * exp_tds * float(pass_sh.get(pos, 0.0))
            for pos in ["RB","QB","WR","TE"]:
                by_pos[pos] += RUSH_FRAC * exp_tds * float(rush_sh.get(pos, 0.0))
            # round for display
            for k in by_pos:
                by_pos[k] = round(by_pos[k], 2)
            # shares to present
            proj_kind = {
                "pass": round(PASS_FRAC, 2),
                "rush": round(RUSH_FRAC, 2),
            }
            hist_kind = None
            if team in kind24:
                hist_kind = {
                    "pass": round(float(kind24[team].get("pass", 0.0)), 2),
                    "rush": round(float(kind24[team].get("rush", 0.0)), 2),
                }
            hist_pos = None
            hist_pos_counts = None
            if team in pos24:
                hist_pos = {k: round(float(pos24[team].get(k, 0.0)), 2) for k in ["RB","WR","TE","QB"]}
            if team in pos24_counts:
                hist_pos_counts = {k: int(pos24_counts[team].get(k, 0)) for k in ["RB","WR","TE","QB"]}
            card = {
                "team": team,
                "opp": opp,
                "team_abbr": _team_abbrev(team),
                "team_class": _team_avatar_class(team),
                "logo_url": logo_url,
                "game_tds": round(float(game_sum.get(gid, 0.0)), 2),
                "team_tds": round(exp_tds, 2),
                "by_pos": by_pos,
                "proj_kind": proj_kind,
                "hist_kind": hist_kind,
                "hist_pos": hist_pos,
                "hist_pos_counts": hist_pos_counts,
                "game_id": gid,
                "date": r.get("date"),
                "is_home": int(r.get("is_home") or 0) == 1,
            }
            cards.append(card)
        # sort by team_tds desc
        cards.sort(key=lambda x: x["team_tds"], reverse=True)
        # render full UI shell
        return render_template("ui.html", tab="teams", cards=cards, season=season, week=week, env=os.environ)

    @app.route("/ui/qbs")
    def ui_qbs():
        season = int(request.args.get("season", "2025"))
        week = int(request.args.get("week", "1"))
        # Use player CSV to derive team expected pass TDs (sum of receivers' expected TDs)
        fp: Optional[Path] = DATA_DIR / f"player_td_likelihood_{season}_wk{week}.csv"
        if not fp.exists():
            fp = _latest_csv("player_td_likelihood")
        if fp is None or not fp.exists():
            return render_template("ui.html", tab="qbs", cards=[], season=season, week=week, env=os.environ)
        df = pd.read_csv(fp)
        # Ensure needed columns
        for c in ["team","opponent","position","player","exp_rec_td","exp_rush_td","game_id"]:
            if c not in df.columns:
                # Render empty if missing required data
                return render_template("ui.html", tab="qbs", cards=[], season=season, week=week, env=os.environ)
        df = df.copy()
        df["exp_rec_td"] = pd.to_numeric(df["exp_rec_td"], errors="coerce").fillna(0.0)
        df["exp_rush_td"] = pd.to_numeric(df["exp_rush_td"], errors="coerce").fillna(0.0)
        # Build depth map to identify starting QBs
        depth_fp = DATA_DIR / f"depth_chart_{season}_wk{week}.csv"
        depth_map: dict[tuple[str,str,str], tuple[int,int]] = {}
        if depth_fp.exists():
            try:
                ddf = pd.read_csv(depth_fp)
                if all(c in ddf.columns for c in ["team","position","player","depth_rank","depth_size"]):
                    ddf = ddf.copy()
                    ddf["team"] = ddf["team"].astype(str).str.lower().str.strip()
                    ddf["position"] = ddf["position"].astype(str)
                    ddf["player"] = ddf["player"].astype(str).str.lower().str.strip()
                    ddf["depth_rank"] = pd.to_numeric(ddf["depth_rank"], errors="coerce").fillna(9999).astype(int)
                    ddf["depth_size"] = pd.to_numeric(ddf["depth_size"], errors="coerce").fillna(0).astype(int)
                    for _, rr in ddf.iterrows():
                        key = (rr.get("team"), rr.get("position"), rr.get("player"))
                        depth_map[key] = (int(rr.get("depth_rank")), int(rr.get("depth_size")))
            except Exception:
                depth_map = {}
        if not depth_map:
            depth_map = _build_depth_map(df, season, week)
        # Helper to normalize
        def _norm(s: object) -> str:
            return str(s or "").strip().lower()
        # Team expected pass TDs by summing receivers
        team_pass_tds = df.groupby(["game_id","team"], dropna=False)["exp_rec_td"].sum().to_dict()
        # Collect QB candidates per team with ranks
        qb_rows = df[df["position"].astype(str).str.upper() == "QB"].copy()
        team_qb_candidates: dict[str, list[tuple[int, dict]] ] = {}
        for _, r in qb_rows.iterrows():
            team = str(r.get("team"))
            key = (_norm(team), "QB", _norm(r.get("player")))
            rk, sz = depth_map.get(key, (9999, None))
            rec = {
                "player": str(r.get("player")),
                "team": team,
                "opp": str(r.get("opponent")),
                "game_id": r.get("game_id"),
                "depth_rank": None if rk is None else int(rk),
                "depth_size": sz,
            }
            team_qb_candidates.setdefault(team, []).append((rk if rk is not None else 9999, rec))
        # Preload 2024 QB stats and career aggregates
        pass24 = _player_pass_td_counts_for_season(2024)
        rush24 = _player_rush_td_counts_for_season(2024)
        # Totals fallback CSV for 2024 rush TDs
        totals24_map: dict[str, int] = {}
        totals_fp = DATA_DIR / "player_td_totals_2024.csv"
        if totals_fp.exists():
            try:
                tdf = pd.read_csv(totals_fp)
                for _, rr in tdf.iterrows():
                    name = str(rr.get("player") or "").strip().lower()
                    if name:
                        rv = pd.to_numeric(rr.get("rush_td"), errors="coerce")
                        totals24_map[name] = 0 if pd.isna(rv) else int(rv)
            except Exception:
                totals24_map = {}
        # Fallback meta for 2024 totals if available (helps QBs with name variants)
        meta = _load_player_meta()
        pass_career = _player_pass_td_career_counts()
        career_rush_map, _, _ = _player_td_career_counts()
        # Build cards per team
        cards: list[dict] = []
        for (gid, team), pass_tds in team_pass_tds.items():
            qb_list = team_qb_candidates.get(team, [])
            qb_rec = None
            if qb_list:
                qb_list.sort(key=lambda x: x[0])
                qb_rec = qb_list[0][1]
            # visuals
            logo_url = _logo_for_team(team)
            team_name = team
            # QB rush prob from exp_rush_td
            qb_rush_lambda = 0.0
            try:
                qb_row = qb_rows[qb_rows["team"].astype(str) == team]
                if qb_rec is not None:
                    qb_row = qb_row[qb_row["player"].astype(str) == qb_rec["player"]]
                qb_rush_lambda = float(qb_row.get("exp_rush_td").sum()) if not qb_row.empty else 0.0
            except Exception:
                qb_rush_lambda = 0.0
            import math as _m
            qb_rush_prob = 0.0 if qb_rush_lambda <= 0 else (1.0 - _m.exp(-qb_rush_lambda))
            # Stats
            qb_name = qb_rec.get("player") if qb_rec else None
            td24_thrown = int(_map_lookup_with_variants(pass24, qb_name) or 0) if qb_name else 0
            td24_rushed = int(_map_lookup_with_variants(rush24, qb_name) or 0) if qb_name else 0
            if qb_name and td24_rushed == 0 and totals24_map:
                td24_rushed = int(_map_lookup_with_variants(totals24_map, qb_name) or 0)
            if qb_name and meta is not None and td24_rushed == 0:
                mrow = meta.get(qb_name.lower())
                if mrow:
                    # meta stores 2024 rushing TDs under td24_rush for all players including QBs
                    try:
                        td24_rushed = int(mrow.get("td24_rush") or 0)
                    except Exception:
                        pass
            career_thrown = int(_map_lookup_with_variants(pass_career, qb_name) or 0) if qb_name else 0
            career_rushed = int(_map_lookup_with_variants(career_rush_map, qb_name) or 0) if qb_name else 0
            # Card
            card = {
                "player": qb_rec.get("player") if qb_rec else "QB (TBD)",
                "team": team_name,
                "opp": qb_rec.get("opp") if qb_rec else None,
                "team_abbr": _team_abbrev(team_name),
                "team_class": _team_avatar_class(team_name),
                "logo_url": logo_url,
                "exp_pass_tds": float(pass_tds),
                "qb_rush_prob": qb_rush_prob,
                "qb_rush_lambda": qb_rush_lambda,
                "depth_rank": qb_rec.get("depth_rank") if qb_rec else None,
                "depth_size": qb_rec.get("depth_size") if qb_rec else None,
                "headshot": _headshot_for(qb_rec.get("player"), season) if qb_rec else None,
                "game_id": gid,
                # 2024 and career
                "td24_thrown": td24_thrown,
                "td24_rushed": td24_rushed,
                "career_thrown": career_thrown,
                "career_rushed": career_rushed,
            }
            cards.append(card)
        cards.sort(key=lambda x: x["exp_pass_tds"], reverse=True)
        return render_template("ui.html", tab="qbs", cards=cards, season=season, week=week, env=os.environ)

    @app.route("/trends")
    def trends():
        sort = request.args.get("sort", "team")
        order = request.args.get("order", "asc")
        team_filter = request.args.get("team")

        # Find latest cached team position shares
        if not DATA_DIR.exists():
            return render_template("trends.html", rows=[], columns=[], file_name=None, sort=sort, order=order, team_filter=team_filter, message="No data directory found")

        files = sorted(DATA_DIR.glob("team_pos_td_shares_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return render_template("trends.html", rows=[], columns=[], file_name=None, sort=sort, order=order, team_filter=team_filter, message="No trends cache found; build with build_td_trends.py")
        fp = files[0]

        try:
            tps = pd.read_csv(fp)
        except Exception:
            return render_template("trends.html", rows=[], columns=[], file_name=fp.name, sort=sort, order=order, team_filter=team_filter, message="Failed to read trends cache")

        # Expect columns: team, kind (pass/rush), WR/TE/RB/QB/OTHER shares
        tps = tps.copy()
        for c in ["team","kind"]:
            if c not in tps.columns:
                return render_template("trends.html", rows=[], columns=[], file_name=fp.name, sort=sort, order=order, team_filter=team_filter, message="Trends cache missing required columns")

        # Build a wide summary: pass_WR/TE/RB and rush_RB/QB/WR/TE
        pass_df = tps[tps["kind"].astype(str) == "pass"].copy()
        rush_df = tps[tps["kind"].astype(str) == "rush"].copy()
        def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
            out = df.copy()
            for c in cols:
                if c not in out.columns:
                    out[c] = pd.NA
            return out
        pass_df = ensure_cols(pass_df, ["WR","TE","RB"]).rename(columns={"WR":"pass_WR","TE":"pass_TE","RB":"pass_RB"})
        rush_df = ensure_cols(rush_df, ["RB","QB","WR","TE"]).rename(columns={"RB":"rush_RB","QB":"rush_QB","WR":"rush_WR","TE":"rush_TE"})
        view = pass_df.merge(rush_df, on="team", how="outer")

        # Filter by team
        if team_filter:
            view = view[view["team"].astype(str).str.contains(team_filter, case=False, na=False)]

        # Sorting
        view, sort = _sort_view(view, sort, default_sort="team", order=order)

        rows = view.to_dict(orient="records")
        return render_template("trends.html", rows=rows, columns=view.columns, file_name=fp.name, sort=sort, order=order, team_filter=team_filter, message=None)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
