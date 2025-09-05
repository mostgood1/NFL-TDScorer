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

    @app.route("/admin/refresh-live-tds", methods=["POST", "GET"])
    def admin_refresh_live_tds():
        env_flag = os.environ.get("TD_ADMIN", "").lower()
        # Fallbacks: query param admin=1 or key match (if ADMIN_KEY env set)
        qp_admin = (request.args.get("admin") or request.form.get("admin") or "").lower()
        admin_key_env = os.environ.get("ADMIN_KEY", "")
        supplied_key = request.args.get("key") or request.form.get("key") or ""
        authorized = False
        if env_flag in {"1","true","yes"}:
            authorized = True
        elif qp_admin in {"1","true","yes"}:
            authorized = True
        elif admin_key_env and supplied_key and (admin_key_env == supplied_key):
            authorized = True
        if not authorized:
            # Debug hint header so we can see what server sees
            return ({"ok": False, "error": "unauthorized", "debug_env_flag": env_flag, "hint": "set TD_ADMIN=1 or pass ?admin=1"}, 403)
        try:
            s = int(request.args.get("season") or request.form.get("season") or 2025)
            w = int(request.args.get("week") or request.form.get("week") or 1)
            names = _fetch_espn_td_scorers(s, w, sweep=bool(request.args.get('sweep') or request.form.get('sweep')))
            from datetime import datetime
            _save_live_td_scorers(s, w, names, meta={"refreshed_at": datetime.utcnow().isoformat() + "Z", "source": "espn"})
            return {"ok": True, "count": len(names)}
        except Exception as e:
            return {"ok": False, "error": str(e)}, 500

    @app.route("/admin/live-log")
    def admin_live_log():
        if os.environ.get("TD_ADMIN", "").lower() not in {"1","true","yes"}:
            return {"ok": False, "error": "unauthorized"}, 403
        try:
            s = int(request.args.get("season") or 2025)
            w = int(request.args.get("week") or 1)
            fp = _live_td_log_file(s, w)
            if not fp.exists():
                return {"ok": False, "error": "log not found"}, 404
            from flask import send_file
            return send_file(str(fp), mimetype="text/csv", as_attachment=True, download_name=fp.name)
        except Exception as e:
            return {"ok": False, "error": str(e)}, 500

    @app.route("/admin/seed-live-tds", methods=["POST","GET"])
    def admin_seed_live_tds():
        if os.environ.get("TD_ADMIN", "").lower() not in {"1","true","yes"}:
            return {"ok": False, "error": "unauthorized"}, 403
        try:
            s = int(request.args.get("season") or request.form.get("season") or 2025)
            w = int(request.args.get("week") or request.form.get("week") or 1)
            raw = request.args.get("names") or request.form.get("names") or ""
            # Accept comma or newline separated
            parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
            if not parts:
                return {"ok": False, "error": "no names provided"}, 400
            existing = _load_live_td_scorers(s, w)
            before_ct = len(existing)
            # Merge case-insensitive
            for nm in parts:
                if nm.lower() not in existing:
                    existing.add(nm.lower())
            merged_list = sorted({p for p in parts} | {n for n in existing})
            from datetime import datetime as _dt
            _save_live_td_scorers(s, w, list(merged_list), meta={"refreshed_at": _dt.utcnow().isoformat()+"Z", "source": "manual_seed", "added": parts})
            return {"ok": True, "added": parts, "total": len(merged_list), "previous": before_ct}
        except Exception as e:
            return {"ok": False, "error": str(e)}, 500

    @app.route("/health")
    def health():
        started = False
        try:
            started = bool(globals().get("_LIVE_BG_THREAD_STARTED"))
        except Exception:
            started = False
        return {"ok": True, "live_thread": started}

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
        # Build abbr map too from same assets
        nonlocal _TEAM_ABBR_MAP
        am: dict[str, str] = {}
        try:
            fp = DATA_DIR / "nfl_team_assets.json"
            if fp.exists():
                data = json.load(open(fp, "r", encoding="utf-8"))
                items = []
                def add_abbr_entry(name: Optional[str], abbr: Optional[str]):
                    if not name or not abbr:
                        return
                    am[str(name).lower()] = str(abbr).upper()
                    am[str(abbr).lower()] = str(abbr).upper()
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("teams") or data.get("data") or []
                    if not isinstance(items, list):
                        # try dict mapping
                        for k, v in data.items():
                            if isinstance(v, dict):
                                add_abbr_entry(k, v.get("abbr") or v.get("abbreviation"))
                        items = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    name = it.get("display_name") or it.get("full_name") or it.get("team") or it.get("name")
                    abbr = it.get("abbr") or it.get("abbreviation")
                    add_abbr_entry(name, abbr)
        except Exception:
            pass
        _TEAM_ABBR_MAP = am

    # Also keep a name->abbr map for robust keying across datasets
    _TEAM_ABBR_MAP: dict[str, str] = {}

    # Hard fallback map for full team names and common variants -> standard abbreviations
    _HARD_ABBR_MAP: dict[str, str] = {
        # NFC
        "arizona cardinals": "ARI",
        "atlanta falcons": "ATL",
        "carolina panthers": "CAR",
        "chicago bears": "CHI",
        "dallas cowboys": "DAL",
        "detroit lions": "DET",
        "green bay packers": "GB",
        "los angeles rams": "LAR",
        "minnesota vikings": "MIN",
        "new orleans saints": "NO",
        "new york giants": "NYG",
        "philadelphia eagles": "PHI",
        "san francisco 49ers": "SF",
        "seattle seahawks": "SEA",
        "tampa bay buccaneers": "TB",
        "washington commanders": "WAS",
        # AFC
        "baltimore ravens": "BAL",
        "buffalo bills": "BUF",
        "cincinnati bengals": "CIN",
        "cleveland browns": "CLE",
        "denver broncos": "DEN",
        "houston texans": "HOU",
        "indianapolis colts": "IND",
        "jacksonville jaguars": "JAX",
        "kansas city chiefs": "KC",
        "las vegas raiders": "LV",
        "los angeles chargers": "LAC",
        "miami dolphins": "MIA",
        "new england patriots": "NE",
        "new york jets": "NYJ",
        "pittsburgh steelers": "PIT",
        "tennessee titans": "TEN",
        # Common abbr/legacy aliases
        "la": "LAR",
        "lar": "LAR",
        "lac": "LAC",
        "lv": "LV",
        "oak": "LV",
        "sd": "LAC",
        "nof": "NO",
        "nor": "NO",
        "was": "WAS",
        "wsh": "WAS",
        "sf": "SF",
        "gb": "GB",
    }

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

    def _team_std_abbr(team_name: str) -> Optional[str]:
        # Try assets-provided abbr mapping first
        nonlocal _TEAM_ABBR_MAP
        if not _TEAM_ABBR_MAP:
            _load_team_logos()
        if _TEAM_ABBR_MAP:
            ab = _TEAM_ABBR_MAP.get(str(team_name).lower())
            if ab:
                return ab.upper()
        # Fallback to hardcoded full-name map
        t = str(team_name or "").strip()
        if t:
            ab2 = _HARD_ABBR_MAP.get(t.lower())
            if ab2:
                return ab2
        # Fallback heuristic (may not be standard)
        parts = t.split()
        if len(parts) == 1:
            return parts[0][:3].upper()
        return None

    # Live TD scorers (weekly) — cache to data/live_td_{season}_wk{week}.json
    def _live_td_file(season: int, week: int) -> Path:
        return DATA_DIR / f"live_td_{season}_wk{week}.json"

    # Event log file (append scoring events with timestamp + player names)
    def _live_td_log_file(season: int, week: int) -> Path:
        return DATA_DIR / f"live_td_log_{season}_wk{week}.csv"

    def _save_live_td_scorers(season: int, week: int, names: list[str], meta: Optional[dict] = None) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "names": sorted(list({str(n).strip() for n in names if str(n).strip()})),
                "meta": meta or {},
            }
            with open(_live_td_file(season, week), "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def _append_live_td_events(season: int, week: int, new_names: list[str]) -> None:
        """Append newly seen TD scorer names to a CSV log with UTC timestamp."""
        if not new_names:
            return
        try:
            from datetime import datetime, timezone
            fp = _live_td_log_file(season, week)
            exists = fp.exists()
            with open(fp, "a", encoding="utf-8") as f:
                if not exists:
                    f.write("ts_utc,player\n")
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                for nm in sorted({n for n in new_names if n}):
                    f.write(f"{now},{nm}\n")
        except Exception:
            pass

    def _load_live_td_scorers(season: int, week: int) -> set[str]:
        try:
            fp = _live_td_file(season, week)
            if not fp.exists():
                return set()
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            names = obj.get("names") or []
            return {str(n).strip().lower() for n in names if str(n).strip()}
        except Exception:
            return set()

    def _live_td_age_secs(season: int, week: int) -> Optional[float]:
        try:
            fp = _live_td_file(season, week)
            if not fp.exists():
                return None
            import time
            return max(0.0, time.time() - fp.stat().st_mtime)
        except Exception:
            return None

    # Helper: are we currently within typical NFL game windows (ET)?
    def _is_game_window_now_et() -> bool:
        try:
            from datetime import datetime
            try:
                from zoneinfo import ZoneInfo  # py3.9+
                tz = ZoneInfo("America/New_York")
            except Exception:
                # Fallback: naive local time
                tz = None
            now = datetime.now(tz) if tz else datetime.now()
            day = now.weekday()  # 0=Mon .. 6=Sun
            hr = now.hour
            # Windows (ET): Sun 09:00-23:59, Mon 19:00-23:59, Thu 19:00-23:59, Sat 13:00-23:59
            if day == 6 and hr >= 9:  # Sunday
                return True
            if day == 0 and hr >= 19:  # Monday
                return True
            if day == 3 and hr >= 19:  # Thursday
                return True
            if day == 5 and hr >= 13:  # Saturday
                return True
            return False
        except Exception:
            return False

    def _fetch_espn_td_scorers(season: int, week: int, sweep: bool = False) -> list[str]:
        """Fetch this week's TD scorers via ESPN scoreboard/summary endpoints.

        sweep=True widens search if normal probes return nothing:
          - Tries nearby weeks (week-1, week, week+1) with season types 1,2,3
          - Uses date-based scoreboard queries for yesterday & today (UTC)
        """
        try:
            import requests  # type: ignore
        except Exception:
            return []
        base = "https://site.api.espn.com/apis/v2/sports/football/nfl"
        names: set[str] = set()
        # Get events for the week
        event_ids: list[str] = []
        try:
            # Expanded probe set: prioritize regular season (seasontype=2), include group=80 (NFL), then fallbacks.
            probe_primary = [
                f"{base}/scoreboard?week={int(week)}&season={int(season)}&seasontype=2&groups=80",
                f"{base}/scoreboard?week={int(week)}&year={int(season)}&seasontype=2&groups=80",
                f"{base}/scoreboard?week={int(week)}&season={int(season)}&seasontype=2",
                f"{base}/scoreboard?week={int(week)}&year={int(season)}&seasontype=2",
                f"{base}/scoreboard?seasontype=2&week={int(week)}&groups=80",
                f"{base}/scoreboard?seasontype=2&week={int(week)}",
                f"{base}/scoreboard?year={int(season)}&groups=80",
                f"{base}/scoreboard?season={int(season)}&groups=80",
                f"{base}/scoreboard?year={int(season)}",
                f"{base}/scoreboard?season={int(season)}",
            ]
            collected: list[str] = []
            for sb_url in probe_primary:
                try:
                    r = requests.get(sb_url, timeout=10)
                except Exception:
                    continue
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except Exception:
                        continue
                    events = data.get("events") or []
                    if events:
                        collected.extend([str(e.get("id")) for e in events if e and e.get("id")])
                        if collected:
                            break
            # If still empty, probe preseason (1) and postseason (3)
            if not collected:
                for st in (1, 3):
                    fallback_urls = [
                        f"{base}/scoreboard?week={int(week)}&season={int(season)}&seasontype={st}&groups=80",
                        f"{base}/scoreboard?week={int(week)}&year={int(season)}&seasontype={st}&groups=80",
                        f"{base}/scoreboard?seasontype={st}&week={int(week)}&groups=80",
                    ]
                    for sb_url in fallback_urls:
                        try:
                            r = requests.get(sb_url, timeout=10)
                        except Exception:
                            continue
                        if r.status_code == 200:
                            try:
                                data = r.json()
                            except Exception:
                                continue
                            events = data.get("events") or []
                            if events:
                                collected.extend([str(e.get("id")) for e in events if e and e.get("id")])
                                if collected:
                                    break
                    if collected:
                        break
            # Deduplicate preserving order
            seen = set()
            event_ids = []
            for eid in collected:
                if eid not in seen:
                    seen.add(eid)
                    event_ids.append(eid)
        except Exception:
            event_ids = []
        # If no events and sweep requested, broaden search heuristically
        if (not event_ids) and sweep:
            try:
                import datetime as _dt
                today = _dt.datetime.utcnow().date()
                dates = [today - _dt.timedelta(days=1), today]
                date_strs = [d.strftime('%Y%m%d') for d in dates]
                nearby_weeks = sorted({w for w in [week, week-1, week+1] if w >= 0})
                seasontypes = [2,1,3]  # regular, preseason, postseason
                extra_urls: list[str] = []
                for ds in date_strs:
                    extra_urls.append(f"{base}/scoreboard?dates={ds}&groups=80")
                for st in seasontypes:
                    for wk in nearby_weeks:
                        extra_urls.append(f"{base}/scoreboard?week={int(wk)}&season={int(season)}&seasontype={st}&groups=80")
                for url in extra_urls:
                    try:
                        r = requests.get(url, timeout=10)
                    except Exception:
                        continue
                    if r.status_code != 200:
                        continue
                    try:
                        data = r.json()
                    except Exception:
                        continue
                    events = data.get('events') or []
                    if events:
                        for e in events:
                            eid = str(e.get('id')) if e and e.get('id') else None
                            if eid:
                                event_ids.append(eid)
                # Deduplicate
                event_ids = list(dict.fromkeys(event_ids))
            except Exception:
                pass

        # For each event, parse scoring plays for touchdowns
        for eid in event_ids:
            try:
                s_url = f"{base}/summary?event={eid}"
                rs = requests.get(s_url, timeout=10)
                if rs.status_code != 200:
                    continue
                summ = rs.json()
                scoring = summ.get("scoringPlays") or []
                for p in scoring:
                    txt_raw = str(p.get("text") or "")
                    txt = txt_raw.lower()
                    stype = (p.get("scoringType") or {}).get("displayName") or (p.get("type") or {}).get("text")
                    if ("touchdown" not in txt) and (not stype or "touchdown" not in str(stype).lower()):
                        continue
                    # Prefer player list with roles
                    players = p.get("players") or []
                    for pl in players:
                        role = str(pl.get("type") or "").lower()
                        ath = (pl.get("athlete") or {})
                        nm = str(ath.get("displayName") or "").strip()
                        if nm and any(k in role for k in ["rush", "rushing", "reception", "catch", "return", "run"]):
                            names.add(nm)
                    # Fallback: top-level athlete
                    ath = (p.get("athlete") or {})
                    nm = str(ath.get("displayName") or "").strip()
                    if nm:
                        names.add(nm)
                    # Fallback 2: parse from text when structures are missing
                    if not nm and not players:
                        try:
                            import re as _re
                            # Receiver TD: "Name pass from Name" => first name is scorer
                            m = _re.search(r"([A-Z][A-Za-z\.'-]+\s+[A-Z][A-Za-z\.'-]+)\s+pass\s+from\s+([A-Z][A-Za-z\.'-]+\s+[A-Z][A-Za-z\.'-]+)", txt_raw)
                            if m:
                                names.add(m.group(1))
                            else:
                                # Rush TD: "Name X Yd Run" or contains 'rush'/'run'
                                m2 = _re.search(r"^([A-Z][A-Za-z\.'-]+\s+[A-Z][A-Za-z\.'-]+).*(?:run|rush)", txt_raw)
                                if m2:
                                    names.add(m2.group(1))
                                else:
                                    # Return TD: "Name return" patterns
                                    m3 = _re.search(r"^([A-Z][A-Za-z\.'-]+\s+[A-Z][A-Za-z\.'-]+).*(?:return)", txt_raw)
                                    if m3:
                                        names.add(m3.group(1))
                        except Exception:
                            pass
            except Exception:
                continue
        return sorted(list(names))

    def _fetch_nfl_liveupdate_td_scorers(dates: Optional[list[str]] = None) -> list[str]:
        """Fallback: gather TD scorers from NFL legacy liveupdate (scorestrip + game-center).

        Args:
            dates: list of YYYYMMDD strings (UTC) to filter; if None uses today,yesterday,2 days ago.
        Returns:
            Sorted list of scorer names (best-effort, may be partial or empty).
        """
        try:
            import requests, re, datetime as _dt
        except Exception:
            return []
        if dates is None:
            today = _dt.datetime.utcnow().date()
            dates = [(today - _dt.timedelta(days=i)).strftime('%Y%m%d') for i in range(0,3)]
        # Fetch scorestrip from multiple known endpoints, aggregate games
        games: set[str] = set()
        endpoints = [
            "https://www.nfl.com/liveupdate/scorestrip/scorestrip.json",
            "https://www.nfl.com/liveupdate/scorestrip/ss.json",
        ]
        for ep in endpoints:
            try:
                r = requests.get(ep, timeout=10)
                if r.status_code != 200:
                    continue
                js = r.json()
                # Known containers: 'ss', 'scores', 'gms', 'games'
                cands = []
                for key in ['ss','scores','gms','games']:
                    arr = js.get(key)
                    if isinstance(arr, list) and arr:
                        cands = arr
                        break
                if not cands:
                    continue
                for g in cands:
                    try:
                        # Accept list or dict forms
                        if isinstance(g, list):
                            eid = str(g[0]) if g else None
                        elif isinstance(g, dict):
                            eid = str(g.get('eid') or g.get('game_id') or g.get('id') or '')
                        else:
                            eid = None
                        if not eid or len(eid) < 8:
                            continue
                        if any(eid.startswith(d) for d in dates):
                            games.add(eid)
                    except Exception:
                        continue
            except Exception:
                continue
        if not games:
            return []
        names: set[str] = set()
        for eid in sorted(games):
            try:
                gc_url_candidates = [
                    f"https://www.nfl.com/liveupdate/game-center/{eid}/{eid}_gtd.json",
                    f"https://www.nfl.com/liveupdate/game-center/{eid}/{eid}.json",
                ]
                gj = None
                for u in gc_url_candidates:
                    try:
                        gr = requests.get(u, timeout=10)
                    except Exception:
                        continue
                    if gr.status_code == 200:
                        try:
                            gj = gr.json()
                        except Exception:
                            gj = None
                    if gj:
                        break
                if not gj:
                    continue
                game_obj = gj.get(eid) or gj  # some variants root at eid, others direct
                drives = game_obj.get('drives') or {}
                for dv in drives.values():
                    if not isinstance(dv, dict):
                        continue
                    plays = dv.get('plays') or {}
                    for pv in plays.values():
                        if not isinstance(pv, dict):
                            continue
                        desc = str(pv.get('desc') or '')
                        if 'touchdown' not in desc.lower():
                            continue
                        # Extract name heuristically: first capitalized two-word sequence
                        m = re.search(r"([A-Z][A-Za-z\.'-]+\s+[A-Z][A-Za-z\.'-]+)", desc)
                        if m:
                            names.add(m.group(1))
            except Exception:
                continue
        return sorted(names)

    # Background polling thread (env-gated) to keep live TD file warm without admin page reloads
    _LIVE_BG_THREAD_STARTED = False
    def _maybe_start_live_thread():
        """Start background live TD polling if ENABLE_LIVE_THREAD env is truthy.

        Safeguards:
        - Only starts once per process.
        - Creates a lock file inside DATA_DIR to reduce duplicate starts across multi-worker setups.
        - Lock file is best-effort; if creation fails (already exists), we skip.
        """
        nonlocal _LIVE_BG_THREAD_STARTED
        if _LIVE_BG_THREAD_STARTED:
            return
        if os.environ.get("ENABLE_LIVE_THREAD", "").lower() not in {"1", "true", "yes"}:
            return
        lock_path = DATA_DIR / "live_thread.lock"
        try:
            if lock_path.exists():
                # Another worker likely started it; mark as started logically for health reporting
                _LIVE_BG_THREAD_STARTED = True
                return
            # Attempt atomic creation
            with open(lock_path, "x", encoding="utf-8") as _lk:
                _lk.write(str(os.getpid()))
        except Exception:
            # Can't create lock: assume already running elsewhere
            _LIVE_BG_THREAD_STARTED = True
            return

        import threading, time
        def _loop():
            while True:
                try:
                    if _is_game_window_now_et():
                        try:
                            s = int(os.environ.get("TD_LIVE_SEASON", "2025"))
                        except Exception:
                            s = 2025
                        try:
                            w = int(os.environ.get("TD_LIVE_WEEK", "1"))
                        except Exception:
                            w = 1
                        existing = _load_live_td_scorers(s, w)
                        fresh = _fetch_espn_td_scorers(s, w, sweep=False)
                        if fresh:
                            new = [n for n in fresh if n.lower() not in existing]
                            merged = sorted(list({*(n for n in existing), *fresh}))
                            _save_live_td_scorers(s, w, merged, meta={"bg_thread": True})
                            if new:
                                _append_live_td_events(s, w, new)
                        sleep_for = int(os.environ.get("LIVE_THREAD_ACTIVE_SLEEP", "15"))
                    else:
                        sleep_for = int(os.environ.get("LIVE_THREAD_IDLE_SLEEP", "60"))
                except Exception:
                    sleep_for = 60
                time.sleep(max(5, sleep_for))
        try:
            t = threading.Thread(target=_loop, name="live-td-bg", daemon=True)
            t.start()
            _LIVE_BG_THREAD_STARTED = True
        except Exception:
            # Cleanup lock on failure
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except Exception:
                pass
    # Attempt to start background thread (non-blocking, env-gated)
    try:
        _maybe_start_live_thread()
    except Exception:
        pass

    # Warm 2024 team distributions cache at startup (best-effort)
    try:
        _get_team_2024_distributions(force_rebuild=False)
    except Exception:
        pass

    # 2024 team TD distribution caches
    _TEAM_TD24_KIND: Optional[dict[str, dict[str, float]]] = None  # team -> {pass, rush} shares
    _TEAM_TD24_KIND_COUNTS: Optional[dict[str, dict[str, int]]] = None  # team -> {pass, rush} counts
    _TEAM_TD24_POS: Optional[dict[str, dict[str, float]]] = None   # team -> {RB,WR,TE,QB} shares
    _TEAM_TD24_POS_COUNTS: Optional[dict[str, dict[str, int]]] = None  # team -> {RB,WR,TE,QB} counts
    def _get_team_2024_distributions(force_rebuild: bool = False) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        """Compute team-level 2024 TD distributions.

        - Kind shares: pass vs rush
        - Position shares: WR/TE/RB/QB, mapped via seasonal_rosters positions
        Results are normalized to 1.0 when possible.
        """
        nonlocal _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
        # Return cached only if all present and non-empty (avoid sticky empty cache from earlier failures)
        if (not force_rebuild) and (
            isinstance(_TEAM_TD24_KIND, dict) and _TEAM_TD24_KIND and
            isinstance(_TEAM_TD24_POS, dict) and _TEAM_TD24_POS and
            isinstance(_TEAM_TD24_POS_COUNTS, dict) and _TEAM_TD24_POS_COUNTS and
            isinstance(_TEAM_TD24_KIND_COUNTS, dict) and _TEAM_TD24_KIND_COUNTS
        ):
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
        # Try loading from cached CSVs first (local or sibling NFL-Betting repo)
        try:
            ks_fp = DATA_DIR / "team_td_2024_kind_shares.csv"
            kc_fp = DATA_DIR / "team_td_2024_kind_counts.csv"
            ps_fp = DATA_DIR / "team_td_2024_pos_shares.csv"
            pc_fp = DATA_DIR / "team_td_2024_pos_counts.csv"
            # sibling path fallback
            if not (ks_fp.exists() and kc_fp.exists() and ps_fp.exists() and pc_fp.exists()):
                sib = BASE_DIR.parent / "NFL-Betting" / "nfl_compare" / "data"
                ks2 = sib / "team_td_2024_kind_shares.csv"
                kc2 = sib / "team_td_2024_kind_counts.csv"
                ps2 = sib / "team_td_2024_pos_shares.csv"
                pc2 = sib / "team_td_2024_pos_counts.csv"
                if ks2.exists() and kc2.exists() and ps2.exists() and pc2.exists():
                    ks_fp, kc_fp, ps_fp, pc_fp = ks2, kc2, ps2, pc2
            if (not force_rebuild) and ks_fp.exists() and kc_fp.exists() and ps_fp.exists() and pc_fp.exists():
                ksd = pd.read_csv(ks_fp)
                kcd = pd.read_csv(kc_fp)
                psd = pd.read_csv(ps_fp)
                pcd = pd.read_csv(pc_fp)
                kind_out = {}
                kind_counts_out = {}
                pos_out = {}
                pos_counts_out = {}
                for _, rr in ksd.iterrows():
                    tm = str(rr.get("team") or "").strip()
                    if tm:
                        kind_out[tm] = {"pass": float(rr.get("pass", 0.0)), "rush": float(rr.get("rush", 0.0))}
                for _, rr in kcd.iterrows():
                    tm = str(rr.get("team") or "").strip()
                    if tm:
                        kind_counts_out[tm] = {"pass": int(rr.get("pass", 0) or 0), "rush": int(rr.get("rush", 0) or 0)}
                for _, rr in psd.iterrows():
                    tm = str(rr.get("team") or "").strip()
                    if tm:
                        pos_out[tm] = {k: float(rr.get(k, 0.0) or 0.0) for k in ["RB","WR","TE","QB"]}
                for _, rr in pcd.iterrows():
                    tm = str(rr.get("team") or "").strip()
                    if tm:
                        pos_counts_out[tm] = {k: int(rr.get(k, 0) or 0) for k in ["RB","WR","TE","QB"]}
                if kind_out and pos_out and pos_counts_out and kind_counts_out:
                    _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS = kind_out, pos_out, pos_counts_out, kind_counts_out
                    return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
        except Exception:
            pass
        kind: dict[str, dict[str, int]] = {}
        pos: dict[str, dict[str, int]] = {}
        # Load PBP 2024
        pbp_fp = DATA_DIR / "pbp_2024.csv"
        if not pbp_fp.exists():
            _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS = {}, {}, {}, {}
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
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
                _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS = {}, {}, {}, {}
                return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
        if pbp is None or pbp.empty:
            _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS = {}, {}, {}, {}
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
        # Identify team column
        team_col = None
        for c in ["posteam", "offense_team", "team"]:
            if c in pbp.columns:
                team_col = c
                break
        if team_col is None:
            _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS = {}, {}, {}, {}
            return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS
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
        # Build alias map to handle pbp name styles like "A.Brown" vs roster "A.J. Brown"
        pos_map_alias: dict[str, str] = dict(pos_map)
        def _add_alias(key: str, val: str):
            k = (key or "").strip().lower()
            if k and k not in pos_map_alias:
                pos_map_alias[k] = val
        for nm, p in list(pos_map.items()):
            raw = nm
            nm2 = raw.replace(".", "").replace(",", "").replace("-", " ").strip()
            parts = [t for t in nm2.split() if t]
            if parts:
                last = parts[-1]
                initials = "".join([t[0] for t in parts[:-1]]) if len(parts) > 1 else (parts[0][0] if parts[0] else "")
                # Common alias variants
                variants = set()
                if initials:
                    variants.add(f"{initials}.{last}".lower())
                    variants.add(f"{initials}{last}".lower())
                    variants.add(f"{initials} {last}".lower())
                variants.add(f"{parts[0][0]}.{last}".lower())
                variants.add(f"{parts[0][0]}{last}".lower())
                variants.add(f"{parts[0][0]} {last}".lower())
                variants.add(f"{parts[0]} {last}".lower())
                variants.add(f"{parts[0]}{last}".lower())
                variants.add(f"{raw}".lower())
                variants.add(raw.replace(" ", "").lower())
                for v in variants:
                    _add_alias(v, p)
        # Iterate plays and build counts
        for _, r in pbp.iterrows():
            tm = str(r.get(team_col) or "").strip()
            if not tm:
                continue
            ab_tm = _team_std_abbr(tm) or _team_abbrev(tm) or tm
            # kind shares
            if int(r.get("rush_touchdown") or 0) == 1:
                kind.setdefault(ab_tm, {"pass": 0, "rush": 0})["rush"] += 1
                n = str(r.get("rusher_player_name") or "").strip().lower()
                p = pos_map_alias.get(n) or pos_map_alias.get(n.replace(".", "").replace(" ", ""))
                if p not in {"RB", "WR", "TE", "QB"}:
                    # Default rusher to RB when unknown
                    p = "RB"
                pos.setdefault(ab_tm, {}).setdefault(p, 0)
                pos[ab_tm][p] += 1
            if int(r.get("pass_touchdown") or 0) == 1:
                kind.setdefault(ab_tm, {"pass": 0, "rush": 0})["pass"] += 1
                n = str(r.get("receiver_player_name") or "").strip().lower()
                p = pos_map_alias.get(n) or pos_map_alias.get(n.replace(".", "").replace(" ", ""))
                if p not in {"RB", "WR", "TE", "QB"}:
                    # Default receiver to WR when unknown
                    p = "WR"
                pos.setdefault(ab_tm, {}).setdefault(p, 0)
                pos[ab_tm][p] += 1
        # Normalize to shares
        kind_out: dict[str, dict[str, float]] = {}
        kind_counts_out: dict[str, dict[str, int]] = {}
        pos_out: dict[str, dict[str, float]] = {}
        pos_counts_out: dict[str, dict[str, int]] = {}
        for tm, kv in kind.items():
            tot = float(kv.get("pass", 0) + kv.get("rush", 0))
            if tot > 0:
                kind_out[tm] = {"pass": kv.get("pass", 0) / tot, "rush": kv.get("rush", 0) / tot}
            kind_counts_out[tm] = {"pass": int(kv.get("pass", 0)), "rush": int(kv.get("rush", 0))}
        for tm, kv in pos.items():
            tot = float(sum(kv.values()))
            if tot > 0:
                pos_out[tm] = {k: v / tot for k, v in kv.items() if k in {"RB", "WR", "TE", "QB"}}
                pos_counts_out[tm] = {k: int(v) for k, v in kv.items() if k in {"RB", "WR", "TE", "QB"}}
            # ensure all keys present
            if tm not in pos_out:
                pos_out[tm] = {}
            if tm not in pos_counts_out:
                pos_counts_out[tm] = {}
            for k in ["RB", "WR", "TE", "QB"]:
                pos_out[tm][k] = float(pos_out[tm].get(k, 0.0))
                pos_counts_out[tm][k] = int(pos_counts_out[tm].get(k, 0))
        _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS = kind_out, pos_out, pos_counts_out, kind_counts_out
        # Persist to CSVs for faster future loads
        try:
            ks_fp = DATA_DIR / "team_td_2024_kind_shares.csv"
            kc_fp = DATA_DIR / "team_td_2024_kind_counts.csv"
            ps_fp = DATA_DIR / "team_td_2024_pos_shares.csv"
            pc_fp = DATA_DIR / "team_td_2024_pos_counts.csv"
            pd.DataFrame([
                {"team": tm, "pass": v.get("pass", 0.0), "rush": v.get("rush", 0.0)} for tm, v in _TEAM_TD24_KIND.items()
            ]).to_csv(ks_fp, index=False)
            pd.DataFrame([
                {"team": tm, "pass": v.get("pass", 0), "rush": v.get("rush", 0)} for tm, v in _TEAM_TD24_KIND_COUNTS.items()
            ]).to_csv(kc_fp, index=False)
            pd.DataFrame([
                {"team": tm, **{k: _TEAM_TD24_POS[tm].get(k, 0.0) for k in ["RB","WR","TE","QB"]}} for tm in _TEAM_TD24_POS.keys()
            ]).to_csv(ps_fp, index=False)
            pd.DataFrame([
                {"team": tm, **{k: _TEAM_TD24_POS_COUNTS[tm].get(k, 0) for k in ["RB","WR","TE","QB"]}} for tm in _TEAM_TD24_POS_COUNTS.keys()
            ]).to_csv(pc_fp, index=False)
        except Exception:
            pass
        return _TEAM_TD24_KIND, _TEAM_TD24_POS, _TEAM_TD24_POS_COUNTS, _TEAM_TD24_KIND_COUNTS

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

    # Root: redirect to UI
    @app.route("/")
    def root():
        from flask import redirect, url_for
        return redirect(url_for("ui_home"))

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
        # Load Bovada ATD snapshot (optional)
        bovada_map: dict[tuple[str,str], dict] = {}
        try:
            bovada_fp = DATA_DIR / f"bovada_atd_{season}_wk{week}.csv"
            if bovada_fp.exists():
                bdf = pd.read_csv(bovada_fp)
                # normalize
                for _, rr in bdf.iterrows():
                    player = str(rr.get("player") or '').strip()
                    team = str(rr.get("team") or '').strip()
                    if not player:
                        continue
                    key = (player.lower(), team.upper())
                    bovada_map[key] = {
                        "american": None if pd.isna(rr.get("american_odds")) else int(rr.get("american_odds")),
                        "implied_prob": None if pd.isna(rr.get("implied_prob")) else float(rr.get("implied_prob")),
                        "decimal": None if pd.isna(rr.get("decimal_odds")) else float(rr.get("decimal_odds")),
                    }
        except Exception:
            bovada_map = {}
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
        # Load live TD scorers for badge
        live_names = _load_live_td_scorers(season, week)
        # If empty or stale, try on-demand fetch as a fallback (not strictly gated by window)
        stale = _live_td_age_secs(season, week)
        if ((not live_names) or (stale is None) or (stale > 15)):
            try:
                fetched = _fetch_espn_td_scorers(season, week, sweep=False)
                if fetched:
                    live_names = {str(n).strip().lower() for n in fetched if str(n).strip()}
                    _save_live_td_scorers(season, week, list(live_names), meta={"on_demand": True})
                else:
                    # Ensure we don't loop: keep any existing names
                    live_names = _load_live_td_scorers(season, week)
            except Exception:
                pass
        # Build a variant key set for robust matching (e.g., Ken/Kenneth, particles, hyphens)
        live_keys: set[str] = set()
        for nm in list(live_names):
            for key in _name_key_variants(nm):
                live_keys.add(key)
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
            # Bovada odds lookup
            # Original join used (full team name).upper(), but snapshot stores true abbreviations (KC, LAC, etc.).
            # Normalize to a standard abbreviation and try several variants plus player-only fallback.
            bovada_rec = None
            t_up_full = (team or '').upper()
            # Standard abbreviation from helper / hard map
            try:
                team_abbr_std = _team_std_abbr(team)  # type: ignore
            except Exception:
                team_abbr_std = None
            # Candidate team tokens in priority order
            cand_tokens: list[str] = []
            if team_abbr_std:
                cand_tokens.append(team_abbr_std.upper())
                # Handle Rams edge case where some feeds use LA vs LAR
                if team_abbr_std.upper() == 'LAR':
                    cand_tokens.append('LA')
            # Add raw uppercase full token (may coincidentally match if snapshot used it)
            if t_up_full and t_up_full not in cand_tokens:
                cand_tokens.append(t_up_full)
            # Always finish with empty token (player-only key)
            cand_tokens.append('')
            pl_key = p.lower()
            for tk in cand_tokens:
                key = (pl_key, tk)
                if key in bovada_map:
                    bovada_rec = bovada_map[key]
                    break
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
                # opponent visuals
                "opp_logo_url": _logo_for_team(opp) if opp else None,
                "opp_abbr": _team_abbrev(opp) if opp else None,
                "game_id": game_id,
                "td_scored": any((k in live_keys) for k in _name_key_variants(str(p or ""))),
                "bovada_american": None if not bovada_rec else bovada_rec.get("american"),
                "bovada_implied": None if not bovada_rec else bovada_rec.get("implied_prob"),
                "bovada_decimal": None if not bovada_rec else bovada_rec.get("decimal"),
                # difference between model ATD and market implied probability (pct points)
                "atd_edge_pct": None if (not bovada_rec or bovada_rec.get("implied_prob") is None) else round((adj_atd - bovada_rec.get("implied_prob")) * 100, 1),
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
        # Include last fetch time if present
        last_fetch_iso = None
        try:
            lf = _live_td_file(season, week)
            if lf.exists():
                with open(lf, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                meta_block = obj.get("meta") or {}
                # Attempt known keys
                last_fetch_iso = meta_block.get("refreshed_at") or meta_block.get("fetched_at") or meta_block.get("updated_at")
        except Exception:
            last_fetch_iso = None
        live_meta = {"count": len(live_keys), "age": stale, "last": last_fetch_iso}
        return render_template("ui.html", tab="players", cards=cards, season=season, week=week, pos=pos_filter, sort=sort_by, team=team_filter, game=game_filter, teams=teams, games=games, env=os.environ, live_meta=live_meta)

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
        # Ensure required columns and adjusted values
        df = df.copy()
        if "expected_tds" not in df.columns:
            df["expected_tds"] = 0.0
        if "opponent" not in df.columns:
            df["opponent"] = None
        df["__adj_expected_tds"] = df.apply(lambda r: _adj_team_tds(str(r.get("team")), str(r.get("opponent")) if r.get("opponent") is not None else None, float(r.get("expected_tds") or 0.0)), axis=1)
        # Load team position shares (last N seasons); if missing, fallback later to 2024 shares
        tps_files = sorted(DATA_DIR.glob("team_pos_td_shares_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        tps = pd.read_csv(tps_files[0]) if tps_files else pd.DataFrame()
        def get_pos_shares(team: str, kind: str) -> dict[str, float]:
            if tps is None or tps.empty:
                return {}
            tt = tps.copy()
            tt["team"] = tt["team"].astype(str)
            tt["kind"] = tt["kind"].astype(str)
            m = tt[(tt["team"] == team) & (tt["kind"] == kind)]
            if m.empty:
                ab = _team_std_abbr(team) or _team_abbrev(team)
                m = tt[(tt["team"] == ab) & (tt["kind"] == kind)]
            if m.empty:
                return {}
            row = m.iloc[0]
            keys = ["WR","TE","RB","QB"]
            out = {k: float(row.get(k) or 0.0) for k in keys}
            s = sum(out.values())
            if s > 0:
                out = {k: v/s for k, v in out.items()}
            return out
        # Default pass/rush split when not present
        PASS_FRAC = 0.58
        RUSH_FRAC = 1.0 - PASS_FRAC
        # Default position shares when team-specific data missing
        DEFAULT_PASS_SH = {"WR": 0.65, "TE": 0.2, "RB": 0.15, "QB": 0.0}
        DEFAULT_RUSH_SH = {"RB": 0.85, "QB": 0.15, "WR": 0.0, "TE": 0.0}
        # 2024 historical distributions
        kind24, pos24, pos24_counts, kind24_counts = _get_team_2024_distributions()
        if not kind24:
            kind24 = {}
        if not pos24:
            pos24 = {}
        if not pos24_counts:
            pos24_counts = {}
        if not kind24_counts:
            kind24_counts = {}
        # Load market lines for the week (spread/total/moneyline)
        lines_map: dict[str, dict] = {}
        real_lines_map: dict[tuple[str, str], dict] = {}
        try:
            # Primary: CSV pre-joined by game_id
            lines_fp = DATA_DIR / "lines.csv"
            if lines_fp.exists():
                ldf = pd.read_csv(lines_fp)
                if "game_id" in ldf.columns:
                    for _, lr in ldf.iterrows():
                        gid = str(lr.get("game_id") or "")
                        if gid:
                            lines_map[gid] = {k: lr.get(k) for k in lr.index}
            # Fallback: parse real_betting_lines_*.json and key by (home_full, away_full)
            json_files = sorted(DATA_DIR.glob("real_betting_lines_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                except Exception:
                    continue
                lines_obj = obj.get("lines") if isinstance(obj, dict) else None
                if not isinstance(lines_obj, dict):
                    continue
                for matchup, payload in lines_obj.items():
                    # Expect format: "Away Team @ Home Team"
                    try:
                        if " @ " not in matchup:
                            continue
                        away_name, home_name = matchup.split(" @ ", 1)
                        home = str(home_name).strip()
                        away = str(away_name).strip()
                        markets = payload.get("markets") or []
                        total_val = None
                        spread_home = None
                        money_home = None
                        money_away = None
                        # Try direct fields
                        ml = payload.get("moneyline") or {}
                        if isinstance(ml, dict):
                            money_home = ml.get("home")
                            money_away = ml.get("away")
                        tr = payload.get("total_runs") or {}
                        if isinstance(tr, dict):
                            total_val = tr.get("line")
                        rl = payload.get("run_line") or {}
                        if isinstance(rl, dict):
                            spread_home = rl.get("home")
                        # Parse standard markets if present
                        if markets and isinstance(markets, list):
                            for mkt in markets:
                                key = str(mkt.get("key") or "").lower()
                                outs = mkt.get("outcomes") or []
                                if key == "totals" and outs:
                                    # take first point
                                    try:
                                        total_val = outs[0].get("point", total_val)
                                    except Exception:
                                        pass
                                if key == "spreads" and outs:
                                    # find home team record for spread
                                    try:
                                        for o in outs:
                                            nm = str(o.get("name") or "").strip()
                                            if nm == home:
                                                spread_home = o.get("point", spread_home)
                                                break
                                    except Exception:
                                        pass
                                if key == "h2h" and outs:
                                    try:
                                        for o in outs:
                                            nm = str(o.get("name") or "").strip()
                                            if nm == home:
                                                money_home = o.get("price", money_home)
                                            elif nm == away:
                                                money_away = o.get("price", money_away)
                                    except Exception:
                                        pass
                        # Save latest per-matchup
                        real_lines_map[(home, away)] = {
                            "close_total": total_val,
                            "close_spread_home": spread_home,
                            "moneyline_home": money_home,
                            "moneyline_away": money_away,
                        }
                    except Exception:
                        continue
        except Exception:
            pass
        # Group by game for game sum
        game_sum = df.groupby("game_id")["__adj_expected_tds"].sum().to_dict()
        cards: list[dict] = []
        for _, r in df.iterrows():
            team = str(r.get("team"))
            opp = str(r.get("opponent"))
            exp_tds = float(r.get("__adj_expected_tds") or 0.0)
            gid = r.get("game_id")
            logo_url = _logo_for_team(team)
            # position projections
            pass_sh = get_pos_shares(team, "pass")
            rush_sh = get_pos_shares(team, "rush")
            local_pass_frac = PASS_FRAC
            local_rush_frac = RUSH_FRAC
            if (not pass_sh or not rush_sh):
                ab = _team_std_abbr(team) or _team_abbrev(team)
                if ab in pos24:
                    k = kind24.get(ab, {})
                    local_pass_frac = float(k.get("pass", PASS_FRAC))
                    local_rush_frac = float(k.get("rush", RUSH_FRAC))
                    pshares = pos24.get(ab, {})
                    pass_sh = {x: float(pshares.get(x, 0.0)) for x in ["WR","TE","RB","QB"]}
                    rush_sh = {x: float(pshares.get(x, 0.0)) for x in ["RB","QB","WR","TE"]}
                    sp = sum(pass_sh.values()) or 1.0
                    pass_sh = {k2: v2/sp for k2, v2 in pass_sh.items()}
                    sr = sum(rush_sh.values()) or 1.0
                    rush_sh = {k2: v2/sr for k2, v2 in rush_sh.items()}
            # Absolute fallback if still missing
            if not pass_sh:
                pass_sh = DEFAULT_PASS_SH.copy()
            if not rush_sh:
                rush_sh = DEFAULT_RUSH_SH.copy()
            # expected by position
            by_pos = {"RB":0.0,"WR":0.0,"TE":0.0,"QB":0.0}
            for pos in ["WR","TE","RB"]:
                by_pos[pos] += local_pass_frac * exp_tds * float(pass_sh.get(pos, 0.0))
            for pos in ["RB","QB","WR","TE"]:
                by_pos[pos] += local_rush_frac * exp_tds * float(rush_sh.get(pos, 0.0))
            for k in by_pos:
                by_pos[k] = round(by_pos[k], 2)
            proj_kind = {
                "pass": round(local_pass_frac, 2),
                "rush": round(local_rush_frac, 2),
            }
            # 2024 historical shares/counts using standard abbr
            _ab = _team_std_abbr(team) or _team_abbrev(team)
            hist_kind = None
            hist_kind_counts = None
            if _ab in kind24:
                hist_kind = {
                    "pass": round(float(kind24[_ab].get("pass", 0.0)), 2),
                    "rush": round(float(kind24[_ab].get("rush", 0.0)), 2),
                }
            else:
                # If we lack 2024 kind shares, mirror current projection as a placeholder
                hist_kind = {"pass": round(float(proj_kind["pass"]), 2), "rush": round(float(proj_kind["rush"]), 2)}
            if _ab in kind24_counts:
                hist_kind_counts = {
                    "pass": int(kind24_counts[_ab].get("pass", 0)),
                    "rush": int(kind24_counts[_ab].get("rush", 0)),
                }
            hist_pos = None
            hist_pos_counts = None
            if _ab in pos24:
                hist_pos = {k: round(float(pos24[_ab].get(k, 0.0)), 2) for k in ["RB","WR","TE","QB"]}
            if _ab in pos24_counts:
                hist_pos_counts = {k: int(pos24_counts[_ab].get(k, 0)) for k in ["RB","WR","TE","QB"]}
            # Ensure team_tds is non-zero if base is present but adjustment failed
            team_tds_val = round(exp_tds if exp_tds and exp_tds > 0 else float(r.get("expected_tds") or 0.0), 2)
            # Market lines
            mkt = None
            try:
                row = lines_map.get(gid)
                if row:
                    # prefer close_ fields, fallback to current
                    total = row.get("close_total")
                    if pd.isna(total) if isinstance(total, float) else total is None:
                        total = row.get("total")
                    spr_home = row.get("close_spread_home")
                    if pd.isna(spr_home) if isinstance(spr_home, float) else spr_home is None:
                        spr_home = row.get("spread_home")
                    # coerce to float
                    try:
                        total_f = float(total) if total is not None else None
                    except Exception:
                        total_f = None
                    try:
                        spr_home_f = float(spr_home) if spr_home is not None else None
                    except Exception:
                        spr_home_f = None
                    is_home = bool(int(r.get("is_home") or 0) == 1)
                    # team spread from home perspective
                    team_spread = None
                    if spr_home_f is not None:
                        team_spread = spr_home_f if is_home else -spr_home_f
                    # moneyline
                    ml_home = row.get("moneyline_home")
                    ml_away = row.get("moneyline_away")
                    try:
                        ml = int(ml_home) if is_home else int(ml_away)
                    except Exception:
                        try:
                            ml = int(float(ml_home)) if is_home else int(float(ml_away))
                        except Exception:
                            ml = None
                    # implied points
                    imp_pts = None
                    if total_f is not None and spr_home_f is not None:
                        if is_home:
                            imp_pts = total_f/2 - spr_home_f/2
                        else:
                            imp_pts = total_f/2 + spr_home_f/2
                    mkt = {
                        "total": total_f,
                        "spread": team_spread,
                        "moneyline": ml,
                        "implied_pts": None if imp_pts is None else round(float(imp_pts), 1),
                    }
                else:
                    # Fallback: match from real_lines_map using home/away names
                    is_home = bool(int(r.get("is_home") or 0) == 1)
                    home_name = team if is_home else opp
                    away_name = opp if is_home else team
                    row2 = real_lines_map.get((home_name, away_name))
                    if row2:
                        total = row2.get("close_total")
                        spr_home = row2.get("close_spread_home")
                        try:
                            total_f = float(total) if total is not None else None
                        except Exception:
                            total_f = None
                        try:
                            spr_home_f = float(spr_home) if spr_home is not None else None
                        except Exception:
                            spr_home_f = None
                        team_spread = None
                        if spr_home_f is not None:
                            team_spread = spr_home_f if is_home else -spr_home_f
                        ml_home = row2.get("moneyline_home")
                        ml_away = row2.get("moneyline_away")
                        try:
                            ml = int(ml_home) if is_home else int(ml_away)
                        except Exception:
                            try:
                                ml = int(float(ml_home)) if is_home else int(float(ml_away))
                            except Exception:
                                ml = None
                        imp_pts = None
                        if total_f is not None and spr_home_f is not None:
                            if is_home:
                                imp_pts = total_f/2 - spr_home_f/2
                            else:
                                imp_pts = total_f/2 + spr_home_f/2
                        mkt = {
                            "total": total_f,
                            "spread": team_spread,
                            "moneyline": ml,
                            "implied_pts": None if imp_pts is None else round(float(imp_pts), 1),
                        }
            except Exception:
                mkt = None
            card = {
                "team": team,
                "opp": opp,
                "team_abbr": _team_abbrev(team),
                "team_class": _team_avatar_class(team),
                "logo_url": logo_url,
                "game_tds": round(float(game_sum.get(gid, 0.0)), 2),
                "team_tds": team_tds_val,
                "by_pos": by_pos,
                "proj_kind": proj_kind,
                "hist_kind": hist_kind,
                "hist_pos": hist_pos,
                "hist_pos_counts": hist_pos_counts,
                "hist_kind_counts": hist_kind_counts,
                "market": mkt,
                "game_id": gid,
                "date": r.get("date"),
                "is_home": int(r.get("is_home") or 0) == 1,
            }
            cards.append(card)
        cards.sort(key=lambda x: x["team_tds"], reverse=True)
        return render_template("ui.html", tab="teams", cards=cards, season=season, week=week, env=os.environ)

    @app.route("/admin/rebuild-2024", methods=["POST", "GET"])
    def admin_rebuild_2024():
        # Simple admin gate by env var
        if not (os.environ.get("TD_ADMIN", "").lower() in ("1","true","yes")):
            return ("forbidden", 403)
        try:
            _get_team_2024_distributions(force_rebuild=True)
            return ("ok", 200)
        except Exception as e:
            return (f"error: {e}", 500)

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
