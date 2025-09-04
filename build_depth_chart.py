from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from typing import Optional
import re

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _map_pos(p: object) -> str:
    p = str(p or "").upper()
    if p in {"HB", "FB"}:
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


def load_injuries(season: int) -> pd.DataFrame | None:
    try:
        import nfl_data_py as nfl  # type: ignore
        try:
            inj = nfl.import_injury_reports([season])
        except TypeError:
            inj = nfl.import_injury_reports(seasons=[season])
        return inj
    except Exception:
        return None


def load_depth_charts(season: int) -> pd.DataFrame | None:
    try:
        import nfl_data_py as nfl  # type: ignore
        try:
            dc = nfl.import_depth_charts([season])
        except TypeError:
            dc = nfl.import_depth_charts(seasons=[season])
        return dc
    except Exception:
        return None

TEAM_SLUGS = {
    "Arizona Cardinals": "arizona-cardinals",
    "Atlanta Falcons": "atlanta-falcons",
    "Baltimore Ravens": "baltimore-ravens",
    "Buffalo Bills": "buffalo-bills",
    "Carolina Panthers": "carolina-panthers",
    "Chicago Bears": "chicago-bears",
    "Cincinnati Bengals": "cincinnati-bengals",
    "Cleveland Browns": "cleveland-browns",
    "Dallas Cowboys": "dallas-cowboys",
    "Denver Broncos": "denver-broncos",
    "Detroit Lions": "detroit-lions",
    "Green Bay Packers": "green-bay-packers",
    "Houston Texans": "houston-texans",
    "Indianapolis Colts": "indianapolis-colts",
    "Jacksonville Jaguars": "jacksonville-jaguars",
    "Kansas City Chiefs": "kansas-city-chiefs",
    "Las Vegas Raiders": "las-vegas-raiders",
    "Los Angeles Chargers": "los-angeles-chargers",
    "Los Angeles Rams": "los-angeles-rams",
    "Miami Dolphins": "miami-dolphins",
    "Minnesota Vikings": "minnesota-vikings",
    "New England Patriots": "new-england-patriots",
    "New Orleans Saints": "new-orleans-saints",
    "New York Giants": "new-york-giants",
    "New York Jets": "new-york-jets",
    "Philadelphia Eagles": "philadelphia-eagles",
    "Pittsburgh Steelers": "pittsburgh-steelers",
    "San Francisco 49ers": "san-francisco-49ers",
    "Seattle Seahawks": "seattle-seahawks",
    "Tampa Bay Buccaneers": "tampa-bay-buccaneers",
    "Tennessee Titans": "tennessee-titans",
    "Washington Commanders": "washington-commanders",
}


# ESPN team abbreviations used in depth chart URLs
ESPN_TEAM_ABBR = {
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "lv",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "wsh",
}


def _clean_name(txt: str) -> str:
    # Remove injury suffixes (Q, D, O, IR, PUP, NFI, SUSP), asterisks, and extra spaces
    t = (txt or "").strip()
    # Common markers are separated by space
    t = re.sub(r"\s+(Q|D|O|IR|PUP|NFI|SUSP|DNP)$", "", t, flags=re.IGNORECASE)
    t = t.replace("*", "").strip()
    return t


def _scrape_espn_depth_for_team(team_name: str) -> dict[str, list[str]]:
    """Return pos -> ordered list scraped from ESPN using formation + table mapping."""
    import requests
    from bs4 import BeautifulSoup
    abbr = ESPN_TEAM_ABBR.get(team_name)
    if not abbr:
        return {}
    url = f"https://www.espn.com/nfl/team/depth/_/name/{abbr}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception:
        return {}

    # Parse formation text (e.g., "3WR 1TE" or "2RB 2WR 1TE") near the offense section
    soup = BeautifulSoup(resp.text, "lxml")
    formation_text = ""
    try:
        # search for a node containing both WR and TE tokens
        txt_nodes = soup.find_all(string=True)
        cand = [t.strip() for t in txt_nodes if t and "WR" in t and ("TE" in t or "RB" in t)]
        # prefer the shortest such token (e.g., "3WR 1TE")
        cand = sorted(cand, key=lambda s: len(s))
        if cand:
            formation_text = cand[0]
    except Exception:
        formation_text = ""
    # default to 3WR 1TE 1RB if unknown
    wr_count = 3
    te_count = 1
    rb_count = 1
    if formation_text:
        wr_m = re.search(r"(\d+)\s*WR", formation_text, flags=re.IGNORECASE)
        te_m = re.search(r"(\d+)\s*TE", formation_text, flags=re.IGNORECASE)
        rb_m = re.search(r"(\d+)\s*RB", formation_text, flags=re.IGNORECASE)
        if wr_m:
            wr_count = max(1, int(wr_m.group(1)))
        if te_m:
            te_count = max(1, int(te_m.group(1)))
        if rb_m:
            rb_count = max(1, int(rb_m.group(1)))

    # Extract the offense table with tiers via pandas
    try:
        tables = pd.read_html(resp.text)
    except Exception:
        return {}
    # pick first table that has Starter/2nd columns
    df = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any(c in {"Starter", "2nd"} for c in cols):
            df = t.copy()
            break
    if df is None:
        return {}
    # Ensure 4 tier columns exist
    # If exactly 4 columns, assume they are tiers without POS
    if df.shape[1] >= 4:
        df = df.iloc[:, :4].copy()
        df.columns = ["Starter", "2nd", "3rd", "4th"]
    else:
        # Not expected; abort
        return {}
    # Clean names
    for c in ["Starter", "2nd", "3rd", "4th"]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"-": ""})
        df[c] = df[c].apply(_clean_name)
    # Determine how many skill rows from formation
    skill_rows = 1 + rb_count + wr_count + te_count
    skill_rows = min(skill_rows, len(df))
    df_skill = df.iloc[:skill_rows].reset_index(drop=True)

    pos_lists: dict[str, list[str]] = {"QB": [], "RB": [], "WR": [], "TE": []}
    # Row 0: QB
    if skill_rows >= 1:
        for tier in ["Starter", "2nd", "3rd", "4th"]:
            nm = df_skill.loc[0, tier]
            if nm and nm not in pos_lists["QB"]:
                pos_lists["QB"].append(nm)
    # Next RB rows
    idx = 1
    for _ in range(rb_count):
        if idx >= skill_rows:
            break
        for tier in ["Starter", "2nd", "3rd", "4th"]:
            nm = df_skill.loc[idx, tier]
            if nm and nm not in pos_lists["RB"]:
                pos_lists["RB"].append(nm)
        idx += 1
    # WR rows, aggregate tier-by-tier across WR rows
    wr_rows = []
    for _ in range(wr_count):
        if idx >= skill_rows:
            break
        wr_rows.append(df_skill.loc[idx, ["Starter", "2nd", "3rd", "4th"]].tolist())
        idx += 1
    if wr_rows:
        for tier_idx in range(4):
            for r in wr_rows:
                nm = (r[tier_idx] or "").strip()
                if nm and nm not in pos_lists["WR"]:
                    pos_lists["WR"].append(nm)
    # TE rows
    for _ in range(te_count):
        if idx >= skill_rows:
            break
        for tier in ["Starter", "2nd", "3rd", "4th"]:
            nm = df_skill.loc[idx, tier]
            if nm and nm not in pos_lists["TE"]:
                pos_lists["TE"].append(nm)
        idx += 1
    pos_lists = {k: v for k, v in pos_lists.items() if v}
    return pos_lists


def build_depth_chart_from_espn(season: int, week: int) -> pd.DataFrame:
    rows = []
    for team in ESPN_TEAM_ABBR.keys():
        pos_lists = _scrape_espn_depth_for_team(team)
        for pos, players in pos_lists.items():
            depth_size = len(players)
            for i, player in enumerate(players, start=1):
                rows.append({
                    "season": season,
                    "week": week,
                    "team": team,
                    "position": pos,
                    "player": player,
                    "depth_rank": i,
                    "depth_size": depth_size,
                    "status": "",
                    "active": True,
                })
    return pd.DataFrame(rows)


def _scrape_nfl_com_depth_for_team(team_name: str) -> dict[str, list[str]]:
    """Return a mapping pos -> ordered list of players from NFL.com depth chart page.
    We flatten the offense table row-wise left-to-right so WR starters (multiple columns) become WR1, WR2, WR3, etc.
    """
    import requests
    from bs4 import BeautifulSoup

    slug = TEAM_SLUGS.get(team_name)
    if not slug:
        return {}
    url = f"https://www.nfl.com/teams/{slug}/depth-chart/"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return {}
    soup = BeautifulSoup(resp.text, "lxml")
    # Find the offense table: prefer a section with header containing 'Offense'
    offense_table = None
    # Heuristic: tables with many position headers like QB, RB, WR, TE
    tables = soup.find_all("table")
    best_score = -1
    for t in tables:
        hdr = t.find("thead")
        if not hdr:
            continue
        ths = [th.get_text(strip=True) for th in hdr.find_all("th")]
        score = sum(1 for k in ths if re.match(r"^(QB|RB|FB|WR|TE)$", k))
        if score > best_score:
            offense_table = t
            best_score = score
    if offense_table is None:
        return {}
    # Build column index -> normalized pos
    header_ths = offense_table.find("thead").find_all("th")
    col_pos: dict[int, str] = {}
    for i, th in enumerate(header_ths):
        txt = th.get_text(strip=True).upper()
        if not txt:
            continue
        if txt.startswith("WR"):
            col_pos[i] = "WR"
        elif txt in {"RB", "HB", "FB"}:
            col_pos[i] = "RB"
        elif txt.startswith("TE"):
            col_pos[i] = "TE"
        elif txt.startswith("QB"):
            col_pos[i] = "QB"
    # Flatten rows into ordered lists per position
    pos_lists: dict[str, list[str]] = {"WR": [], "RB": [], "TE": [], "QB": []}
    tbody = offense_table.find("tbody")
    if not tbody:
        return {}
    for tr in tbody.find_all("tr"):
        tds = tr.find_all(["td", "th"])  # sometimes first cell is th
        for idx, td in enumerate(tds):
            pos = col_pos.get(idx)
            if not pos:
                continue
            # extract player names (first anchor or text)
            # handle multiple players in one cell 'A. B. / C. D.'
            cell_text = td.get_text("/", strip=True)
            parts = [p.strip() for p in cell_text.split("/") if p.strip()]
            for name in parts:
                # filter out non-player text (e.g., empty or dashes)
                if not name or set(name) == {"-"}:
                    continue
                # avoid duplicates while preserving order
                if name not in pos_lists[pos]:
                    pos_lists[pos].append(name)
    # prune empties
    for k in list(pos_lists.keys()):
        if not pos_lists[k]:
            pos_lists.pop(k, None)
    return pos_lists


def build_depth_chart_from_nfl(season: int, week: int) -> pd.DataFrame:
    rows = []
    for team in TEAM_SLUGS.keys():
        pos_lists = _scrape_nfl_com_depth_for_team(team)
        for pos, players in pos_lists.items():
            depth_size = len(players)
            for i, player in enumerate(players, start=1):
                rows.append({
                    "season": season,
                    "week": week,
                    "team": team,
                    "position": pos,
                    "player": player,
                    "depth_rank": i,
                    "depth_size": depth_size,
                    "status": "",
                    "active": True,
                })
    return pd.DataFrame(rows)


def build_depth_chart(season: int, week: int, source: str = "auto") -> pd.DataFrame:
    dc = load_depth_charts(season)
    inj = load_injuries(season)

    # Injury status map (best-effort)
    status_map: dict[tuple[str, str, int], str] = {}
    if inj is not None and not inj.empty:
        df_inj = inj.copy()
        # normalize
        for col in ["team", "recent_team", "club", "team_abbr", "team_abbreviation"]:
            if col in df_inj.columns:
                df_inj[col] = df_inj[col].astype(str)
        # player name columns
        name_cols = [c for c in ["player", "player_name", "full_name", "player_display_name"] if c in df_inj.columns]
        team_col = None
        for c in ["team", "recent_team", "club", "team_abbr", "team_abbreviation"]:
            if c in df_inj.columns:
                team_col = c
                break
        wk_col = "week" if "week" in df_inj.columns else None
        stat_col = None
        for c in ["injury_status", "status", "game_status"]:
            if c in df_inj.columns:
                stat_col = c
                break
        if name_cols and team_col and stat_col:
            dfw = df_inj.copy()
            if wk_col:
                try:
                    dfw = dfw[dfw[wk_col].astype(int) == int(week)]
                except Exception:
                    pass
            for _, r in dfw.iterrows():
                nm = _norm(r.get(name_cols[0]))
                tm = _norm(r.get(team_col))
                status = str(r.get(stat_col) or "").strip()
                if nm and tm:
                    status_map[(tm, nm, week)] = status

    rows = []
    # ESPN source if requested
    if source == "espn":
        espn_df = build_depth_chart_from_espn(season, week)
        if not espn_df.empty:
            return espn_df
    # NFL.com source if requested
    if source == "nfl":
        nfl_df = build_depth_chart_from_nfl(season, week)
        if not nfl_df.empty:
            return nfl_df
    if dc is not None and not dc.empty:
        df = dc.copy()
        # detect columns
        name_cols = [c for c in ["player", "player_name", "full_name", "player_display_name"] if c in df.columns]
        team_col = "team" if "team" in df.columns else ("recent_team" if "recent_team" in df.columns else None)
        pos_col = None
        for c in ["position", "pos", "position_group", "depth_chart_position"]:
            if c in df.columns:
                pos_col = c
                break
        order_col = None
        for c in ["depth_chart_order", "order", "depth", "depth_team_rank"]:
            if c in df.columns:
                order_col = c
                break
        if name_cols and team_col and pos_col:
            df[team_col] = df[team_col].astype(str)
            df[pos_col] = df[pos_col].astype(str).map(_map_pos)
            if order_col is None:
                # Try to join with weekly player likelihoods to rank by expected_td
                df_core = df[[team_col, pos_col, name_cols[0]]].copy()
                df_core.rename(columns={team_col: 'team', pos_col: 'position', name_cols[0]: 'player'}, inplace=True)
                df_core['team'] = df_core['team'].astype(str)
                df_core['position'] = df_core['position'].astype(str)
                df_core['player'] = df_core['player'].astype(str)
                rk_col = '__rk'
                like_fp = DATA_DIR / f"player_td_likelihood_{season}_wk{week}.csv"
                if like_fp.exists():
                    try:
                        pl = pd.read_csv(like_fp)
                        pl = pl.copy()
                        pl['team'] = pl['team'].astype(str)
                        pl['position'] = pl['position'].astype(str)
                        pl['player'] = pl['player'].astype(str)
                        pl['expected_td'] = pd.to_numeric(pl.get('expected_td'), errors='coerce').fillna(0.0)
                        merged = df_core.merge(pl[['team','position','player','expected_td']], on=['team','position','player'], how='left')
                        merged['expected_td'] = merged['expected_td'].fillna(0.0)
                        merged.sort_values(['team','position','expected_td','player'], ascending=[True, True, False, True], inplace=True)
                        merged[rk_col] = merged.groupby(['team','position']).cumcount() + 1
                        df = merged.rename(columns={'team': team_col, 'position': pos_col, 'player': name_cols[0]}).copy()
                        order_col = rk_col
                    except Exception:
                        # fallback alphabetical
                        df = df[[team_col, pos_col, name_cols[0]]].copy()
                        df.sort_values([team_col, pos_col, name_cols[0]], inplace=True)
                        df[rk_col] = df.groupby([team_col, pos_col]).cumcount() + 1
                        order_col = rk_col
                else:
                    # fallback alphabetical
                    df = df[[team_col, pos_col, name_cols[0]]].copy()
                    df.sort_values([team_col, pos_col, name_cols[0]], inplace=True)
                    df[rk_col] = df.groupby([team_col, pos_col]).cumcount() + 1
                    order_col = rk_col
            else:
                df[order_col] = pd.to_numeric(df[order_col], errors="coerce").fillna(9999).astype(int)
            # filter week if present
            if "week" in df.columns:
                try:
                    df = df[df["week"].astype(int) == int(week)]
                except Exception:
                    pass
            for (tm, pos), grp in df.groupby([team_col, pos_col], dropna=False):
                g = grp.sort_values(order_col, ascending=True).reset_index(drop=True)
                total = len(g)
                for idx, r in g.iterrows():
                    player = str(r.get(name_cols[0]) or "").strip()
                    if not player:
                        continue
                    team_key = _norm(tm)
                    player_key = _norm(player)
                    status = status_map.get((team_key, player_key, week), "")
                    active = str(status).strip().lower() not in {"out", "ir", "pup", "suspended", "inactive"}
                    rows.append({
                        "season": season,
                        "week": week,
                        "team": str(tm),
                        "position": str(r.get(pos_col)),
                        "player": player,
                        "depth_rank": int(r.get(order_col) or (idx+1)),
                        "depth_size": total,
                        "status": status,
                        "active": bool(active),
                    })
    # fallback from player_td_likelihood
    if not rows:
        fp = DATA_DIR / f"player_td_likelihood_{season}_wk{week}.csv"
        if fp.exists():
            try:
                pl = pd.read_csv(fp)
                pl["team"] = pl["team"].astype(str)
                pl["position"] = pl["position"].astype(str).map(_map_pos)
                pl["player"] = pl["player"].astype(str)
                pl["expected_td"] = pd.to_numeric(pl.get("expected_td"), errors="coerce").fillna(0.0)
                for (tm, pos), grp in pl.groupby(["team", "position" ], dropna=False):
                    g = grp.sort_values("expected_td", ascending=False).reset_index(drop=True)
                    total = len(g)
                    for idx, r in g.iterrows():
                        player = str(r.get("player") or "").strip()
                        if not player:
                            continue
                        team_key = _norm(tm)
                        player_key = _norm(player)
                        status = status_map.get((team_key, player_key, week), "")
                        active = str(status).strip().lower() not in {"out", "ir", "pup", "suspended", "inactive"}
                        rows.append({
                            "season": season,
                            "week": week,
                            "team": str(tm),
                            "position": str(pos),
                            "player": player,
                            "depth_rank": int(idx+1),
                            "depth_size": total,
                            "status": status,
                            "active": bool(active),
                        })
            except Exception:
                pass
    out = pd.DataFrame(rows)
    # Apply optional overrides
    ov_fp = DATA_DIR / "depth_chart_overrides.csv"
    if not out.empty and ov_fp.exists():
        try:
            ov = pd.read_csv(ov_fp)
            if all(c in ov.columns for c in ['team','position','player','depth_rank']):
                ov = ov.copy()
                ov['team'] = ov['team'].astype(str).str.strip()
                ov['position'] = ov['position'].astype(str)
                ov['player'] = ov['player'].astype(str).str.strip()
                ov['depth_rank'] = pd.to_numeric(ov['depth_rank'], errors='coerce').astype('Int64')
                # apply per team/pos
                out = out.copy()
                for (tm, pos), g in out.groupby(['team','position']):
                    key = (tm, pos)
                    ovg = ov[(ov['team'] == tm) & (ov['position'] == pos)]
                    if ovg.empty:
                        continue
                    # map overrides
                    rank_map = { (row['player']): int(row['depth_rank']) for _, row in ovg.iterrows() if pd.notna(row['depth_rank']) }
                    if not rank_map:
                        continue
                    mask = (out['team'] == tm) & (out['position'] == pos)
                    seg = out[mask].copy()
                    # assign override ranks
                    seg['__ovr'] = seg['player'].map(rank_map)
                    # players with overrides first by specified rank, then others by current depth_rank
                    # also re-sequence to 1..N with overrides fixed
                    fixed = seg[seg['__ovr'].notna()].copy()
                    fixed['__ovr'] = fixed['__ovr'].astype(int)
                    rest = seg[seg['__ovr'].isna()].copy()
                    rest.sort_values(['depth_rank','player'], inplace=True)
                    # build final order
                    used = set(int(x) for x in fixed['__ovr'].tolist())
                    cur_rank = 1
                    new_ranks = {}
                    # assign in order, filling gaps
                    for _, r in fixed.sort_values('__ovr').iterrows():
                        want = int(r['__ovr'])
                        # fill any gaps before this override
                        while cur_rank < want and not rest.empty:
                            idx, rr = rest.iloc[0].name, rest.iloc[0]
                            new_ranks[idx] = cur_rank
                            rest = rest.iloc[1:]
                            cur_rank += 1
                        # place the override at its desired rank
                        new_ranks[r.name] = want
                        cur_rank = want + 1
                    # assign remaining rest
                    for idx, rr in rest.iterrows():
                        new_ranks[idx] = cur_rank
                        cur_rank += 1
                    # write back
                    out.loc[mask, 'depth_rank'] = out.loc[mask].apply(lambda row: new_ranks.get(row.name, row['depth_rank']), axis=1)
                    # normalize ordering
                    seg2 = out[mask].sort_values('depth_rank').reset_index(drop=True)
                    seg2['depth_rank'] = range(1, len(seg2)+1)
                    out.loc[mask, 'depth_rank'] = seg2['depth_rank'].values
        except Exception:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build weekly depth chart file")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--source", choices=["auto","espn","nfl","nfl_data_py"], default="auto", help="Depth chart source")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.source == "nfl_data_py":
        df = build_depth_chart(args.season, args.week, source="nfl_data_py")
    elif args.source == "espn":
        df = build_depth_chart(args.season, args.week, source="espn")
    elif args.source == "nfl":
        df = build_depth_chart(args.season, args.week, source="nfl")
    else:
        # try ESPN, then NFL.com, then fallback
        df = build_depth_chart(args.season, args.week, source="espn")
        if df.empty:
            df = build_depth_chart(args.season, args.week, source="nfl")
        if df.empty:
            df = build_depth_chart(args.season, args.week, source="nfl_data_py")
    out = DATA_DIR / f"depth_chart_{args.season}_wk{args.week}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
