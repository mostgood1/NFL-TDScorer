#!/usr/bin/env python
"""Fetch Bovada NFL Anytime TD (ATD) odds and persist historically.

For now we only target Week 1, but the script supports arbitrary (season, week).
Output:
  - data/bovada_atd_{season}_wk{week}.csv (snapshot for the week at pull time)
  - data/bovada_atd_history.csv (append/update master history)

We identify bets by (season, week, event_id, outcome_id). If the same key is re-fetched,
we update the record (last_update + odds fields) instead of duplicating.

Bovada structure reference (public JSON):
  https://www.bovada.lv/services/sports/event/v2/events/A/description/FOOTBALL/NFL
This returns nested arrays; each sport -> league -> events. Each event has displayGroups
which include markets, where one market's description (case-insensitive) is typically
"Anytime Touchdown Scorer" (sometimes variants like "To Score a Touchdown"). We match
these descriptions by lowercasing and checking substrings: 'anytime touchdown' or
'to score a touchdown'.

Each market outcome has a price dict with 'american', 'decimal'. We compute implied probability
from the American price and store both.

NOTE: Bovada rotates endpoints occasionally. If this breaks, inspect the raw JSON saved via
--dump-json to adapt the parsing rules.

Usage examples:
  python fetch_bovada_atd.py --season 2025 --week 1
  python fetch_bovada_atd.py --season 2025 --week 1 --dry-run (no network, just shows target URL)
  python fetch_bovada_atd.py --season 2025 --week 1 --dump-json raw_week1.json

"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except ImportError:  # fallback minimal
    requests = None  # type: ignore
    import urllib.request  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ATD_MARKET_KEYWORDS = [
    "anytime touchdown",  # common phrasing
    "to score a touchdown",  # alternate phrasing
]

# Simple team name normalization; Bovada often uses full names.
TEAM_ABBR_MAP = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


def american_to_implied_prob(american: int | float) -> Optional[float]:
    try:
        a = float(american)
    except Exception:
        return None
    if a == 0:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    else:
        return -a / (-a + 100.0)


DEFAULT_ENDPOINTS = [
    # Primary documented variant (was working previously)
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/FOOTBALL/NFL",
    # Lowercase path variant
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/football/nfl",
    # Without description segment
    "https://www.bovada.lv/services/sports/event/v2/events/A/FOOTBALL/NFL",
    # Query param variants (lang, market filtering attempts)
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/FOOTBALL/NFL?lang=en",
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/FOOTBALL/NFL?lang=en-US",
    # Broader FOOTBALL feed (filter later)
    "https://www.bovada.lv/services/sports/event/v2/events/A/description/FOOTBALL",
]


def _http_get(url: str) -> tuple[int, bytes, dict]:
    if requests:
        r = requests.get(url, timeout=25, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        return r.status_code, r.content, dict(r.headers)
    else:  # urllib fallback
        req = urllib.request.Request(url, headers={  # type: ignore
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Cache-Control": "no-cache",
        })
        with urllib.request.urlopen(req, timeout=25) as resp:  # type: ignore
            headers = {k: v for k, v in resp.getheaders()}
            return getattr(resp, 'status', 200), resp.read(), headers


def fetch_bovada_json(verbose: bool = True, debug: bool = False) -> Any:
    endpoints = []
    override = os.environ.get("BOVADA_ATD_ENDPOINTS")
    if override:
        endpoints = [e.strip() for e in override.split("|") if e.strip()]
    if not endpoints:
        endpoints = list(DEFAULT_ENDPOINTS)
    last_err = None
    attempt_idx = 0
    for url in endpoints:
        attempt_idx += 1
        try:
            status, body, headers = _http_get(url)
            if verbose:
                print(f"Attempt {attempt_idx}: {url} -> status {status} bytes {len(body)} content-type={headers.get('Content-Type')} ")
            if debug:
                dbg_dir = DATA_DIR / "bovada_debug"
                dbg_dir.mkdir(exist_ok=True)
                with open(dbg_dir / f"attempt_{attempt_idx}.meta.txt", "w", encoding="utf-8") as mf:
                    mf.write(f"URL: {url}\nStatus: {status}\nHeaders: {json.dumps(headers, indent=2)}\nBytes: {len(body)}\n")
                if body:
                    with open(dbg_dir / f"attempt_{attempt_idx}.raw", "wb") as bf:
                        bf.write(body)
            if status != 200 or not body:
                continue
            # Attempt JSON decode
            try:
                return json.loads(body.decode("utf-8"))
            except Exception as je:  # maybe HTML; keep error
                last_err = je
                continue
        except Exception as e:  # network error
            last_err = e
            continue
    # HTML fallback: attempt to fetch the public NFL page and scrape inline JSON
    try:
        nfl_page = "https://www.bovada.lv/sports/football/nfl"
        status, body, headers = _http_get(nfl_page)
        if verbose:
            print(f"HTML fallback {nfl_page} -> status {status} bytes {len(body)} content-type={headers.get('Content-Type')} ")
        if debug:
            dbg_dir = DATA_DIR / "bovada_debug"
            dbg_dir.mkdir(exist_ok=True)
            with open(dbg_dir / "html_fallback.meta.txt", "w", encoding="utf-8") as mf:
                mf.write(f"URL: {nfl_page}\nStatus: {status}\nHeaders: {json.dumps(headers, indent=2)}\nBytes: {len(body)}\n")
            if body:
                with open(dbg_dir / "html_fallback.raw", "wb") as bf:
                    bf.write(body)
        txt = body.decode("utf-8", errors="ignore")
        import re
        # Try to find any window.__data style JSON (not guaranteed on Bovada)
        m = re.search(r'(\{\s*"sports".*?\})\s*</script>', txt, re.DOTALL)
        if not m:
            m = re.search(r'(\[\{.*?\}\])', txt, re.DOTALL)
        if m:
            snippet = m.group(1)
            try:
                return json.loads(snippet)
            except Exception:
                pass
    except Exception as e:
        last_err = e
    if verbose:
        print(f"All Bovada endpoint attempts failed or returned empty. Last error: {last_err}")
    return []  # return empty list so caller can proceed gracefully


def find_atd_markets(raw: Any) -> List[Dict[str, Any]]:
    """Traverse Bovada nested JSON, returning list of dicts with event + market context.

    Structure (as of writing): raw is a list; each element has 'events'. Each event has:
      - id
      - description (e.g., "Dallas Cowboys @ Philadelphia Eagles")
      - startTime
      - competitors (list with id, name, home/away)
      - displayGroups -> list -> markets (with description, outcomes)
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for sport_block in raw:
        events = sport_block.get("events") if isinstance(sport_block, dict) else None
        if not isinstance(events, list):
            continue
        for ev in events:
            dg_list = ev.get("displayGroups") if isinstance(ev, dict) else None
            if not isinstance(dg_list, list):
                continue
            for dg in dg_list:
                markets = dg.get("markets") if isinstance(dg, dict) else None
                if not isinstance(markets, list):
                    continue
                for mkt in markets:
                    desc = str(mkt.get("description") or "").lower()
                    if any(k in desc for k in ATD_MARKET_KEYWORDS):
                        out.append({
                            "event": ev,
                            "market": mkt,
                        })
    return out


def parse_outcomes(atd_markets: List[Dict[str, Any]], season: int, week: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    for item in atd_markets:
        ev = item.get("event", {})
        mkt = item.get("market", {})
        event_id = ev.get("id")
        ev_desc = ev.get("description")
        start_time = ev.get("startTime")
        competitors = ev.get("competitors") or []
        comp_map: Dict[str, Dict[str, Any]] = {}
        home_team_abbr = None
        away_team_abbr = None
        for comp in competitors:
            cid = str(comp.get("id"))
            name = str(comp.get("name") or "").strip()
            lower_name = name.lower()
            abbr = TEAM_ABBR_MAP.get(lower_name)
            if not abbr:
                # fallback initials
                parts = [p for p in lower_name.split() if p]
                abbr = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else (parts[0][:3].upper() if parts else "UNK")
            comp_map[cid] = {"name": name, "abbr": abbr, "home": comp.get("home")}
        # Identify home/away abbreviations
        for c in comp_map.values():
            if c.get("home") is True:
                home_team_abbr = c.get("abbr")
            else:
                away_team_abbr = c.get("abbr")
        outcomes = mkt.get("outcomes") or []
        for oc in outcomes:
            outcome_id = oc.get("id")
            player = oc.get("description") or oc.get("name")
            price = oc.get("price") or {}
            american = price.get("american") or price.get("us")
            decimal_odds = price.get("decimal")
            try:
                american_int = int(american)
            except Exception:
                try:
                    american_int = int(float(american))
                except Exception:
                    american_int = None  # type: ignore
            implied = american_to_implied_prob(american_int) if american_int is not None else None
            # Attempt to link to competitor
            comp_id = oc.get("competitorId")
            team_abbr = None
            is_home = None
            opp_abbr = None
            if comp_id and str(comp_id) in comp_map:
                team_abbr = comp_map[str(comp_id)]["abbr"]
                is_home = comp_map[str(comp_id)]["home"] is True
                opp_abbr = home_team_abbr if not is_home else away_team_abbr
            # Heuristic: if team_abbr missing, try to parse trailing parenthetical '(XXX)' in player string
            if (not team_abbr) and isinstance(player, str):
                import re
                m = re.search(r"^(.*)\(([A-Z0-9]{2,4})\)\s*$", player.strip())
                if m:
                    base_name = m.group(1).strip()
                    cand = m.group(2).upper().strip()
                    # Validate candidate against known team map values to avoid capturing nicknames
                    valid_abbrs = set(TEAM_ABBR_MAP.values())
                    if cand in valid_abbrs:
                        team_abbr = cand
                        player = base_name  # strip the parenthetical
            row = {
                "season": season,
                "week": week,
                "event_id": event_id,
                "event_description": ev_desc,
                "start_time": start_time,
                "market_id": mkt.get("id"),
                "market_description": mkt.get("description"),
                "outcome_id": outcome_id,
                "player": player,
                "team": team_abbr,
                "opp": opp_abbr,
                "is_home": is_home,
                "american_odds": american_int,
                "decimal_odds": float(decimal_odds) if decimal_odds not in (None, "") else None,
                "implied_prob": round(implied, 4) if implied is not None else None,
                "last_update": now_iso,
                "source": "bovada",
            }
            rows.append(row)
    return rows


def load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd  # optional speed path
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        # Fallback csv.DictReader
        out: List[Dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                out.append(dict(r))
        return out


def merge_history(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx: Dict[Tuple[Any, Any, Any, Any], int] = {}
    for i, r in enumerate(existing):
        key = (r.get("season"), r.get("week"), r.get("event_id"), r.get("outcome_id"))
        idx[key] = i
    for nr in new_rows:
        key = (nr.get("season"), nr.get("week"), nr.get("event_id"), nr.get("outcome_id"))
        if key in idx:
            existing[idx[key]] = nr  # replace / update
        else:
            existing.append(nr)
    return existing


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    cols = [
        "season","week","event_id","event_description","start_time","market_id","market_description","outcome_id","player","team","opp","is_home","american_odds","decimal_odds","implied_prob","last_update","source"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main():
    ap = argparse.ArgumentParser(description="Fetch Bovada NFL Anytime TD odds")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="Do not perform network fetch; just report endpoints tried")
    ap.add_argument("--dump-json", type=str, help="Also dump raw Bovada JSON to this file path")
    ap.add_argument("--from-file", type=str, help="Parse ATD markets from a previously saved Bovada JSON file (skips fetch)")
    ap.add_argument("--debug", action="store_true", help="Enable verbose attempt logging and save raw responses to data/bovada_debug/")
    args = ap.parse_args()

    if args.dry_run:
        print("[dry-run] Endpoints that would be attempted (in order):")
        for u in (os.environ.get("BOVADA_ATD_ENDPOINTS"," ").split("|") if os.environ.get("BOVADA_ATD_ENDPOINTS") else DEFAULT_ENDPOINTS):
            if u.strip():
                print("  -", u.strip())
        print("Plus HTML fallback: https://www.bovada.lv/sports/football/nfl")
        return

    if args.from_file:
        try:
            with open(args.from_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            print(f"Loaded raw JSON from file: {args.from_file}")
        except Exception as e:
            print(f"ERROR: failed to read --from-file JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        raw = fetch_bovada_json(debug=args.debug)
        if not raw:
            print("ERROR: No data returned from Bovada endpoints (possibly geo-blocked / challenge). Provide a raw JSON via --from-file.", file=sys.stderr)
            # Still proceed to attempt parse (will yield 0 markets) so pipeline doesn't hard fail.

    if args.dump_json:
        try:
            with open(args.dump_json, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            print(f"Saved raw JSON -> {args.dump_json}")
        except Exception as e:
            print(f"WARN: could not dump raw JSON: {e}")

    markets = find_atd_markets(raw)
    if not markets:
        print("WARN: No ATD markets found. Endpoint/structure may have changed or fetch returned empty.")
    rows = parse_outcomes(markets, args.season, args.week)
    print(f"Parsed {len(rows)} ATD player odds rows.")

    snapshot_path = DATA_DIR / f"bovada_atd_{args.season}_wk{args.week}.csv"
    write_csv(snapshot_path, rows)
    print(f"Wrote snapshot -> {snapshot_path}")

    history_path = DATA_DIR / "bovada_atd_history.csv"
    hist = load_history(history_path)
    merged = merge_history(hist, rows)
    write_csv(history_path, merged)
    print(f"Updated history -> {history_path} (total rows: {len(merged)})")


if __name__ == "__main__":
    main()
