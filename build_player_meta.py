from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def name_key_variants(full_name: str) -> list[str]:
    s = str(full_name or "").strip()
    if not s:
        return []
    low = s.lower()
    tokens = [t for t in s.split() if t]
    out = [low]
    if tokens:
        first = tokens[0]
        finit = (first[0] + ".").lower() if first else ""
        last = tokens[-1]
        # default: initial + last word
        if finit and last:
            out.append(f"{finit}{last.lower()}")
            out.append(f"{finit} {last.lower()}")
        # handle common multi-word last-name particles (e.g., "St. Brown", "Van Noy")
        particles = {"st.", "st", "van", "von", "de", "de.", "da", "del", "della", "la", "le", "mac", "mc", "o'"}
        if len(tokens) >= 3:
            last2 = tokens[-2].lower()
            if last2 in particles:
                combo = f"{tokens[-2]} {tokens[-1]}".lower()
                out.append(f"{finit}{combo}")
        # also try hyphenated last names keeping hyphen
        if "-" in last:
            out.append(f"{finit}{last.lower()}")
    # de-dupe, preserve order
    seen = set()
    uniq = []
    for k in out:
        if k and k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def load_2024_totals() -> Dict[str, Tuple[int, int, int]]:
    fp = DATA / "player_td_totals_2024.csv"
    out: Dict[str, Tuple[int, int, int]] = {}
    if not fp.exists():
        return out
    try:
        df = pd.read_csv(fp)
    except Exception:
        return out
    for _, r in df.iterrows():
        name = str(r.get("player") or "").strip().lower()
        if not name:
            continue
        try:
            rush = int(pd.to_numeric(r.get("rush_td"), errors="coerce").fillna(0))
        except Exception:
            rush = int(r.get("rush_td") or 0)
        try:
            rec = int(pd.to_numeric(r.get("rec_td"), errors="coerce").fillna(0))
        except Exception:
            rec = int(r.get("rec_td") or 0)
        try:
            total = int(pd.to_numeric(r.get("total_td"), errors="coerce").fillna(0))
        except Exception:
            total = int(r.get("total_td") or 0)
        out[name] = (rush, rec, total)
    return out


def load_career_counts() -> Dict[str, Tuple[int, int, int]]:
    # Aggregate across all pbp_*.csv in data
    rush_map: Dict[str, int] = {}
    rec_map: Dict[str, int] = {}
    total_map: Dict[str, int] = {}
    files = sorted(DATA.glob("pbp_*.csv"))
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
    # Build tuple map
    out: Dict[str, Tuple[int, int, int]] = {}
    for name in set(list(rush_map.keys()) + list(rec_map.keys()) + list(total_map.keys())):
        out[name] = (rush_map.get(name, 0), rec_map.get(name, 0), total_map.get(name, 0))
    return out


def build_player_meta() -> Path:
    td24 = load_2024_totals()  # name -> (rush, rec, total)
    career = load_career_counts()  # name -> (rush, rec, total)

    # Union of all base names
    all_names = set(td24.keys()) | set(career.keys())

    rows = []
    for base in sorted(all_names):
        r24, c24, t24 = td24.get(base, (0, 0, 0))
        rc, cc, tc = career.get(base, (0, 0, 0))
        # add base
        rows.append({
            "player": base,
            "td24_rush": r24,
            "td24_rec": c24,
            "td24_total": t24,
            "career_rush": rc,
            "career_rec": cc,
            "career_total": tc,
        })
        # add variants
        for v in name_key_variants(base):
            if v == base:
                continue
            rows.append({
                "player": v,
                "td24_rush": r24,
                "td24_rec": c24,
                "td24_total": t24,
                "career_rush": rc,
                "career_rec": cc,
                "career_total": tc,
            })

    out = DATA / "player_meta.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates(subset=["player"])\
        .sort_values(["player"]).to_csv(out, index=False)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    build_player_meta()
