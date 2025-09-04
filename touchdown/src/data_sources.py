from __future__ import annotations

import json
from datetime import datetime
import os
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
from .schemas import GameRow, TeamStatRow, LineRow
from .team_normalizer import normalize_team_name

# Point to NFL-Touchdown/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _field_names(model_cls) -> list:
    try:
        return list(model_cls.model_fields.keys())  # Pydantic v2
    except AttributeError:
        try:
            return list(model_cls.__fields__.keys())  # Pydantic v1
        except Exception:
            return []


def load_games() -> pd.DataFrame:
    fp = DATA_DIR / "games.csv"
    if not fp.exists():
        return pd.DataFrame(columns=_field_names(GameRow))
    df = pd.read_csv(fp)
    return df


def load_team_stats() -> pd.DataFrame:
    fp = DATA_DIR / "team_stats.csv"
    if not fp.exists():
        return pd.DataFrame(columns=_field_names(TeamStatRow))
    df = pd.read_csv(fp)
    return df


def _parse_real_lines_json(blob: Dict[str, Any]) -> pd.DataFrame:
    lines = blob.get('lines', {}) if isinstance(blob, dict) else {}
    rows: List[Dict[str, Any]] = []
    for matchup_key, game_lines in lines.items():
        try:
            if '@' in matchup_key:
                away_team, home_team = [p.strip() for p in matchup_key.split('@', 1)]
            elif 'vs' in matchup_key:
                away_team, home_team = [p.strip() for p in matchup_key.split('vs', 1)]
            else:
                continue
            ml_home = None
            ml_away = None
            spread_home = None
            total_line = None
            spread_home_price = None
            spread_away_price = None
            total_over_price = None
            total_under_price = None
            if isinstance(game_lines, dict) and 'moneyline' in game_lines:
                try:
                    ml = game_lines['moneyline']
                    ml_home = ml.get('home')
                    ml_away = ml.get('away')
                except Exception:
                    pass
            if isinstance(game_lines, dict) and 'total_runs' in game_lines:
                tr = game_lines['total_runs'] or {}
                total_line = tr.get('line', total_line)
                total_over_price = tr.get('over', total_over_price)
                total_under_price = tr.get('under', total_under_price)
            if isinstance(game_lines, dict) and 'total' in game_lines and isinstance(game_lines['total'], dict):
                total_line = game_lines['total'].get('line', total_line)
            if isinstance(game_lines, dict) and 'run_line' in game_lines:
                rl = game_lines['run_line']
                spread_home = rl.get('home', rl.get('line', spread_home))
            markets = []
            if isinstance(game_lines, dict) and 'markets' in game_lines and isinstance(game_lines['markets'], list):
                markets = game_lines['markets']
            def _is_full_game(m: Dict[str, Any]) -> bool:
                key = str(m.get('key', '')).lower()
                desc = str(m.get('description', '')).lower()
                bad_tokens = ['1h', '2h', 'half', 'q1', 'q2', 'q3', 'q4', 'quarter']
                return not any(t in key or t in desc for t in bad_tokens)
            for m in markets:
                if not _is_full_game(m):
                    continue
                key = m.get('key')
                outcomes = m.get('outcomes', []) or []
                if key in ('h2h', 'moneyline'):
                    for o in outcomes:
                        name = str(o.get('name', ''))
                        price = o.get('price')
                        if not name or price is None:
                            continue
                        if name.strip().lower() == home_team.lower():
                            ml_home = price
                        elif name.strip().lower() == away_team.lower():
                            ml_away = price
                elif key in ('spreads', 'spread'):
                    for o in outcomes:
                        name = str(o.get('name', ''))
                        pt = o.get('point')
                        price = o.get('price')
                        if name.strip().lower() == home_team.lower():
                            if pt is not None:
                                spread_home = pt
                            spread_home_price = price if price is not None else spread_home_price
                        elif name.strip().lower() == away_team.lower():
                            spread_away_price = price if price is not None else spread_away_price
                elif key in ('totals', 'total'):
                    pts = [o.get('point') for o in outcomes if o.get('point') is not None]
                    if pts:
                        total_line = pts[0]
                    for o in outcomes:
                        nm = str(o.get('name','')).strip().lower()
                        price = o.get('price')
                        if nm.startswith('over'):
                            total_over_price = price if price is not None else total_over_price
                        elif nm.startswith('under'):
                            total_under_price = price if price is not None else total_under_price
            try:
                if total_line is not None:
                    total_line = float(total_line)
            except Exception:
                pass
            try:
                if spread_home is not None:
                    spread_home = float(spread_home)
            except Exception:
                pass
            rows.append({
                'away_team': away_team,
                'home_team': home_team,
                'moneyline_home': ml_home,
                'moneyline_away': ml_away,
                'spread_home': spread_home,
                'total': total_line,
                'spread_home_price': spread_home_price,
                'spread_away_price': spread_away_price,
                'total_over_price': total_over_price,
                'total_under_price': total_under_price,
            })
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        'away_team','home_team','moneyline_home','moneyline_away','spread_home','total',
        'spread_home_price','spread_away_price','total_over_price','total_under_price'
    ])


def _try_load_latest_real_lines() -> pd.DataFrame:
    if os.getenv('DISABLE_JSON_ODDS', '').strip() in ('1','true','True','yes','Y'):
        return pd.DataFrame(columns=['away_team','home_team','moneyline_home','moneyline_away','spread_home','total'])
    candidates = []
    today_us = datetime.now().strftime('%Y_%m_%d')
    candidates.append(DATA_DIR / f'real_betting_lines_{today_us}.json')
    today_hy = datetime.now().strftime('%Y-%m-%d')
    candidates.append(DATA_DIR / f'real_betting_lines_{today_hy}.json')
    candidates.append(DATA_DIR / 'real_betting_lines.json')
    for fp in candidates:
        if fp.exists():
            try:
                blob = json.loads(fp.read_text(encoding='utf-8'))
                return _parse_real_lines_json(blob)
            except Exception:
                continue
    try:
        files = sorted(DATA_DIR.glob('real_betting_lines_*.json'))
        for fp in reversed(files):
            try:
                blob = json.loads(fp.read_text(encoding='utf-8'))
                df = _parse_real_lines_json(blob)
                if not df.empty:
                    return df
            except Exception:
                continue
    except Exception:
        pass
    return pd.DataFrame(columns=['away_team','home_team','moneyline_home','moneyline_away','spread_home','total'])


def load_lines() -> pd.DataFrame:
    csv_cols = _field_names(LineRow)
    df_csv = pd.DataFrame(columns=csv_cols)
    fp = DATA_DIR / "lines.csv"
    if fp.exists():
        try:
            df_csv = pd.read_csv(fp)
            if 'home_team' in df_csv.columns:
                df_csv['home_team'] = df_csv['home_team'].astype(str).apply(normalize_team_name)
            if 'away_team' in df_csv.columns:
                df_csv['away_team'] = df_csv['away_team'].astype(str).apply(normalize_team_name)
        except Exception:
            df_csv = pd.DataFrame(columns=csv_cols)
    df_json = _try_load_latest_real_lines()
    if df_json.empty:
        return df_csv
    if not df_csv.empty:
        merged = df_csv.merge(
            df_json,
            on=['home_team','away_team'],
            how='left',
            suffixes=('', '_json')
        )
        for col in ['spread_home','total','moneyline_home','moneyline_away','spread_home_price','spread_away_price','total_over_price','total_under_price']:
            json_col = f'{col}_json'
            if json_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[json_col])
        drop_cols = [c for c in merged.columns if c.endswith('_json')]
        merged = merged.drop(columns=drop_cols)
        try:
            key_cols = ['home_team', 'away_team']
            present = set(tuple(x) for x in merged[key_cols].astype(str).values.tolist())
            add_rows = df_json[~df_json[key_cols].astype(str).apply(tuple, axis=1).isin(present)].copy()
            if not add_rows.empty:
                for col in ['season','week','game_id','close_spread_home','close_total']:
                    if col not in add_rows.columns:
                        add_rows[col] = pd.NA
                add_rows = add_rows.reindex(columns=merged.columns, fill_value=pd.NA)
                merged = pd.concat([merged, add_rows], ignore_index=True)
        except Exception:
            pass
        return merged
    else:
        for col in ['season','week','game_id','close_spread_home','close_total']:
            df_json[col] = pd.NA
        return df_json[[c for c in csv_cols if c in df_json.columns] + [c for c in df_json.columns if c not in csv_cols]]
