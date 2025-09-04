from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

# Local data path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class WeatherCols:
    temp_f: str = "wx_temp_f"
    wind_mph: str = "wx_wind_mph"
    precip_pct: str = "wx_precip_pct"
    roof: str = "roof"
    surface: str = "surface"


def _ensure_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.roof, WeatherCols.surface]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def load_stadium_meta() -> pd.DataFrame:
    fp = DATA_DIR / "stadium_meta.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["team","roof","surface","lat","lon","tz","altitude_ft"])
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame(columns=["team","roof","surface","lat","lon","tz","altitude_ft"])
    return df


def load_weather_for_date(date_str: str) -> pd.DataFrame:
    candidates = [
        DATA_DIR / f"weather_{date_str}.csv",
        DATA_DIR / f"weather_{date_str.replace('-', '_')}.csv",
        DATA_DIR / "weather.csv",
    ]
    for fp in candidates:
        if fp.exists():
            try:
                return pd.read_csv(fp)
            except Exception:
                continue
    return pd.DataFrame(columns=["date","home_team", WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct])


def load_weather_for_games(games: pd.DataFrame) -> pd.DataFrame:
    if games is None or games.empty:
        return pd.DataFrame(columns=["game_id","date","home_team","away_team", WeatherCols.temp_f, WeatherCols.wind_mph, WeatherCols.precip_pct, WeatherCols.roof, WeatherCols.surface])

    out_rows = []
    stad = load_stadium_meta()
    stad_map = {}
    if not stad.empty and "team" in stad.columns:
        stad_map = stad.set_index(stad["team"].astype(str).str.strip()).to_dict(orient="index")

    for date_str, gdf in games.groupby(games.get("date", pd.Series(index=games.index, dtype=str))):
        wdf = load_weather_for_date(str(date_str))
        if not wdf.empty:
            wdf["home_team"] = wdf["home_team"].astype(str).str.strip()
        for _, row in gdf.iterrows():
            r = {
                "game_id": row.get("game_id"),
                "date": row.get("date"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                WeatherCols.temp_f: pd.NA,
                WeatherCols.wind_mph: pd.NA,
                WeatherCols.precip_pct: pd.NA,
                WeatherCols.roof: pd.NA,
                WeatherCols.surface: pd.NA,
                "neutral_site": pd.NA,
            }
            ht = str(row.get("home_team", "")).strip()
            if not wdf.empty:
                m = wdf[wdf["home_team"] == ht]
                if not m.empty:
                    r[WeatherCols.temp_f] = m.iloc[0].get(WeatherCols.temp_f)
                    r[WeatherCols.wind_mph] = m.iloc[0].get(WeatherCols.wind_mph)
                    r[WeatherCols.precip_pct] = m.iloc[0].get(WeatherCols.precip_pct)
                    if 'neutral_site' in m.columns:
                        r["neutral_site"] = m.iloc[0].get('neutral_site')
            if ht in stad_map:
                r[WeatherCols.roof] = stad_map[ht].get("roof")
                r[WeatherCols.surface] = stad_map[ht].get("surface")
            out_rows.append(r)

    out = pd.DataFrame(out_rows)
    return _ensure_weather_columns(out)
