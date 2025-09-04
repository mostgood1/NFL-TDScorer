from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from touchdown.src.pbp_td_trends import compute_td_trends, last_n_seasons


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Summarize last-N seasons TD trends by team (position shares) and top scorers.")
    ap.add_argument("--season", type=int, required=True, help="Current season (e.g., 2025); trends use prior N seasons.")
    ap.add_argument("--years", type=int, default=5, help="Number of seasons back (default 5).")
    args = ap.parse_args(argv)

    seasons = last_n_seasons(args.season, args.years)
    res = compute_td_trends(args.season, seasons)
    tps: pd.DataFrame = res.get("team_pos_shares", pd.DataFrame())
    pc: pd.DataFrame = res.get("player_td_counts", pd.DataFrame())
    start_season = res.get("start_season", seasons[0])
    end_season = res.get("end_season", seasons[-1])

    base_dir = Path(__file__).resolve().parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Team position shares summary: combine pass and rush into one row per team
    if tps is not None and not tps.empty:
        tps_pass = tps[tps.get("kind") == "pass"].copy()
        tps_rush = tps[tps.get("kind") == "rush"].copy()
        cols_pos = ["WR", "TE", "RB", "QB", "OTHER"]
        for c in cols_pos:
            if c not in tps_pass.columns: tps_pass[c] = 0.0
            if c not in tps_rush.columns: tps_rush[c] = 0.0
        pass_cols = {c: f"pass_{c}" for c in ["WR", "TE", "RB"]}
        rush_cols = {c: f"rush_{c}" for c in ["RB", "QB", "WR", "TE"]}
        tps_pass = tps_pass[["team", "kind", *pass_cols.keys()]].rename(columns=pass_cols)
        tps_rush = tps_rush[["team", "kind", *rush_cols.keys()]].rename(columns=rush_cols)
        team_summary = (
            tps_pass.merge(tps_rush, on=["team"], how="outer")
                    .drop(columns=[c for c in ["kind_x", "kind_y"] if c in tps_pass.columns or c in tps_rush.columns], errors="ignore")
        )
    else:
        team_summary = pd.DataFrame()

    # Top scorers per team (overall), plus per-kind optional
    def topk(df: pd.DataFrame, k: int = 3) -> str:
        if df is None or df.empty:
            return ""
        s = df.sort_values("td_count", ascending=False).head(k)
        return "; ".join(f"{r.player}:{int(r.td_count)}" for _, r in s.iterrows())

    if pc is not None and not pc.empty:
        agg_any = pc.groupby(["team", "player"]).agg(td_count=("td_count", "sum")).reset_index()
        top_any = (
            agg_any.sort_values(["team", "td_count"], ascending=[True, False])
                   .groupby("team")
                   .apply(lambda g: topk(g, 5))
                   .reset_index(name="top_scorers_any")
        )
        # Optional separate pass/rush
        top_pass = (
            pc[pc["kind"] == "pass"].groupby(["team", "player"]).agg(td_count=("td_count", "sum")).reset_index()
              .sort_values(["team", "td_count"], ascending=[True, False])
              .groupby("team").apply(lambda g: topk(g, 3)).reset_index(name="top_scorers_rec")
        )
        top_rush = (
            pc[pc["kind"] == "rush"].groupby(["team", "player"]).agg(td_count=("td_count", "sum")).reset_index()
              .sort_values(["team", "td_count"], ascending=[True, False])
              .groupby("team").apply(lambda g: topk(g, 3)).reset_index(name="top_scorers_rush")
        )
        score_summary = top_any.merge(top_pass, on="team", how="outer").merge(top_rush, on="team", how="outer")
    else:
        score_summary = pd.DataFrame()

    # Join and write
    if team_summary is not None and not team_summary.empty:
        out = team_summary.merge(score_summary, on="team", how="left") if not score_summary.empty else team_summary
    else:
        out = score_summary

    trends_fp = base_dir / f"td_trends_summary_{start_season}_{end_season}.csv"
    players_fp = base_dir / f"td_trends_players_{start_season}_{end_season}.csv"
    if out is not None and not out.empty:
        out.to_csv(trends_fp, index=False)
        print(f"Wrote team trends summary: {trends_fp}")
        # Show a small sample
        try:
            cols = [c for c in out.columns if c.startswith("pass_") or c.startswith("rush_")]
            print(out[["team", *cols]].head(6).to_string(index=False))
        except Exception:
            pass
    if pc is not None and not pc.empty:
        pc.to_csv(players_fp, index=False)
        print(f"Wrote player TD counts: {players_fp}")


if __name__ == "__main__":
    main()
