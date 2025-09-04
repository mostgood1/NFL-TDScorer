from __future__ import annotations

import argparse
from pathlib import Path
from touchdown.src.player_td_likelihood import compute_player_td_likelihood


def _ensure_import(_: Path) -> None:
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Compute player anytime TD likelihoods from local model and usage priors in ./data.')
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    ap.add_argument('--out', type=str, default=None, help='Optional output CSV path; defaults to ./data/player_td_likelihood_<season>_wk<week>.csv')
    # no external repo dependency
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    # imports resolved statically from local package

    df = compute_player_td_likelihood(args.season, args.week)
    if df is None or df.empty:
        print('No player anytime TD results produced.')
        return

    out_fp = Path(args.out) if args.out else (here / 'data' / f'player_td_likelihood_{args.season}_wk{args.week}.csv')
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f'Wrote player anytime TD likelihoods to {out_fp}')
    try:
        print(df.groupby(['team','position']).head(1)[['team','player','position','anytime_td_prob','expected_td']].to_string(index=False))
    except Exception:
        pass


if __name__ == '__main__':
    main()
