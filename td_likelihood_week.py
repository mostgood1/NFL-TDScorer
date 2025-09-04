from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from touchdown.src.td_likelihood import compute_td_likelihood


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Compute team TD likelihoods from local touchdown package and ./data inputs.')
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    ap.add_argument('--out', type=str, default=None, help='Optional output CSV path; defaults to ./data/td_likelihood_<season>_wk<week>.csv')
    # no external repo dependency
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    # imports resolved statically from local package

    df = compute_td_likelihood(args.season, args.week)
    if df is None or df.empty:
        print('No results produced.')
        return

    out_fp = Path(args.out) if args.out else (here / 'data' / f'td_likelihood_{args.season}_wk{args.week}.csv')
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f'Wrote team TD likelihoods to {out_fp}')
    try:
        preview_cols = [c for c in ['team','opponent','td_likelihood','td_score','implied_points','expected_tds'] if c in df.columns]
        print(df.head(12)[preview_cols].to_string(index=False))
    except Exception:
        pass


if __name__ == '__main__':
    main()
