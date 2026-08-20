# data/build_rapm.py
"""
build_rapm.py — Rebuild on/off and RAPM for every WNBA season from the stint
history in data/stints_out/.

The ridge penalty is tuned ONCE by cross-validation on a single reference
season and then held fixed across all seasons, so results are comparable
year to year: a player's RAPM should differ from another season's because the
basketball differed, not because the regularization did.

Usage:
    python -m data.build_rapm --tune-only            # just report the alpha
    python -m data.build_rapm --alpha 4000           # skip tuning, use a value
    python -m data.build_rapm                        # tune, then run all seasons
    python -m data.build_rapm --season 2024 --season 2025
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from data.rapm import compute_rapm, find_best_alpha

STINTS_DIR = Path("data/stints_out")

# Season the ridge penalty is tuned on. A recent, complete, fully verified
# season of typical length — tuning on all 30 at once would pick a penalty
# suited to a 188k-stint sample and under-regularize every individual season.
REFERENCE_SEASON = "2024"


def load_stints(season: str | None = None,
                season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Read stint parquet back into the shape compute_rapm expects.

    Parquet stores the lineups as arrays; the design matrix does set algebra on
    them (`home_lineup | away_lineup`), so they have to become frozensets again.
    """
    pattern = (f"stints_{season}_{season_type.replace(' ', '_')}.parquet"
               if season else f"stints_*_{season_type.replace(' ', '_')}.parquet")
    paths = sorted(glob.glob(str(STINTS_DIR / pattern)))
    if not paths:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for col in ("home_lineup", "away_lineup"):
        df[col] = df[col].map(lambda a: frozenset(int(x) for x in a))
    return df


def load_names() -> dict[int, str]:
    path = STINTS_DIR / "player_names.csv"
    if not path.exists():
        return {}
    names = pd.read_csv(path)
    return dict(zip(names["PERSON_ID"].astype(int), names["PLAYER_NAME"]))


def available_seasons(season_type: str = "Regular Season") -> list[str]:
    suffix = season_type.replace(" ", "_")
    seasons = []
    for p in sorted(glob.glob(str(STINTS_DIR / f"stints_*_{suffix}.parquet"))):
        stem = Path(p).stem.replace("stints_", "").replace(f"_{suffix}", "")
        seasons.append(stem)
    return sorted(seasons)


def tune(season: str, season_type: str, cv: int, min_stint_poss: float):
    stints = load_stints(season, season_type)
    if stints.empty:
        raise SystemExit(f"No stints for {season} {season_type}")
    print(f"Tuning ridge alpha on {season} {season_type} "
          f"({len(stints):,} stints)\n")
    best, scores = find_best_alpha(
        stints, cv=cv, min_stint_poss=min_stint_poss, verbose=True
    )

    # A flat curve means the choice barely matters; a sharp one means it does.
    lo, hi = scores["cv_rmse"].min(), scores["cv_rmse"].max()
    print(f"\nCV RMSE range across grid: {lo:.4f} - {hi:.4f} "
          f"({100 * (hi - lo) / lo:.2f}% spread)")
    return best, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons")
    ap.add_argument("--season-type", default="Regular Season")
    ap.add_argument("--alpha", type=float, default=None,
                    help="skip tuning and use this alpha")
    ap.add_argument("--tune-only", action="store_true")
    ap.add_argument("--tune-season", default=REFERENCE_SEASON)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--min-stint-poss", type=float, default=0.5)
    ap.add_argument("--out", default="data/rapm_out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Alpha ────────────────────────────────────────────────────────────────
    if args.alpha is not None:
        alpha = args.alpha
        print(f"Using supplied alpha={alpha:g}\n")
    else:
        alpha, scores = tune(args.tune_season, args.season_type,
                             args.cv, args.min_stint_poss)
        scores.to_csv(out_dir / "alpha_cv_scores.csv", index=False)
        print(f"\nSelected alpha={alpha:g} "
              f"(tuned on {args.tune_season}, applied to every season)\n")

    if args.tune_only:
        return

    # ── Per-season RAPM ──────────────────────────────────────────────────────
    seasons = args.seasons or available_seasons(args.season_type)
    names = load_names()
    summary = []

    for season in seasons:
        stints = load_stints(season, args.season_type)
        if stints.empty:
            continue
        print("=" * 66)
        print(f"{season} {args.season_type}")
        print("=" * 66)
        try:
            res = compute_rapm(
                season,
                season_type=args.season_type,
                ridge_alpha=alpha,
                min_stint_poss=args.min_stint_poss,
                stints_df=stints,
                name_lookup=names,
            )
        except Exception as e:
            print(f"  [warn] {season} failed: {type(e).__name__}: {e}")
            continue

        res.insert(0, "SEASON", season)
        path = out_dir / f"rapm_{season}_{args.season_type.replace(' ', '_')}.csv"
        res.to_csv(path, index=False)
        summary.append({"season": season, "players": len(res),
                        "stints": len(stints), "alpha": alpha})
        print(f"  -> {len(res)} players written to {path.name}\n")

    if not summary:
        print("nothing computed")
        return

    s = pd.DataFrame(summary)
    s.to_csv(out_dir / "rapm_summary.csv", index=False)

    combined = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(out_dir / "rapm_*_*.csv")))
         if "summary" not in p and "alpha" not in p],
        ignore_index=True,
    )
    combined.to_csv(out_dir / "rapm_all_seasons.csv", index=False)

    print("=" * 66)
    print("RAPM BUILD SUMMARY")
    print("=" * 66)
    print(f"  alpha            : {alpha:g} (tuned on {args.tune_season})")
    print(f"  seasons          : {s['season'].min()}-{s['season'].max()} "
          f"({len(s)})")
    print(f"  player-seasons   : {s['players'].sum():,}")
    print(f"  stints used      : {s['stints'].sum():,}")
    print(f"  written to       : {out_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()
