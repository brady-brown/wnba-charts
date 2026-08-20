# data/build_all.py
"""
build_all.py — Build the full WNBA stint history from play-by-play.

One API call per game (PlayByPlayV2) and nothing else. GameRotation is no
longer part of the pipeline; `data.pbp_rotation` reproduces its on-court sets
from the PBP, verified second-by-second (`python -m data.verify_rotation`).

Everything is cached to data/cache/, so a re-run costs no API calls and a run
interrupted partway resumes where it stopped.

Usage:
    python -m data.build_all                      # every season, 1997-present
    python -m data.build_all --season 2024 --season 2025
    python -m data.build_all --fetch-only         # warm the PBP cache, no build
    python -m data.build_all --out data/stints    # parquet output directory
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from data.cache import CACHE_DIR, _cache_key, get_game_ids, get_pbp
from data.stints import _game_stints

# WNBA play-by-play is available from the league's first season.
FIRST_SEASON = 1997


def all_seasons(through: int | None = None) -> list[str]:
    through = through or pd.Timestamp.today().year
    return [str(y) for y in range(FIRST_SEASON, through + 1)]


def _pbp_cached(game_id: str) -> bool:
    return (CACHE_DIR / _cache_key("pbp", game_id=game_id)).exists()


def season_game_ids(season: str, season_type: str) -> list[str]:
    """
    The schedule for one season and type, or [] if the league played none.

    [] means the league answered and had nothing to report — 1997 had no
    play-in, a season in progress has no playoffs yet. It must never mean the
    request failed: an empty schedule builds an empty season, writes no
    parquet, and reports success, which is indistinguishable from a healthy
    quiet night. data.cache.ApiUnavailable is deliberately not an Exception,
    so the handler below cannot absorb one.
    """
    try:
        df = get_game_ids(season, season_type=season_type)
    except Exception as e:
        print(f"  [warn] {season} {season_type}: game ids unusable "
              f"({type(e).__name__}: {str(e)[:70]})")
        return []
    if df.empty:
        return []
    return df["GAME_ID"].astype(str).str.zfill(10).tolist()


def fetch_season(season: str, season_type: str, pause: float = 0.0) -> dict:
    """Warm the PBP cache for one season. Returns per-season counts."""
    gids = season_game_ids(season, season_type)
    missing = [g for g in gids if not _pbp_cached(g)]
    stats = {"games": len(gids), "already_cached": len(gids) - len(missing),
             "fetched": 0, "failed": 0}
    if not missing:
        return stats

    print(f"  {season} {season_type}: {len(missing)} of {len(gids)} games to fetch")
    for i, gid in enumerate(missing, 1):
        try:
            df = get_pbp(gid)
            if df is None or df.empty:
                stats["failed"] += 1
            else:
                stats["fetched"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"    [warn] {gid}: {type(e).__name__}: {str(e)[:70]}")
        if pause:
            time.sleep(pause)
        if i % 50 == 0:
            print(f"    ... {i}/{len(missing)}")
    return stats


def build_season(season: str, season_type: str) -> tuple[pd.DataFrame, dict]:
    """Build stints for one season from whatever PBP is cached or fetchable."""
    gids = season_game_ids(season, season_type)
    rows: list[dict] = []
    names: dict[int, str] = {}
    n_ok = 0

    for gid in gids:
        stints, game_names = _game_stints(gid)
        if stints:
            n_ok += 1
            rows.extend(stints)
            names.update(game_names)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["season"] = season
        df["season_type"] = season_type
    return df, {"games": len(gids), "games_with_stints": n_ok,
                "stints": len(df), "names": names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons",
                    help="repeatable; default is every season from 1997")
    ap.add_argument("--season-type", action="append", dest="season_types",
                    default=None, help="default: Regular Season and Playoffs")
    ap.add_argument("--fetch-only", action="store_true",
                    help="warm the PBP cache without building stints")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="build only from games already cached")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="extra seconds between API calls")
    ap.add_argument("--out", type=str, default="data/stints_out",
                    help="directory for per-season parquet output")
    args = ap.parse_args()

    seasons = args.seasons or all_seasons()
    season_types = args.season_types or ["Regular Season", "Playoffs"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seasons : {seasons[0]}-{seasons[-1]} ({len(seasons)})")
    print(f"Types   : {', '.join(season_types)}")
    print(f"Output  : {out_dir}\n")

    summary: list[dict] = []
    all_names: dict[int, str] = {}

    for season in seasons:
        for season_type in season_types:
            label = f"{season} {season_type}"

            if not args.skip_fetch:
                fs = fetch_season(season, season_type, pause=args.pause)
                if fs["games"] == 0:
                    continue
                if fs["fetched"] or fs["failed"]:
                    print(f"  {label}: fetched {fs['fetched']}, "
                          f"failed {fs['failed']}, "
                          f"cached {fs['already_cached']}")

            if args.fetch_only:
                continue

            df, info = build_season(season, season_type)
            if info["games"] == 0:
                continue
            if not df.empty:
                path = out_dir / f"stints_{season}_{season_type.replace(' ', '_')}.parquet"
                df.drop(columns=[]).assign(
                    home_lineup=df["home_lineup"].map(lambda s: sorted(s)),
                    away_lineup=df["away_lineup"].map(lambda s: sorted(s)),
                ).to_parquet(path, index=False)
            all_names.update(info.pop("names"))
            summary.append({"season": season, "season_type": season_type, **info})
            print(f"  {label}: {info['stints']:>6,} stints from "
                  f"{info['games_with_stints']}/{info['games']} games")

    if args.fetch_only:
        print("\nfetch complete")
        return

    if not summary:
        # Silence here used to be a successful exit. For an unattended run
        # asking for one named season that is the worst possible answer: the
        # season exists, the league played, and the build produced nothing —
        # exactly what a dead API looks like from the outside. An explicit
        # --season that yields nothing is a failure; a bare full-history run
        # finding nothing new is not.
        print("\nnothing built")
        if args.seasons:
            raise SystemExit(
                f"nothing built for {', '.join(args.seasons)} — the seasons were "
                f"named explicitly, so an empty build is a failure, not a quiet night")
        return

    s = pd.DataFrame(summary)
    s.to_csv(out_dir / "build_summary.csv", index=False)
    pd.DataFrame(
        [{"PERSON_ID": k, "PLAYER_NAME": v} for k, v in sorted(all_names.items())]
    ).to_csv(out_dir / "player_names.csv", index=False)

    print("\n" + "=" * 64)
    print("BUILD SUMMARY")
    print("=" * 64)
    for st in s["season_type"].unique():
        sub = s[s["season_type"] == st]
        print(f"\n{st}")
        print(f"  seasons          : {sub['season'].min()}-{sub['season'].max()}")
        print(f"  games with stints: {sub['games_with_stints'].sum():,} / "
              f"{sub['games'].sum():,}")
        print(f"  stints           : {sub['stints'].sum():,}")
    print(f"\nplayers          : {len(all_names):,}")
    print(f"written to       : {out_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
