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
import unicodedata
from pathlib import Path

import pandas as pd

from data.build_common import merge_rows
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


def _fold(name: str) -> str:
    """A name with its accents removed, for comparing two spellings of one."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(name))
                   if not unicodedata.combining(c))


def _prefer(old: str, new: str) -> str:
    """
    Which of two spellings of one player id to keep.

    The feed is inconsistent about diacritics from season to season — the same
    id is "Dorka Juhász" in one year and "Dorka Juhasz" in another — so plain
    last-wins quietly strips the accents off a name because of which season
    happened to be processed last. That is a loss of information nobody chose.

    When two spellings differ ONLY by accents, keep the accented one. When they
    differ for a real reason — Megan Gustafson married and the feed now says
    Megan DiLeo — the newer one wins, which is the behaviour that was already
    there and is a genuine editorial choice, not an accident.
    """
    if old == new:
        return new
    if _fold(old) == _fold(new):
        return old if _fold(old) != old else new
    return new


def _merge_names(path: Path, fresh: dict[int, str]) -> dict[int, str]:
    """Union of every player ever seen, across every season ever built."""
    names: dict[int, str] = {}
    if path.exists():
        try:
            prior = pd.read_csv(path)
            names = dict(zip(prior["PERSON_ID"].astype(int), prior["PLAYER_NAME"]))
        except Exception as e:
            print(f"  [warn] {path.name} unreadable ({type(e).__name__}); "
                  f"rewriting from this run only")
    for pid, name in fresh.items():
        names[pid] = _prefer(names[pid], name) if pid in names else name
    pd.DataFrame([{"PERSON_ID": k, "PLAYER_NAME": v}
                  for k, v in sorted(names.items())]).to_csv(path, index=False)
    return names


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
            # _prefer, not dict.update: plain last-wins here would collapse the
            # two spellings before _merge_names ever sees them, so the accent
            # rule below would have nothing left to choose between.
            for pid, name in info.pop("names").items():
                all_names[pid] = (_prefer(all_names[pid], name)
                                  if pid in all_names else name)
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

    # ── League-wide tables ───────────────────────────────────────────────────
    # These two describe every season, but a run may have built only one. Written
    # plainly they were overwritten with whatever the run happened to cover, so a
    # nightly `--season 2026` reduced the 1997-2026 name table to the 401 players
    # who appeared in 2026 and the build summary to a single row. Nothing
    # downstream noticed, because the historical RAPM CSVs already had their
    # names baked in — until the day someone rebuilt a past season and got a
    # table full of blanks.
    #
    # So: merge, never replace. Rows for what this run built win; rows for
    # seasons it did not touch survive.
    s = merge_rows(out_dir / "build_summary.csv", pd.DataFrame(summary),
                   key=["season", "season_type"])
    _merge_names(out_dir / "player_names.csv", all_names)

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
