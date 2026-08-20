# data/refresh_live.py
"""
refresh_live.py — Invalidate the cache entries that go stale while a season is
still being played.

data/cache.py is a write-once cache: if a key is on disk it is served forever
and the endpoint is never called again. That is exactly right for 1997-2025,
where the answer cannot change, and exactly wrong for the season in progress.
Without this module a nightly run would re-derive the same finished games every
night and never see a new one, which looks identical to a healthy build from
the outside — same JSON, no diff, no deploy, no error.

Four things go stale, and only these four:

* **game_ids** — the season's schedule-to-date. This is the one that actually
  hides new games; everything downstream iterates the list it returns.
* **player_stats** — the LeagueDashPlayerStats benchmark feed merged into the
  RAPM table for per-game and per-100 columns.
* **shot_chart** — season-to-date shot coordinates behind the shot charts.
* **partial PBP** — a game fetched while it was still being played is cached
  as a half-finished game and, being cached, is never corrected. Detected by
  structure rather than by timestamp: a finished game reaches period 4 (or
  higher, with overtime) and closes it with an end-of-period event.

Team identity (data/cache/teams/{season}.json) is deliberately NOT invalidated.
Rosters move, franchises do not, and a mid-season re-derive would only spend two
API calls to write the same file back.

Usage:
    python -m data.refresh_live                  # current calendar year
    python -m data.refresh_live --season 2026
    python -m data.refresh_live --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.cache import CACHE_DIR, _cache_key

SEASON_TYPES = ("Regular Season", "Playoffs")

# Every (per_mode, measure_type) pair the project asks LeagueDashPlayerStats
# for. PerGame/Per100Possessions feed the RAPM table; Totals feeds
# data/verify_box.py. Listing them beats globbing because cache keys are
# md5 digests of the parameter set — the filename says nothing about the season.
PLAYER_FEED_MODES = (
    ("PerGame", "Base"),
    ("Per100Possessions", "Base"),
    ("Totals", "Base"),
    ("PerGame", "Advanced"),
    ("Totals", "Advanced"),
)


def _volatile_keys(season: str) -> list[str]:
    """Cache keys whose answer changes as long as the season is being played."""
    keys = []
    for season_type in SEASON_TYPES:
        keys.append(_cache_key("game_ids", season=season, season_type=season_type))
        keys.append(_cache_key("shot_chart", season=season, player_id=0, team_id=0,
                               season_type=season_type))
        for per_mode, measure_type in PLAYER_FEED_MODES:
            keys.append(_cache_key("player_stats", season=season,
                                   season_type=season_type,
                                   per_mode=per_mode, measure_type=measure_type))
    return keys


def _season_game_ids(season: str) -> list[str]:
    """Game ids from the cached schedule, read before it is invalidated."""
    gids: list[str] = []
    for season_type in SEASON_TYPES:
        path = CACHE_DIR / _cache_key("game_ids", season=season,
                                      season_type=season_type)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty and "GAME_ID" in df.columns:
            gids.extend(df["GAME_ID"].astype(str).str.zfill(10).tolist())
    return gids


def _is_complete(path: Path) -> bool:
    """
    Does this cached play-by-play cover a finished game?

    A finished game reaches regulation (period 4) and closes its final period
    with an end-of-period event (EVENTMSGTYPE 13). A game captured at halftime
    satisfies neither, and nothing else in the pipeline would ever notice.
    """
    try:
        df = pd.read_csv(path, usecols=["EVENTMSGTYPE", "PERIOD"])
    except Exception:
        return False
    if df.empty:
        return False
    last_period = int(df["PERIOD"].max())
    if last_period < 4:
        return False
    return bool((df.loc[df["PERIOD"] == last_period, "EVENTMSGTYPE"] == 13).any())


def partial_games(season: str) -> list[str]:
    """Game ids whose cached PBP stops before the final buzzer."""
    return [gid for gid in _season_game_ids(season)
            if (CACHE_DIR / _cache_key("pbp", game_id=gid)).exists()
            and not _is_complete(CACHE_DIR / _cache_key("pbp", game_id=gid))]


def refresh(season: str, dry_run: bool = False) -> dict:
    """Drop the stale entries for one season. Returns what was removed."""
    partials = partial_games(season)
    stale = _volatile_keys(season) + [_cache_key("pbp", game_id=g) for g in partials]

    removed = []
    for key in stale:
        path = CACHE_DIR / key
        if not path.exists():
            continue
        removed.append(key)
        if not dry_run:
            path.unlink()

    return {"season": season, "removed": len(removed),
            "partial_games": partials, "dry_run": dry_run}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=str(pd.Timestamp.today().year),
                    help="season being played; default is the current year")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be dropped, delete nothing")
    args = ap.parse_args()

    info = refresh(args.season, dry_run=args.dry_run)
    verb = "would drop" if args.dry_run else "dropped"
    print(f"{args.season}: {verb} {info['removed']} stale cache entries")
    if info["partial_games"]:
        print(f"  partial games re-queued: {', '.join(info['partial_games'])}")


if __name__ == "__main__":
    main()
