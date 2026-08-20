# data/build_advanced.py
"""
build_advanced.py — Materialise PBP-derived box and advanced rates per season.

data/advanced.py replays every game's play-by-play and joins it to the stint
lineups, which costs ~10s a season. The site build runs over 30 seasons and
three scopes, so the result is cached here as parquet and read back by
build_site.py, the same way rapm_out/ caches the ridge.

Usage:
    python -m data.build_advanced                     # every season, every scope
    python -m data.build_advanced --season 2026
    python -m data.build_advanced --season 2024 --scope reg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.advanced import build_season
from data.build_common import PROJECT_ROOT, built_seasons

ADVANCED_DIR = PROJECT_ROOT / "data" / "advanced_out"

# Site scope key -> the season_type build_season() understands. "all" is
# reg + playoffs merged at the event level, not two rate tables averaged.
SCOPE_TYPES = {
    "reg": "Regular Season",
    "playoffs": "Playoffs",
    "all": "All",
}


def path_for(season: str, scope: str) -> Path:
    return ADVANCED_DIR / f"advanced_{season}_{scope}.parquet"


def load(season: str, scope: str = "reg") -> pd.DataFrame | None:
    """Read a cached table, or None if it has not been built."""
    p = path_for(season, scope)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def build(season: str, scope: str = "reg", progress: bool = False) -> int:
    df = build_season(season, SCOPE_TYPES[scope], progress=progress)
    if df is None or df.empty:
        print(f"  {season} {scope:<9}: no stints")
        return 0
    ADVANCED_DIR.mkdir(parents=True, exist_ok=True)
    out = path_for(season, scope)
    df.to_parquet(out, index=False)
    print(f"  {season} {scope:<9}: {len(df):>3} players -> {out.name}")
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons")
    ap.add_argument("--scope", action="append", dest="scopes",
                    choices=sorted(SCOPE_TYPES))
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave already-built tables alone (resume a long run)")
    args = ap.parse_args()

    seasons = args.seasons or built_seasons(newest_first=False)
    scopes = args.scopes or ["reg", "playoffs", "all"]
    for season in seasons:
        for scope in scopes:
            if args.skip_existing and path_for(season, scope).exists():
                print(f"  {season} {scope:<9}: cached")
                continue
            build(season, scope, progress=args.progress)


if __name__ == "__main__":
    main()
