# data/build_common.py
"""
build_common.py — Shared helpers for every WNBA charts build script.

Ported from nba_charts/pipeline/build_common.py. Same job: one home for paths,
season keys, scopes, the NaN-safe JSON writer and slugify, so none of them can
drift between build scripts. The NaN rule especially must never diverge — the
browser's JSON.parse rejects bare NaN and the page hangs on "Loading...".

TEAM IDENTITY IS THE HARD PART HERE, much more so than in the NBA:

* `nba_api.stats.static.teams` contains ZERO WNBA teams, so there is no static
  table to fall back on at all.
* `leaguestandings` does not serve league_id="10", so conference can't come
  from there either.
* 20 franchises have appeared across 30 seasons and the league has gone
  8 -> 16 -> 12 -> 15 teams, with five franchises now defunct.
* One franchise id can carry three identities: 1611661319 is the Utah Starzz,
  then the San Antonio Silver Stars, then the Las Vegas Aces.
* Franchise id 1611661327 was the Portland Fire (2000-2002), went dormant, and
  is REUSED for the revived Portland Fire in 2026.
* Phoenix changed abbreviation from PHO to PHX in 2026, so anything keyed on
  abbreviation across seasons breaks.

So identity is derived per season from data rather than hardcoded: name and
abbreviation from that season's play-by-play (always available offline, and
era-correct by construction), conference from LeagueDashTeamStats filtered by
conference (two calls per season, cached to disk).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache import CACHE_DIR, _cache_key, get_game_ids

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_PKG_DIR.parent
SITE_DIR = PROJECT_ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"
STINTS_DIR = PROJECT_ROOT / "data" / "stints_out"
RAPM_DIR = PROJECT_ROOT / "data" / "rapm_out"
TEAM_CACHE_DIR = CACHE_DIR / "teams"


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
FIRST_SEASON = 1997      # the league's first season; PBP exists from here


def current_season() -> str:
    """
    The WNBA season key that is live today.

    The season runs roughly May through October, so a date in the first four
    months of a year still belongs to the previous season — in March 2027 the
    most recent season is still 2026.
    """
    override = os.environ.get("OVERRIDE_SEASON")
    if override:
        return override
    today = date.today()
    return str(today.year if today.month >= 5 else today.year - 1)


def all_seasons(newest_first: bool = True) -> list[str]:
    """
    Every season key, NEWEST FIRST.

    This order drives the season dropdown and decides which season `_headers`
    treats as live (the rest are frozen and cached forever). Unlike the NBA
    build's hand-maintained list, it is generated — the WNBA adds a season every
    year and a stale constant would silently stop building the newest one.
    """
    seasons = [str(y) for y in range(FIRST_SEASON, int(current_season()) + 1)]
    return list(reversed(seasons)) if newest_first else seasons


def built_seasons(newest_first: bool = True) -> list[str]:
    """Seasons that actually have stint output on disk."""
    import glob
    found = set()
    for path in glob.glob(str(STINTS_DIR / "stints_*.parquet")):
        stem = Path(path).stem.replace("stints_", "")
        found.add(stem.split("_")[0])
    seasons = sorted(s for s in found if s.isdigit())
    return list(reversed(seasons)) if newest_first else seasons


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------
#   key      -> (season_type(s), json filename suffix)
SCOPES = {
    "reg":      (["Regular Season"],             ""),
    "playoffs": (["Playoffs"],                   "-playoffs"),
    "all":      (["Regular Season", "Playoffs"], "-all"),
}

# Scopes the Lineup Builder ships raw stint data for. "all" is deliberately
# excluded: it is exactly reg+playoffs concatenated, so it nearly doubles the
# payload to answer a question nobody asks — mixing regular-season and playoff
# possessions into one on/off split isn't a meaningful unit of analysis.
ONOFF_SCOPES = ("reg", "playoffs")


def scope_filename(stem: str, scope: str) -> str:
    """('player-stats', 'playoffs') -> 'player-stats-playoffs.json'"""
    return f"{stem}{SCOPES[scope][1]}.json"


# ---------------------------------------------------------------------------
# Team metadata
# ---------------------------------------------------------------------------
# Abbreviations that changed while the franchise did not. Normalising these
# would ERASE era-correct identity, so they are recorded rather than applied —
# a season map should report the abbreviation actually used that year.
KNOWN_ABBR_CHANGES = {
    1611661317: {"PHO": "PHX"},   # Phoenix Mercury, PHO through 2025, PHX from 2026
}


def _identity_from_pbp(season: str, max_games: int = 40) -> dict[int, dict]:
    """
    team_id -> {name, city, abbr} for one season, straight from cached PBP.

    The play-by-play carries the city, nickname and abbreviation the franchise
    used in that game, so identity comes out era-correct with no table to
    maintain and no network access. A few dozen games is plenty to see every
    team in a 12-16 team league.
    """
    try:
        games = get_game_ids(season)
    except Exception:
        return {}
    if games is None or games.empty:
        return {}

    cols = ["PLAYER1_TEAM_ID", "PLAYER1_TEAM_CITY",
            "PLAYER1_TEAM_NICKNAME", "PLAYER1_TEAM_ABBREVIATION"]
    out: dict[int, dict] = {}
    for gid in games["GAME_ID"].astype(str).str.zfill(10)[:max_games]:
        path = CACHE_DIR / _cache_key("pbp", game_id=gid)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=cols).dropna()
        except Exception:
            continue
        for tid, city, nick, abbr in df.itertuples(index=False):
            tid = int(tid)
            if tid in out:
                continue
            out[tid] = {"city": str(city).strip(),
                        "nickname": str(nick).strip(),
                        "name": f"{str(city).strip()} {str(nick).strip()}",
                        "abbr": str(abbr).strip()}
    return out


def _conferences(season: str, pause: float = 0.7) -> dict[int, str]:
    """
    team_id -> "East"/"West" for one season.

    LeagueDashTeamStats accepts a conference filter, so asking for each
    conference separately reveals the split without any hardcoded table. This is
    the only part of team identity that needs the network, and it is cached.
    """
    from nba_api.stats.endpoints import leaguedashteamstats

    out: dict[int, str] = {}
    for conf in ("East", "West"):
        try:
            time.sleep(pause)
            df = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable="10", season=season,
                conference_nullable=conf, per_mode_detailed="PerGame",
                timeout=60,
            ).get_data_frames()[0]
        except Exception as e:
            print(f"  [warn] {season} {conf} conference lookup failed: "
                  f"{type(e).__name__}: {str(e)[:60]}")
            continue
        for tid in df["TEAM_ID"]:
            out[int(tid)] = conf
    return out


def season_team_map(season: str, refresh: bool = False) -> dict[str, dict]:
    """
    Era-correct abbreviation -> {id, name, city, abbr, conf, slug} for a season.

    Cached to data/cache/teams/{season}.json, so a rebuild costs nothing and
    only the first run touches the network (twice, for the conference split).

    Keyed by the abbreviation the franchise used THAT season, so 1611661319 is
    "UTA" in 1997, "SAN" in 2008 and "LVA" in 2024 — each key resolving to the
    right name for its era. Every franchise appears exactly once, so an
    id -> entry map built off .values() is unambiguous.
    """
    TEAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TEAM_CACHE_DIR / f"{season}.json"
    if cache_path.exists() and not refresh:
        with open(cache_path) as f:
            return json.load(f)

    identity = _identity_from_pbp(season)
    if not identity:
        print(f"  [warn] {season}: no cached PBP, team map will be empty")
        return {}

    conf_by_id = _conferences(season)

    out: dict[str, dict] = {}
    for tid, meta in identity.items():
        out[meta["abbr"]] = {
            "id": tid,
            "name": meta["name"],
            "city": meta["city"],
            "abbr": meta["abbr"],
            "conf": conf_by_id.get(tid),
            "slug": slugify(meta["name"]),
        }

    with open(cache_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def team_map_by_id(season: str, refresh: bool = False) -> dict[int, dict]:
    """Same data as season_team_map, keyed by franchise id."""
    return {v["id"]: v for v in season_team_map(season, refresh=refresh).values()}


def franchise_history(seasons=None) -> pd.DataFrame:
    """
    Every (season, franchise) identity as a table — the audit view.

    Makes relocations and rebrands visible in one place, which is how you catch
    a franchise id quietly changing meaning between two seasons.
    """
    rows = []
    for season in (seasons or built_seasons(newest_first=False)):
        for meta in season_team_map(season).values():
            rows.append({"season": season, **meta})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["id", "season"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None and numpy scalars with Python ones.

    Never remove this: json.dumps emits bare `NaN`/`Infinity`, which are invalid
    JSON — the browser throws a SyntaxError and the page stalls on "Loading...".
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return sanitize_for_json(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    # pandas missing-value scalars (pd.NA / pd.NaT) aren't Python floats, so the
    # isinstance(float) check above misses them and json.dumps(allow_nan=False)
    # raises "NAType is not JSON serializable". Reached only for scalars —
    # dict/list/tuple are recursed above — so pd.isna returns a plain bool here.
    try:
        if obj is not None and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def write_json(data, filename, directory) -> Path:
    """Write compact, NaN-safe JSON and log the size."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with open(path, "w") as f:
        f.write(json.dumps(sanitize_for_json(data), separators=(",", ":"),
                           allow_nan=False))
    print(f"  {filename:<32s} {os.path.getsize(path) / 1024:8.1f} KB")
    return path


def slugify(name: str) -> str:
    """URL-safe slug. MUST stay identical to slugify() in site/js/common.js —
    per-team JSON files are looked up by this slug from the browser."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def rnd(v, d):
    """Round, passing None/NaN through as None."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, d)


def prune_stale(directory, keep: set[str]) -> int:
    """Delete files in `directory` not in `keep`. Renamed franchises leave their
    old slug behind otherwise, and a stale file looks live to the browser."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
            removed += 1
    return removed


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
# SITE_DIR        Path  static site root (served as-is).
# SITE_DATA_DIR   Path  site/data — parent of per-season JSON dirs.
# STINTS_DIR      Path  data/stints_out — per-season stint parquet.
# RAPM_DIR        Path  data/rapm_out — per-season RAPM csv.
# TEAM_CACHE_DIR  Path  data/cache/teams — cached per-season team maps.
# SCOPES          dict  scope key -> (season_type list, filename suffix).
# ONOFF_SCOPES    tuple scopes that ship raw stint data ("all" excluded).
