# data/build_site.py
"""
build_site.py — Export WNBA artifacts to the static JSON the site fetches.

Stage 1 (upstream, hits the API and caches): data/stints.py, data/rapm.py,
data/shots.py. Stage 2 is this script: read what those produced, shape it,
write JSON. Nothing here touches the network except the cached team map.

Following nba_charts/pipeline/build_site.py:
  * one shared build_common (season keys / slugify / NaN-safe write_json)
  * per-scope data files
  * build stamp = latest GAME DATE, not wall clock, so a run with no new games
    rewrites byte-identical JSON -> no git diff -> no deploy. Load-bearing;
    never stamp anything with now().

Outputs:
    seasons.json                         season index for the dropdown
    {season}/player-stats.json           Player Stats table (box + on/off)
    {season}/player-impact.json          RAPM, merged into the table by id

Usage:
    python -m data.build_site
    python -m data.build_site --season 2024 --season 2025
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from data.build_common import (RAPM_DIR, SCOPES, SITE_DATA_DIR, SITE_DIR,
                               STINTS_DIR, built_seasons, rnd, scope_filename,
                               season_team_map, write_json)
from data.build_advanced import load as load_advanced
from data.cache import CACHE_DIR, _cache_key, get_game_ids

# Applies to player-impact ONLY. A ridge coefficient below this many possessions
# is mostly prior, so it is not worth publishing. The stats table itself ships
# every player — box and on/off are plain aggregates that stay honest at any
# sample size, and a playoff run is short enough that a floor would empty it.
MIN_POSS_EMIT = 100


# ---------------------------------------------------------------------------
# Build stamp
# ---------------------------------------------------------------------------
def _data_freshness(season: str) -> str:
    """
    ISO date of the most recent game in a season — the build stamp.

    Using the data date rather than wall clock keeps the JSON byte-identical
    when no new games have arrived, so a nightly run with nothing new produces
    no git diff and triggers no deploy.
    """
    latest = None
    for season_type in ("Regular Season", "Playoffs"):
        path = CACHE_DIR / _cache_key("game_ids", season=season,
                                      season_type=season_type)
        if not path.exists():
            continue
        try:
            dates = pd.to_datetime(pd.read_csv(path)["GAME_DATE"], errors="coerce")
        except Exception:
            continue
        if dates.notna().any():
            d = dates.max()
            if latest is None or d > latest:
                latest = d
    return latest.date().isoformat() if latest is not None else f"{season}-05-01"


def _feed_alignment_warning(season: str, df: pd.DataFrame) -> str | None:
    """
    Warn when the box feed and the stint feed disagree on how much season exists.

    These are two independent pulls — GP_PG/MIN_PG/PTS_PG come from
    LeagueDashPlayerStats, while POSS and every on/off and RAPM column come from
    the cached play-by-play. Refresh one and not the other and they silently
    merge into a single row: 2026 shipped for three months with 35 games of box
    score next to 5.7 games of possessions, and nothing in the build complained.

    Games are counted two ways and compared. The box side is
    max(GP_PG) * n_teams / 2; the stint side is the distinct game_id count in
    the season's stint parquet — deliberately that file and not the game_ids
    cache, because the parquet is what POSS is actually summed from. Refreshing
    game_ids without re-running build_all leaves the parquet short, and a check
    against game_ids would call that healthy. A short playoff or an in-progress
    season keeps both sides equally short, so the ratio stays near 1 — it only
    drops when the two feeds are from different dates.
    """
    if "GP_PG" not in df.columns:
        return None
    gp = pd.to_numeric(df["GP_PG"], errors="coerce").max()
    n_teams = df["TEAM_ABBREVIATION"].nunique() if "TEAM_ABBREVIATION" in df.columns else 0
    if not gp or gp <= 0 or n_teams < 2:
        return None
    box_games = gp * n_teams / 2

    path = STINTS_DIR / f"stints_{season}_Regular_Season.parquet"
    if not path.exists():
        return None
    try:
        stint_games = pd.read_parquet(path, columns=["game_id"])["game_id"].nunique()
    except Exception:
        return None
    if not stint_games:
        return None

    ratio = stint_games / box_games
    if ratio >= 0.9:
        return None
    return (f"  [STALE] {season}: box feed implies ~{box_games:.0f} games "
            f"(max GP {gp:.0f} x {n_teams} teams / 2) but the stint parquet "
            f"has {stint_games}. POSS, on/off and RAPM are built on "
            f"{ratio * 100:.0f}% of the games the box stats cover. "
            f"Re-pull with get_game_ids(..., force_refresh=True) then "
            f"`python -m data.build_all --season {season}`.")


def _int(v):
    """int() that passes None/NaN through — box feeds are sparse in old seasons."""
    if v is None or (isinstance(v, float) and v != v) or pd.isna(v):
        return None
    return int(v)


def _meta(season: str, **extra) -> dict:
    return {"season": season, "built_at": _data_freshness(season), **extra}


# ---------------------------------------------------------------------------
# Player Stats — {season}/player-stats.json
# ---------------------------------------------------------------------------
def _rapm_frame(season: str) -> pd.DataFrame | None:
    """Regular-season RAPM output — the only scope the ridge is solved for."""
    path = RAPM_DIR / f"rapm_{season}_Regular_Season.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _scope_frame(season: str, scope: str) -> pd.DataFrame | None:
    """
    The table behind one scope.

    'reg' reuses the RAPM CSV, which already carries on/off and box stats
    alongside the ridge output. 'playoffs' and 'all' come from
    data/build_scopes.py, which computes the same splits WITHOUT the ridge —
    a WNBA playoff run is too few possessions for RAPM to mean anything.
    """
    if scope == "reg":
        return _rapm_frame(season)
    path = RAPM_DIR / f"onoff_{season}_{scope}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_player_stats(season: str, scope: str = "reg") -> int:
    """
    Box + on/off table for one season.

    RAPM is deliberately NOT in this file — it goes to player-impact.json, which
    the page merges in lazily by player id so the table paints without waiting
    on it. Same split nba_charts uses.
    """
    df = _scope_frame(season, scope)
    if df is None or df.empty:
        print(f"  [skip] player-stats ({scope}): nothing built for {season}")
        return 0

    if scope == "reg":
        warn = _feed_alignment_warning(season, df)
        if warn:
            print(warn)

    teams = season_team_map(season)
    df = df.copy()

    # Box score and every rate come from the play-by-play (data/advanced.py),
    # not from the stats API: the on-floor denominators behind USG%, AST%, REB%,
    # STL% and BLK% only exist because we know the lineup for every second, and
    # mixing a league-sourced numerator with our own denominator would produce
    # rates that reconcile against nothing. The API feeds are the benchmark
    # instead — see data/verify_box.py. Falls back to the RAPM frame's
    # API-derived columns when a season has no advanced table built yet.
    adv = load_advanced(season, scope)
    A = {}
    if adv is not None and not adv.empty:
        A = {int(p): row for p, row in zip(adv["player_id"],
                                           adv.to_dict("records"))}
    else:
        # Say so rather than shipping a half-API, half-PBP row. Silently mixing
        # two feeds is exactly how 2026 came to carry 35 games of box score
        # beside six games of possessions.
        print(f"  [NO ADVANCED] {season} {scope}: run "
              f"`python -m data.build_advanced --season {season}` — shooting and "
              f"advanced rates will be blank and the box falls back to the API feed.")

    def av(pid, key, default=None):
        v = A.get(pid, {}).get(key, default)
        return None if v is None or (isinstance(v, float) and v != v) else v

    records = []
    for r in df.itertuples(index=False):
        abbr = str(getattr(r, "TEAM_ABBREVIATION", "") or "")
        tm = teams.get(abbr, {})
        pid = int(r.PLAYER_ID)
        records.append({
            "id": pid,
            "n": r.PLAYER_NAME,
            "tid": tm.get("id"),
            "t": tm.get("name", abbr),
            "ta": abbr,
            "conf": tm.get("conf"),
            "slug": tm.get("slug"),
            "gp": _int(av(pid, "gp", getattr(r, "GP_PG", None))),
            "min": rnd(av(pid, "min_pg", getattr(r, "MIN_PG", None)), 1),
            "poss": int(round(r.POSS)),
            "usg": rnd(av(pid, "usg", getattr(r, "USG_PCT", None)), 1),
            # Per game
            "pts": rnd(av(pid, "pts_pg", getattr(r, "PTS_PG", None)), 1),
            "reb": rnd(av(pid, "reb_pg", getattr(r, "REB_PG", None)), 1),
            "oreb": rnd(av(pid, "oreb_pg"), 1), "dreb": rnd(av(pid, "dreb_pg"), 1),
            "ast": rnd(av(pid, "ast_pg", getattr(r, "AST_PG", None)), 1),
            "stl": rnd(av(pid, "stl_pg", getattr(r, "STL_PG", None)), 1),
            "blk": rnd(av(pid, "blk_pg", getattr(r, "BLK_PG", None)), 1),
            "tov": rnd(av(pid, "tov_pg", getattr(r, "TOV_PG", None)), 1),
            "pf": rnd(av(pid, "pf_pg"), 1),
            # Shooting — percentages computed from season totals, not averaged
            "fgm": rnd(av(pid, "fgm_pg"), 1),
            "fga": rnd(av(pid, "fga_pg", getattr(r, "FGA_PG", None)), 1),
            "fg3m": rnd(av(pid, "fg3m_pg"), 1), "fg3a": rnd(av(pid, "fg3a_pg"), 1),
            "ftm": rnd(av(pid, "ftm_pg"), 1),
            "fta": rnd(av(pid, "fta_pg", getattr(r, "FTA_PG", None)), 1),
            "fg": rnd(av(pid, "fg_pct", getattr(r, "FG_PCT_PG", None)), 1),
            "fg3": rnd(av(pid, "fg3_pct", getattr(r, "FG3_PCT_PG", None)), 1),
            "ft": rnd(av(pid, "ft_pct"), 1),
            "efg": rnd(av(pid, "efg"), 1), "ts": rnd(av(pid, "ts"), 1),
            "par3": rnd(av(pid, "par3"), 1), "ftr": rnd(av(pid, "ftr"), 1),
            # Advanced rates — exact on-floor denominators from the stints
            "astp": rnd(av(pid, "astp"), 1), "orbp": rnd(av(pid, "orbp"), 1),
            "drbp": rnd(av(pid, "drbp"), 1), "trbp": rnd(av(pid, "trbp"), 1),
            "stlp": rnd(av(pid, "stlp"), 1), "blkp": rnd(av(pid, "blkp"), 1),
            "tovp": rnd(av(pid, "tovp"), 1),
            # Per 100 possessions. Offensive production is divided by offensive
            # possessions and defensive by defensive — the stint engine counts
            # them separately and they are not equal.
            "pts100": rnd(av(pid, "pts_100", getattr(r, "PTS_100", None)), 1),
            "reb100": rnd(av(pid, "reb_100", getattr(r, "REB_100", None)), 1),
            "ast100": rnd(av(pid, "ast_100", getattr(r, "AST_100", None)), 1),
            "stl100": rnd(av(pid, "stl_100", getattr(r, "STL_100", None)), 1),
            "blk100": rnd(av(pid, "blk_100", getattr(r, "BLK_100", None)), 1),
            "tov100": rnd(av(pid, "tov_100", getattr(r, "TOV_100", None)), 1),
            "stocks100": rnd(av(pid, "stocks_100", getattr(r, "STOCKS_100", None)), 1),
            # Raw on/off — the unregularized inputs RAPM is fit on
            "on_o": rnd(r.ON_ORTG, 1), "on_d": rnd(r.ON_DRTG, 1), "on": rnd(r.ON_NET, 1),
            "off_o": rnd(r.OFF_ORTG, 1), "off_d": rnd(r.OFF_DRTG, 1),
            "off": rnd(r.OFF_NET, 1), "onoff": rnd(r.ON_OFF, 1),
        })

    records.sort(key=lambda x: (x["pts"] is None, -(x["pts"] or 0)))
    write_json({"players": records,
                "meta": _meta(season, scope=scope, n_players=len(records))},
               scope_filename("player-stats", scope), SITE_DATA_DIR / season)
    return len(records)


def build_impact(season: str) -> int:
    """
    RAPM for one season, fetched lazily and merged into the table by id.

    Regular season only, and shown on every scope with that stated on the page —
    the same call nba_charts makes. Playoff samples are far too small for a
    ridge fit to describe the player rather than the prior.
    """
    df = _rapm_frame(season)
    if df is None or df.empty:
        print(f"  [skip] player-impact: no RAPM output for {season}")
        return 0

    teams = season_team_map(season)
    df = df[df["POSS"] >= MIN_POSS_EMIT].sort_values("RAPM", ascending=False)

    records = []
    for r in df.itertuples(index=False):
        abbr = str(getattr(r, "TEAM_ABBREVIATION", "") or "")
        tm = teams.get(abbr, {})
        records.append({
            "id": int(r.PLAYER_ID),
            "n": r.PLAYER_NAME,
            "tid": tm.get("id"),
            "t": tm.get("name", abbr),
            "conf": tm.get("conf"),
            # Namespaced on purpose. This file is merged into EVERY scope, but
            # the ridge is solved on the regular season only — so a plain
            # "poss"/"onoff" here would overwrite a playoff row's own numbers
            # with regular-season ones and nobody would see it happen.
            "rposs": int(round(r.POSS)),
            "orapm": rnd(r.ORAPM, 2), "drapm": rnd(r.DRAPM, 2), "rapm": rnd(r.RAPM, 2),
        })

    write_json({"players": records,
                "meta": _meta(season, n_players=len(records),
                              # The page shows this so a reader knows the
                              # ratings are comparable across seasons.
                              alpha=2310.13)},
               "player-impact.json", SITE_DATA_DIR / season)
    return len(records)


# ---------------------------------------------------------------------------
# Season index
# ---------------------------------------------------------------------------
def write_seasons_index() -> list[str]:
    """Season keys that actually have data on disk, newest first."""
    if not SITE_DATA_DIR.exists():
        return []
    built = sorted(
        (p.name for p in SITE_DATA_DIR.iterdir()
         if p.is_dir() and re.fullmatch(r"\d{4}", p.name)),
        reverse=True,
    )
    write_json(built, "seasons.json", SITE_DATA_DIR)
    return built


def write_headers(built: list[str]) -> None:
    """
    Emit site/_headers so browsers and the CDN stop serving stale season JSON.

    Without an explicit Cache-Control, both apply RFC 9111 *heuristic* freshness
    — roughly 10% of the file's age — so a day-old payload is treated as fresh
    for hours and fetch() never revalidates. That is how a rebuilt season can
    look unchanged in the browser while the JSON on disk is already correct.

    Completed seasons are frozen and cacheable forever; the live season changes
    whenever a build runs. seasons.json must stay short-lived — it decides which
    season is live, so a stale copy would pin the whole site to last year.
    Generated from what is actually on disk rather than hand-maintained, so it
    cannot rot when a new season starts.
    """
    live = built[0] if built else None
    lines = [
        "# Generated by build_site.py — do not edit by hand.",
        "",
        "/data/seasons.json",
        "  Cache-Control: public, max-age=300",
        "",
    ]
    for s in built[1:]:          # every season except the live one is frozen
        lines += [f"/data/{s}/*",
                  "  Cache-Control: public, max-age=31536000, immutable", ""]
    if live:
        lines += [f"/data/{live}/*", "  Cache-Control: public, max-age=3600", ""]
    lines += [
        "/js/*", "  Cache-Control: public, max-age=3600", "",
        "/css/*", "  Cache-Control: public, max-age=3600", "",
    ]
    (SITE_DIR / "_headers").write_text("\n".join(lines))
    print(f"  _headers written (live={live}, frozen={len(built[1:])} seasons)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# Scopes the shot charts ship. "all" is excluded on purpose: the page draws one
# season type at a time, and reg+playoffs concatenated would nearly double the
# coordinate payload to answer a question the UI never asks. site/js/player-card.js
# assumes exactly this pair.
SHOT_SCOPES = ("reg", "playoffs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons",
                    help="WNBA season key e.g. 2024 (repeatable)")
    ap.add_argument("--no-shots", action="store_true",
                    help="skip the shot-chart export (it is the only stage "
                         "that can still hit the network)")
    args = ap.parse_args()

    seasons = args.seasons or built_seasons()
    for season in seasons:
        print(f"\n=== {season} ===")
        built_any = False
        for scope in SCOPES:
            if build_player_stats(season, scope):
                built_any = True
        if build_impact(season):
            built_any = True
        if not args.no_shots:
            for scope in SHOT_SCOPES:
                if build_shots(season, scope):
                    built_any = True
        if not built_any:
            print("  nothing built")

    built = write_seasons_index()
    write_headers(built)
    print(f"\nSeasons available: {len(built)} "
          f"({built[-1] if built else '-'}-{built[0] if built else '-'})")
    print("Preview with:  python -m http.server 8000 --directory site")


# ---------------------------------------------------------------------------
# Shot Charts — {season}/shots-index.json + {season}/shots/{slug}.json
# ---------------------------------------------------------------------------
def build_shots(season: str, scope: str = "reg") -> int:
    """
    Per-team shot payloads plus a slim index.

    Split per team for the same reason nba_charts does it: a whole season of
    raw coordinates is megabytes, and the page only ever shows one team at a
    time. Each file carries the era's arc geometry so the browser draws the
    line that was actually on the floor that year — the JS mirror must not
    infer it from the season number.
    """
    from data.shots import season_shots
    from data.zones import ZONE_NAMES, baselines, geometry_for_season, zone_records

    # SCOPES[scope][0] is a LIST of season types — "all" is ["Regular Season",
    # "Playoffs"]. Taking [0][0] would quietly build regular-season-only data and
    # stamp it scope="all", which is indistinguishable from a correct build once
    # written. Concatenate every type the scope names instead.
    frames = []
    for season_type in SCOPES[scope][0]:
        try:
            part = season_shots(season, season_type=season_type)
        except Exception as e:
            print(f"  [skip] shots ({scope}/{season_type}): {type(e).__name__}: {str(e)[:60]}")
            continue
        if part is not None and not part.empty:
            frames.append(part)
    df = pd.concat(frames, ignore_index=True) if frames else None
    if df is None or df.empty:
        print(f"  [skip] shots ({scope}): no shot data for {season}")
        return 0

    # The flat coordinate array below is written in row order, so the payload
    # inherits whatever order the feed happened to return. That order is not
    # stable between calls, which made two builds of identical data produce
    # different JSON — the site's whole no-new-games-no-diff property depends on
    # this being deterministic. (GAME_ID, GAME_EVENT_ID) identifies a shot.
    df = df.sort_values(["GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)

    geom = geometry_for_season(season)
    teams = season_team_map(season)
    by_id = {v["id"]: v for v in teams.values()}
    out_dir = SITE_DATA_DIR / season / "shots"

    # League baselines are computed WITHIN the season — a 19'9" three and a
    # 22'1.75" three are different shots, so a pooled baseline describes neither.
    league_base = baselines(df["made"].values, df["zone"].values)

    index, written = [], set()
    for tid, g in df.groupby("TEAM_ID"):
        meta = by_id.get(int(tid))
        if not meta:
            continue
        # GAME_ID breaks the tie: a team plays at most once a day, but sorting
        # on date alone leaves same-date rows in feed order, which would permute
        # every game index in the payload for no reason.
        games = (g[["GAME_ID", "GAME_DATE"]].drop_duplicates()
                 .sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True))
        game_pos = {gid: i for i, gid in enumerate(games["GAME_ID"])}
        players = sorted(g["PLAYER_NAME"].dropna().unique().tolist())
        player_pos = {p: i for i, p in enumerate(players)}
        name_to_id = (g.dropna(subset=["PLAYER_NAME"])
                       .drop_duplicates("PLAYER_NAME")
                       .set_index("PLAYER_NAME")["PLAYER_ID"].to_dict())

        flat = []
        for r in g.itertuples(index=False):
            flat.extend([
                int(game_pos.get(r.GAME_ID, -1)),
                int(r.LOC_X), int(r.LOC_Y), int(r.made),
                int(player_pos.get(r.PLAYER_NAME, -1)),
            ])

        payload = {
            "team": meta["name"], "team_id": int(tid), "slug": meta["slug"],
            "season": season, "scope": scope,
            "era": geom.label, "arc_r": geom.arc_r, "corner_x": geom.corner_x,
            "gp": int(len(games)),
            "players": players,
            "player_ids": [int(name_to_id.get(p, 0)) for p in players],
            "shots": flat,
            "team_zones": zone_records(g["made"].values, g["zone"].values),
            "player_zones": {
                p: zone_records(pg["made"].values, pg["zone"].values)
                for p, pg in g.groupby("PLAYER_NAME") if len(pg) >= 30
            },
        }
        fname = f"{meta['slug']}{SCOPES[scope][1]}.json"
        write_json(payload, fname, out_dir)
        written.add(fname)
        index.append({"team": meta["name"], "team_id": int(tid),
                      "slug": meta["slug"], "conf": meta.get("conf"),
                      "gp": int(len(games)), "fga": int(len(g))})

    if not index:
        return 0

    index.sort(key=lambda x: x["team"])
    write_json({"teams": index, "zone_names": ZONE_NAMES,
                "league_baselines": league_base,
                "era": geom.label, "arc_r": geom.arc_r, "corner_x": geom.corner_x,
                "meta": _meta(season, scope=scope, n_teams=len(index))},
               scope_filename("shots-index", scope), SITE_DATA_DIR / season)
    return len(index)


if __name__ == "__main__":
    main()
