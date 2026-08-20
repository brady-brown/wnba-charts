# data/build_scopes.py
"""
build_scopes.py — On/off + box splits for the Playoffs and All Games scopes.

RAPM is deliberately NOT computed here. It is solved on regular-season
play-by-play only, for the same reason nba_charts does it that way: a WNBA
playoff run is a handful of games, and a ridge fit on that few possessions
produces ratings that mostly reflect the prior, not the player. The site shows
regular-season RAPM on every scope and says so.

What the non-regular-season scopes DO get is everything that survives a small
sample honestly: raw on/off splits and box stats, both of which are simple
aggregates rather than model output.

Writes data/rapm_out/onoff_{season}_{scope}.csv in the same column shape as the
RAPM CSVs (minus ORAPM/DRAPM/RAPM), so data/build_site.py can read either.

Usage:
    python -m data.build_scopes                       # playoffs + all, every season
    python -m data.build_scopes --scope playoffs
    python -m data.build_scopes --season 2024
"""

from __future__ import annotations

import argparse

import pandas as pd

from data.build_common import RAPM_DIR, SCOPES
from data.build_rapm import load_names, load_stints
from data.rapm import _compute_on_off, _fetch_box_stats

MIN_STINT_POSS = 0.5      # must match data.rapm.compute_rapm's default


def scope_stints(season: str, scope: str) -> pd.DataFrame:
    """Stints for one scope. 'all' is regular season and playoffs concatenated."""
    season_types = SCOPES[scope][0]
    frames = [load_stints(season, st) for st in season_types]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_scope(season: str, scope: str) -> pd.DataFrame:
    """On/off splits + box stats for one (season, scope). No ridge."""
    stints = scope_stints(season, scope)
    if stints.empty:
        return pd.DataFrame()

    total_poss = stints["home_poss"] + stints["away_poss"]
    stints = stints[total_poss >= MIN_STINT_POSS].copy()
    if stints.empty:
        return pd.DataFrame()

    names = load_names()
    out = _compute_on_off(stints)
    out.insert(0, "SEASON", season)
    out.insert(1, "PLAYER_NAME", out["PLAYER_ID"].map(names).fillna(
        out["PLAYER_ID"].astype(str)))

    # Possessions on the floor, matching how compute_rapm counts them.
    player_poss: dict[int, float] = {}
    for s in stints.itertuples(index=False):
        p = float(s.home_poss) + float(s.away_poss)
        for pid in s.home_lineup | s.away_lineup:
            player_poss[pid] = player_poss.get(pid, 0.0) + p
    out["POSS"] = out["PLAYER_ID"].map(player_poss).round(1)

    # Box stats. Best-effort: the endpoint has no playoff rows for some early
    # seasons, and a missing join must not take down the scope.
    season_type = SCOPES[scope][0]
    try:
        if len(season_type) == 1:
            pg, p100 = _fetch_box_stats(season, season_type[0])
        else:
            # "all" — combine each season_type's feed the same way multi-season
            # requests are combined.
            pgs, p100s = [], []
            for st in season_type:
                # A season_type with no games at all (an in-progress season has
                # no playoffs yet) must not sink the whole scope — take what
                # exists and carry on.
                try:
                    a, b = _fetch_box_stats(season, st)
                except Exception:
                    continue
                if a is not None and not a.empty:
                    pgs.append(a)
                if b is not None and not b.empty:
                    p100s.append(b)
            if not pgs:
                raise ValueError("no box feed for any season type")
            pg = pd.concat(pgs, ignore_index=True).drop_duplicates("PLAYER_ID", keep="first")
            p100 = pd.concat(p100s, ignore_index=True).drop_duplicates("PLAYER_ID", keep="first")
        out = out.merge(pg.drop(columns=["PLAYER_NAME"], errors="ignore"),
                        on="PLAYER_ID", how="left")
        out = out.merge(p100.drop(columns=["PLAYER_NAME"], errors="ignore"),
                        on="PLAYER_ID", how="left")
        if all(c in out.columns for c in ("FGA_PG", "FTA_PG", "TOV_PG", "GP_PG", "ON_POSS_O")):
            uses = (out["FGA_PG"] + 0.44 * out["FTA_PG"] + out["TOV_PG"]) * out["GP_PG"]
            out["USG_PCT"] = (uses / out["ON_POSS_O"] * 100).round(1)
        for pct in ("FG_PCT_PG", "FG3_PCT_PG"):
            if pct in out.columns:
                out[pct] = (out[pct] * 100).round(1)
        if all(c in out.columns for c in ("STL_100", "BLK_100")):
            out["STOCKS_100"] = out["STL_100"] + out["BLK_100"]
    except Exception as e:
        print(f"    [warn] {season} {scope}: box stats unavailable: "
              f"{type(e).__name__}: {str(e)[:60]}")

    return out.sort_values("ON_OFF", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons")
    ap.add_argument("--scope", action="append", dest="scopes",
                    choices=["playoffs", "all"],
                    help="default: both non-regular-season scopes")
    args = ap.parse_args()

    from data.build_common import built_seasons
    seasons = args.seasons or built_seasons(newest_first=False)
    scopes = args.scopes or ["playoffs", "all"]
    RAPM_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        for scope in scopes:
            df = build_scope(season, scope)
            if df.empty:
                print(f"  {season} {scope:<9}: no stints")
                continue
            path = RAPM_DIR / f"onoff_{season}_{scope}.csv"
            df.to_csv(path, index=False)
            print(f"  {season} {scope:<9}: {len(df):>3} players -> {path.name}")


if __name__ == "__main__":
    main()
