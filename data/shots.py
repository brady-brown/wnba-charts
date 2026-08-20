# data/shots.py
"""
shots.py — Shot-chart aggregates: zone rollups for league, team and player.

Layering follows nba_charts (pipeline/shots.py): geometry lives ONLY in
data/zones.py, and this module does aggregation only. Keeping them apart is
what stops the classifier from being reimplemented three times with three
slightly different arcs.

Two WNBA-specific rules this module enforces:

* **Everything is per season.** The three-point line moved in 2004 and again in
  2013, so a zone means different things in different eras. Zones are assigned
  with that season's geometry, and league baselines are computed within a
  season — pooling a 19'9" three with a 22'1.75" three yields a baseline that
  describes neither shot.
* **Points come from SHOT_TYPE, not from our zone.** The feed knows whether a
  shot counted for two or three; our classifier only knows where it was taken.
  Near the arc the two can disagree, and the feed wins on value.

Usage:
    from data.shots import season_shots, zone_table, league_baselines
    df = season_shots("2024")
    zone_table(df, by="PLAYER_ID")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.cache import get_shot_chart
from data.zones import (
    THREE_POINT_ZONES,
    ZONE_INDEX,
    ZONE_NAMES,
    baselines,
    classify,
    geometry_for_season,
    zone_records,
)

MIN_PLAYER_FGA = 30      # below this a player's zone chart is noise


# ── Loading and preparation ───────────────────────────────────────────────────

def prepare(shots: pd.DataFrame, season) -> pd.DataFrame:
    """Attach era-correct zones and drop unusable rows (heaves, bad coords)."""
    geom = geometry_for_season(season)
    df = shots.dropna(subset=["LOC_X", "LOC_Y"]).copy()
    df["zone"] = classify(df["LOC_X"].values, df["LOC_Y"].values, geom)
    df = df[~df["zone"].isin(["Heave", "Unknown"])].copy()

    df["made"] = df["SHOT_MADE_FLAG"].astype(float)
    # Shot value from the feed, not from our zone — see module docstring.
    if "SHOT_TYPE" in df.columns:
        df["shot_value"] = np.where(
            df["SHOT_TYPE"].astype(str).str.startswith("3PT"), 3.0, 2.0
        )
    else:
        df["shot_value"] = np.where(df["zone"].isin(THREE_POINT_ZONES), 3.0, 2.0)
    df["points"] = df["made"] * df["shot_value"]
    df["season"] = str(season)
    df["era"] = geom.label
    return df


def season_shots(season, season_type: str = "Regular Season") -> pd.DataFrame:
    """Every prepared shot for one season (one API call, then cached)."""
    raw = get_shot_chart(str(season), season_type=season_type)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return prepare(raw, season)


def load_seasons(seasons, season_type: str = "Regular Season") -> pd.DataFrame:
    """Prepared shots for several seasons, each classified in its own era."""
    if isinstance(seasons, (str, int)):
        seasons = [seasons]
    frames = []
    for season in seasons:
        df = season_shots(season, season_type=season_type)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Aggregation ───────────────────────────────────────────────────────────────

def league_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    League FG% and points-per-attempt per zone, per season.

    Points-per-attempt matters more than FG% for comparing zones to each other:
    a 35% three (1.05 PPA) beats a 45% long two (0.90 PPA). FG% is still the
    right scale for comparing a player to the league WITHIN a zone.
    """
    rows = []
    for season, g in df.groupby("season"):
        fg = baselines(g["made"].values, g["zone"].values)
        for name in ZONE_NAMES:
            zg = g[g["zone"] == name]
            rows.append({
                "season": season,
                "zone": name,
                "zone_index": ZONE_INDEX[name],
                "LG_FGA": len(zg),
                "LG_FG_PCT": fg[ZONE_INDEX[name]],
                "LG_PPA": round(float(zg["points"].mean()), 4) if len(zg) else None,
            })
    return pd.DataFrame(rows)


def zone_table(df: pd.DataFrame, by: str | list[str] = "PLAYER_ID",
               min_fga: int = MIN_PLAYER_FGA) -> pd.DataFrame:
    """
    Per-entity zone splits with league-relative columns.

    `by` is any grouping key present in the data — "PLAYER_ID", "TEAM_ID", or a
    list. Returns one row per (entity, season, zone) with FGA/FGM/FG% plus
    points-per-attempt and the deltas against that season's league baseline,
    which is what a shot chart actually colours by.

    `min_fga` filters on the entity's TOTAL attempts in the season, not per
    zone — otherwise a player's rim volume would silently delete his corner-three
    row and the chart would show gaps where the truth is "small sample".
    """
    keys = [by] if isinstance(by, str) else list(by)
    group = keys + ["season", "zone"]

    agg = (df.groupby(group, dropna=False)
             .agg(FGM=("made", "sum"), FGA=("made", "size"),
                  PTS=("points", "sum"))
             .reset_index())

    if min_fga:
        totals = agg.groupby(keys + ["season"])["FGA"].transform("sum")
        agg = agg[totals >= min_fga].copy()
    if agg.empty:
        return agg

    agg["FG_PCT"] = (agg["FGM"] / agg["FGA"]).round(4)
    agg["PPA"] = (agg["PTS"] / agg["FGA"]).round(4)
    agg["zone_index"] = agg["zone"].map(ZONE_INDEX)

    lg = league_baselines(df)[["season", "zone", "LG_FG_PCT", "LG_PPA"]]
    agg = agg.merge(lg, on=["season", "zone"], how="left")
    agg["FG_PCT_VS_LG"] = (agg["FG_PCT"] - agg["LG_FG_PCT"]).round(4)
    agg["PPA_VS_LG"] = (agg["PPA"] - agg["LG_PPA"]).round(4)

    # Share of the entity's own attempts taken from this zone — shot diet.
    agg["FGA_SHARE"] = (
        agg["FGA"] / agg.groupby(keys + ["season"])["FGA"].transform("sum")
    ).round(4)

    sort_keys = keys + ["season", "zone_index"]
    return agg.sort_values(sort_keys).reset_index(drop=True)


def compact_payload(df: pd.DataFrame, season) -> dict:
    """
    Compact per-season encoding, in the shape a browser wants.

    Mirrors nba_charts' format: a flat int array of
    [game_index, x, y, made, player_index] quintuples with names and games
    listed once, so the payload stays small. Not needed for notebook work —
    this exists so a future site can consume the same pipeline.
    """
    g = df[df["season"] == str(season)]
    if g.empty:
        return {}

    games = (g[["GAME_ID", "GAME_DATE"]].drop_duplicates()
             .sort_values("GAME_DATE").reset_index(drop=True))
    game_pos = {gid: i for i, gid in enumerate(games["GAME_ID"])}

    players = sorted(g["PLAYER_NAME"].dropna().unique().tolist())
    player_pos = {p: i for i, p in enumerate(players)}
    name_to_id = (g.dropna(subset=["PLAYER_NAME"])
                   .drop_duplicates("PLAYER_NAME")
                   .set_index("PLAYER_NAME")["PLAYER_ID"].to_dict())

    flat = np.column_stack([
        g["GAME_ID"].map(game_pos).fillna(-1).astype(int).values,
        g["LOC_X"].astype(int).values,
        g["LOC_Y"].astype(int).values,
        g["made"].astype(int).values,
        g["PLAYER_NAME"].map(player_pos).fillna(-1).astype(int).values,
    ]).ravel().tolist()

    geom = geometry_for_season(season)
    return {
        "season": str(season),
        "era": geom.label,
        # The browser has to draw the arc at the right radius for this season.
        "arc_r": geom.arc_r,
        "corner_x": geom.corner_x,
        "gp": int(len(games)),
        "games": [{"id": str(r.GAME_ID), "date": str(r.GAME_DATE)}
                  for r in games.itertuples()],
        "players": players,
        "player_ids": [int(name_to_id.get(p, 0)) for p in players],
        "shots": flat,
        "league_zones": zone_records(g["made"].values, g["zone"].values),
        "zone_names": ZONE_NAMES,
    }


def territory(df: pd.DataFrame) -> pd.DataFrame:
    """Top scorer in each zone per season — who owns which patch of floor."""
    rows = []
    for (season, zone), zg in df.groupby(["season", "zone"]):
        by_player = zg.groupby("PLAYER_NAME")["points"].sum()
        if by_player.empty or by_player.max() <= 0:
            continue
        rows.append({"season": season, "zone": zone,
                     "zone_index": ZONE_INDEX.get(zone),
                     "PLAYER_NAME": by_player.idxmax(),
                     "PTS": int(by_player.max())})
    return pd.DataFrame(rows).sort_values(["season", "zone_index"]).reset_index(drop=True)
