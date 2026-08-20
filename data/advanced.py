# data/advanced.py
"""
advanced.py — Advanced rate stats computed against real on-court context.

The conventional formulas for AST%, REB%, STL% and BLK% (Basketball-Reference's)
all contain the same fudge: they need "what the team did while this player was
on the floor" and, lacking lineup data, approximate it with a minutes share —

    AST% = AST / ((MP / (TmMP / 5)) * TmFG - FG)

That factor assumes the player was on the floor for a representative slice of
the team's production. It is wrong in exactly the cases people care about: a
centre who only plays next to the starting point guard has her teammates' FG
total overstated, a bench unit that plays at a different pace gets the wrong
denominator entirely.

We already reconstruct the exact five-on-five lineup for every second of every
game, so none of that is necessary. Every event from data/box_pbp.py is stamped
with the second it happened; every stint carries the ten players on the floor.
Joining the two gives the true on-court totals for teammates AND opponents, and
each rate becomes a plain quotient of things that actually happened.

Where a rate has no context term (TS%, eFG%, 3PAr, FTr, TOV%) the definition is
the standard one and is computed straight from the player's own totals.

Usage:
    from data.advanced import season_advanced
    df = season_advanced("2024")            # one row per player
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.box_pbp import STAT_COLS, game_events
from data.build_common import STINTS_DIR
from data.cache import get_game_ids, get_pbp

# Team-side aggregates accumulated per stint, then summed over the stints a
# player appears in. "tm_" is her own team's production while she was on;
# "opp_" is what the other five did over the same seconds.
_CTX = ["fgm", "fga", "fg3a", "fta", "oreb", "dreb", "tov", "ast"]


def _events_with_stint(game_id: str, stints: pd.DataFrame) -> pd.DataFrame:
    """Attach a stint index to every box event in one game."""
    ev = game_events(get_pbp(game_id))
    if ev.empty:
        return ev
    gs = stints[stints["game_id"] == game_id]
    if gs.empty:
        return ev.iloc[0:0]
    # Half-open [t_start, t_end): an event exactly on a substitution second
    # belongs to the stint that is starting, matching the stint engine.
    idx = np.searchsorted(gs["t_start"].values, ev["elapsed"].values, side="right") - 1
    idx = np.clip(idx, 0, len(gs) - 1)
    ev = ev.copy()
    ev["stint_row"] = gs.index.values[idx]
    return ev


def _stint_team_totals(ev: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Per (stint, team) sums of the context stats."""
    if ev.empty:
        return pd.DataFrame(columns=["stint_row", "team_id"] + _CTX)
    return ev.groupby(["stint_row", "team_id"], as_index=False)[_CTX].sum()


def season_context(season: str, season_type: str = "Regular Season",
                   progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (player_box, player_context).

    player_box     : per-player season totals of her own counting stats.
    player_context : per-player on-court totals for her team and her opponent,
                     summed over every stint she appeared in, plus the on-court
                     offensive and defensive possessions the stint engine
                     already measured.
    """
    path = STINTS_DIR / f"stints_{season}_{season_type.replace(' ', '_')}.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    stints = pd.read_parquet(path).reset_index(drop=True)

    gids = get_game_ids(season, season_type=season_type)
    if gids is None or gids.empty:
        return pd.DataFrame(), pd.DataFrame()
    gids = gids["GAME_ID"].astype(str).str.zfill(10).tolist()

    box_parts, ctx_parts, appearances = [], [], []
    for n, gid in enumerate(gids, 1):
        if progress and n % 50 == 0:
            print(f"    {season}: {n}/{len(gids)} games")
        ev = _events_with_stint(gid, stints)
        if ev.empty:
            continue
        # Team rebounds ride along as player_id 0: they belong in the team
        # context totals below but never in a player's own box score.
        players = ev[ev["player_id"] > 0]
        box_parts.append(players.groupby(["player_id", "team_id"],
                                         as_index=False)[STAT_COLS].sum())
        # Games played comes from the lineups, not from the box events: a player
        # who checks in and records nothing still played. That is also why GP
        # cannot be counted off `players` above.
        gs_lineups = stints[stints["game_id"] == gid]
        played = set()
        for row in gs_lineups.itertuples():
            played.update(int(p) for p in row.home_lineup)
            played.update(int(p) for p in row.away_lineup)
        appearances.append(pd.DataFrame({"player_id": sorted(played), "gp": 1}))

        # Team totals per stint, then push them onto each of the five players
        # who were on the floor for that stint.
        tt = _stint_team_totals(ev, stints)
        gs = stints[stints["game_id"] == gid]
        rows = []
        for row in gs.itertuples():
            here = tt[tt["stint_row"] == row.Index]
            by_team = {int(t): here[here["team_id"] == t][_CTX].sum()
                       for t in here["team_id"].unique() if t}
            zero = pd.Series({c: 0 for c in _CTX})
            for side, lineup, own_tid, opp_tid, own_poss, opp_poss in (
                ("home", row.home_lineup, row.home_team_id, row.away_team_id,
                 row.home_poss, row.away_poss),
                ("away", row.away_lineup, row.away_team_id, row.home_team_id,
                 row.away_poss, row.home_poss),
            ):
                own = by_team.get(int(own_tid), zero)
                opp = by_team.get(int(opp_tid), zero)
                for pid in lineup:
                    rec = {"player_id": int(pid),
                           "on_poss_o": own_poss, "on_poss_d": opp_poss,
                           "secs": row.t_end - row.t_start}
                    for c in _CTX:
                        rec[f"tm_{c}"] = own.get(c, 0)
                        rec[f"opp_{c}"] = opp.get(c, 0)
                    rows.append(rec)
        if rows:
            ctx_parts.append(pd.DataFrame(rows).groupby("player_id", as_index=False).sum())

    if not box_parts:
        return pd.DataFrame(), pd.DataFrame()
    box = pd.concat(box_parts, ignore_index=True).groupby("player_id", as_index=False)[STAT_COLS].sum()
    ctx = pd.concat(ctx_parts, ignore_index=True).groupby("player_id", as_index=False).sum()
    gp = pd.concat(appearances, ignore_index=True).groupby("player_id", as_index=False)["gp"].sum()
    # A player with lineup time but no box events still belongs in the table.
    box = gp.merge(box, on="player_id", how="left")
    box[STAT_COLS] = box[STAT_COLS].fillna(0)
    return box, ctx


def _safe(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where((den > 0) & np.isfinite(den), num / den.replace(0, np.nan), np.nan)


def season_advanced(season: str, season_type: str = "Regular Season",
                    progress: bool = False) -> pd.DataFrame:
    """Per-player advanced rates for one season, all from PBP + stints."""
    box, ctx = season_context(season, season_type, progress=progress)
    if box.empty:
        return pd.DataFrame()
    return _rates(box.merge(ctx, on="player_id", how="left"))


def _rates(df: pd.DataFrame) -> pd.DataFrame:
    """Every derived rate, from counting totals plus on-floor context."""
    if df.empty:
        return df
    fga, fta, tov = df["fga"], df["fta"], df["tov"]
    df["reb"] = df["oreb"] + df["dreb"]
    df["min"] = df["secs"] / 60.0

    # ── Shooting: own totals only, standard definitions ────────────────────
    df["fg_pct"]  = _safe(df["fgm"], fga) * 100
    df["fg3_pct"] = _safe(df["fg3m"], df["fg3a"]) * 100
    df["ft_pct"]  = _safe(df["ftm"], fta) * 100
    df["efg"]     = _safe(df["fgm"] + 0.5 * df["fg3m"], fga) * 100
    df["ts"]      = _safe(df["pts"], 2 * (fga + 0.44 * fta)) * 100
    df["par3"]    = _safe(df["fg3a"], fga) * 100
    df["ftr"]     = _safe(fta, fga) * 100

    # Scoring possessions the player used, the shared denominator for USG/TOV%.
    uses = fga + 0.44 * fta + tov
    df["tovp"] = _safe(tov, uses) * 100

    # ── Context rates: exact on-court denominators, no minutes-share ───────
    # USG%: share of her team's used possessions that were hers, over the
    # seconds she was actually out there.
    tm_uses = df["tm_fga"] + 0.44 * df["tm_fta"] + df["tm_tov"]
    df["usg"] = _safe(uses, tm_uses) * 100

    # AST%: share of teammate baskets she assisted. Her own makes cannot be
    # assisted by her, so they come out of the denominator.
    df["astp"] = _safe(df["ast"], df["tm_fgm"] - df["fgm"]) * 100

    # Rebound rates: her boards over every board available while she was on.
    df["orbp"] = _safe(df["oreb"], df["tm_oreb"] + df["opp_dreb"]) * 100
    df["drbp"] = _safe(df["dreb"], df["tm_dreb"] + df["opp_oreb"]) * 100
    df["trbp"] = _safe(df["reb"],
                       df["tm_oreb"] + df["tm_dreb"] + df["opp_oreb"] + df["opp_dreb"]) * 100

    # STL%: opponent possessions she ended with a steal.
    df["stlp"] = _safe(df["stl"], df["on_poss_d"]) * 100
    # BLK%: opponent two-point attempts she blocked. Threes are excluded
    # because they are nearly unblockable and would flatten the rate.
    df["blkp"] = _safe(df["blk"], df["opp_fga"] - df["opp_fg3a"]) * 100

    # ── Per game ───────────────────────────────────────────────────────────
    gp = df["gp"].replace(0, np.nan)
    df["min_pg"] = df["min"] / gp
    for c in ("pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
              "fga", "fta", "fgm", "fg3m", "fg3a", "ftm"):
        df[f"{c}_pg"] = df[c] / gp

    # ── Per 100 possessions ────────────────────────────────────────────────
    # Offensive possessions for offensive production, defensive for defensive —
    # a steal is not something you get on offence, and dividing both by the same
    # figure quietly assumes they are equal.
    off_p = df["on_poss_o"].replace(0, np.nan)
    def_p = df["on_poss_d"].replace(0, np.nan)
    for c in ("pts", "ast", "oreb", "tov", "fga"):
        df[f"{c}_100"] = df[c] / off_p * 100
    for c in ("stl", "blk", "dreb"):
        df[f"{c}_100"] = df[c] / def_p * 100
    df["reb_100"] = df["reb"] / (off_p + def_p) * 100
    df["stocks_100"] = (df["stl"] + df["blk"]) / def_p * 100
    return df


def build_season(season: str, season_type: str = "Regular Season",
                 progress: bool = False) -> pd.DataFrame:
    """
    season_advanced() for one scope, with "all" handled as reg + playoffs.

    The two scopes are concatenated at the *event* level rather than by adding
    two rate tables together — averaging two percentages would weight a
    three-game playoff run the same as a full season.
    """
    if season_type != "All":
        return season_advanced(season, season_type, progress=progress)

    boxes, ctxs = [], []
    for st in ("Regular Season", "Playoffs"):
        b, c = season_context(season, st, progress=progress)
        if not b.empty:
            boxes.append(b)
            ctxs.append(c)
    if not boxes:
        return pd.DataFrame()
    box = pd.concat(boxes, ignore_index=True).groupby("player_id", as_index=False).sum()
    ctx = pd.concat(ctxs, ignore_index=True).groupby("player_id", as_index=False).sum()
    return _rates(box.merge(ctx, on="player_id", how="left"))
