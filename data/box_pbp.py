# data/box_pbp.py
"""
box_pbp.py — Per-player box totals derived from play-by-play, not from the API.

Why derive what the stats API will hand over for free: everything else on this
site is computed from the raw PBP by our own engine, and a box score that comes
from somewhere else cannot be reconciled with it. USG%, AST%, REB%, STL% and
BLK% all need a numerator (the player's own events) and a denominator (what her
team and her opponent did *while she was on the floor*). The API only publishes
season totals, so the conventional formulas approximate the on-floor part with a
minutes share — `player_stat * team_minutes / (player_minutes * 5)`. We already
know the exact five-on-five lineup for every second of every game, so the same
rates can be computed against real on-court totals instead of an estimate.

That makes the API feeds a *benchmark* rather than a source. `data.verify_box`
reconciles this module's output against LeagueDashPlayerStats.

Event attribution (WNBA PlayByPlayV2, verified against the 2024 feed):

    type 1  made FG        PLAYER1 scorer, PLAYER2 assister
    type 2  missed FG      PLAYER1 shooter, PLAYER3 blocker
    type 3  free throw     PLAYER1 shooter; "MISS" prefix marks a miss
    type 4  rebound        PLAYER1 rebounder (absent on team rebounds)
    type 5  turnover       PLAYER1 committer, PLAYER2 stealer
    type 6  foul           PLAYER1 fouler, PLAYER2 drew it

Shot value comes from "3PT" in the description, never from our shot zones — the
feed knows what the basket counted for and the geometry does not.

Usage:
    from data.box_pbp import game_events, season_box
    ev = game_events(get_pbp("1022400001"))
    box = season_box("2024")
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data.pbp_rotation import _chronological

# Counting stats this module produces. Everything downstream (rates, per-100,
# the site export) is derived from these and from stint on-floor context.
STAT_COLS = [
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pts",
]

# Rebounds can only follow a missed field goal or a missed final free throw.
# Anything else (a made basket, a turnover, the end of a period) clears the
# pending miss so a stray rebound is not attributed to the wrong team.
_MISS_TYPES = {2, 3}

# Foul EVENTMSGACTIONTYPEs that are technicals rather than personal fouls. The
# official box score's PF column counts personal fouls only, and the feed agrees
# with itself here: a technical prints as "T.FOUL (P0.PN)", leaving the running
# personal count untouched. Flagrants (14, 15) DO count as personal fouls and
# are deliberately absent from this set.
_TECH_FOUL_TYPES = {11, 12, 13, 16, 17, 18, 19, 25}

# Every player rebound in the feed — all 30 seasons, checked — is annotated with
# that player's running counts, "Jones REBOUND (Off:0 Def:1)". Those absolute
# counters are the authority on whether a board was offensive, because deciding
# it from the preceding event is wrong whenever the feed reorders within a
# second: an offensive rebound and the putback it set up share a timestamp, and
# when the putback sorts first the rebound looks like it followed a made basket
# and gets dropped. That cost 4.6-9.6% of offensive rebounds before 2016.
_REB_COUNTS = re.compile(r"\(Off:(\d+)\s+Def:(\d+)\)")


def _desc(row) -> str:
    parts = [row.HOMEDESCRIPTION, row.NEUTRALDESCRIPTION, row.VISITORDESCRIPTION]
    return " ".join(p for p in parts if isinstance(p, str))


# PERSONnTYPE tells a player apart from a team. 4 = home player, 5 = away
# player; 2 and 3 are the home and away *teams*. On a team event the feed puts
# the TEAM id in PLAYERn_ID and leaves PLAYERn_TEAM_ID null, so reading the id
# alone turns every team rebound into a phantom player with a dozen boards.
_PLAYER_TYPES = {4, 5}
_TEAM_TYPES = {2, 3}


def _tid(v) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _actor(pid, ptype, tid) -> tuple[int, int]:
    """(player_id, team_id) for an event slot, or (0, 0) if it isn't a player."""
    try:
        t = int(ptype)
    except (TypeError, ValueError):
        return 0, 0
    if t not in _PLAYER_TYPES:
        return 0, 0
    try:
        n = int(pid)
    except (TypeError, ValueError):
        return 0, 0
    return (n, _tid(tid)) if n > 0 else (0, 0)


def _team_actor(pid, ptype) -> int:
    """
    Team id for a team event, else 0.

    Team rebounds (a miss out of bounds, a buzzer) belong to no player, but they
    still consumed a rebounding opportunity. They are emitted with player_id 0 so
    they land in the team context totals that REB% divides by, without ever
    reaching a player's own box score. Leaving them out inflates every rebound
    rate — measurably, DREB% ran ~1.9 points high against the league's.
    """
    try:
        t = int(ptype)
    except (TypeError, ValueError):
        return 0
    if t not in _TEAM_TYPES:
        return 0
    try:
        n = int(pid)          # on team rows the feed puts the TEAM id here
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _resolve_rebounds(df: pd.DataFrame) -> dict[int, str]:
    """
    Classify every player rebound as offensive or defensive from the feed's own
    running counters, independent of event order.

    Returns {row position -> "oreb" | "dreb"}. Rows with no annotation are left
    out and fall back to the preceding-miss rule.

    For each player the counters only ever increase, so sorting her rebounds by
    Off+Def recovers the true sequence even when the chronological sort has
    swapped two events inside one second. Whichever counter went up names the
    board.
    """
    per_player: dict[int, list[tuple[int, int, int]]] = {}
    for pos, r in enumerate(df.itertuples(index=False)):
        if (int(r.EVENTMSGTYPE) if pd.notna(r.EVENTMSGTYPE) else 0) != 4:
            continue
        pid, _ = _actor(r.PLAYER1_ID, r.PERSON1TYPE, r.PLAYER1_TEAM_ID)
        if not pid:
            continue
        m = _REB_COUNTS.search(_desc(r))
        if not m:
            continue
        per_player.setdefault(pid, []).append((int(m.group(1)), int(m.group(2)), pos))

    out: dict[int, str] = {}
    for pid, events in per_player.items():
        prev_o = prev_d = 0
        for off, dfn, pos in sorted(events):          # sorts by (off, def, pos)
            if off > prev_o:
                out[pos] = "oreb"
            elif dfn > prev_d:
                out[pos] = "dreb"
            prev_o, prev_d = max(prev_o, off), max(prev_d, dfn)
    return out


def game_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (player, event) box contribution for a single game.

    Columns: game_id, elapsed, player_id, team_id + STAT_COLS. A single PBP row
    can emit several rows — a blocked layup produces an `fga` for the shooter and
    a `blk` for the defender, on opposite teams.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["game_id", "elapsed", "player_id", "team_id"] + STAT_COLS)

    df = _chronological(pbp)
    reb_kind = _resolve_rebounds(df)
    rows = []
    pending_miss_team = 0        # team that missed the shot now available to rebound

    def add(pid, tid, elapsed, **stats):
        if not pid:
            return
        rec = {"player_id": pid, "team_id": tid, "elapsed": elapsed}
        rec.update({c: 0 for c in STAT_COLS})
        rec.update(stats)
        rows.append(rec)

    for pos, r in enumerate(df.itertuples(index=False)):
        et = int(r.EVENTMSGTYPE) if pd.notna(r.EVENTMSGTYPE) else 0
        elapsed = float(r.ELAPSED)
        desc = _desc(r)
        is3 = "3PT" in desc

        p1, t1 = _actor(r.PLAYER1_ID, r.PERSON1TYPE, r.PLAYER1_TEAM_ID)
        p2, t2 = _actor(r.PLAYER2_ID, r.PERSON2TYPE, r.PLAYER2_TEAM_ID)
        p3, t3 = _actor(r.PLAYER3_ID, r.PERSON3TYPE, r.PLAYER3_TEAM_ID)

        if et == 1:                                   # made field goal
            pts = 3 if is3 else 2
            add(p1, t1, elapsed, fgm=1, fga=1, pts=pts,
                fg3m=int(is3), fg3a=int(is3))
            add(p2, t2, elapsed, ast=1)
            pending_miss_team = 0

        elif et == 2:                                 # missed field goal
            add(p1, t1, elapsed, fga=1, fg3a=int(is3))
            add(p3, t3, elapsed, blk=1)               # PLAYER3 is the blocker
            pending_miss_team = t1

        elif et == 3:                                 # free throw
            made = not desc.strip().startswith("MISS")
            add(p1, t1, elapsed, fta=1, ftm=int(made), pts=int(made))
            # Only the last free throw of a trip is reboundable. "N of M" tells
            # us which; a technical ("1 of 1") is handled by the same rule.
            last_of_trip = True
            if " of " in desc:
                try:
                    n, m = desc.split(" of ")[0].split()[-1], desc.split(" of ")[1].split()[0]
                    last_of_trip = int(n) == int(m)
                except (ValueError, IndexError):
                    last_of_trip = True
            pending_miss_team = 0 if (made or not last_of_trip) else t1

        elif et == 4:                                 # rebound
            if p1:
                kind = reb_kind.get(pos)
                if kind is None and pending_miss_team:
                    kind = "oreb" if t1 == pending_miss_team else "dreb"
                if kind:
                    add(p1, t1, elapsed, **{kind: 1})
            else:
                # Team rebound: no annotation to read, so the pending miss is
                # the only signal. Recorded against player_id 0.
                team = _team_actor(r.PLAYER1_ID, r.PERSON1TYPE)
                if team and pending_miss_team:
                    kind = "oreb" if team == pending_miss_team else "dreb"
                    rec = {"player_id": 0, "team_id": team, "elapsed": elapsed}
                    rec.update({c: 0 for c in STAT_COLS})
                    rec[kind] = 1
                    rows.append(rec)
            pending_miss_team = 0

        elif et == 5:                                 # turnover
            add(p1, t1, elapsed, tov=1)
            add(p2, t2, elapsed, stl=1)               # PLAYER2 is the stealer
            pending_miss_team = 0

        elif et == 6:                                 # foul
            at = int(r.EVENTMSGACTIONTYPE) if pd.notna(r.EVENTMSGACTIONTYPE) else 0
            if at not in _TECH_FOUL_TYPES:
                add(p1, t1, elapsed, pf=1)
            # A shooting foul is followed by free throws; leave pending as is.

        else:
            # Substitutions, timeouts, jump balls, period markers: no box impact,
            # but a period boundary must not leave a miss dangling into the next.
            if et in (12, 13):
                pending_miss_team = 0

    if not rows:
        return pd.DataFrame(columns=["game_id", "elapsed", "player_id", "team_id"] + STAT_COLS)

    out = pd.DataFrame(rows)
    out["game_id"] = str(df["GAME_ID"].iloc[0]).zfill(10)
    return out[["game_id", "elapsed", "player_id", "team_id"] + STAT_COLS]


def game_box(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-player totals for one game. Team rows (player_id 0) are dropped."""
    ev = game_events(pbp)
    if ev.empty:
        return ev
    ev = ev[ev["player_id"] > 0]
    box = (ev.groupby(["game_id", "player_id", "team_id"], as_index=False)[STAT_COLS]
             .sum())
    box["reb"] = box["oreb"] + box["dreb"]
    return box
