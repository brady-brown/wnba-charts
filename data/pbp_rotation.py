# data/pbp_rotation.py
"""
Reconstruct on-court rotations for WNBA games from play-by-play alone.

Why: GameRotation is a second API call per game and is missing for some games
(notably 2026, where only 2 of 46 games have it cached). PlayByPlayV2 is the
one call we already need for scoring/possessions, so deriving lineups from it
halves the request count and covers games GameRotation doesn't.

WNBA PBP is PlayByPlayV2, which makes this much easier than the NBA V3 feed:
a substitution row carries BOTH player IDs (PLAYER1 = out, PLAYER2 = in), so
there is no name-string parsing or same-surname disambiguation to do.

The remaining problem is opening lineups, which PBP never states directly.
See `_detect_starters` for how they're inferred.

Verified against GameRotation on 1,170 cached games (2021-2026): 1,169 match
the API's on-court sets for every second of every game. The one exception
(1022300041, ATL vs NYL 2023) is not recoverable from PBP — Naz Hillmon played
the entire first quarter without recording a single event and was substituted
only at the quarter break, which emits no PBP row, so the feed contains no
trace of her being on the floor. That period is emitted as a 4-player lineup
with a warning rather than a guessed fifth; callers that need strict 5v5
should drop lineups whose length is not 5.

Run `python -m data.verify_rotation` to reproduce.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from data.cache import get_pbp

QUARTER_SEC = 600   # WNBA 10-min quarters (2006 onward)
HALF_SEC = 1200     # WNBA 20-min halves (1997-2005)
OT_SEC = 300        # 5-min OT, every era

# Regulation period lengths seen in WNBA play-by-play, shortest first:
# 10-minute quarters (2006+), 12-minute quarters (a handful of games carry
# NBA-length periods), 20-minute halves (1997-2005). A period's true length is
# the smallest of these covering the largest clock reading observed in it —
# play rarely starts at exactly 10:00, so the observed max is a lower bound.
_PERIOD_LENGTHS = (QUARTER_SEC, 720, HALF_SEC)

SUB = 8             # EVENTMSGTYPE for a substitution

# Event types whose PLAYER1/2/3 must have been on the floor: made shot, missed
# shot, free throw, rebound, turnover, foul, jump ball. Everything else is
# excluded because PLAYER1 is often a team id rather than a player (timeout,
# period start/end, instant replay) or the player need not be on the court
# (ejection).
_ONCOURT_EVENTS = {1, 2, 3, 4, 5, 6, 10}

FOUL = 6

# Technical fouls are the exception to "a foul means you were playing" — they
# can be assessed to a player sitting on the bench. One of these (a bench
# technical on Alyssa Thomas, CON @ WAS 2022-06-14) was enough to put her in
# our lineup for an entire quarter she never played, so fouls of these action
# types are not treated as evidence of being on the floor.
#   11, 17 → "T.FOUL" / "T.Foul"      16 → double technical
#   18, 19, 25, 30 → delay of game, taunting, and other administrative techs
_TECHNICAL_FOUL_TYPES = {11, 16, 17, 18, 19, 25, 30}

# PERSON*TYPE values that mean "a player on the floor"
_PLAYER_PERSON_TYPES = {4, 5}


# ── Time helpers ──────────────────────────────────────────────────────────────

def _clock_seconds(pctimestring) -> float | None:
    """PCTIMESTRING is time REMAINING in the period ('6:54')."""
    try:
        parts = str(pctimestring).split(":")
        return int(parts[0]) * 60 + float(parts[1])
    except (TypeError, ValueError, IndexError):
        return None


def _period_scheme(pbp: pd.DataFrame) -> tuple[int, int]:
    """
    Work out this game's period format: (regulation_period_length, n_regulation).

    The WNBA played two 20-minute halves from 1997 through 2005 and switched to
    four 10-minute quarters in 2006, so period length cannot be a constant —
    assuming 10-minute quarters on a 2003 game puts every event in the wrong
    period and makes the elapsed clock run backwards.

    Rather than hardcode the cutover year (and get the transition or any future
    rule change wrong), read it off the game: the longest clock reading in
    period 1 tells us how long a period is.
    """
    # Largest clock reading in each period.
    seen: dict[int, float] = {}
    for period, clock in zip(pbp["PERIOD"], pbp["PCTIMESTRING"]):
        secs = _clock_seconds(clock)
        if secs is None:
            continue
        try:
            period = int(period)
        except (TypeError, ValueError):
            continue
        if secs > seen.get(period, -1.0):
            seen[period] = secs

    if not seen:
        return QUARTER_SEC, 4

    # Overtime is 5 minutes in every era, so any period whose clock ever reads
    # above 5:00 is regulation. Taking the highest such period index (rather
    # than a count) keeps periods 1..n contiguous even if one is oddly sparse.
    regulation = [p for p, mx in seen.items() if mx > OT_SEC]
    if not regulation:
        return QUARTER_SEC, 4
    n_reg = max(regulation)

    # Use the longest clock across ALL regulation periods, not just period 1 —
    # one period may have no event near its opening tip, but four rarely do.
    observed = max(seen[p] for p in regulation)
    reg_len = next((L for L in _PERIOD_LENGTHS if observed <= L), observed)
    return reg_len, n_reg


def _period_offset(period: int, reg_len: int = QUARTER_SEC, n_reg: int = 4) -> float:
    if period <= n_reg:
        return (period - 1) * reg_len
    return n_reg * reg_len + (period - n_reg - 1) * OT_SEC


def _period_len(period: int, reg_len: int = QUARTER_SEC, n_reg: int = 4) -> float:
    return reg_len if period <= n_reg else OT_SEC


def _pct_to_elapsed(period, pctimestring,
                    reg_len: int = QUARTER_SEC, n_reg: int = 4) -> float | None:
    """Convert a period + remaining clock into seconds elapsed since tip."""
    remaining = _clock_seconds(pctimestring)
    if remaining is None:
        return None
    try:
        period = int(period)
    except (TypeError, ValueError):
        return None
    return _period_offset(period, reg_len, n_reg) + (
        _period_len(period, reg_len, n_reg) - remaining
    )


def _chronological(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the PBP into true game order and attach an `ELAPSED` column.

    EVENTNUM is NOT reliably chronological in the WNBA V2 feed: some
    substitutions carry event numbers from a separate high block (e.g. 520 and
    521 sitting among events numbered 140-175). Sorting on EVENTNUM drops those
    subs at the end of the period, which silently corrupts starter detection —
    a player subbed out early looks like they were never on the floor. Sort on
    the game clock, and keep EVENTNUM only to break ties within a second.
    """
    # The feed occasionally emits a row twice, identical in every column and
    # sharing one EVENTNUM — 59 such rows across 55 games in the cache. Each
    # one double-counts its event (a duplicated made 3PT inflated one 2015 game
    # by 3 points), so drop them before anything reads the frame.
    df = pbp.drop_duplicates().copy()
    reg_len, n_reg = _period_scheme(df)
    df["ELAPSED"] = [
        _pct_to_elapsed(p, c, reg_len, n_reg)
        for p, c in zip(df["PERIOD"], df["PCTIMESTRING"])
    ]
    df = df[df["ELAPSED"].notna()]
    df = df.sort_values(["PERIOD", "ELAPSED", "EVENTNUM"]).reset_index(drop=True)
    df.attrs["period_scheme"] = (reg_len, n_reg)
    return df


# ── Home / away identification ────────────────────────────────────────────────

def _home_away_from_pbp(pbp: pd.DataFrame) -> tuple[int | None, int | None]:
    """
    Return (home_team_id, away_team_id) from the PBP itself.

    A substitution's text lands in HOMEDESCRIPTION or VISITORDESCRIPTION
    depending on which bench made it, and PLAYER1_TEAM_ID names that team —
    so subs alone pin down both sides without trusting PERSON1TYPE or the
    GameRotation frame order (which is what we're trying to verify).
    """
    votes: dict[str, dict[int, int]] = {"home": {}, "away": {}}
    for row in pbp.itertuples(index=False):
        if int(row.EVENTMSGTYPE) != SUB:
            continue
        try:
            tid = int(row.PLAYER1_TEAM_ID)
        except (TypeError, ValueError):
            continue
        if not tid:
            continue
        home_desc = getattr(row, "HOMEDESCRIPTION", None)
        away_desc = getattr(row, "VISITORDESCRIPTION", None)
        side = None
        if isinstance(home_desc, str) and home_desc.strip():
            side = "home"
        elif isinstance(away_desc, str) and away_desc.strip():
            side = "away"
        if side:
            votes[side][tid] = votes[side].get(tid, 0) + 1

    home = max(votes["home"], key=votes["home"].get) if votes["home"] else None
    away = max(votes["away"], key=votes["away"].get) if votes["away"] else None
    if home is not None and home == away:
        return None, None
    return home, away


def _player_teams(pbp: pd.DataFrame) -> dict[int, int]:
    """pid → team_id, by majority vote over every slot the player appears in."""
    votes: dict[int, dict[int, int]] = {}
    for slot in (1, 2, 3):
        pid_col, tid_col = f"PLAYER{slot}_ID", f"PLAYER{slot}_TEAM_ID"
        if pid_col not in pbp.columns:
            continue
        sub = pbp[[pid_col, tid_col]].dropna()
        for pid, tid in sub.itertuples(index=False):
            try:
                pid, tid = int(pid), int(tid)
            except (TypeError, ValueError):
                continue
            if pid <= 0 or tid <= 0:
                continue
            votes.setdefault(pid, {})
            votes[pid][tid] = votes[pid].get(tid, 0) + 1
    return {pid: max(v, key=v.get) for pid, v in votes.items()}


def _name_lookup(pbp: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for slot in (1, 2, 3):
        pid_col, name_col = f"PLAYER{slot}_ID", f"PLAYER{slot}_NAME"
        if name_col not in pbp.columns:
            continue
        sub = pbp[[pid_col, name_col]].dropna()
        for pid, name in sub.itertuples(index=False):
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            if pid > 0 and isinstance(name, str) and name.strip():
                names[pid] = name.strip()
    return names


# ── Opening-lineup detection ──────────────────────────────────────────────────

def _is_technical(row) -> bool:
    """True if a foul row is a technical, which a bench player can draw.

    Checked two ways because the action-type coding is not fully consistent
    across seasons: the numeric type, and the "T.FOUL" tag the feed writes into
    the description.
    """
    try:
        if int(row.EVENTMSGACTIONTYPE) in _TECHNICAL_FOUL_TYPES:
            return True
    except (TypeError, ValueError):
        pass
    for col in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
        desc = getattr(row, col, None)
        if isinstance(desc, str) and ("T.FOUL" in desc.upper()
                                      or "TECHNICAL" in desc.upper()):
            return True
    return False

def _detect_starters(period_pbp: pd.DataFrame, team_id: int,
                     pid_to_team: dict[int, int]) -> tuple[set[int], list[int]]:
    """
    Infer the five players who opened a period for one team.

    PBP never states a starting lineup, so it's recovered by elimination.
    Walking the period in order:

      - a player subbed OUT who was never subbed IN this period was on the
        floor when the period began → starter
      - a player appearing in any other on-court event, not yet subbed in
        → starter
      - a player subbed IN is recorded so later appearances don't count

    Returns the confirmed players ordered by first appearance. Normally there
    are exactly 5; `_resolve_opening` handles the over- and under-determined
    cases.
    """
    subbed_in: set[int] = set()
    confirmed: list[int] = []
    seen: set[int] = set()

    def _confirm(pid: int) -> None:
        if pid > 0 and pid not in subbed_in and pid not in seen:
            seen.add(pid)
            confirmed.append(pid)

    for row in period_pbp.itertuples(index=False):
        etype = int(row.EVENTMSGTYPE)

        if etype == SUB:
            try:
                out_pid = int(row.PLAYER1_ID)
                in_pid = int(row.PLAYER2_ID)
            except (TypeError, ValueError):
                continue
            if pid_to_team.get(out_pid) != team_id:
                continue
            _confirm(out_pid)
            if in_pid > 0:
                subbed_in.add(in_pid)
            continue

        if etype not in _ONCOURT_EVENTS:
            continue
        if etype == FOUL and _is_technical(row):
            continue

        for slot in (1, 2, 3):
            try:
                pid = int(getattr(row, f"PLAYER{slot}_ID"))
                ptype = int(getattr(row, f"PERSON{slot}TYPE"))
            except (TypeError, ValueError):
                continue
            if pid <= 0 or ptype not in _PLAYER_PERSON_TYPES:
                continue
            if pid_to_team.get(pid) != team_id:
                continue
            _confirm(pid)

    return confirmed


def _chain_valid(lineup: frozenset, subs, team_id: int) -> bool:
    """
    Replay a period's substitutions against a candidate opening lineup.

    Every sub is a strong consistency check: the outgoing player has to be on
    the floor and the incoming player has to be off it. A candidate lineup that
    survives the whole period's sub chain is almost certainly the real one,
    which is a far better test than "keep whoever showed up first".
    """
    for _, out_pid, in_pid, s_tid in subs:
        if s_tid != team_id:
            continue
        if out_pid not in lineup or in_pid in lineup:
            return False
        lineup = frozenset((lineup - {out_pid}) | {in_pid})
    return True


def _resolve_opening(confirmed: list[int], subs, team_id: int,
                     prior_close: frozenset) -> tuple[frozenset, bool]:
    """
    Turn the confirmed-player list into a definite 5-player opening lineup.

    Returns (lineup, certain). `certain` marks lineups the evidence pins down
    on its own, so the backward pass knows which ones it may overwrite and
    which it can trust as an anchor.
    """
    pool = list(dict.fromkeys(confirmed))

    if len(pool) == 5 and _chain_valid(frozenset(pool), subs, team_id):
        return frozenset(pool), True

    # Over-determined: more than five players look like starters, so at least
    # one piece of evidence is bad. Prefer the 5-subset that replays cleanly.
    # combinations() over an appearance-ordered pool tries the earliest five
    # first, which is the right tiebreak when several subsets are valid.
    if len(pool) > 5:
        for combo in combinations(pool[:9], 5):
            if _chain_valid(frozenset(combo), subs, team_id):
                return frozenset(combo), True
        return frozenset(pool[:5]), False

    # Under-determined: a player can go a whole period without recording a stat
    # or being subbed. Everyone confirmed is definitely on, so hold them fixed
    # and search the previous period's closing lineup for the remainder.
    extras = [p for p in prior_close if p not in pool]
    need = 5 - len(pool)
    if need <= len(extras):
        for combo in combinations(extras, need):
            cand = frozenset(pool) | frozenset(combo)
            if _chain_valid(cand, subs, team_id):
                return cand, False
    return frozenset(pool + extras[:need]), False


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct_intervals(pbp: pd.DataFrame) -> dict:
    """
    Rebuild every on-court interval for a game from its PBP.

    Returns a dict with:
        intervals    : list of {t_start, t_end, lineups: {team_id: frozenset}}
        home_team_id / away_team_id
        name_lookup  : pid → player name
        warnings     : list of str describing anything we had to paper over
    """
    pbp = _chronological(pbp)
    reg_len, n_reg = pbp.attrs["period_scheme"]

    home_team_id, away_team_id = _home_away_from_pbp(pbp)
    pid_to_team = _player_teams(pbp)
    names = _name_lookup(pbp)
    warnings: list[str] = []

    if home_team_id is None or away_team_id is None:
        return {
            "intervals": [], "home_team_id": home_team_id,
            "away_team_id": away_team_id, "name_lookup": names,
            "warnings": ["could not determine home/away from PBP"],
            "pbp": pbp, "period_scheme": (reg_len, n_reg),
        }

    teams = [home_team_id, away_team_id]
    periods = [int(p) for p in sorted(pbp["PERIOD"].dropna().unique())]

    # ── Stage 1: per-period substitutions and detected starters ──────────────
    subs_by: dict[int, list[tuple[float, int, int, int]]] = {}
    detected_by: dict[int, dict[int, set[int]]] = {}

    for period in periods:
        period_pbp = pbp[pbp["PERIOD"] == period]
        detected_by[period] = {
            tid: _detect_starters(period_pbp, tid, pid_to_team) for tid in teams
        }

        decoded: list[tuple[float, int, int, int]] = []
        for row in period_pbp[period_pbp["EVENTMSGTYPE"] == SUB].itertuples(index=False):
            try:
                out_pid, in_pid = int(row.PLAYER1_ID), int(row.PLAYER2_ID)
                tid = int(row.PLAYER1_TEAM_ID)
            except (TypeError, ValueError):
                continue
            if tid not in teams or out_pid <= 0 or in_pid <= 0:
                continue
            decoded.append((float(row.ELAPSED), out_pid, in_pid, tid))
        subs_by[period] = decoded

    def _apply(lineup: frozenset, subs, tid: int) -> frozenset:
        for _, out_pid, in_pid, s_tid in subs:
            if s_tid == tid:
                lineup = frozenset((lineup - {out_pid}) | {in_pid})
        return lineup

    def _undo(lineup: frozenset, subs, tid: int) -> frozenset:
        for _, out_pid, in_pid, s_tid in reversed(subs):
            if s_tid == tid:
                lineup = frozenset((lineup - {in_pid}) | {out_pid})
        return lineup

    # ── Stage 2: forward pass ────────────────────────────────────────────────
    # A period's opening lineup is whatever we could confirm, topped up from
    # the previous period's closing lineup.
    opening: dict[int, dict[int, frozenset]] = {}
    certain: dict[int, dict[int, bool]] = {}
    prev_close: dict[int, frozenset] = {t: frozenset() for t in teams}

    for period in periods:
        opening[period], certain[period] = {}, {}
        for tid in teams:
            lineup, is_certain = _resolve_opening(
                detected_by[period][tid], subs_by[period], tid, prev_close[tid]
            )
            opening[period][tid] = lineup
            certain[period][tid] = is_certain
            prev_close[tid] = _apply(lineup, subs_by[period], tid)

    # ── Stage 3: backward pass ───────────────────────────────────────────────
    # Substitutions made between periods produce no PBP event, so a player who
    # comes on at the break and never records a stat is invisible going
    # forward — the forward pass wrongly keeps whoever they replaced.
    #
    # The next period usually pins them down, though. Rewinding that period's
    # opening lineup through this period's subs recovers what the floor must
    # have looked like when this period began. Only accept it when it still
    # contains everyone we positively confirmed; that guards against a
    # between-period sub at the *next* break.
    for period in reversed(periods[:-1]):
        nxt = periods[periods.index(period) + 1]
        for tid in teams:
            if certain[period][tid] or not certain[nxt][tid]:
                continue
            cand = _undo(opening[nxt][tid], subs_by[period], tid)
            if (len(cand) == 5
                    and cand >= set(detected_by[period][tid])
                    and _chain_valid(cand, subs_by[period], tid)):
                opening[period][tid] = cand
                certain[period][tid] = True

    # ── Stage 4: emit intervals ──────────────────────────────────────────────
    lineups: dict[int, frozenset] = {}
    intervals: list[dict] = []

    for period in periods:
        period_start = _period_offset(period, reg_len, n_reg)
        period_end = period_start + _period_len(period, reg_len, n_reg)
        lineups = dict(opening[period])

        for tid in teams:
            if len(lineups[tid]) != 5:
                warnings.append(
                    f"P{period} team {tid}: only {len(lineups[tid])} players "
                    f"identified at period start"
                )
            elif not certain[period][tid]:
                warnings.append(f"P{period} team {tid}: opening lineup inferred, "
                                f"not confirmed")

        t_current = period_start

        # Subs made during one stoppage all share a clock value. Emit a single
        # interval up to that moment, then apply the whole wave at once —
        # otherwise the second sub of a pair looks like a zero-length stint.
        decoded = [s for s in subs_by[period] if s[0] >= t_current]

        i = 0
        while i < len(decoded):
            elapsed = decoded[i][0]
            j = i
            while j < len(decoded) and decoded[j][0] == elapsed:
                j += 1
            wave, i = decoded[i:j], j

            if elapsed > t_current:
                intervals.append({
                    "t_start": t_current,
                    "t_end": elapsed,
                    "lineups": dict(lineups),
                })
            t_current = elapsed

            for _, out_pid, in_pid, tid in wave:
                if out_pid not in lineups[tid] and lineups[tid]:
                    warnings.append(
                        f"P{period} t={elapsed:.0f}: {names.get(out_pid, out_pid)} "
                        f"subbed out but was not on the floor"
                    )
                lineups[tid] = frozenset((lineups[tid] - {out_pid}) | {in_pid})

        if period_end > t_current:
            intervals.append({
                "t_start": t_current,
                "t_end": period_end,
                "lineups": dict(lineups),
            })

    return {
        "intervals": intervals,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "name_lookup": names,
        "warnings": warnings,
        # Returned so callers reuse this exact parse — re-deriving ELAPSED with
        # a different period scheme would silently misalign events to stints.
        "pbp": pbp,
        "period_scheme": (reg_len, n_reg),
    }


def reconstruct_rotation(game_id: str, pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Reconstruct a game's rotation in the same shape GameRotation returns, so
    the two can be compared or swapped for one another.

    Columns: GAME_ID, TEAM_ID, PERSON_ID, PLAYER_NAME, IN_TIME_REAL,
    OUT_TIME_REAL (tenths of a second, matching the API), HOME_AWAY.
    """
    if pbp is None:
        pbp = get_pbp(game_id)
    if pbp is None or pbp.empty:
        return pd.DataFrame()

    res = reconstruct_intervals(pbp)
    if not res["intervals"]:
        return pd.DataFrame()

    home_id, away_id = res["home_team_id"], res["away_team_id"]
    names = res["name_lookup"]

    # Per player, stitch consecutive intervals into maximal on-court runs.
    runs: list[dict] = []
    open_run: dict[tuple[int, int], float] = {}   # (tid, pid) → run start
    prev_end: float | None = None

    for iv in res["intervals"]:
        on_now = {(tid, pid) for tid, lu in iv["lineups"].items() for pid in lu}
        for key in list(open_run):
            if key not in on_now or (prev_end is not None and iv["t_start"] != prev_end):
                tid, pid = key
                runs.append({"TEAM_ID": tid, "PERSON_ID": pid,
                             "IN_TIME_REAL": open_run.pop(key) * 10.0,
                             "OUT_TIME_REAL": prev_end * 10.0})
        for key in on_now:
            open_run.setdefault(key, iv["t_start"])
        prev_end = iv["t_end"]

    for (tid, pid), start in open_run.items():
        runs.append({"TEAM_ID": tid, "PERSON_ID": pid,
                     "IN_TIME_REAL": start * 10.0,
                     "OUT_TIME_REAL": prev_end * 10.0})

    df = pd.DataFrame(runs)
    if df.empty:
        return df
    df["GAME_ID"] = str(game_id)
    df["PLAYER_NAME"] = df["PERSON_ID"].map(names)
    df["HOME_AWAY"] = df["TEAM_ID"].map({home_id: "home", away_id: "away"})
    return df.sort_values(["TEAM_ID", "PERSON_ID", "IN_TIME_REAL"]).reset_index(drop=True)[
        ["GAME_ID", "TEAM_ID", "PERSON_ID", "PLAYER_NAME",
         "IN_TIME_REAL", "OUT_TIME_REAL", "HOME_AWAY"]
    ]
