# data/stints.py
"""
Reconstruct 5v5 lineup stints for WNBA games.

Lineups and box events both come from PlayByPlayV2 — ONE API call per game.
Lineups used to come from GameRotation, which meant a second call per game and
left us unable to build stints for any game the endpoint hadn't covered yet.
`data.pbp_rotation` reproduces GameRotation's on-court sets from the PBP
exactly: verified second-by-second on 1,170 games, 1,169 of them perfect.
See `python -m data.verify_rotation`.

Output schema is unchanged, so `data.rapm` and `data.evaluation` are unaffected.
"""

from bisect import bisect_right

import pandas as pd

from data.cache import get_game_ids, get_pbp
from data.pbp_rotation import reconstruct_intervals

# EVENTMSGTYPE values we score off
MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER = 1, 2, 3, 4, 5

# PLAYER*_ID values at or above this are team ids, not players — the WNBA feed
# puts a team id in PLAYER1_ID for team rebounds and team turnovers.
_TEAM_ID_FLOOR = 1_610_000_000


def _row_team(row) -> int | None:
    """Team credited with an event, tolerating team-level rows."""
    try:
        tid = int(row.PLAYER1_TEAM_ID)
        if tid > 0:
            return tid
    except (TypeError, ValueError):
        pass
    # Team rebound / team turnover: PLAYER1_TEAM_ID is null and the team id
    # sits in PLAYER1_ID instead.
    try:
        pid = int(row.PLAYER1_ID)
        return pid if pid >= _TEAM_ID_FLOOR else None
    except (TypeError, ValueError):
        return None


def _free_throw_made(row, desc: str) -> bool:
    """
    Decide whether a free-throw row went in.

    "Not tagged MISS" is not enough. The last free throw of a game is sometimes
    written as a bare "Free Throw 2 of 2" — no MISS, no running point total, no
    score change — even though it missed, which silently credited a phantom
    point. Positive evidence is required instead:

      MISS in the text          -> missed
      a "(N PTS)" running total -> made  (modern feeds)
      the SCORE column moved    -> made  (1990s feeds omit the PTS suffix)

    Anything else is treated as a miss, since a made free throw always leaves
    one of those two traces.
    """
    if "MISS" in desc:
        return False
    if "PTS)" in desc:
        return True
    score = getattr(row, "SCORE", None)
    return isinstance(score, str) and bool(score.strip())


def _description(row) -> str:
    return " ".join(
        str(getattr(row, c, "") or "")
        for c in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION")
    )


# ── Per-game stint builder ─────────────────────────────────────────────────────

def _game_stints(game_id: str) -> tuple[list[dict], dict]:
    try:
        pbp = get_pbp(game_id)
    except Exception as e:
        print(f"  [warn] {game_id} pbp: {e}")
        return [], {}
    if pbp is None or pbp.empty:
        return [], {}

    res = reconstruct_intervals(pbp)
    # Reuse the frame reconstruct_intervals parsed. It carries ELAPSED computed
    # under this game's period scheme — 20-minute halves before 2006, 10-minute
    # quarters after — so re-deriving it here could misalign events to stints.
    pbp = res["pbp"]
    home_team_id, away_team_id = res["home_team_id"], res["away_team_id"]
    name_lookup = res["name_lookup"]

    if home_team_id is None or away_team_id is None:
        print(f"  [warn] {game_id} could not determine home/away from PBP")
        return [], {}

    # Keep only true 5v5 intervals. A period the PBP can't fully pin down comes
    # back short-handed rather than guessed (see pbp_rotation's docstring), and
    # a 4-player lineup would silently corrupt every per-possession rate.
    intervals = [
        iv for iv in res["intervals"]
        if len(iv["lineups"].get(home_team_id, ())) == 5
        and len(iv["lineups"].get(away_team_id, ())) == 5
    ]
    if not intervals:
        return [], name_lookup

    n = len(intervals)
    h_pts = [0.0] * n; a_pts = [0.0] * n
    # Oliver possession inputs: POSS = FGA - OREB + TOV + 0.44 * FTA
    h_fga = [0.0] * n; a_fga = [0.0] * n
    h_oreb = [0.0] * n; a_oreb = [0.0] * n
    h_tov = [0.0] * n; a_tov = [0.0] * n
    h_fta = [0.0] * n; a_fta = [0.0] * n

    starts = [iv["t_start"] for iv in intervals]
    last_end = intervals[-1]["t_end"]

    def find_interval(elapsed: float) -> int | None:
        if elapsed < starts[0] or elapsed > last_end:
            return None
        idx = bisect_right(starts, elapsed) - 1
        return idx if idx >= 0 else None

    # Team of the most recent missed shot — classifies TEAM rebounds, which
    # carry no per-player tally.
    last_miss_team: int | None = None
    # Per-player running offensive-rebound count, read from the "(Off:N Def:N)"
    # tally the feed appends to every player rebound. A rebound is offensive iff
    # that counter ticks up — more reliable than guessing from the last miss,
    # which gets tip-outs and multi-rebound scrambles wrong.
    player_oreb_seen: dict[int, int] = {}

    for row in pbp.itertuples(index=False):
        elapsed = float(row.ELAPSED)
        etype = int(row.EVENTMSGTYPE)
        if etype not in (MADE_SHOT, MISSED_SHOT, FREE_THROW, REBOUND, TURNOVER):
            continue

        tid = _row_team(row)
        if tid is None:
            continue
        is_home, is_away = tid == home_team_id, tid == away_team_id
        idx = find_interval(elapsed)
        desc = _description(row).upper()

        if etype == MADE_SHOT:
            if idx is not None and (is_home or is_away):
                pts = 3.0 if "3PT" in desc else 2.0
                if is_home:
                    h_pts[idx] += pts; h_fga[idx] += 1.0
                else:
                    a_pts[idx] += pts; a_fga[idx] += 1.0
            last_miss_team = None

        elif etype == MISSED_SHOT:
            if idx is not None and (is_home or is_away):
                (h_fga if is_home else a_fga)[idx] += 1.0
            last_miss_team = tid

        elif etype == FREE_THROW:
            made = _free_throw_made(row, desc)
            if idx is not None and (is_home or is_away):
                if is_home:
                    h_fta[idx] += 1.0
                    if made:
                        h_pts[idx] += 1.0
                else:
                    a_fta[idx] += 1.0
                    if made:
                        a_pts[idx] += 1.0
            last_miss_team = None if made else tid

        elif etype == REBOUND:
            offensive = None
            try:
                pid = int(row.PLAYER1_ID)
            except (TypeError, ValueError):
                pid = 0
            if pid and pid < _TEAM_ID_FLOOR and "OFF:" in desc:
                # "(Off:0 Def:1)" — offensive iff the offensive counter moved.
                try:
                    off_n = int(desc.split("OFF:")[1].split()[0].strip(") "))
                    offensive = off_n > player_oreb_seen.get(pid, 0)
                    player_oreb_seen[pid] = max(off_n, player_oreb_seen.get(pid, 0))
                except (ValueError, IndexError):
                    offensive = None
            if offensive is None:
                offensive = last_miss_team is not None and tid == last_miss_team

            if idx is not None and offensive and (is_home or is_away):
                (h_oreb if is_home else a_oreb)[idx] += 1.0
            last_miss_team = None

        elif etype == TURNOVER:
            if idx is not None and (is_home or is_away):
                (h_tov if is_home else a_tov)[idx] += 1.0
            last_miss_team = None

    def poss(fga, oreb, tov, fta) -> float:
        return fga - oreb + tov + 0.44 * fta

    # Cumulative score at each boundary (index i = score before interval i).
    h_cum = [0.0] * (n + 1)
    a_cum = [0.0] * (n + 1)
    for i in range(n):
        h_cum[i + 1] = h_cum[i] + h_pts[i]
        a_cum[i + 1] = a_cum[i] + a_pts[i]

    stints: list[dict] = []
    for i, iv in enumerate(intervals):
        # Oliver's formula subtracts offensive rebounds, so a short stint with
        # one made shot and one offensive board nets to exactly zero
        # possessions. Dropping on `poss <= 0` therefore threw away real
        # scoring — about 0.3 points per game, always in the same direction.
        # Emit anything with activity and let callers set a possession floor;
        # data.rapm already filters at min_stint_poss, so its input is
        # unchanged by this.
        home_poss = max(poss(h_fga[i], h_oreb[i], h_tov[i], h_fta[i]), 0.0)
        away_poss = max(poss(a_fga[i], a_oreb[i], a_tov[i], a_fta[i]), 0.0)
        activity = (
            h_fga[i] + a_fga[i] + h_fta[i] + a_fta[i]
            + h_tov[i] + a_tov[i] + h_oreb[i] + a_oreb[i]
            + h_pts[i] + a_pts[i]
        )
        if activity <= 0:
            continue
        stints.append({
            "game_id": game_id,
            "t_start": iv["t_start"],
            "t_end": iv["t_end"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_lineup": iv["lineups"][home_team_id],
            "away_lineup": iv["lineups"][away_team_id],
            "home_pts": h_pts[i],
            "away_pts": a_pts[i],
            "home_poss": home_poss,
            "away_poss": away_poss,
            "score_margin": h_cum[i] - a_cum[i],   # home − away at stint start
        })

    return stints, name_lookup


# ── Public API ────────────────────────────────────────────────────────────────

def build_stints(
    seasons,
    season_type: str = "Regular Season",
) -> tuple[pd.DataFrame, dict]:
    """
    Build all 5v5 lineup stints for the given season(s).

    Parameters
    ----------
    seasons     : str or list of str, e.g. "2024" or ["2023", "2024"]
    season_type : "Regular Season" | "Playoffs"

    Returns
    -------
    stints_df   : DataFrame with one row per stint
    name_lookup : dict mapping PERSON_ID → player name string
    """
    game_ids_df = get_game_ids(seasons, season_type=season_type)
    if game_ids_df.empty:
        return pd.DataFrame(), {}
    game_ids = game_ids_df["GAME_ID"].astype(str).str.zfill(10).tolist()

    all_stints: list[dict] = []
    combined_names: dict = {}
    empty_games: list[str] = []

    for i, gid in enumerate(game_ids):
        if i % 25 == 0:
            print(f"  [{i+1}/{len(game_ids)}] game {gid}")
        game_stints, game_names = _game_stints(gid)
        if not game_stints:
            empty_games.append(gid)
        all_stints.extend(game_stints)
        combined_names.update(game_names)

    if empty_games:
        print(f"  [warn] {len(empty_games)} game(s) produced no stints: "
              f"{', '.join(empty_games[:5])}"
              f"{' ...' if len(empty_games) > 5 else ''}")

    if not all_stints:
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_stints)
    print(f"Built {len(df):,} stints across "
          f"{len(game_ids) - len(empty_games)}/{len(game_ids)} games.")
    return df, combined_names
