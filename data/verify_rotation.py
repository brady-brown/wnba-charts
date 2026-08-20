# data/verify_rotation.py
"""
verify_rotation.py — Check our PBP-reconstructed rotations against GameRotation.

GameRotation is the league's own on-court record (IN_TIME_REAL / OUT_TIME_REAL
per player), derived independently of the play-by-play text. That makes it a
gold standard for whatever we rebuild from PBP: if the two agree second by
second, the PBP reconstruction can stand in for the API call.

Comparison is per SECOND and per TEAM_ID (never per home/away label — the
labels are exactly what's in question; see the HOME_AWAY audit below).

Usage:
    python -m data.verify_rotation                  # every cached game
    python -m data.verify_rotation --season 2024
    python -m data.verify_rotation --limit 50 --verbose
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache import CACHE_DIR, _cache_key
from data.pbp_rotation import reconstruct_intervals

SEASONS = ["2021", "2022", "2023", "2024", "2025", "2026"]


# ── Cached-game discovery (no API calls) ──────────────────────────────────────

def cached_games(seasons=None, season_type: str = "Regular Season") -> pd.DataFrame:
    """Games that have BOTH pbp and rotation already on disk."""
    frames = []
    for season in (seasons or SEASONS):
        path = CACHE_DIR / _cache_key("game_ids", season=season,
                                      season_type=season_type)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    games = pd.concat(frames, ignore_index=True)
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    games["pbp_path"] = [CACHE_DIR / _cache_key("pbp", game_id=g)
                         for g in games["GAME_ID"]]
    games["rot_path"] = [CACHE_DIR / _cache_key("rotation", game_id=g)
                         for g in games["GAME_ID"]]
    keep = [p.exists() and r.exists()
            for p, r in zip(games["pbp_path"], games["rot_path"])]
    return games[keep].reset_index(drop=True)


# ── Occupancy grids ───────────────────────────────────────────────────────────

def _grid_from_runs(runs, n_sec: int) -> dict[tuple[int, int], np.ndarray]:
    """(team_id, pid) → bool array marking each second the player is on court.

    Second t covers [t, t+1); a run [in, out) claims seconds ceil(in)..out-1.
    """
    grid: dict[tuple[int, int], np.ndarray] = {}
    for tid, pid, t_in, t_out in runs:
        lo, hi = int(np.ceil(t_in)), int(np.floor(t_out))
        if hi <= lo:
            continue
        arr = grid.setdefault((int(tid), int(pid)), np.zeros(n_sec, dtype=bool))
        arr[max(lo, 0):min(hi, n_sec)] = True
    return grid


def _api_runs(rot: pd.DataFrame):
    for r in rot.itertuples(index=False):
        yield (int(r.TEAM_ID), int(r.PERSON_ID),
               float(r.IN_TIME_REAL) / 10.0, float(r.OUT_TIME_REAL) / 10.0)


def _pbp_runs(intervals):
    for iv in intervals:
        for tid, lineup in iv["lineups"].items():
            for pid in lineup:
                yield (tid, pid, iv["t_start"], iv["t_end"])


# ── Per-game comparison ───────────────────────────────────────────────────────

def compare_game(game_id: str, pbp: pd.DataFrame, rot: pd.DataFrame) -> dict:
    res = reconstruct_intervals(pbp)
    intervals = res["intervals"]
    out = {
        "game_id": game_id,
        "n_warnings": len(res["warnings"]),
        "warnings": res["warnings"],
    }
    if not intervals:
        return {**out, "status": "no_intervals"}

    # Game length: trust the API's last exit, but never truncate our own.
    n_sec = int(max(rot["OUT_TIME_REAL"].max() / 10.0,
                    intervals[-1]["t_end"]))

    api = _grid_from_runs(list(_api_runs(rot)), n_sec)
    ours = _grid_from_runs(list(_pbp_runs(intervals)), n_sec)

    keys = sorted(set(api) | set(ours))
    zero = np.zeros(n_sec, dtype=bool)
    A = np.array([api.get(k, zero) for k in keys])
    B = np.array([ours.get(k, zero) for k in keys])

    teams = sorted({k[0] for k in keys})
    team_rows = {t: np.array([k[0] == t for k in keys]) for t in teams}

    # A second is "matched" for a team when the exact set of five agrees.
    per_team = {}
    for t in teams:
        rows = team_rows[t]
        diff = (A[rows] != B[rows]).any(axis=0)
        # Only judge seconds the API actually covers 5 players for; a gap in
        # the official feed is not our error.
        covered = A[rows].sum(axis=0) == 5
        per_team[t] = {
            "covered_sec": int(covered.sum()),
            "match_sec": int((~diff & covered).sum()),
            "ours_five_sec": int((B[rows].sum(axis=0) == 5).sum()),
        }

    covered_all = np.array([A[team_rows[t]].sum(axis=0) == 5 for t in teams]).all(axis=0)
    diff_all = (A != B).any(axis=0)

    # Player-minute deltas (ours − API), in seconds.
    player_delta = {
        k: int(ours.get(k, zero).sum()) - int(api.get(k, zero).sum())
        for k in keys
    }
    worst = max(player_delta.items(), key=lambda kv: abs(kv[1])) if player_delta else ((0, 0), 0)

    first_bad = int(np.argmax(diff_all & covered_all)) if (diff_all & covered_all).any() else None

    return {
        **out,
        "status": "ok",
        "n_sec": n_sec,
        "covered_sec": int(covered_all.sum()),
        "match_sec": int((~diff_all & covered_all).sum()),
        "mismatch_sec": int((diff_all & covered_all).sum()),
        "first_mismatch_sec": first_bad,
        "max_player_delta_sec": worst[1],
        "max_player_delta_pid": worst[0][1],
        "n_players_off": sum(1 for v in player_delta.values() if v != 0),
        "per_team": per_team,
        "api_home_teams": sorted(rot.loc[rot["HOME_AWAY"] == "home", "TEAM_ID"].unique().tolist()),
        "pbp_home_team": res["home_team_id"],
        "pbp_away_team": res["away_team_id"],
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def run(seasons=None, limit: int | None = None, verbose: bool = False) -> pd.DataFrame:
    games = cached_games(seasons)
    if games.empty:
        print("No games with both pbp and rotation cached.")
        return pd.DataFrame()
    if limit:
        games = games.head(limit)

    print(f"Verifying {len(games)} games "
          f"({', '.join(sorted(games['season'].unique()))})\n")

    rows = []
    for i, g in enumerate(games.itertuples(index=False)):
        if i % 100 == 0:
            print(f"  [{i+1}/{len(games)}] {g.GAME_ID}")
        try:
            pbp = pd.read_csv(g.pbp_path)
            rot = pd.read_csv(g.rot_path)
        except Exception as e:
            rows.append({"game_id": g.GAME_ID, "status": f"read_error: {e}"})
            continue
        if pbp.empty or rot.empty:
            rows.append({"game_id": g.GAME_ID, "status": "empty"})
            continue
        try:
            r = compare_game(g.GAME_ID, pbp, rot)
        except Exception as e:
            rows.append({"game_id": g.GAME_ID, "status": f"error: {type(e).__name__}: {e}"})
            continue
        r["season"] = g.season
        r["matchup"] = g.MATCHUP
        rows.append(r)

    df = pd.DataFrame(rows)
    report(df, verbose=verbose)
    return df


def report(df: pd.DataFrame, verbose: bool = False) -> None:
    ok = df[df["status"] == "ok"].copy()
    print("\n" + "=" * 72)
    print("PBP-RECONSTRUCTED ROTATION  vs  GameRotation API")
    print("=" * 72)
    print(f"games compared      : {len(ok)} / {len(df)}")
    bad_status = df[df["status"] != "ok"]
    if len(bad_status):
        print(f"games not compared  : {len(bad_status)}")
        for s, n in bad_status["status"].value_counts().items():
            print(f"    {n:>4}  {s}")
    if ok.empty:
        return

    ok["pct"] = 100.0 * ok["match_sec"] / ok["covered_sec"].clip(lower=1)
    total_cov = ok["covered_sec"].sum()
    total_match = ok["match_sec"].sum()

    print(f"\nseconds compared    : {total_cov:,}")
    print(f"seconds matching    : {total_match:,}  ({100.0*total_match/total_cov:.4f}%)")
    print(f"perfect games       : {(ok['mismatch_sec'] == 0).sum()} / {len(ok)} "
          f"({100.0*(ok['mismatch_sec']==0).mean():.1f}%)")
    print(f"games >= 99% match  : {(ok['pct'] >= 99).sum()}")
    print(f"games <  95% match  : {(ok['pct'] < 95).sum()}")

    print("\nper-game match % distribution")
    for q in [0.0, 0.01, 0.05, 0.25, 0.50]:
        print(f"    p{int(q*100):<3}  {ok['pct'].quantile(q):7.3f}%")

    # ── HOME_AWAY label audit ────────────────────────────────────────────────
    # Read straight off the cached CSV, so this reports the label as it was
    # written to disk. PBP tells us which bench made each substitution, so its
    # home/away is independent of how GameRotation orders its two frames.
    agree = ok.apply(
        lambda r: bool(r["api_home_teams"]) and r["api_home_teams"][0] == r["pbp_home_team"],
        axis=1,
    )
    print(f"\nHOME_AWAY as stored in cache, agreeing with PBP : "
          f"{agree.sum()} / {len(ok)}")
    if agree.sum() != len(ok):
        print("    ^ these cache files were written with the two GameRotation")
        print("      frames swapped. get_game_rotation() now re-derives the")
        print("      label from PBP on read, so consumers get the right one.")

    worst = ok.nsmallest(15, "pct")
    if (worst["pct"] < 100).any():
        print("\nworst games")
        print(f"    {'game_id':<12}{'season':<8}{'matchup':<16}{'match%':>9}"
              f"{'bad_s':>7}{'1st_bad':>9}{'warns':>7}")
        for r in worst.itertuples(index=False):
            if r.pct >= 100:
                continue
            print(f"    {r.game_id:<12}{r.season:<8}{str(r.matchup):<16}"
                  f"{r.pct:>8.2f}%{r.mismatch_sec:>7}{str(r.first_mismatch_sec):>9}"
                  f"{r.n_warnings:>7}")
            if verbose and r.warnings:
                for w in r.warnings[:5]:
                    print(f"        - {w}")

    print(f"\nplayer-seconds: games with any player off by >0s : "
          f"{(ok['n_players_off'] > 0).sum()}")
    print(f"largest single-player error                      : "
          f"{ok['max_player_delta_sec'].abs().max()}s")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons",
                    help="repeatable, e.g. --season 2024 --season 2025")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verbose", action="store_true",
                    help="print reconstruction warnings for the worst games")
    ap.add_argument("--out", type=str, help="write per-game results to CSV")
    args = ap.parse_args()

    df = run(seasons=args.seasons, limit=args.limit, verbose=args.verbose)
    if args.out and not df.empty:
        df.drop(columns=["warnings", "per_team"], errors="ignore").to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
