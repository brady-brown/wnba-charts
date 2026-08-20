# data/audit_stints.py
"""
audit_stints.py — Whole-history consistency audit of the built stint tables.

Runs entirely off cached PBP and the parquet output — no API calls — so it can
be re-run freely after any change to the reconstruction.

Three checks, none of which need GameRotation (which does not exist before
~2012, see data/verify_rotation.py):

  points     every game's stint points must sum to the final score in its own
             play-by-play. Catches scoring dropped by interval gaps or filters.
  floor      total on-court time must equal exactly 10 players x game length.
             Catches lineups that drift to 4 or 6 players.
  continuity intervals must tile the game with no gaps or overlaps.

Usage:
    python -m data.audit_stints
    python -m data.audit_stints --stints-dir data/stints_out --verbose
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

from data.cache import CACHE_DIR, _cache_key
from data.pbp_rotation import reconstruct_intervals


def _score_total(text) -> float | None:
    try:
        home, away = str(text).split("-")
        return float(home) + float(away)
    except ValueError:
        return None


def _final_score(game_id: str) -> float | None:
    """
    Combined final score from the game's own PBP SCORE column.

    Returns None when that column is untrustworthy. In some feeds SCORE does
    not start at 0-0 — one 2019 game opens at "32 - 29" — meaning the running
    score was truncated even though the scoring EVENTS are all present. Spot
    checks against BoxScoreTraditionalV2 confirmed our stint totals were right
    and the SCORE column was wrong in every such case, so these games must be
    excluded from the baseline rather than counted against us.

    The first scoring play can be worth at most 3 points, so a first SCORE
    above that means the column started mid-game.
    """
    path = CACHE_DIR / _cache_key("pbp", game_id=game_id)
    if not path.exists():
        return None
    try:
        scores = pd.read_csv(path, usecols=["SCORE"])["SCORE"].dropna()
    except Exception:
        return None
    if scores.empty:
        return None
    first = _score_total(scores.iloc[0])
    if first is None or first > 3:
        return None
    return _score_total(scores.iloc[-1])


def audit_points(stints_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(stints_dir / "stints_*.parquet"))):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        season = str(df["season"].iloc[0])
        season_type = str(df["season_type"].iloc[0])
        totals = df.groupby("game_id")[["home_pts", "away_pts"]].sum().sum(axis=1)
        for gid, ours in totals.items():
            official = _final_score(str(gid).zfill(10))
            if official is None:
                continue
            rows.append({"season": season, "season_type": season_type,
                         "game_id": gid, "ours": float(ours),
                         "official": official, "delta": float(ours) - official})
    return pd.DataFrame(rows)


def audit_structure(seasons: list[str] | None = None) -> pd.DataFrame:
    """Floor-time and continuity checks, straight from cached PBP."""
    rows = []
    for path in sorted(glob.glob(str(CACHE_DIR / "pbp_*.csv"))):
        try:
            pbp = pd.read_csv(path)
        except Exception:
            continue
        if pbp.empty or "GAME_ID" not in pbp.columns:
            continue
        game_id = str(pbp["GAME_ID"].iloc[0]).zfill(10)
        try:
            res = reconstruct_intervals(pbp)
        except Exception as e:
            rows.append({"game_id": game_id, "status": f"error: {type(e).__name__}"})
            continue
        intervals = res["intervals"]
        if not intervals:
            rows.append({"game_id": game_id, "status": "no_intervals"})
            continue

        game_len = intervals[-1]["t_end"] - intervals[0]["t_start"]
        floor = sum(
            (iv["t_end"] - iv["t_start"]) * sum(len(l) for l in iv["lineups"].values())
            for iv in intervals
        )
        gaps = sum(
            1 for a, b in zip(intervals[:-1], intervals[1:])
            if abs(b["t_start"] - a["t_end"]) > 1e-9
        )
        sizes = {len(l) for iv in intervals for l in iv["lineups"].values()}
        rows.append({
            "game_id": game_id,
            "status": "ok",
            "period_scheme": str(res["period_scheme"]),
            "game_len": game_len,
            "on_floor": floor / game_len if game_len else 0.0,
            "gaps": gaps,
            "lineup_sizes": ",".join(str(s) for s in sorted(sizes)),
            "warnings": len(res["warnings"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stints-dir", default="data/stints_out")
    ap.add_argument("--skip-structure", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stints_dir = Path(args.stints_dir)

    print("=" * 70)
    print("POINTS — stint totals vs each game's final score")
    print("=" * 70)
    pts = audit_points(stints_dir)
    if pts.empty:
        print("  no stint parquet found")
    else:
        for season, sub in pts.groupby("season"):
            exact = (sub["delta"].abs() < 0.5).mean() * 100
            flag = "" if exact == 100 else "   <-- "
            print(f"  {season}  games={len(sub):>4}  exact={exact:6.2f}%  "
                  f"mean_delta={sub['delta'].mean():+6.3f}  "
                  f"max|delta|={sub['delta'].abs().max():4.0f}{flag}")
        print(f"\n  ALL  games={len(pts):,}  "
              f"exact={100*(pts['delta'].abs()<0.5).mean():.3f}%  "
              f"total_delta={pts['delta'].sum():+.0f} points")
        if args.verbose:
            bad = pts[pts["delta"].abs() >= 0.5]
            if len(bad):
                print("\n  games not reconciling:")
                print(bad.head(20).to_string(index=False))

    if args.skip_structure:
        return

    print("\n" + "=" * 70)
    print("STRUCTURE — floor time and interval continuity")
    print("=" * 70)
    st = audit_structure()
    if st.empty:
        print("  no cached pbp found")
        return
    ok = st[st["status"] == "ok"]
    print(f"  games audited      : {len(ok):,} / {len(st):,}")
    for status, n in st[st["status"] != "ok"]["status"].value_counts().items():
        print(f"      {n:>4}  {status}")
    exact10 = (ok["on_floor"] - 10).abs() < 1e-9
    print(f"  exactly 10 on floor: {exact10.sum():,} "
          f"({100*exact10.mean():.2f}%)")
    print(f"  games with gaps    : {int((ok['gaps'] > 0).sum())}")
    print(f"  strict 5v5 only    : {int((ok['lineup_sizes'] == '5').sum()):,}")
    print(f"  period schemes     : {ok['period_scheme'].value_counts().to_dict()}")
    if args.verbose:
        bad = ok[~exact10]
        if len(bad):
            print("\n  games not exactly 10 on floor:")
            print(bad.head(20)[
                ["game_id", "period_scheme", "game_len", "on_floor",
                 "lineup_sizes", "warnings"]
            ].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
