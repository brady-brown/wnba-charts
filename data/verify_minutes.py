# data/verify_minutes.py
"""
verify_minutes.py — Validate PBP-reconstructed lineups against box-score minutes.

`verify_rotation.py` checks us against GameRotation, which is the stronger test
but only exists for recent seasons — the endpoint returns nothing before ~2008
and returns corrupt rows in the halves era (negative IN_TIME_REAL, exits before
entries, half a game of coverage). That leaves 1997-2011 unverified, which
matters because those are exactly the seasons the PBP-only pipeline unlocks.

The official box score covers every season back to 1997 and records each
player's minutes independently of the play-by-play, so summing our on-court
intervals per player and comparing to MIN is a true out-of-sample check.

It is a weaker test than GameRotation — it constrains totals, not who shared
the floor with whom — so agreement here means "no player is on for the wrong
length of time", not "every lineup is exactly right".

Usage:
    python -m data.verify_minutes --season 2002 --limit 20
    python -m data.verify_minutes --season 1998 --season 2003 --limit 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data.cache import get_box_score, get_game_ids, get_pbp
from data.pbp_rotation import reconstruct_intervals


def _min_to_seconds(value) -> float | None:
    """Box-score MIN is 'MM:SS' in most seasons and a bare number in some."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan"):
        return 0.0
    if ":" in text:
        try:
            m, s = text.split(":")[:2]
            return int(m) * 60 + float(s)
        except ValueError:
            return None
    try:
        return float(text) * 60.0
    except ValueError:
        return None


def compare_game(game_id: str) -> dict | None:
    pbp = get_pbp(game_id)
    if pbp is None or pbp.empty:
        return None
    res = reconstruct_intervals(pbp)
    if not res["intervals"]:
        return None

    ours: dict[int, float] = {}
    for iv in res["intervals"]:
        dur = iv["t_end"] - iv["t_start"]
        for lineup in iv["lineups"].values():
            for pid in lineup:
                ours[pid] = ours.get(pid, 0.0) + dur

    box = get_box_score(game_id)
    if box is None or box.empty:
        return None

    rows = []
    for r in box.itertuples(index=False):
        try:
            pid = int(r.PLAYER_ID)
        except (TypeError, ValueError):
            continue
        official = _min_to_seconds(getattr(r, "MIN", None))
        if official is None:
            continue
        rows.append({
            "game_id": game_id,
            "player_id": pid,
            "player": getattr(r, "PLAYER_NAME", ""),
            "official_sec": official,
            "ours_sec": ours.get(pid, 0.0),
        })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["delta"] = df["ours_sec"] - df["official_sec"]
    # Players the box score credits with 0:00 and we also never place on the
    # floor are trivially correct; keep them out of the error rate so a long
    # inactive list can't flatter the numbers.
    played = df[(df["official_sec"] > 0) | (df["ours_sec"] > 0)]

    return {
        "game_id": game_id,
        "players": len(played),
        "exact": int((played["delta"].abs() < 1).sum()),
        "within_30s": int((played["delta"].abs() <= 30).sum()),
        "mean_abs_delta": float(played["delta"].abs().mean()),
        "max_abs_delta": float(played["delta"].abs().max()),
        # Total floor time must be 5 players x 2 teams x game length.
        "total_delta_sec": float(played["delta"].sum()),
        "detail": played,
    }


def run(seasons, season_type: str, limit: int, verbose: bool) -> None:
    frames, results = [], []
    for season in seasons:
        games = get_game_ids(season, season_type=season_type)
        if games.empty:
            print(f"{season}: no games")
            continue
        gids = games["GAME_ID"].astype(str).str.zfill(10).tolist()[:limit]
        print(f"\n{season}: checking {len(gids)} games")
        for gid in gids:
            try:
                r = compare_game(gid)
            except Exception as e:
                print(f"  [warn] {gid}: {type(e).__name__}: {str(e)[:60]}")
                continue
            if r is None:
                print(f"  [warn] {gid}: no data")
                continue
            r["season"] = season
            frames.append(r.pop("detail").assign(season=season))
            results.append(r)
            print(f"  {gid}: {r['exact']}/{r['players']} exact, "
                  f"mean |Δ| {r['mean_abs_delta']:.1f}s, "
                  f"max |Δ| {r['max_abs_delta']:.0f}s")

    if not results:
        print("\nnothing verified")
        return

    res = pd.DataFrame(results)
    detail = pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 68)
    print("PBP-RECONSTRUCTED MINUTES  vs  OFFICIAL BOX SCORE")
    print("=" * 68)
    for season, sub in res.groupby("season"):
        d = detail[detail["season"] == season]
        played = d[(d["official_sec"] > 0) | (d["ours_sec"] > 0)]
        exact = (played["delta"].abs() < 1).mean() * 100
        w30 = (played["delta"].abs() <= 30).mean() * 100
        print(f"  {season}  games={len(sub):>3}  players={len(played):>4}  "
              f"exact={exact:6.2f}%  within_30s={w30:6.2f}%  "
              f"mean|Δ|={played['delta'].abs().mean():5.1f}s  "
              f"max|Δ|={played['delta'].abs().max():5.0f}s")

    if verbose:
        worst = detail.reindex(detail["delta"].abs().sort_values(ascending=False).index)
        print("\nlargest per-player disagreements")
        print(worst.head(15)[
            ["season", "game_id", "player", "official_sec", "ours_sec", "delta"]
        ].to_string(index=False))
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons", required=True)
    ap.add_argument("--season-type", default="Regular Season")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args.seasons, args.season_type, args.limit, args.verbose)


if __name__ == "__main__":
    main()
