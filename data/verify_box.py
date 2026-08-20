# data/verify_box.py
"""
verify_box.py — Reconcile PBP-derived box totals against the official feeds.

The site computes its box score from play-by-play (data/box_pbp.py) so that the
counting stats and the on-floor rates come from one source. That only holds up
if the derivation is exact, so the league's own numbers are kept as a benchmark
and checked here rather than shipped.

Two levels, because they fail differently:

* `--games`  per-player rows against BoxScoreTraditionalV2 for individual games.
  Catches attribution bugs — a rebound credited to the wrong team, a blocker
  read out of the wrong PLAYER slot. One API call per game, so it samples.
* `--season` season totals against LeagueDashPlayerStats. Catches drift that a
  handful of games would hide, and costs two calls per season.

Usage:
    python -m data.verify_box --season 2024
    python -m data.verify_box --season 2024 --games 25
    python -m data.verify_box --all-seasons
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

from data.box_pbp import STAT_COLS, game_box
from data.cache import get_box_score, get_game_ids, get_pbp, get_player_feed

# PBP name -> official box name. The traditional box calls turnovers "TO".
_OFF_COLS = {
    "fgm": "FGM", "fga": "FGA", "fg3m": "FG3M", "fg3a": "FG3A",
    "ftm": "FTM", "fta": "FTA", "oreb": "OREB", "dreb": "DREB",
    "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TO", "pf": "PF",
    "pts": "PTS",
}
# LeagueDashPlayerStats uses "TOV" and publishes per-game, not totals.
_API_COLS = dict(_OFF_COLS, tov="TOV")


def _season_game_ids(season: str, season_type: str) -> list[str]:
    df = get_game_ids(season, season_type=season_type)
    if df is None or df.empty:
        return []
    return df["GAME_ID"].astype(str).str.zfill(10).tolist()


def verify_games(season: str, n: int, season_type: str = "Regular Season",
                 seed: int = 0) -> bool:
    """Per-player reconciliation against the official box for `n` sampled games."""
    gids = _season_game_ids(season, season_type)
    if not gids:
        print(f"  {season}: no games")
        return True
    rng = random.Random(seed)
    sample = gids if n >= len(gids) else rng.sample(gids, n)

    totals = {c: [0, 0] for c in _OFF_COLS}     # [mismatched players, players]
    worst = []
    id_only = unjoined = 0
    for gid in sorted(sample):
        pbp = get_pbp(gid)
        mine = game_box(pbp)
        off = get_box_score(gid)
        if off is None or off.empty:
            continue
        # The 1997-99 feeds sometimes give a player one id in the play-by-play
        # and another in the box score, which reads as two half-populated rows
        # and inflates every stat's mismatch count. Fall back to the name so the
        # check measures the arithmetic rather than the feed's id hygiene.
        names = {}
        for i in (1, 2, 3):
            ids, nms = pbp.get(f"PLAYER{i}_ID"), pbp.get(f"PLAYER{i}_NAME")
            if ids is None or nms is None:
                continue
            for pid, nm in zip(ids, nms):
                if isinstance(nm, str) and nm.strip():
                    try:
                        names[int(pid)] = nm.strip()
                    except (TypeError, ValueError):
                        pass
        mine = mine.copy()
        mine["_name"] = mine["player_id"].map(names)
        off = off.copy()
        remap = {}
        off_ids = set(off["PLAYER_ID"])
        by_name = {str(n).strip(): int(p) for n, p in
                   zip(off.get("PLAYER_NAME", []), off.get("PLAYER_ID", []))}
        for pid, nm in zip(mine["player_id"], mine["_name"]):
            if pid not in off_ids and nm in by_name:
                remap[pid] = by_name[nm]
        if remap:
            id_only += len(remap)
            mine["player_id"] = mine["player_id"].replace(remap)
            mine = mine.groupby("player_id", as_index=False)[list(STAT_COLS)].sum()
        off = off.rename(columns={v: f"{k}_off" for k, v in _OFF_COLS.items()
                                  if v in off.columns})
        keep = ["PLAYER_ID", "PLAYER_NAME"] + [f"{k}_off" for k in _OFF_COLS
                                               if f"{k}_off" in off.columns]
        both = mine.merge(off[keep], left_on="player_id", right_on="PLAYER_ID",
                          how="outer", indicator=True)
        # A row on only one side is an identity problem, not an arithmetic one.
        # Most are simply DNPs — the official box lists every dressed player and
        # someone who never checked in generates no PBP events — so only count
        # the ones actually carrying stats. Those are real: the 1997 feed has at
        # least one player under a different id with a corrupted name
        # ("Penny McCray, Nikki (" where the box score says Penny Moore).
        solo = both[both["_merge"] != "both"]
        if len(solo):
            cols = [c for c in list(STAT_COLS) + [f"{k}_off" for k in _OFF_COLS]
                    if c in solo.columns]
            unjoined += int((solo[cols].fillna(0).abs().sum(axis=1) > 0).sum())
        cmp = both[both["_merge"] == "both"]
        for k in _OFF_COLS:
            b = f"{k}_off"
            if b not in cmp.columns:
                continue
            a_v = cmp[k].fillna(0)
            b_v = cmp[b].fillna(0)
            n_bad = int((a_v != b_v).sum())
            totals[k][0] += n_bad
            totals[k][1] += len(cmp)
            if n_bad:
                for _, row in cmp[a_v != b_v].iterrows():
                    worst.append((gid, row.get("PLAYER_NAME"), k,
                                  row.get(k), row.get(b)))

    note = []
    if id_only:
        note.append(f"{id_only} joined by name")
    if unjoined:
        note.append(f"{unjoined} unjoinable (feed id mismatch)")
    print(f"  {season} {season_type}: {len(sample)} games, per-player rows"
          + (f" [{', '.join(note)}]" if note else ""))
    ok = True
    for k, (bad, tot) in totals.items():
        if tot and bad:
            ok = False
            print(f"    {k:<5} {bad}/{tot} rows differ")
    if ok:
        print("    all stats exact")
    for w in worst[:15]:
        print(f"      {w[0]} {w[1]}: {w[2]} pbp={w[3]} official={w[4]}")
    return ok


def verify_season(season: str, season_type: str = "Regular Season") -> bool:
    """Season totals against LeagueDashPlayerStats (per-game x GP)."""
    gids = _season_game_ids(season, season_type)
    if not gids:
        return True
    boxes = [game_box(get_pbp(g)) for g in gids]
    boxes = [b for b in boxes if b is not None and not b.empty]
    if not boxes:
        print(f"  {season}: no PBP")
        return True
    mine = pd.concat(boxes, ignore_index=True).groupby("player_id",
                                                       as_index=False)[STAT_COLS].sum()

    # Totals, not PerGame x GP — the per-game feed is rounded to one decimal.
    api = get_player_feed(season, season_type=season_type, per_mode="Totals").copy()
    for k, src in _API_COLS.items():
        if src in api.columns:
            api[k] = api[src]

    cmp = mine.merge(api[["PLAYER_ID", "PLAYER_NAME"] +
                         [k for k in _API_COLS if k in api.columns]],
                     left_on="player_id", right_on="PLAYER_ID",
                     suffixes=("_pbp", "_api"), how="inner")
    print(f"  {season} {season_type}: {len(cmp)} players matched "
          f"({len(mine)} pbp / {len(api)} api)")
    ok = True
    for k in STAT_COLS:
        a, b = f"{k}_pbp", f"{k}_api"
        if b not in cmp.columns:
            continue
        s1, s2 = cmp[a].sum(), cmp[b].sum()
        pct = 100 * (s1 - s2) / max(s2, 1)
        # A handful of events per season is the two feeds disagreeing with each
        # other, not a parse error — the per-game reconciliation is the strict
        # check and it is exact. Flag only a systematic gap.
        flag = "" if abs(pct) < 0.1 else "   <<<"
        if flag:
            ok = False
        print(f"    {k:<5} pbp={s1:>8.0f} api={s2:>8.0f} "
              f"diff={s1 - s2:>+5.0f} {pct:+6.2f}%{flag}")
    return ok


# Rates with no on-floor term. Their definitions are identical to the league's,
# so they must reproduce the published value; a gap here is an arithmetic bug.
_EXACT_RATES = {"ts": "TS_PCT", "efg": "EFG_PCT"}
# Rates whose denominator is "what happened while she was on the floor". The
# league estimates that with a minutes share, we measure it from the lineups, so
# these are EXPECTED to differ. They should still track closely — a low
# correlation would mean our on-floor join is wrong, not merely sharper.
_ESTIMATED_RATES = {"usg": "USG_PCT", "astp": "AST_PCT", "orbp": "OREB_PCT",
                    "drbp": "DREB_PCT", "trbp": "REB_PCT"}


def verify_rates(season: str, season_type: str = "Regular Season",
                 min_min: float = 200.0) -> bool:
    """Compare PBP-derived rates against the league's Advanced feed."""
    from data.advanced import season_advanced

    mine = season_advanced(season, season_type)
    if mine.empty:
        print(f"  {season}: no advanced rows")
        return True
    adv = get_player_feed(season, season_type=season_type, per_mode="PerGame",
                          measure_type="Advanced")
    cmp = mine.merge(adv, left_on="player_id", right_on="PLAYER_ID", how="inner")
    cmp = cmp[cmp["min"] >= min_min]
    if cmp.empty:
        print(f"  {season}: nobody over {min_min:.0f} minutes")
        return True

    print(f"  {season} {season_type}: {len(cmp)} players over {min_min:.0f} min")
    ok = True
    print("    exact (same definition, must match):")
    for k, col in _EXACT_RATES.items():
        if col not in cmp.columns:
            continue
        diff = (cmp[k] - cmp[col] * 100).abs()
        bad = int((diff > 0.15).sum())
        if bad:
            ok = False
        print(f"      {k:<5} max|diff|={diff.max():5.2f}  over 0.15: {bad}/{len(cmp)}"
              + ("   <<<" if bad else ""))

    print("    on-floor (ours exact, theirs a minutes-share estimate):")
    for k, col in _ESTIMATED_RATES.items():
        if col not in cmp.columns:
            continue
        theirs = cmp[col] * 100
        diff = cmp[k] - theirs
        r = cmp[k].corr(theirs)
        if pd.notna(r) and r < 0.95:
            ok = False
        print(f"      {k:<5} r={r:5.3f}  median diff={diff.median():+5.2f}  "
              f"p90|diff|={diff.abs().quantile(0.9):5.2f}"
              + ("   <<<" if pd.notna(r) and r < 0.95 else ""))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", dest="seasons")
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--season-type", default="Regular Season")
    ap.add_argument("--games", type=int, default=0,
                    help="also reconcile N sampled games per season")
    ap.add_argument("--rates", action="store_true",
                    help="also compare advanced rates against the Advanced feed")
    ap.add_argument("--skip-box", action="store_true",
                    help="skip the counting-stat reconciliation")
    args = ap.parse_args()

    if args.all_seasons:
        from data.build_common import built_seasons
        seasons = built_seasons(newest_first=False)
    else:
        seasons = args.seasons or ["2024"]

    all_ok = True
    for s in seasons:
        if args.games:
            all_ok &= verify_games(s, args.games, args.season_type)
        if not args.skip_box:
            all_ok &= verify_season(s, args.season_type)
        if args.rates:
            all_ok &= verify_rates(s, args.season_type)
    print("\nRESULT:", "all clean" if all_ok else "differences found")


if __name__ == "__main__":
    main()
