# data/health_check.py
"""
health_check.py — Refuse to publish a build that is broken in a way the
pipeline itself would not notice.

Every stage in data/nightly.py exits non-zero when it throws, so the chain
already catches crashes. What it does not catch is the quiet failure: a stage
that succeeds and writes a file that is wrong. Those are the ones that reach
the browser.

The four checks here are the four ways this project has actually been able to
ship a broken page:

* **Bare NaN in the JSON.** json.dumps of a numpy NaN emits the literal `NaN`,
  which Python happily reads back and JSON.parse rejects outright. The page
  does not error — it hangs on "Loading..." forever. data/build_common.py has
  a NaN-safe writer for exactly this; this asserts it was the one used.
* **A stamp that did not move.** build_site stamps `meta.built_at` with the
  date of the most recent GAME, not wall clock. If the league played and the
  stamp is still last week's, the cache refresh silently failed and the build
  is a no-op wearing a fresh timestamp.
* **A season that emptied out.** A table that parses but holds no players is a
  successful build of nothing.
* **A moved ridge penalty.** RAPM is only comparable across seasons because
  alpha is fixed. A build that re-tuned would publish numbers that cannot be
  compared to the other twenty-nine seasons on the same page.

Usage:
    python -m data.health_check --season 2026
    python -m data.health_check --season 2026 --max-stale-days 5
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from data.build_common import SITE_DATA_DIR
from data.nightly import ALPHA

RAPM_SUMMARY = Path("data/rapm_out/rapm_summary.csv")

# JSON.parse rejects all three; Python's json module accepts all three.
BAD_LITERALS = re.compile(r"\b(NaN|-?Infinity)\b")


class Failed(Exception):
    """A check that should stop the deploy."""


def _load(path: Path) -> dict:
    if not path.exists():
        raise Failed(f"missing {path}")
    raw = path.read_text()
    if BAD_LITERALS.search(raw):
        bad = BAD_LITERALS.search(raw).group(0)
        raise Failed(f"{path.name} contains the bare literal {bad!r} — "
                     f"JSON.parse will reject it and the page will hang")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise Failed(f"{path.name} is not valid JSON: {e}")


def check_season(season: str, max_stale_days: int) -> list[str]:
    """Every check for one season. Returns the lines to print on success."""
    notes = []
    season_dir = SITE_DATA_DIR / season

    index = _load(SITE_DATA_DIR / "seasons.json")
    if season not in index:
        raise Failed(f"{season} is missing from seasons.json — the dropdown "
                     f"will not offer it")
    notes.append(f"seasons.json lists {len(index)} seasons, {season} included")

    payload = _load(season_dir / "player-stats.json")
    players = payload.get("players") or payload.get("rows") or payload.get("data")
    if not players:
        # Field name has moved before; fall back to the largest list in the doc
        # rather than passing a build that shipped an empty table.
        lists = [v for v in payload.values() if isinstance(v, list)]
        players = max(lists, key=len) if lists else []
    if len(players) < 50:
        raise Failed(f"{season} player-stats.json holds {len(players)} players — "
                     f"a season in progress has hundreds")
    notes.append(f"player-stats.json: {len(players)} players")

    stamp = (payload.get("meta") or {}).get("built_at")
    if not stamp:
        raise Failed(f"{season} player-stats.json has no meta.built_at stamp")
    try:
        stamped = datetime.fromisoformat(stamp).date()
    except ValueError:
        raise Failed(f"meta.built_at is not an ISO date: {stamp!r}")
    stale = (date.today() - stamped).days
    if stale > max_stale_days:
        raise Failed(
            f"latest game in {season} is {stamp} — {stale} days old. Either the "
            f"season is over (re-point --season, or let the schedule idle) or "
            f"data/refresh_live.py failed to invalidate the schedule cache and "
            f"the build re-derived games it already had.")
    notes.append(f"latest game {stamp} ({stale}d ago)")

    impact = _load(season_dir / "player-impact.json")
    rated = [v for v in impact.values() if isinstance(v, list)]
    n_rated = len(max(rated, key=len)) if rated else 0
    notes.append(f"player-impact.json: {n_rated} rated players")

    return notes


def check_alpha() -> str:
    """The ridge penalty every published season must share."""
    if not RAPM_SUMMARY.exists():
        raise Failed(f"missing {RAPM_SUMMARY}")
    used = pd.read_csv(RAPM_SUMMARY)["alpha"].unique()
    if len(used) != 1:
        raise Failed(f"seasons were solved with different alphas ({sorted(used)}) — "
                     f"their RAPM cannot be compared on the same page")
    if abs(float(used[0]) - ALPHA) > 1e-6:
        raise Failed(f"alpha is {used[0]}, expected {ALPHA} — a re-tune moved the "
                     f"penalty, so this build is not comparable to the published "
                     f"history. Rebuild every season or restore the pin.")
    return f"alpha {used[0]:g} across every season"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=str(pd.Timestamp.today().year))
    ap.add_argument("--max-stale-days", type=int, default=4,
                    help="how old the newest game may be before the build is "
                         "treated as stale; default 4 covers the All-Star break")
    args = ap.parse_args()

    try:
        notes = check_season(args.season, args.max_stale_days)
        notes.append(check_alpha())
    except Failed as e:
        print(f"HEALTH CHECK FAILED\n  {e}")
        raise SystemExit(1)

    print(f"HEALTH CHECK PASSED — {args.season}")
    for note in notes:
        print(f"  {note}")


if __name__ == "__main__":
    main()
