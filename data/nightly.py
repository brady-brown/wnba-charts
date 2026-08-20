# data/nightly.py
"""
nightly.py — One command that takes the season in progress from "the league
played games last night" to "the site has them".

The full pipeline is six stages and every one of them has flags that matter.
Spreading those across six CI steps means the schedule of record lives in a
YAML file, where it cannot be run locally, cannot be tested, and drifts from
what a human types. So the schedule lives here and CI calls this.

Stages, in the only order that works:

    1. refresh_live   drop the cache entries that go stale mid-season, or
                      every stage below rebuilds last week's games forever
    2. build_all      fetch the new play-by-play, rebuild stints
    3. build_rapm     ridge on the regular season, FIXED alpha
    4. build_scopes   playoff + all-games on/off (no ridge, by design)
    5. build_advanced PBP-derived box and advanced rates
    6. build_site     shape it all into the JSON the browser fetches

ALPHA IS NOT RE-TUNED. data/build_rapm.py tunes by cross-validation when no
alpha is supplied, and a nightly job that re-tuned would hand every player a
slightly different number every night for reasons that have nothing to do with
basketball. The value below is the one every season in data/rapm_out was built
with; changing it is a decision to rebuild all thirty seasons, not a nightly.

Only stage 2 and the shot-chart half of stage 6 touch the network. Everything
else reads what they cached, so a run with a warm cache and no new games is
pure local computation and rewrites byte-identical JSON.

Usage:
    python -m data.nightly                       # current season, full chain
    python -m data.nightly --season 2026
    python -m data.nightly --no-refresh          # trust the cache as-is
    python -m data.nightly --dry-run             # print the chain, run nothing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# The ridge penalty every season in data/rapm_out/ was solved with. Held fixed
# so a player's RAPM moves between seasons because the basketball differed, not
# because the regularization did. See data/build_rapm.py.
ALPHA = 2310.13

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def stages(season: str, refresh: bool = True, shots: bool = True) -> list[tuple[str, list[str]]]:
    """(label, argv) for each stage, in dependency order."""
    chain: list[tuple[str, list[str]]] = []
    if refresh:
        chain.append(("refresh", ["-m", "data.refresh_live", "--season", season]))
    chain += [
        ("stints",   ["-m", "data.build_all", "--season", season]),
        ("rapm",     ["-m", "data.build_rapm", "--season", season,
                      "--alpha", str(ALPHA)]),
        ("scopes",   ["-m", "data.build_scopes", "--season", season]),
        ("advanced", ["-m", "data.build_advanced", "--season", season]),
        ("site",     ["-m", "data.build_site", "--season", season]
                     + ([] if shots else ["--no-shots"])),
    ]
    return chain


def run(label: str, argv: list[str]) -> float:
    """Run one stage, streaming its output. Raises on a non-zero exit."""
    print(f"\n{'=' * 64}\n[{label}] python {' '.join(argv)}\n{'=' * 64}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, *argv], cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"\n[{label}] FAILED after {elapsed:.0f}s "
                         f"(exit {proc.returncode}) — nothing downstream ran, "
                         f"so the site still serves the last good build.")
    print(f"[{label}] ok in {elapsed:.0f}s", flush=True)
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=str(pd.Timestamp.today().year),
                    help="season to update; default is the current year")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip stage 1 and trust the cache as it stands")
    ap.add_argument("--no-shots", action="store_true",
                    help="skip the shot-chart export in stage 6")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the chain without running it")
    args = ap.parse_args()

    chain = stages(args.season, refresh=not args.no_refresh, shots=not args.no_shots)

    if args.dry_run:
        print(f"nightly chain for {args.season}:")
        for label, argv in chain:
            print(f"  {label:9s} python {' '.join(argv)}")
        return

    print(f"WNBA nightly — season {args.season}, alpha {ALPHA:g}")
    t0 = time.time()
    timings = [(label, run(label, argv)) for label, argv in chain]

    print(f"\n{'=' * 64}\nNIGHTLY COMPLETE in {time.time() - t0:.0f}s\n{'=' * 64}")
    for label, elapsed in timings:
        print(f"  {label:9s} {elapsed:6.0f}s")


if __name__ == "__main__":
    main()
