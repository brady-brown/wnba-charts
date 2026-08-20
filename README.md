# WNBA Charts

Free WNBA analytics for every season since 1997 — box score, shooting, advanced
rates, on/off splits, and RAPM — built from the league's public play-by-play
feed and published as a static site.

**Everything is derived from play-by-play.** The league's own aggregate
endpoints are used as a benchmark to check the derivation against, never as the
source. That is what makes rebound rates and usage divide by what actually
happened on the floor rather than by a share of minutes, and it is why the
history reaches back to 1997 rather than to whichever season the modern
endpoints happen to cover.

## Layout

```
data/          pipeline — fetch, cache, derive, export
  cache.py         write-once cache over the league API
  refresh_live.py  invalidates the entries that go stale mid-season
  pbp_rotation.py  reconstructs who was on the floor, from PBP alone
  stints.py        lineup-change segments
  rapm.py          ridge on stints
  advanced.py      PBP-derived box and advanced rates
  shots.py         zone rollups (era-aware: the arc moved in 2004 and 2013)
  build_*.py       materialise each layer
  nightly.py       the whole chain, in order
  health_check.py  refuses to publish a quietly broken build
  verify_*.py      derivation vs. the league's own numbers
viz/           matplotlib court + shot chart rendering
site/          the static site; site/data/ is generated
```

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python -m data.nightly --season 2026      # update the season in progress
python -m data.health_check --season 2026 # verify what it produced
python -m http.server 8000 --directory site
```

A full rebuild of all thirty seasons is `python -m data.build_all` followed by
`build_rapm`, `build_scopes`, `build_advanced`, `build_site` — hours, and a
few thousand API calls. The nightly path touches only the current season.

### The ridge penalty is pinned

`data/nightly.py` passes a fixed `alpha` to `build_rapm`. Supplied no alpha,
`build_rapm` re-tunes by cross-validation, which is correct once and wrong every
night after: a player's rating would move because the regularization moved, not
because the basketball did. Changing the pin is a decision to rebuild all thirty
seasons, and `data/health_check.py` fails the build if any published season
disagrees with it.

## Nightly updates

`.github/workflows/nightly.yml` runs at 11:00 UTC (07:00 ET) from May through
October — the league does not play November through April, and a job that ran
anyway would fail its own staleness gate nightly until a red build stopped
meaning anything. Off-season rebuilds go through `workflow_dispatch`.

Each run refreshes the live season's stale cache, rebuilds the chain, verifies
the output, commits whatever changed, and publishes `site/` to GitHub Pages.

Two things make this cheap enough to run unattended:

* **The cache is write-once.** A finished game never changes, so it is fetched
  once and read forever. A nightly costs one API call per game actually played.
  `data/refresh_live.py` is what keeps that from also meaning "never notices a
  new game": it drops the four things that do change while a season is in
  progress — the schedule, the benchmark feeds, the shot coordinates, and any
  play-by-play captured before the final buzzer.
* **Builds are stamped with data, not clocks.** `meta.built_at` is the date of
  the most recent game, so a night with no games rewrites byte-identical JSON,
  produces no diff, and commits nothing.

### What is in git and what is not

`data/cache/` is roughly 615 MB, nearly all of it raw play-by-play and shot
coordinates for seasons that will never change again. Git holds the derived
output (`data/stints_out`, `rapm_out`, `advanced_out`, `site/data`) plus the
small metadata a build needs to run offline — team identity, the schedule index,
the benchmark feeds. The bulk lives in the Actions cache, restored by prefix on
every run and saved back grown.

The consequence worth knowing: a fresh clone can rebuild and publish the site
without a single API call, but a rebuild of *history* re-fetches the
play-by-play it was derived from.

## Data notes

Things that look like bugs and are not:

* **Home/away.** Lineups are ordered `[away, home]`. Any cached `rotation_*.csv`
  from the retired GameRotation path is inverted and is git-ignored for that
  reason.
* **The three-point line moved** in 2004 and again in 2013. Zones are assigned
  with the season's own geometry and league baselines are computed within a
  season; a pooled baseline across eras describes neither shot.
* **Team identity is per-season.** One franchise id can carry three names —
  1611661319 is the Utah Starzz, the San Antonio Silver Stars, and the Las Vegas
  Aces. Phoenix changed abbreviation from PHO to PHX in 2026. Nothing is keyed
  on abbreviation across seasons.
* **RAPM is regular-season only**, and is shown on every scope with that stated
  on the page. A playoff run is too few possessions for a ridge fit to describe
  the player rather than the prior; the playoff scopes ship raw on/off and box
  splits instead, which stay honest at small samples.
