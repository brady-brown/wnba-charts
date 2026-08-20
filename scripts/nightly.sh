#!/bin/bash
#
# nightly.sh — the fetch half of the nightly, run from a machine the league
# will actually talk to.
#
# GitHub-hosted runners cannot reach stats.nba.com (see README). Everything
# downstream of the fetch is happy in CI, so this script does the minimum that
# has to happen here — rebuild, verify, push — and lets .github/workflows/
# pages.yml publish off the push.
#
# Installed as a LaunchAgent; see scripts/com.bradybrown.wnba-nightly.plist.
# Safe to run by hand at any time.
#
#   scripts/nightly.sh              # current season
#   scripts/nightly.sh 2026         # a named one

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
SEASON="${1:-$(date +%Y)}"
LOG="$HOME/Library/Logs/wnba-nightly.log"
LOCK="/tmp/wnba-nightly.lock"

mkdir -p "$(dirname "$LOG")"

# Keep the log to the last ~2000 lines. A year of nightlies is otherwise a file
# nobody opens because it is too big to skim.
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 4000 ]; then
  tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

exec >> "$LOG" 2>&1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# A run that overlaps another would fetch the same games twice and race on the
# push. Keep the second one out rather than queueing it — if last night's run is
# somehow still going, tonight's has nothing to add.
#
# shlock, not flock: macOS ships shlock in /usr/bin and does not ship flock at
# all. The trap clears the lock on every ordinary exit, success or failure; the
# age check covers the one case the trap cannot — a run killed outright by a
# reboot or a force quit, which would otherwise leave a lock that skips every
# night forever. shlock is documented to reclaim a lock whose pid is dead, but
# it declines to in practice here, so the age check is what actually does it.
if [ -f "$LOCK" ] && [ -n "$(find "$LOCK" -mmin +360 2>/dev/null)" ]; then
  say "clearing stale lock (older than 6h, pid $(cat "$LOCK" 2>/dev/null))"
  rm -f "$LOCK"
fi
if ! /usr/bin/shlock -f "$LOCK" -p $$; then
  say "another run holds $LOCK (pid $(cat "$LOCK" 2>/dev/null)); skipping"
  exit 0
fi
trap 'rm -f "$LOCK"' EXIT

fail() {
  say "FAILED: $1"
  # The entire lesson of setting this up: a nightly that fails quietly is worse
  # than no nightly. Put it on screen.
  osascript -e "display notification \"$1\" with title \"WNBA nightly failed\"" 2>/dev/null || true
  exit 1
}

say "=== nightly start (season $SEASON) ==="
cd "$REPO"

[ -x "$PYTHON" ] || fail "no venv at $PYTHON"

"$PYTHON" -m data.nightly --season "$SEASON" || fail "build chain failed — see $LOG"
"$PYTHON" -m data.health_check --season "$SEASON" || fail "health check rejected the build — nothing pushed"

git add -A
if git diff --cached --quiet; then
  # Nothing new to record. That is the ordinary quiet night — build_site stamps
  # meta.built_at with the latest game date, so a night with no games rewrites
  # byte-identical JSON. But "nothing to commit" is not "nothing to push": a
  # night that committed and then failed to push would otherwise sit unpublished
  # forever, since every night after it also has nothing to commit.
  git fetch -q origin main 2>/dev/null || true
  if [ -n "$(git log --oneline origin/main..HEAD 2>/dev/null)" ]; then
    say "no new data, but $(git rev-list --count origin/main..HEAD) local commit(s) unpushed"
  else
    say "no changes — no new games since the last run"
    say "=== nightly done ==="
    exit 0
  fi
else
  git commit -q -m "nightly: $SEASON through $(date +%F)"
fi

# Push, resyncing if the remote moved. site/data is fully regenerated each run,
# so our tree is always the intended final state — reset onto the remote and lay
# our files down as one commit rather than replaying, which cannot conflict.
for attempt in 1 2 3; do
  if git push -q origin main 2>/dev/null; then
    say "pushed — pages.yml will publish"
    say "=== nightly done ==="
    exit 0
  fi
  say "push attempt $attempt failed (remote moved); resyncing"
  git fetch -q origin main
  git reset -q --soft origin/main
  git add -A
  if git diff --cached --quiet; then
    say "remote already has it; nothing to push"
    say "=== nightly done ==="
    exit 0
  fi
  git commit -q -m "nightly: $SEASON through $(date +%F)"
  sleep 3
done

fail "could not push after 3 attempts"
