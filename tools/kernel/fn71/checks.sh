#!/usr/bin/env bash
# FN71b: the four host checks (tools/gates/fn72_checks.sh: fmt, clippy -D warnings, test --workspace,
# structural) UNDER the box lock (both cards: heavy host compute only under both locks).
#   tools/kernel/fn71/checks.sh   -> ~/saber-cache/ws-fn71b/checks-<sha>-<stamp>.log
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [ -n "${FN71B_CHECKS_ACTIVE:-}" ] || [ -n "${SABER_GPU_LOCK_HELD:-}" ]; then
  echo "fn71b checks: re-entered; refusing" >&2; exit 9
fi
export FN71B_CHECKS_ACTIVE=1
SHA=$(git -C "$REPO" rev-parse --short HEAD)
LOG=${SABER_SHARED_CACHE:-~/saber-cache}/ws-fn71b/checks-$SHA-$(date +%m%d-%H%M%S).log
mkdir -p "$(dirname "$LOG")" "$REPO/.cache"
# shellcheck source=../../capture/gpu_idle.sh
. "$REPO/tools/capture/gpu_idle.sh"
echo "== $(date '+%F %T') FN71b host checks at $SHA -> $LOG (waiting for cards 0,1)"
gpu_lock_run 0,1 -- env SABER_GPU_LOCK_HELD=0,1 nice -n 10 bash "$REPO/tools/gates/fn72_checks.sh" > "$LOG" 2>&1
rc=$?
cat "$LOG"
echo "== $(date '+%F %T') FN71b host checks exit $rc"
exit $rc
