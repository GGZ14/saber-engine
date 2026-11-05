#!/usr/bin/env bash
# FN71: build + sign ONLY the amdgpu module of the user's kernel tree (7.1.3-p2p) with the
# p2pmem patch applied. A 24-thread module build is heavy host compute, so it runs ONLY under
# BOTH card locks (`gpu_lock_run 0,1`), niced, and releases at once. No container, no GPU use.
#   tools/kernel/fn71/build_module.sh            -> ~/saber-cache/ws-fn71/build-<stamp>/
# The inner phase waits (up to FN71_READY_WAIT s, default 3600) for the marker
# $SHARED/ws-fn71/PATCH_READY, written once the patch is applied to the tree, so the sleeper can
# queue while the patch is still being written; it never builds an unmarked tree.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SHARED=${SABER_SHARED_CACHE:-~/saber-cache}
KTREE=${FN71_KTREE:-~/kbuild/linux-7.1.3}
MDIR=drivers/gpu/drm/amd/amdgpu
if [ "${1:-}" != "--inner" ]; then
  if [ -n "${FN71_BUILD_ACTIVE:-}" ] || [ -n "${SABER_GPU_LOCK_HELD:-}" ]; then
    echo "fn71 build: outer phase re-entered; refusing" >&2; exit 9
  fi
  export FN71_BUILD_ACTIVE=1
  OUT=$SHARED/ws-fn71/build-$(date +%m%d-%H%M%S)
  mkdir -p "$OUT"
  # shellcheck source=../../capture/gpu_idle.sh
  . "$REPO/tools/capture/gpu_idle.sh"
  echo "== $(date '+%F %T') FN71 module build, out $OUT (waiting for cards 0,1)"
  gpu_lock_run 0,1 -- env SABER_GPU_LOCK_HELD=0,1 FN71_OUT="$OUT" bash "$REPO/tools/kernel/fn71/build_module.sh" --inner \
    > "$OUT/build.log" 2>&1
  rc=$?
  echo "== $(date '+%F %T') FN71 module build exit $rc; results $OUT"
  exit $rc
fi

# ---------------------------------------------------------------- INNER (inside the lock)
if [ -n "${FN71_INNER:-}" ] || [ -z "${SABER_GPU_LOCK_HELD:-}" ]; then
  echo "fn71 build: inner phase re-entered or outside the lock; refusing" >&2; exit 9
fi
export FN71_INNER=1
OUT=${FN71_OUT:?}
MAKEPID=
trap '[ -n "$MAKEPID" ] && kill -TERM "$MAKEPID" 2>/dev/null; exit 143' TERM INT
echo "=== lock taken $(date -Is); waiting for $SHARED/ws-fn71/PATCH_READY"
waited=0
while [ ! -f "$SHARED/ws-fn71/PATCH_READY" ]; do
  if [ "$waited" -ge "${FN71_READY_WAIT:-3600}" ]; then
    echo "no PATCH_READY after ${waited}s; releasing without building"; exit 4
  fi
  sleep 30
  waited=$((waited + 30))
done
cat "$SHARED/ws-fn71/PATCH_READY"
cd "$KTREE" || exit 1
echo "=== make $(date -Is)"
nice -n 19 make -j24 M=$MDIR modules > "$OUT/make.log" 2>&1 &
MAKEPID=$!
wait "$MAKEPID"
rc=$?
MAKEPID=
echo "=== make exit $rc $(date -Is)"
tail -30 "$OUT/make.log"
[ "$rc" -eq 0 ] || exit "$rc"
KO=$KTREE/$MDIR/amdgpu.ko
nice -n 19 scripts/sign-file sha512 certs/signing_key.pem certs/signing_key.x509 "$KO" || exit 5
sha256sum "$KO" | tee "$OUT/amdgpu.ko.sha256"
ls -la "$KO"
modinfo "$KO" | grep -E '^(filename|vermagic|sig_id|signer|sig_hashalgo|parm:.*p2pmem)' | tee "$OUT/modinfo.txt"
echo "=== done $(date -Is)"
