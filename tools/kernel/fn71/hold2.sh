#!/usr/bin/env bash
# FN71b hold 2 (CARD 1, ONE blocking take): `saber_hsa::NvmeVram` on the POOL route --
#   crates/saber-hsa/examples/fn70_nvme.rs FILE 700 /fn71/pool: the open picks Pool (probe: drive ->
#   the host view, SDMA copy-back from the GPU view), 400 records byte-EQUAL vs a host pread,
#   completion word, 1 GiB stream; then the bounce on the same reader as the control -> prim.log
# Build first, off the lock (compile only):
#   podman run --rm --security-opt label=disable -v $REPO:/work -w /work -e HIP_VISIBLE_DEVICES= \
#     -e CARGO_TARGET_DIR=/work/.cache/target-container -v $REPO/.cache/cargo-registry:/opt/rust/cargo/registry \
#     --entrypoint cargo localhost/saber-dev:latest build --release -p saber-hsa --example fn70_nvme
#   tools/kernel/fn71/hold2.sh   -> ~/saber-cache/ws-fn71b/hold2-<sha>-<stamp>/
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
SHARED=${SABER_SHARED_CACHE:-~/saber-cache}
FILE=${FN71_FILE:-/mnt/nvme/saber-packs/qwen38-27b-nvfp4/rank0.pack}
POOL=/sys/bus/pci/devices/0000:07:00.0/p2pmem/allocate   # card 1
PRIM=$REPO/.cache/target-container/release/examples/fn70_nvme
if [ "${1:-}" != "--inner" ]; then
  if [ -n "${FN71B_HOLD_ACTIVE:-}" ] || [ -n "${SABER_GPU_LOCK_HELD:-}" ]; then
    echo "fn71b hold2: outer phase re-entered; refusing" >&2; exit 9
  fi
  export FN71B_HOLD_ACTIVE=1
  SHA=$(git -C "$REPO" rev-parse --short HEAD)
  OUT=$SHARED/ws-fn71b/hold2-$SHA-$(date +%m%d-%H%M%S)
  mkdir -p "$OUT"
  [ -x "$PRIM" ] || { echo "missing $PRIM: build first" >&2; exit 1; }
  [ "$(cat /sys/module/amdgpu/parameters/p2pmem_mb 2>/dev/null || echo 0)" != 0 ] || {
    echo "no amdgpu.p2pmem_mb on this boot; refusing" >&2; exit 1; }
  [ -r "$POOL" ] && [ -w "$POOL" ] || { echo "$POOL not rw for $(id -un)" >&2; exit 1; }
  UNIT=${SABER_PROD_UNIT:-qwen_prod_nvfp4}
  if [ "$(systemctl --user is-active "$UNIT" 2>/dev/null || true)" = "active" ]; then
    echo "prod's unit is ACTIVE; refusing" >&2; exit 1
  fi
  while podman ps --format '{{.Names}}' | grep -qE '^(saber-27b-serve|saber-flashnext-serve)$'; do
    sleep 60
  done
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
    echo "power cap $f: $(( $(cat "$f") / 1000000 )) W (read, never written)" >> "$OUT/power-cap.txt"
  done
  # shellcheck source=../../capture/gpu_idle.sh
  . "$REPO/tools/capture/gpu_idle.sh"
  echo "== $(date '+%F %T') FN71b hold 2 at $SHA, out $OUT (waiting for card 1)"
  gpu_lock_run 1 -- env SABER_GPU_LOCK_HELD=1 FN71B_OUT="$OUT" bash "$REPO/tools/kernel/fn71/hold2.sh" --inner \
    > "$OUT/hold2.log" 2>&1
  rc=$?
  echo "== $(date '+%F %T') FN71b hold 2 exit $rc; results $OUT"
  exit $rc
fi

# ---------------------------------------------------------------- INNER (inside the lock)
if [ -n "${FN71B_INNER:-}" ] || [ -z "${SABER_GPU_LOCK_HELD:-}" ]; then
  echo "fn71b hold2: inner phase re-entered or outside the lock; refusing" >&2; exit 9
fi
export FN71B_INNER=1
OUT=${FN71B_OUT:?}
ulimit -c 0
# shellcheck source=../../capture/gpu_idle.sh
. "$REPO/tools/capture/gpu_idle.sh"
export SABER_GPU_CNAME=saber-fn71b-h2-$$
trap 'podman kill "$SABER_GPU_CNAME" >/dev/null 2>&1; podman rm -f "$SABER_GPU_CNAME" >/dev/null 2>&1' EXIT
trap 'podman kill "$SABER_GPU_CNAME" >/dev/null 2>&1; exit 143' TERM INT
echo "=== prim $(date -Is)"
GPU_TIMEOUT=600 GPU_KILL_AFTER=30 gpu_podman --rm --security-opt label=disable --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e ROCR_VISIBLE_DEVICES=1 -v "$REPO":/work:ro -v /mnt/nvme/saber-packs:/mnt/nvme/saber-packs:ro \
  --mount "type=bind,src=$POOL,dst=/fn71/pool,rw=true" -w /work \
  --entrypoint /work/.cache/target-container/release/examples/fn70_nvme "$IMG" "$FILE" 700 /fn71/pool \
  > "$OUT/prim.log" 2>&1
rc=$?
cat "$OUT/prim.log"
echo "=== end $(date -Is) rc=$rc"
exit $rc
