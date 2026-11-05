#!/usr/bin/env bash
# FN71 verify (FN71b's hold, CARD 1, ONE blocking take) -- AFTER the user installed the patched
# amdgpu (tools/kernel/fn71/install.sh) and rebooted with amdgpu.p2pmem_mb=N:
#   the boot assert (kernel, module parameter, the two dmesg lines, sysfs p2pmem size/available/
#   published, the pool file handed to the user), then tools/kernel/fn71/fn71_verify.cpp on card 1:
#   GEM handle -> dma-buf -> HIP + HSA import, whole-pool mmap, aliasing, O_DIRECT into the pool,
#   1 GiB sha256 through a HIP kernel, the rewrite law, rates (drive->pinned / bounce / P2P at
#   QD 1/8/32 + 1 GiB stream) and the host-DRAM contention probe -> verify.txt
# Build first, off the lock: tools/kernel/fn71/build_verify.sh
#   tools/kernel/fn71/verify.sh   -> ~/saber-cache/ws-fn71/verify-<sha>-<stamp>/
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
SHARED=${SABER_SHARED_CACHE:-~/saber-cache}
FILE=${FN71_FILE:-/mnt/nvme/saber-packs/qwen38-27b-nvfp4/rank0.pack}
BDF=${FN71_BDF:-0000:07:00.0}           # card 1
POOL=/sys/bus/pci/devices/$BDF/p2pmem/allocate
BIN=$REPO/.cache/fn71/fn71_verify
if [ "${1:-}" != "--inner" ]; then
  if [ -n "${FN71_HOLD_ACTIVE:-}" ] || [ -n "${SABER_GPU_LOCK_HELD:-}" ]; then
    echo "fn71 verify: outer phase re-entered; refusing" >&2; exit 9
  fi
  export FN71_HOLD_ACTIVE=1
  SHA=$(git -C "$REPO" rev-parse --short HEAD)
  OUT=$SHARED/${FN71_WS:-ws-fn71b}/verify-$SHA-$(date +%m%d-%H%M%S)
  mkdir -p "$OUT"
  [ -x "$BIN" ] || { echo "missing $BIN: run tools/kernel/fn71/build_verify.sh" >&2; exit 1; }
  # ---- the per-boot load assert: this boot runs the patched module with the pool on
  {
    echo "kernel $(uname -r)"; cat /proc/cmdline
    echo "amdgpu.p2pmem_mb = $(cat /sys/module/amdgpu/parameters/p2pmem_mb 2>&1)"
    journalctl -k -b --no-pager 2>/dev/null | grep -E 'p2pmem|peer-to-peer DMA memory' || echo "(no p2pmem kernel lines)"
    for d in 0000:03:00.0 0000:07:00.0; do
      s=/sys/bus/pci/devices/$d/p2pmem
      echo "$d size $(cat $s/size 2>&1) available $(cat $s/available 2>&1) published $(cat $s/published 2>&1)"
      ls -la $s/allocate 2>&1
    done
  } > "$OUT/boot.txt" 2>&1
  cat "$OUT/boot.txt"
  [ "$(uname -r)" = 7.1.3-p2p ] || { echo "not the 7.1.3-p2p kernel; refusing" >&2; exit 1; }
  mb=$(cat /sys/module/amdgpu/parameters/p2pmem_mb 2>/dev/null || echo missing)
  case "$mb" in ''|missing|0) echo "amdgpu.p2pmem_mb is '$mb' on this boot: patched module not loaded or off; refusing" >&2; exit 1 ;; esac
  [ -r "$POOL" ] && [ -w "$POOL" ] || { echo "$POOL is not rw for $(id -un) (the udev rule from install.sh)" >&2; exit 1; }
  UNIT=${SABER_PROD_UNIT:-qwen_prod_nvfp4}
  if [ "$(systemctl --user is-active "$UNIT" 2>/dev/null || true)" = "active" ]; then
    echo "prod's unit is ACTIVE; refusing" >&2; exit 1
  fi
  # the user's own servers run outside the agents' lock: never stop them, wait until they are down
  while podman ps --format '{{.Names}}' | grep -qE '^(saber-27b-serve|saber-flashnext-serve)$'; do
    sleep 60
  done
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
    echo "power cap $f: $(( $(cat "$f") / 1000000 )) W (read, never written)" >> "$OUT/power-cap.txt"
  done
  # shellcheck source=../../capture/gpu_idle.sh
  . "$REPO/tools/capture/gpu_idle.sh"
  echo "== $(date '+%F %T') FN71 verify at $SHA, out $OUT (waiting for card 1)"
  gpu_lock_run 1 -- env SABER_GPU_LOCK_HELD=1 FN71_OUT="$OUT" bash "$REPO/tools/kernel/fn71/verify.sh" --inner \
    > "$OUT/verify.log" 2>&1
  rc=$?
  echo "== $(date '+%F %T') FN71 verify exit $rc; results $OUT"
  exit $rc
fi

# ---------------------------------------------------------------- INNER (inside the lock)
if [ -n "${FN71_INNER:-}" ] || [ -z "${SABER_GPU_LOCK_HELD:-}" ]; then
  echo "fn71 verify: inner phase re-entered or outside the lock; refusing" >&2; exit 9
fi
export FN71_INNER=1
OUT=${FN71_OUT:?}
ulimit -c 0
# shellcheck source=../../capture/gpu_idle.sh
. "$REPO/tools/capture/gpu_idle.sh"
export SABER_GPU_CNAME=saber-fn71-v-$$
trap 'podman kill "$SABER_GPU_CNAME" >/dev/null 2>&1; podman rm -f "$SABER_GPU_CNAME" >/dev/null 2>&1' EXIT
trap 'podman kill "$SABER_GPU_CNAME" >/dev/null 2>&1; exit 143' TERM INT
echo "=== verify $(date -Is)"
GPU_TIMEOUT=900 GPU_KILL_AFTER=30 gpu_podman --rm --security-opt label=disable --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e HIP_VISIBLE_DEVICES=1 -v "$REPO":/work:ro -v "$OUT":/out \
  -v /mnt/nvme/saber-packs:/mnt/nvme/saber-packs:ro --mount "type=bind,src=$POOL,dst=/fn71/pool,rw=true" -w /work \
  --entrypoint /work/.cache/fn71/fn71_verify "$IMG" "$FILE" /out /fn71/pool
rc=$?
echo "=== verify exit $rc $(date -Is)"
journalctl -k -b --no-pager --since "-20min" 2>/dev/null | grep -iE 'p2p|amdgpu.*(fault|error)|nvme' | tail -40 \
  > "$OUT/dmesg-after.txt" || true
exit $rc
