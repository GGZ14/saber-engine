#!/usr/bin/env bash
# Boot Saber's served Inkling 2 Mini (the engine's defaults) on two cards (TP=2) or ONE card (TP=1): the
# layout follows the GPU count (two or more AMD GPUs -> TP=2, one -> TP=1); the one download serves both
# (the pack's tp1/ folder is the one-card view of the same files).
#
#   tools/serve_inkling.sh             # build the release binary in the dev image, then serve on :8080
#   tools/serve_inkling.sh --no-build  # reuse the last container build
#   tools/serve_inkling.sh stop        # stop the server container
#   PORT=8100 tools/serve_inkling.sh   # another port
#   SABER_NVME=/data/saber tools/serve_inkling.sh   # the storage root tools/fetch_packs.sh inkling used
#   INK_PACK=/data/packs/my-inkling-pack tools/serve_inkling.sh --no-build   # another pack directory
#   INK_TP=1 tools/serve_inkling.sh    # force one card (DEVICE=<HIP index>, default 0); INK_TP=2 forces two
#
# Extra arguments go to `saber engine` (e.g. --ink-serve-ctx 16384). The pack is the published one
# (Hugging Face Launch80/Inkling-2-Mini-NVFP4-SABER, `tools/fetch_packs.sh inkling`): self-contained, its
# dense records and model files in its own base/ (HF3; `inkpaths:` boot line). A pack without base/ reads
# the checkpoint at the path its manifest records, which must then be mounted (it is, under $SABER_NVME/models).
# INK51: the default drafter is DFlash2 (iu8 layers, depth 7, gate prod:0.4:2) when its pack sits beside the
# model pack ($(dirname $INK_PACK)/inkling-2-mini-dflash2, fetched by `tools/fetch_packs.sh inkling`) or at
# <pack>/dflash2; without it the bundled DSpark drafter serves (boot line `INK51 drafter:`). Rollback: pass
# `--ink-drafter dspark`; another drafter pack: `--ink-dflash2-path DIR` (mount it yourself).
# INK52 / INK57: split-key decode attention + the row-shared verify body (`--ink-p49 ks,rv4`) is the default
# (long-context decode / verify; byte-equal at <= 2,048 keys; rv4 byte-equal to ks per row). Rollbacks: pass
# `--ink-p49 ks` (INK52's per-row verify) or `--ink-p49 none` (no split).
# OpenAI-compatible: /v1/models, /v1/chat/completions, /v1/completions; served name Inkling-2-Mini.
# Logs: podman logs -f saber-inkling-serve. Refuses to start while another Saber server holds the cards.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$REPO/tools/lib/container.sh"
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
NVME=${SABER_NVME:-/mnt/nvme}
PACK=${INK_PACK:-$NVME/saber-packs/inkling-2-mini-nvfp4}
PORT=${PORT:-8080}
CNAME=${CNAME:-saber-inkling-serve}
OUT=${SABER_SHARED:-$HOME/saber-cache}/serve-inkling   # the serve plane's files (rendered template)

if [ "${1:-}" = stop ]; then
  $CTR stop -t 30 "$CNAME" >/dev/null 2>&1 && echo "stopped $CNAME" || echo "$CNAME was not running"
  $CTR rm -f "$CNAME" >/dev/null 2>&1 || true
  exit 0
fi
BUILD=1
[ "${1:-}" = --no-build ] && { BUILD=0; shift; }

if $CTR ps --format '{{.Names}}' | grep -qE '^saber-(flashnext-serve|27b)'; then
  echo "another Saber server holds the cards; stop it first:" >&2
  $CTR ps --format '  {{.Names}} {{.Status}}' | grep saber >&2; exit 1
fi
if $CTR ps --format '{{.Names}}' | grep -qE '^saber-(fn|ink|cons|hf)'; then
  echo "a saber hold container is running (an agent holds the cards); stop it or wait" >&2; exit 1
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT\$"; then echo "port $PORT is already in use" >&2; exit 1; fi
[ -f "$PACK/manifest.json" ] || { echo "no pack at $PACK (tools/fetch_packs.sh inkling; SABER_NVME=$NVME)" >&2; exit 1; }
# the layout from the GPU count (KFD topology nodes with a GPU id), unless INK_TP says
TP=${INK_TP:-auto}
if [ "$TP" = auto ]; then
  ngpu=0
  for g in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do [ -r "$g" ] && [ "$(cat "$g")" != 0 ] && ngpu=$((ngpu + 1)); done
  if [ "$ngpu" -ge 2 ]; then TP=2; elif [ "$ngpu" = 1 ]; then TP=1; else echo "no AMD GPU found (/sys/class/kfd); set INK_TP=1|2" >&2; exit 1; fi
  echo "== $ngpu GPU(s) -> TP=$TP"
fi
case "$TP" in
  2) DEVS=${HIP_DEVICES:-0,1} ;;
  1) DEVS=${DEVICE:-0}
     [[ "$DEVS" =~ ^[0-9]+$ ]] || { echo "DEVICE=$DEVS: TP=1 serves on ONE HIP device index" >&2; exit 2; }
     # the one-card view: <pack>/tp1 (the published layout), or INK_PACK naming a view directly
     if ! grep -q '"layout": "tp1-view"' "$PACK/manifest.json"; then
       [ -f "$PACK/tp1/manifest.json" ] || { echo "$PACK has no tp1/ (the one-card view): re-run tools/fetch_packs.sh inkling" >&2; exit 1; }
       PACK=$PACK/tp1
     fi ;;
  *) echo "INK_TP=$TP: 1 (one card) | 2 (two cards) | auto" >&2; exit 2 ;;
esac

if [ "$BUILD" = 1 ]; then
  mkdir -p "$REPO/.cache/target-container" "$REPO/.cache/cargo-registry"
  echo "== $(date '+%F %T') building the release binary in the dev image ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))"
  $CTR run --rm "${CTR_USER[@]}" --security-opt label=disable -v "$REPO":/work \
    -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry -e CARGO_TARGET_DIR=/work/.cache/target-container \
    -w /work --entrypoint bash "$IMG" -c "cargo build --release -p saber-cli 2>&1 | tail -3"
fi
BIN=$REPO/.cache/target-container/release/saber
[ -x "$BIN" ] || { echo "no release binary at $BIN (run without --no-build)" >&2; exit 1; }

# the pack at its own path; the storage root's models/ and saber-packs/ (a pack without base/ reads the
# recorded checkpoint, a maintainer's pack links its trunk8/ overlay there), read-only
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 1; }
MNT=(-v "$PACK":"$PACK":ro -v "$OUT":/out)
# a one-card view reads its TP=2 pack's files (its manifest's `blobs`): mount that root too
[ "$TP" = 1 ] && [ "$(basename "$PACK")" = tp1 ] && MNT+=(-v "$(dirname "$PACK")":"$(dirname "$PACK")":ro)
for d in "$NVME/models" "$NVME/saber-packs"; do [ -d "$d" ] && MNT+=(-v "$d":"$d":ro); done
# INK51: the DFlash2 drafter pack beside the TP=2 pack (the engine's auto-discovery path)
PROOT=$PACK; [ "$TP" = 1 ] && [ "$(basename "$PACK")" = tp1 ] && PROOT=$(dirname "$PACK")
DF2=$(dirname "$PROOT")/inkling-2-mini-dflash2
[ -f "$DF2/manifest.json" ] && MNT+=(-v "$DF2":"$DF2":ro)
$CTR rm -f "$CNAME" >/dev/null 2>&1 || true
$CTR run -d --name "$CNAME" --security-opt label=disable --security-opt seccomp=unconfined \
  "${CTR_GPU[@]}" "${CTR_USER[@]}" --ipc=host --network=host \
  -e HIP_VISIBLE_DEVICES="$DEVS" -e LD_LIBRARY_PATH=/opt/rocm/lib "${MNT[@]}" \
  -v "$REPO":/work:ro -v "$BIN":/opt/saber/saber:ro -w /work --entrypoint /opt/saber/saber "$IMG" \
  engine --pack "$PACK" --tp "$TP" --prebuilt /work/prebuilt/gfx1201 --hip-lib /opt/rocm/lib/libamdhip64.so.7 \
  --ink-serve "$PORT" --ink-serve-seqs 8 --ink-out /out "$@" >/dev/null || { echo "$CTR run failed" >&2; exit 1; }
echo "== $(date '+%T') booting $CNAME on :$PORT (TP=$TP on HIP device(s) $DEVS; usually under a minute)"
for i in $(seq 1 600); do
  $CTR logs "$CNAME" 2>&1 | grep -q 'inkserve listening' && break
  if [ "$($CTR inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" != true ]; then
    echo "the server exited before it listened (code $($CTR inspect -f '{{.State.ExitCode}}' "$CNAME" 2>/dev/null)):" >&2
    $CTR logs "$CNAME" 2>&1 | tail -30 >&2; exit 1
  fi
  sleep 1
done
$CTR logs "$CNAME" 2>&1 | grep -q 'inkserve listening' || { echo "never listened in 600 s" >&2; exit 1; }
$CTR logs "$CNAME" 2>&1 | grep -E '^inkpaths' | head -1
echo "== $(date '+%T') serving after ${i} s: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1  (model Inkling-2-Mini)"
