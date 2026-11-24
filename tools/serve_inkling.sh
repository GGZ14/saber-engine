#!/usr/bin/env bash
# Boot Saber's served Inkling 2 Mini (the engine's defaults) on both cards.
#
#   tools/serve_inkling.sh             # build the release binary in the dev image, then serve on :8080
#   tools/serve_inkling.sh --no-build  # reuse the last container build
#   tools/serve_inkling.sh stop        # stop the server container
#   PORT=8100 tools/serve_inkling.sh   # another port
#   SABER_NVME=/data/saber tools/serve_inkling.sh   # the storage root tools/fetch_packs.sh inkling used
#   INK_PACK=/data/packs/my-inkling-pack tools/serve_inkling.sh --no-build   # another pack directory
#
# Extra arguments go to `saber engine` (e.g. --ink-serve-ctx 16384). The pack is the published one
# (Hugging Face Launch80/Inkling-2-Mini-NVFP4-SABER, `tools/fetch_packs.sh inkling`): self-contained, its
# dense records and model files in its own base/ (HF3; `inkpaths:` boot line). A pack without base/ reads
# the checkpoint at the path its manifest records, which must then be mounted (it is, under $SABER_NVME/models).
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
for d in "$NVME/models" "$NVME/saber-packs"; do [ -d "$d" ] && MNT+=(-v "$d":"$d":ro); done
$CTR rm -f "$CNAME" >/dev/null 2>&1 || true
$CTR run -d --name "$CNAME" --security-opt label=disable --security-opt seccomp=unconfined \
  "${CTR_GPU[@]}" "${CTR_USER[@]}" --ipc=host --network=host \
  -e HIP_VISIBLE_DEVICES=0,1 -e LD_LIBRARY_PATH=/opt/rocm/lib "${MNT[@]}" \
  -v "$REPO":/work:ro -v "$BIN":/opt/saber/saber:ro -w /work --entrypoint /opt/saber/saber "$IMG" \
  engine --pack "$PACK" --tp 2 --prebuilt /work/prebuilt/gfx1201 --hip-lib /opt/rocm/lib/libamdhip64.so.7 \
  --ink-serve "$PORT" --ink-serve-seqs 8 --ink-out /out "$@" >/dev/null || { echo "$CTR run failed" >&2; exit 1; }
echo "== $(date '+%T') booting $CNAME on :$PORT (usually under a minute)"
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
