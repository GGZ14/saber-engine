#!/usr/bin/env bash
# Run the Saber Console (`saber ui`) as the FRONT process: a browser dashboard on :8090 that
# starts, watches, reconfigures and stops the engine for you (docs/CONSOLE.md).
#
#   tools/serve_console.sh                       # build the release binary in the dev image, then the console on 127.0.0.1:8090
#   tools/serve_console.sh --no-build            # reuse the last container build
#   tools/serve_console.sh --mock                # GPU-free: the console over the mock engine (development, demos)
#   tools/serve_console.sh --preset flashnext    # first run only: seed config.toml with this preset (autostart on,
#                                                #   the wizard skipped); later runs use config.toml as the Settings page left it
#   SABER_CONSOLE=1 tools/serve_flashnext.sh     # the same, from a launcher (`--preset flashnext` / `--preset qwen27b`)
#   tools/serve_console.sh --preset qwen27b --tp 1   # first run only: the 27B on ONE card (engine.tp = 1; DEVICE=N picks
#                                                #   the card, default 0); `SABER_CONSOLE=1 tools/serve_27b.sh --tp 1` does the same
#   HOST=0.0.0.0 tools/serve_console.sh          # reachable from other machines: needs an admin token, made once with
#                                                #   `.cache/target-container/release/saber ui --gen-admin-token`
#   tools/serve_console.sh -- --admin-token X    # anything after `--` goes to `saber ui`
#
# Env: HOST (bind, default the config's or 127.0.0.1), CONSOLE_PORT (default the config's or 8090),
# CONFIG (default ~/.config/saber/config.toml), PORT (the ENGINE's port when seeding, default 8080).
# Extra arguments before `--` are the engine's (seeded into engine.extra_args on a first run).
#
# The console runs on the host (it needs no ROCm: the container-built binary runs there) and starts
# the engine in the dev image exactly as the launchers do. It refuses to start an engine while
# another server or a GPU hold owns the cards (the launchers' guard), never touches the power cap,
# and stops the engine it started when it exits (Ctrl-C / SIGTERM). The headless launchers keep
# working unchanged: the console is additive.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$REPO/tools/lib/container.sh"
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
SABER=${SABER_BIN:-$REPO/.cache/target-container/release/saber}   # SABER_BIN: another build (a host path)
CONFIG=${CONFIG:-$HOME/.config/saber/config.toml}
BUILD=1
MOCK=0
PRESET=""
TP=""
ENGINE_ARGS=()
UI_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --mock) MOCK=1; shift ;;
    --preset) PRESET=${2:-}; shift; [ $# -gt 0 ] && shift ;;
    --preset=*) PRESET=${1#*=}; shift ;;
    --tp) TP=${2:-}; shift; [ $# -gt 0 ] && shift ;;
    --tp=*) TP=${1#*=}; shift ;;
    --) shift; UI_ARGS+=("$@"); break ;;
    *) ENGINE_ARGS+=("$1"); shift ;;
  esac
done
case "$PRESET" in ""|flashnext|qwen27b) ;; 27b) PRESET=qwen27b ;; *) echo "--preset: flashnext or qwen27b" >&2; exit 2 ;; esac
case "${TP:-2}" in 1|2) ;; *) echo "--tp: 1 (one card) or 2 (two cards)" >&2; exit 2 ;; esac
if [ "${TP:-2}" = 1 ] && [ "$PRESET" != qwen27b ]; then
  echo "--tp 1: only the 27B serves on one card (--preset qwen27b); Flash-Next is TP=2 only" >&2; exit 2
fi
DEVICE=${DEVICE:-0}
[[ "$DEVICE" =~ ^[0-9]+$ ]] || { echo "DEVICE=$DEVICE: one HIP device index" >&2; exit 2; }

if [ "$BUILD" = 1 ]; then
  mkdir -p "$REPO/.cache/target-container" "$REPO/.cache/cargo-registry"
  echo "== $(date '+%F %T') building the release binary in the dev image ($(git -C "$REPO" rev-parse --short HEAD))"
  nice -n 19 $CTR run --rm "${CTR_USER[@]}" --security-opt label=disable -v "$REPO":/work \
    -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry -e CARGO_TARGET_DIR=/work/.cache/target-container \
    -w /work --entrypoint bash "$IMG" -c "cargo build --release -p saber-cli 2>&1 | tail -3"
fi
[ -x "$SABER" ] || { echo "no release binary at $SABER (run without --no-build)" >&2; exit 1; }
"$SABER" ui --help >/dev/null 2>&1 || { echo "$SABER has no \`ui\` subcommand: rebuild it (run without --no-build)" >&2; exit 1; }

# a first run from a launcher: seed config.toml with that launcher's preset
if [ -n "$PRESET" ] && [ "$MOCK" = 0 ]; then
  if [ -f "$CONFIG" ]; then
    echo "== $CONFIG exists: the console runs the engine it describes (change it on the Settings page)"
    [ ${#ENGINE_ARGS[@]} -gt 0 ] && echo "   (not applied: ${ENGINE_ARGS[*]}; add them under Settings > Advanced, or to engine.extra_args)"
  else
    mkdir -p "$(dirname "$CONFIG")"
    extra=""
    for a in "${ENGINE_ARGS[@]}"; do extra+="\"${a//\"/\\\"}\", "; done
    umask 077
    cat > "$CONFIG" <<EOF
# Seeded by tools/serve_console.sh --preset $PRESET; the console's Settings page edits it from here.
version = 5

[engine]
preset = "$PRESET"
tp = ${TP:-2}
device = $DEVICE
port = ${PORT:-8080}
autostart = true
extra_args = [${extra%, }]

[engine.paths]
repo = "$REPO"

[setup]
completed = true
skipped = true
step = "done"
EOF
    echo "== seeded $CONFIG (preset $PRESET, TP=${TP:-2}$([ "${TP:-2}" = 1 ] && echo " on HIP device $DEVICE"), engine port ${PORT:-8080}, autostart on)"
  fi
fi

ARGS=(ui --config "$CONFIG")
[ -n "${HOST:-}" ] && ARGS+=(--host "$HOST")
[ -n "${CONSOLE_PORT:-}" ] && ARGS+=(--port "$CONSOLE_PORT")
[ "$MOCK" = 1 ] && ARGS+=(--engine mock)
echo "== $(date '+%T') saber ui (Ctrl-C stops the console and the engine it started)"
exec "$SABER" "${ARGS[@]}" "${UI_ARGS[@]}"
