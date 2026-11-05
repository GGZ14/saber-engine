#!/usr/bin/env bash
# Run a command inside the saber-dev image with the repo at /work, models at /models, both GPUs,
# and a persistent cargo cache. Usage: tools/dev.sh cargo build --release   |   tools/dev.sh   (shell)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/tools/lib/container.sh"
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
mkdir -p "$REPO/.cache/cargo-registry" "$REPO/.cache/target-container"
exec $CTR run --rm -it${SABER_DETACH:+d} \
  "${CTR_GPU[@]}" "${CTR_USER[@]}" --ipc=host --security-opt seccomp=unconfined \
  -v "$REPO":/work -v "${SABER_MODELS:-/mnt/nvme/models}":/models:ro \
  -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry \
  -e CARGO_TARGET_DIR=/work/.cache/target-container \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}" \
  -w /work "$IMG" ${1:+-c "$*"}
