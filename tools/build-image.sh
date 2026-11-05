#!/usr/bin/env bash
# Build localhost/saber-dev:latest (override the tag with SABER_IMAGE) from Containerfile.public:
# a public ROCm 7.x base plus Rust, bindgen and hipcc. Needs only podman or docker and network.
# SABER_ROCM_IMAGE picks another ROCm base tag.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/tools/lib/container.sh"   # CTR = podman or docker
TAG=${SABER_IMAGE:-localhost/saber-dev:latest}
echo "== building $TAG from the public ROCm base (Containerfile.public)"
CTX=$(mktemp -d)   # the image needs nothing from the repo; an empty context keeps target/ and .cache/ out of the upload
trap 'rm -rf "$CTX"' EXIT
"$CTR" build -t "$TAG" -f "$REPO/Containerfile.public" ${SABER_ROCM_IMAGE:+--build-arg SABER_ROCM_IMAGE="$SABER_ROCM_IMAGE"} "$CTX"
"$CTR" run --rm "$TAG" -c 'cargo --version && bindgen --version && hipcc --version | head -1 && ls /opt/rocm/lib/libamdhip64.so.7'
