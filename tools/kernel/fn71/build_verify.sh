#!/usr/bin/env bash
# FN71: build the verify program OFF the lock, inside the dev image (compile only, no GPU):
#   .cache/fn71/fn71_verify   (tools/kernel/fn71/fn71_verify.cpp; scratch, never committed)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
mkdir -p "$REPO/.cache/fn71"
podman run --rm --security-opt label=disable -v "$REPO":/work -w /work -e HIP_VISIBLE_DEVICES= \
  --entrypoint /opt/rocm/bin/hipcc "$IMG" -O2 -std=c++17 -Wall -Wno-unused-result \
  tools/kernel/fn71/fn71_verify.cpp -o .cache/fn71/fn71_verify -L/opt/rocm/lib -lhsa-runtime64
ls -la "$REPO/.cache/fn71/fn71_verify"
