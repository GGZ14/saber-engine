#!/usr/bin/env bash
# Build the `saber ui` SPA (ui/ -> ui/dist, which saber-console embeds with rust-embed).
#
#   tools/console/build_ui.sh            # npm ci + vite build inside the saber-dev image
#   tools/console/build_ui.sh --install  # npm install (refresh ui/package-lock.json after a package.json change)
#
# The host has no node and neither does the dev image, so a pinned Node tarball (sha256-checked)
# is unpacked once under the shared cache and mounted read-only into the image. The container gets
# NO GPU devices: this is a JavaScript build. ui/dist is committed so a plain `cargo build` needs
# no node; re-run this after any change under ui/src and commit the result.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
. "$REPO/tools/lib/container.sh"
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
SHARED=${SABER_SHARED:-$HOME/saber-cache}
NODE_VER=v22.11.0
NODE_SHA=83bf07dd343002a26211cf1fcd46a9d9534219aad42ee02847816940bf610a72
NODE_DIR=$SHARED/console-node
NODE_HOME=$NODE_DIR/node-$NODE_VER-linux-x64
NPM_CACHE=$NODE_DIR/npm-cache
MODE=ci
[ "${1:-}" = --install ] && MODE=install

if [ ! -x "$NODE_HOME/bin/node" ]; then
  mkdir -p "$NODE_DIR"
  tar=$NODE_DIR/node-$NODE_VER-linux-x64.tar.xz
  curl -fsSL -o "$tar.part" "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz"
  echo "$NODE_SHA  $tar.part" | sha256sum -c --quiet || { echo "node tarball sha256 mismatch" >&2; rm -f "$tar.part"; exit 1; }
  mv "$tar.part" "$tar"
  tar -C "$NODE_DIR" -xJf "$tar"
fi
mkdir -p "$NPM_CACHE"

# one name, so a stuck build is easy to find and remove
CNAME=saber-console-ui-build
$CTR rm -f "$CNAME" >/dev/null 2>&1 || true
nice -n 19 $CTR run --rm --name "$CNAME" "${CTR_USER[@]}" --security-opt label=disable \
  -v "$REPO/ui":/ui -v "$NODE_HOME":/opt/node:ro -v "$NPM_CACHE":/npm-cache \
  -e PATH=/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e npm_config_cache=/npm-cache -e npm_config_update_notifier=false -e npm_config_fund=false -e npm_config_audit=false \
  -w /ui --entrypoint bash "$IMG" -c "node --version && npm $MODE --no-progress && npm run build"
ls -la "$REPO/ui/dist" "$REPO/ui/dist/assets"
