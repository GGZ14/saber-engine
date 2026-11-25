#!/usr/bin/env bash
# Download the ready-made weight packs from Hugging Face, so a fresh clone can serve without
# converting a checkpoint (the CPU repack of tools/pack/build_devfinal.sh).
#
#   tools/fetch_packs.sh 27b-macaq-w4a8      # ~28 GB: the DEFAULT 27B (MACaQ W4A8-iu8) target + the W4A8-iu8
#                                     #   DFlash2 drafter (TP=2), and the tokenizer / chat template into .cache/;
#                                     #   DRAFTER=fp8 = the FP8 drafter instead (~29 GB)
#   tools/fetch_packs.sh 27b-macaq-w4a8-tp1  # ~26 GB: the same for ONE card (TP=1)
#   tools/fetch_packs.sh 27b-macaq    # ~29 GB: the MACaQ MXFP4 target (the default before the W4A8-iu8 pack) + its
#                                     #   paired FP8 drafter (TP=2): tools/serve_27b.sh --weights macaq
#   tools/fetch_packs.sh 27b-macaq-tp1  # ~28 GB: the same for ONE card (TP=1)
#   tools/fetch_packs.sh 27b          # ~26 GB: the NVFP4 27B target pack + the DFlash2 drafter pack,
#                                     #   and the tokenizer / chat template into .cache/
#   tools/fetch_packs.sh 27b-tp1      # ~25 GB: the SINGLE-CARD (TP=1) 27B target + drafter packs
#   tools/fetch_packs.sh 27b-quark    # ~29 GB: the Quark AWQ MXFP4 27B target + its paired drafter (TP=2)
#   tools/fetch_packs.sh 27b-quark-tp1  # ~28 GB: the same for ONE card (TP=1)
#   tools/fetch_packs.sh flashnext    # 71.1 GB: the DEFAULT Flash-Next pack (MACaQ int4 experts,
#                                     #   Launch80/Qwen3.8-Flash-Next-MACaQ-SABER) + ONLY the 83 of 131 original
#                                     #   checkpoint shards it reads (139.5 GB, not 182.3): 210.6 GB in all
#                                     #   (alias: flashnext-macaq)
#   tools/fetch_packs.sh flashnext-mxfp4  # 74.2 GB: the MXFP4-expert Flash-Next pack (the ROLLBACK:
#                                     #   FN_EXPERTS_FMT=mxfp4 tools/serve_flashnext.sh) + the same 83-shard
#                                     #   checkpoint subset: 213.7 GB in all
#   tools/fetch_packs.sh inkling      # 20.7 GB: Inkling 2 Mini (Launch80/Inkling-2-Mini-NVFP4-SABER), self-contained:
#                                     #   experts, 8-bit trunk, the bundled drafter and the base/ dense subset; no checkpoint.
#                                     #   ONE download serves one card AND two: its tp1/ folder (286 KB) is the one-card
#                                     #   (TP=1) view of the same files; tools/serve_inkling.sh picks by GPU count
#   tools/fetch_packs.sh inkling-tp1  # the same download (an alias: the one-card layout shares every byte)
#   tools/fetch_packs.sh all          # 27b-macaq-w4a8 (TP=2) + flashnext
#
# Packs land in $SABER_PACKS (default $SABER_NVME/saber-packs = /mnt/nvme/saber-packs), the directory the
# launchers read.
# The download resumes if interrupted. Env:
#   SABER_PACKS     destination directory
#   SABER_NVME      the storage root (default /mnt/nvme): flashnext[-mxfp4] put the pack in $SABER_NVME/saber-packs/
#                   and the checkpoint in $SABER_NVME/models/ (the engine finds it there whatever the root is;
#                   serve with SABER_NVME=<root> tools/serve_flashnext.sh)
#   SABER_HF_OWNER  Hugging Face account that hosts the packs (default Launch80)
#   SABER_FN_CHECKPOINT_REPO  Flash-Next's original checkpoint (default amd/Qwen3.8-Flash-Next-Quark-MXFP4)
#   SABER_FN_CHECKPOINT_REV   its revision (default b2091c38..., the one the packs were cut from; tools/models.json)
#   DRAFTER         27b-macaq-w4a8[-tp1] (and all) only: w4a8 (default) or fp8
#   HF_TOKEN        only needed while a repo is private
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NVME=${SABER_NVME:-/mnt/nvme}
PACKS=${SABER_PACKS:-$NVME/saber-packs}
OWNER=${SABER_HF_OWNER:-Launch80}
R27=$OWNER/Qwen3.8-27B-NVFP4-SABER
RFN=$OWNER/Qwen3.8-Flash-Next-SABER
RQK=$OWNER/Qwen3.8-27B-Quark-AWQ-MXFP4-SABER
RMQ=$OWNER/Qwen3.8-27B-MACaQ-MXFP4-SABER
RW4=$OWNER/Qwen3.8-27B-MACaQ-W4A8-SABER
RDW4=$OWNER/Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER
RDF8=$OWNER/Qwen3.8-27B-DFlash2-Drafter-FP8-SABER
RFNM=$OWNER/Qwen3.8-Flash-Next-MACaQ-SABER
RINK=$OWNER/Inkling-2-Mini-NVFP4-SABER
# the checkpoint revision the Flash-Next packs were built from and verified against (their records carry sha256s)
CKREV=${SABER_FN_CHECKPOINT_REV:-b2091c3809eb51902f46a163973362d8176f9e58}
CKREPO=${SABER_FN_CHECKPOINT_REPO:-amd/Qwen3.8-Flash-Next-Quark-MXFP4}

HF=$(command -v hf || command -v huggingface-cli || true)
[ -n "$HF" ] || { echo "needs the Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2; exit 1; }

want=${1:-}
case "$want" in 27b-macaq-w4a8 | 27b-macaq-w4a8-tp1 | 27b-macaq | 27b-macaq-tp1 | 27b | 27b-tp1 | 27b-quark | 27b-quark-tp1 | flashnext | flashnext-macaq | flashnext-mxfp4 | inkling | inkling-tp1 | all) ;; *) echo "usage: $0 27b-macaq-w4a8|27b-macaq-w4a8-tp1|27b-macaq|27b-macaq-tp1|27b|27b-tp1|27b-quark|27b-quark-tp1|flashnext|flashnext-mxfp4|inkling|inkling-tp1|all  (27b-macaq-w4a8 = the default 27B, flashnext = the MACaQ Flash-Next)" >&2; exit 2 ;; esac
[ "$want" = inkling-tp1 ] && want=inkling   # one repo, both layouts
[ "$want" = flashnext-macaq ] && want=flashnext

# A pack the engine would refuse fails here, by name, instead of mid-boot: every rank file
# must start with the SBRPACK1 magic (docs/abi/PACK.md).
check_packs() {
  local f bad=0
  for f in "$@"; do
    [ -f "$f" ] && [ "$(head -c 8 "$f")" = "SBRPACK1" ] || { echo "bad or missing pack: $f" >&2; bad=1; }
  done
  [ "$bad" = 0 ]
}

# The Flash-Next trunk8 / mtp_fp8 / mtp_x4 / hc8 files are raw record blobs, NOT SBRPACK1 files: they have no
# magic, and the engine reads them through the sibling <name>.json, whose "bytes" is the blob's size. So check
# that the index exists and the size matches (a truncated download fails here). dense.pack and vision/rank*.pack
# in the same repo are real SBRPACK1 files and keep the magic check above.
check_blobs() {
  local f j want have bad=0
  for f in "$@"; do
    j=${f%.pack}.json
    if [ ! -f "$f" ] || [ ! -f "$j" ]; then echo "bad or missing blob: $f (needs $j)" >&2; bad=1; continue; fi
    want=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["bytes"])' "$j" 2>/dev/null) || want=
    have=$(wc -c < "$f" | tr -d ' ')
    [ -n "$want" ] && [ "$want" = "$have" ] || { echo "bad blob size: $f is $have bytes, $j says ${want:-?}" >&2; bad=1; }
  done
  [ "$bad" = 0 ]
}

mkdir -p "$PACKS"

if [ "$want" = 27b-macaq ] || [ "$want" = 27b-macaq-tp1 ]; then
  # The MACaQ MXFP4 pack (tools/serve_27b.sh --weights macaq) and the Quark repo's quarkhead drafter: the MACaQ pack keeps
  # the Quark checkpoint's lm_head byte for byte, so the same drafter pairs with it. The tokenizer is the Quark
  # checkpoint's own (the same file the Quark target uses); the chat template is the shared one.
  T=""; [ "$want" = 27b-macaq-tp1 ] && T=-tp1
  echo "== $RMQ${T:+ (TP=1 packs)} -> $PACKS"
  "$HF" download "$RMQ" --local-dir "$PACKS" --include "qwen38-27b-macaq7-bc3$T/*"
  echo "== $RQK (the paired drafter${T:+, TP=1}) -> $PACKS"
  "$HF" download "$RQK" --local-dir "$PACKS" --include "qwen38-dflash2-drafter-quarkhead$T/*"
  mkdir -p "$REPO/.cache/macaq27b-dl"
  "$HF" download "$RMQ" --local-dir "$REPO/.cache/macaq27b-dl" --include 'tokenizer.json' --include 'qwen-fixed-v22.3.jinja'
  cp -f "$REPO/.cache/macaq27b-dl/tokenizer.json" "$REPO/.cache/quark27b.tokenizer.json"
  [ -f "$REPO/.cache/template_corpus.jinja" ] || cp -f "$REPO/.cache/macaq27b-dl/qwen-fixed-v22.3.jinja" "$REPO/.cache/template_corpus.jinja"
  if [ -n "$T" ]; then
    check_packs "$PACKS"/qwen38-27b-macaq7-bc3-tp1/rank0.pack "$PACKS"/qwen38-dflash2-drafter-quarkhead-tp1/rank0.pack
    echo "27B MACaQ MXFP4 TP=1 packs ready: tools/serve_27b.sh --weights macaq --tp 1"
  else
    check_packs "$PACKS"/qwen38-27b-macaq7-bc3/rank{0,1}.pack "$PACKS"/qwen38-dflash2-drafter-quarkhead/rank{0,1}.pack
    echo "27B MACaQ MXFP4 packs ready: tools/serve_27b.sh --weights macaq"
  fi
fi

# One TP folder of a Hugging Face repo that holds one model (the TP=2 pack in tp2/, the TP=1 pack in tp1/) into
# the flat pack folder the launchers read ($PACKS/NAME): the download goes to a hidden staging folder beside it
# (resumable) and the TP folder is moved into place whole once it is complete. A NAME already holding rank0.pack is
# kept (delete it to download again).
fetch_tp_folder() {
  local repo=$1 d=$2 name=$3
  local dst="$PACKS/$name" stage="$PACKS/.fetch-$name"
  if [ -f "$dst/rank0.pack" ]; then echo "== $dst is already there (delete it to download again)"; return 0; fi
  [ ! -e "$dst" ] || { echo "$dst exists but holds no rank0.pack (a Hugging Face $d/ folder inside it? move $dst/$d to $dst, or delete $dst)" >&2; return 1; }
  echo "== $repo ($d/) -> $dst"
  "$HF" download "$repo" --local-dir "$stage" --include "$d/*"
  [ -f "$stage/$d/rank0.pack" ] || { echo "the download of $repo $d/ left no $stage/$d/rank0.pack" >&2; return 1; }
  mv "$stage/$d" "$dst" && rm -rf "$stage"
}

if [ "$want" = 27b-macaq-w4a8 ] || [ "$want" = 27b-macaq-w4a8-tp1 ] || [ "$want" = all ]; then
  # The MACaQ W4A8-iu8 target (the launcher's default weights) and its drafter: tp2/ -> qwen38-27b-macaq-w4a8 and
  # qwen38-dflash2-drafter-$DR, tp1/ -> the same names + -tp1. DRAFTER=w4a8 (default) = the W4A8-iu8 DFlash2
  # drafter, fp8 = the FP8 one (the same packs as qwen38-dflash2-drafter-quarkhead[-tp1]). Both keep the Quark
  # checkpoint's lm_head; the tokenizer is the Quark checkpoint's own, the chat template the shared one.
  D=tp2; T=""; [ "$want" = 27b-macaq-w4a8-tp1 ] && { D=tp1; T=-tp1; }
  DR=${DRAFTER:-w4a8}
  case "$DR" in w4a8) RD=$RDW4 ;; fp8) RD=$RDF8 ;; *) echo "DRAFTER=$DR: w4a8 or fp8" >&2; exit 2 ;; esac
  fetch_tp_folder "$RW4" "$D" "qwen38-27b-macaq-w4a8$T"
  fetch_tp_folder "$RD" "$D" "qwen38-dflash2-drafter-$DR$T"
  mkdir -p "$REPO/.cache/macaq-w4a8-dl"
  "$HF" download "$RW4" --local-dir "$REPO/.cache/macaq-w4a8-dl" --include 'tokenizer.json' --include 'qwen-fixed-v22.3.jinja'
  cp -f "$REPO/.cache/macaq-w4a8-dl/tokenizer.json" "$REPO/.cache/quark27b.tokenizer.json"
  [ -f "$REPO/.cache/template_corpus.jinja" ] || cp -f "$REPO/.cache/macaq-w4a8-dl/qwen-fixed-v22.3.jinja" "$REPO/.cache/template_corpus.jinja"
  DF=""; [ "$DR" = fp8 ] && DF=" --drafter fp8"
  if [ -n "$T" ]; then
    check_packs "$PACKS"/qwen38-27b-macaq-w4a8-tp1/rank0.pack "$PACKS"/qwen38-dflash2-drafter-$DR-tp1/rank0.pack
    echo "27B MACaQ W4A8 TP=1 packs ready: tools/serve_27b.sh --tp 1$DF"
  else
    check_packs "$PACKS"/qwen38-27b-macaq-w4a8/rank{0,1}.pack "$PACKS"/qwen38-dflash2-drafter-$DR/rank{0,1}.pack
    echo "27B MACaQ W4A8 packs ready: tools/serve_27b.sh$DF"
  fi
fi

if [ "$want" = 27b ]; then
  echo "== $R27 -> $PACKS"
  "$HF" download "$R27" --local-dir "$PACKS" \
    --include 'qwen38-27b-nvfp4-devfinal-fp8head/*' --include 'qwen38-dflash2-drafter-fp8head/*'
  mkdir -p "$REPO/.cache"
  "$HF" download "$R27" --local-dir "$REPO/.cache" \
    --include 'template_corpus.tokenizer.json' --include 'template_corpus.jinja'
  check_packs "$PACKS"/qwen38-27b-nvfp4-devfinal-fp8head/rank{0,1}.pack \
              "$PACKS"/qwen38-dflash2-drafter-fp8head/rank{0,1}.pack
  echo "27B NVFP4 packs ready: tools/serve_27b.sh --weights nvfp4"
fi

if [ "$want" = 27b-tp1 ]; then
  # A pack is cut for one TP degree (rows/columns are already sharded per rank), so the TP=1 packs
  # are separate folders from the TP=2 ones above. One rank file each: rank0.pack.
  echo "== $R27 (TP=1 packs) -> $PACKS"
  "$HF" download "$R27" --local-dir "$PACKS" \
    --include 'qwen38-27b-nvfp4-devfinal-tp1-fp8head/*' --include 'qwen38-dflash2-drafter-tp1-fp8head/*'
  mkdir -p "$REPO/.cache"
  "$HF" download "$R27" --local-dir "$REPO/.cache" \
    --include 'template_corpus.tokenizer.json' --include 'template_corpus.jinja'
  check_packs "$PACKS"/qwen38-27b-nvfp4-devfinal-tp1-fp8head/rank0.pack \
              "$PACKS"/qwen38-dflash2-drafter-tp1-fp8head/rank0.pack
  echo "27B NVFP4 TP=1 packs ready: tools/serve_27b.sh --weights nvfp4 --tp 1"
fi

if [ "$want" = 27b-quark ] || [ "$want" = 27b-quark-tp1 ]; then
  # amd/Qwen3.8-27B-Quark-AWQ-MXFP4 packed for Saber: the target and a drafter pack whose head is
  # THIS checkpoint's lm_head (the engine refuses a drafter whose head differs from the target's).
  T=""; [ "$want" = 27b-quark-tp1 ] && T=-tp1
  echo "== $RQK${T:+ (TP=1 packs)} -> $PACKS"
  "$HF" download "$RQK" --local-dir "$PACKS" \
    --include "qwen38-27b-quark-awq-mxfp4$T/*" --include "qwen38-dflash2-drafter-quarkhead$T/*"
  # The checkpoint's own tokenizer (it differs from the NVFP4 one); the chat template is the same file.
  mkdir -p "$REPO/.cache/quark27b-dl"
  "$HF" download "$RQK" --local-dir "$REPO/.cache/quark27b-dl" --include 'tokenizer.json' --include 'qwen-fixed-v22.3.jinja'
  cp -f "$REPO/.cache/quark27b-dl/tokenizer.json" "$REPO/.cache/quark27b.tokenizer.json"
  [ -f "$REPO/.cache/template_corpus.jinja" ] || cp -f "$REPO/.cache/quark27b-dl/qwen-fixed-v22.3.jinja" "$REPO/.cache/template_corpus.jinja"
  if [ -n "$T" ]; then
    check_packs "$PACKS"/qwen38-27b-quark-awq-mxfp4-tp1/rank0.pack "$PACKS"/qwen38-dflash2-drafter-quarkhead-tp1/rank0.pack
    echo "27B Quark TP=1 packs ready: tools/serve_27b.sh --weights quark --tp 1"
  else
    check_packs "$PACKS"/qwen38-27b-quark-awq-mxfp4/rank{0,1}.pack "$PACKS"/qwen38-dflash2-drafter-quarkhead/rank{0,1}.pack
    echo "27B Quark packs ready: tools/serve_27b.sh --weights quark"
  fi
fi

# The Flash-Next packs are NOT self-contained: the manifest (and ple/ple_index.json) point into the
# original AMD Quark checkpoint (embeddings, norms, router, attention and the 102 GB PLE table are read from its
# shards in place). Those records live in 83 of the checkpoint's 131 shards; the other 48 (the MXFP4 routed
# experts, already in the packs' tiers) are never opened, so only the 83 are fetched, into
# $NVME/models/Qwen3.8-Flash-Next-Quark-MXFP4 (the engine rebases the manifest's recorded path there, HF2).
CK=$NVME/models/Qwen3.8-Flash-Next-Quark-MXFP4
CK_GB=139.5

gb_free() { mkdir -p "$1" && df -P -B1 "$1" | awk 'NR==2 {printf "%.1f", $4 / 1e9}'; }

# fn_space PACK_DIR PACK_GB: refuse up front when the pack + the checkpoint subset (+10 GB) do not fit
fn_space() {
  local have free
  have=$( (du -sb "$1" "$CK" 2>/dev/null || true) | awk '{s += $1} END {printf "%.1f", s / 1e9}')
  free=$(gb_free "$NVME")
  echo "== pack $2 GB -> $1, checkpoint subset $CK_GB GB -> $CK"
  echo "   already on disk ${have:-0} GB, free on $(df -P "$NVME" | awk 'NR==2 {print $6}') $free GB"
  awk -v f="$free" -v n="$2" -v c="$CK_GB" -v h="${have:-0}" 'BEGIN { exit !(f + h >= n + c + 10) }' \
    || { echo "not enough space: needs $2 + $CK_GB GB (+10 GB margin), has $free GB free + ${have:-0} GB already fetched" >&2; exit 1; }
  # a maintainer's pack built in place links its members (symlinks): a download must never write through them
  if [ -d "$1" ] && [ -n "$(find "$1" -maxdepth 2 -type l -print -quit)" ]; then
    echo "$1 holds symlinks (a locally built pack); refusing to download over it -- use another SABER_NVME" >&2; exit 1
  fi
}

# Only dense / vision members carry the SBRPACK1 magic; trunk8 / mtp / hc8 are headless records
# described by their .json (checked non-empty).
check_fn_members() {
  check_packs "$1"/rank{0,1}/dense.pack "$1"/shared/dense.pack "$1"/vision/rank{0,1}.pack
  local f
  for f in manifest.json ple/ple_index.json rank0/trunk8.pack rank1/trunk8.pack rank0/mtp_x4.pack rank1/mtp_x4.pack \
           rank0/mtp_fp8.pack rank1/mtp_fp8.pack shared/hc8.pack "${@:2}"; do
    [ -s "$1/$f" ] || { echo "missing or empty $1/$f" >&2; exit 1; }
  done
}

# fetch_fn_checkpoint PACK_DIR: the checkpoint shards PACK_DIR reads, from its own record tables and PLE index
# (not a hand-kept list; tools/models.json carries the same 83 names), + the small model files
fetch_fn_checkpoint() {
  local P=$1 s miss=0
  mapfile -t SHARDS < <(python3 - "$P" <<'PY2'
import json, sys
p = sys.argv[1]
files = set()
def walk(v):
    if isinstance(v, dict):
        st = v.get("storage")
        if isinstance(st, dict) and isinstance(st.get("checkpoint"), list):
            files.update(e["file"] for e in st["checkpoint"])
        for x in v.values():
            walk(x)
    elif isinstance(v, list):
        for x in v:
            walk(x)
for j in ("rank0/dense.json", "rank1/dense.json", "rank0/aliases.json", "rank1/aliases.json", "shared/dense.json"):
    walk(json.load(open(f"{p}/{j}")))
files.update(s["file"] for s in json.load(open(f"{p}/ple/ple_index.json"))["sources"]["ckpt_bf16"]["shards"])
sizes = {s["name"]: s["size_bytes"] for s in json.load(open(f"{p}/manifest.json"))["checkpoint"]["shards"]}
print(f"# {len(files)} shards, {sum(sizes[f] for f in files) / 1e9:.1f} GB", file=sys.stderr)
for f in sorted(files):
    print(f)
PY2
  )
  INC=(--include LICENSE --include README.md --include config.json --include generation_config.json
       --include chat_template.jinja --include 'tokenizer*' --include vocab.json --include merges.txt
       --include '*preprocessor_config.json' --include model.safetensors.index.json)
  for s in "${SHARDS[@]}"; do INC+=(--include "$s"); done
  echo "== $CKREPO@${CKREV:0:8}: ${#SHARDS[@]} shards + the small files -> $CK (resumable)"
  "$HF" download "$CKREPO" --revision "$CKREV" --local-dir "$CK" "${INC[@]}"
  for s in "${SHARDS[@]}"; do [ -s "$CK/$s" ] || { echo "missing $CK/$s" >&2; miss=1; }; done
  [ "$miss" = 0 ] && [ -f "$CK/model.safetensors.index.json" ] || { echo "checkpoint incomplete under $CK" >&2; exit 1; }
  echo "== on disk: pack $(du -sb "$P" | awk '{printf "%.1f", $1 / 1e9}') GB, checkpoint $(du -sb "$CK" | awk '{printf "%.1f", $1 / 1e9}') GB"
  echo "   verify (optional, ~1 min a 100 GB): (cd $P && sha256sum -c --quiet SHA256SUMS)"
}

if [ "$want" = flashnext ] || [ "$want" = flashnext-macaq ] || [ "$want" = all ]; then
  # The MACaQ pack (the served default since MACFN8; HF1 / HF2): repo root = the pack directory.
  P=$PACKS/qwen38-flashnext-macaq-tp2-trunk8
  fn_space "$P" 71.1
  echo "== $RFNM -> $P (resumable)"
  "$HF" download "$RFNM" --local-dir "$P"
  check_fn_members "$P" rank{0,1}/experts.macaq.xtier rank{0,1}/experts.shq.xtier rank{0,1}/experts.tsh
  fetch_fn_checkpoint "$P"
  echo "Flash-Next MACaQ ready: SABER_NVME=$NVME tools/serve_flashnext.sh"
fi

if [ "$want" = flashnext-mxfp4 ]; then
  # The MXFP4 rollback pack (its tp2 expert tiers are not in the MACaQ repo); the same checkpoint subset.
  P=$PACKS/qwen38-flashnext-quark-trunk8
  fn_space "$P" 74.2
  echo "== $RFN -> $PACKS (resumable)"
  "$HF" download "$RFN" --local-dir "$PACKS" --include 'qwen38-flashnext-quark-trunk8/*'
  check_fn_members "$P" rank{0,1}/experts.tp2.xtier
  fetch_fn_checkpoint "$P"
  echo "Flash-Next MXFP4 ready: SABER_NVME=$NVME FN_EXPERTS_FMT=mxfp4 tools/serve_flashnext.sh"
fi

if [ "$want" = inkling ]; then
  # HF3: the Inkling 2 Mini pack is self-contained -- its base/ holds the dense records' tensors (3.2 GB, byte for
  # byte from the converted checkpoint) + the config / tokenizer / chat template; the engine reads base/ first
  # (`inkpaths:` boot line), so no checkpoint download. Serve with SABER_NVME=<root> tools/serve_inkling.sh.
  P=$PACKS/inkling-2-mini-nvfp4
  NEED_GB=20.8
  mkdir -p "$P"
  have=$(du -sb "$P" 2>/dev/null | awk '{printf "%.1f", $1 / 1e9}')
  free=$(df -P -B1 "$P" | awk 'NR==2 {printf "%.1f", $4 / 1e9}')
  echo "== inkling: $NEED_GB GB -> $P (already on disk ${have:-0} GB, free $free GB)"
  awk -v f="$free" -v n="$NEED_GB" -v h="${have:-0}" 'BEGIN { exit !(f + h >= n + 5) }' \
    || { echo "not enough space: needs $NEED_GB GB (+5 GB margin), has $free GB free + ${have:-0} GB already fetched" >&2; exit 1; }
  if [ -n "$(find "$P" -maxdepth 2 -type l -print -quit)" ]; then
    echo "$P holds symlinks (a locally built pack); refusing to download over it -- use another SABER_NVME" >&2; exit 1
  fi
  echo "== $RINK -> $P (resumable)"
  "$HF" download "$RINK" --local-dir "$P"
  miss=0
  for f in manifest.json base/dense.safetensors base/model.safetensors.index.json base/config.json base/tokenizer.json \
           base/chat_template.jinja draft/manifest.json draft/vocab_rank.u32 trunk8/manifest.json \
           rank0/experts.xtier rank0/experts_mx8.xtier rank1/experts.xtier rank1/experts_mx8.xtier \
           rank0/dense.json rank1/dense.json shared/dense.json trunk8/rank0/trunk8.bin trunk8/rank1/trunk8.bin \
           tp1/manifest.json tp1/shared/dense.json tp1/trunk8.json; do
    [ -s "$P/$f" ] || { echo "missing $P/$f" >&2; miss=1; }
  done
  [ "$miss" = 0 ] || { echo "pack incomplete under $P (re-run to resume)" >&2; exit 1; }
  echo "== on disk: $(du -sb "$P" | awk '{printf "%.1f", $1 / 1e9}') GB"
  echo "   verify (optional, ~1 min): (cd $P && sha256sum -c --quiet SHA256SUMS)"
  echo "Inkling 2 Mini ready (one card or two: the launcher picks by GPU count): SABER_NVME=$NVME tools/serve_inkling.sh"
fi
