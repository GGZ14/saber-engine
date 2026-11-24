#!/usr/bin/env python3
"""HF3: make an Inkling 2 pack self-contained (the published form).

The native pack (`ink6-pack/1`) keeps its experts in `rank*/experts*.xtier` but its dense records
(`shared/` + `rank*/dense.json`: embeddings, attention, norms, router, head -- ~3.2 GB) point INTO
the original checkpoint's four shards (19.5 GB). This writes the published pack's `base/`:

  base/dense.safetensors              every tensor a dense record names, byte for byte (a valid
                                      safetensors file; tensor names and dtypes unchanged)
  base/model.safetensors.index.json   the checkpoint marker (weight_map -> dense.safetensors)
  base/<small files>                  config, generation config, chat template, tokenizer

and the staged pack's `shared/` + `rank*/dense.json`, each record re-pointed at
`dense.safetensors` (same length / stride / count / sha256; offsets moved). Every record's bytes
are re-read from the new file and checked against its sha256. The manifest is NOT rewritten (its
`checkpoint` path stays the plan's input): the engine reads `<pack>/base` when it holds the marker
(`saber_cli::inkling2::resolve_ckpt`), else the recorded path.

    build_ink2_base.py --pack /mnt/nvme/models/ink2-pack-nv \
        --ckpt /mnt/nvme/models/Inkling-2-Mini-nvfp4 --base OUT/base --dense-out OUT
"""
import argparse
import hashlib
import json
import os
import shutil
import struct

SMALL = ["config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json"]
NEW = "dense.safetensors"

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True, help="the served pack (dense.json records into --ckpt)")
ap.add_argument("--ckpt", required=True, help="the checkpoint the pack was built from")
ap.add_argument("--base", required=True, help="output: the base/ directory")
ap.add_argument("--dense-out", required=True, help="output: the dir that gets shared/ rank0/ rank1/ dense.json")
a = ap.parse_args()


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


parts = ["shared", "rank0", "rank1"]
recs = {p: json.load(open(f"{a.pack}/{p}/dense.json")) for p in parts}
hdrs = {}
want = {}  # tensor -> (shard, dtype, shape, abs_start, nbytes)
for p in parts:
    for r in recs[p]:
        f = r["file"]
        if f not in hdrs:
            hdrs[f] = header(f"{a.ckpt}/{f}")
        base_off, h = hdrs[f]
        t = h[r["tensor"]]
        s, e = t["data_offsets"]
        start = base_off + s
        span = r["stride"] * (r["count"] - 1) + r["length"]
        assert start <= r["offset"] and r["offset"] + span <= base_off + e, f"{r['tensor']}: record outside its tensor"
        want[r["tensor"]] = (f, t["dtype"], t["shape"], start, e - s)

names = sorted(want, key=lambda n: (want[n][0], want[n][3]))
meta, off = {}, 0
for n in names:
    _, dt, shape, _, nb = want[n]
    meta[n] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
    off += nb
meta["__metadata__"] = {"format": "pt", "source": "dense records of an Inkling 2 Mini SABER pack"}
hj = json.dumps(meta, separators=(",", ":")).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
data0 = 8 + len(hj)

os.makedirs(a.base, exist_ok=True)
out = f"{a.base}/{NEW}"
with open(out + ".part", "wb") as o:
    o.write(struct.pack("<Q", len(hj)) + hj)
    for n in names:
        f, _, _, start, nb = want[n]
        with open(f"{a.ckpt}/{f}", "rb") as src:
            src.seek(start)
            left = nb
            while left:
                b = src.read(min(left, 64 << 20))
                assert b, f"{n}: short read"
                o.write(b)
                left -= len(b)
os.replace(out + ".part", out)
newabs = {n: data0 + meta[n]["data_offsets"][0] for n in names}

# re-point the records, then check every record's sha256 from the new file
bad = 0
with open(out, "rb") as fo:
    for p in parts:
        for r in recs[p]:
            f, _, _, start, _ = want[r["tensor"]]
            r["offset"] = newabs[r["tensor"]] + (r["offset"] - start)
            r["file"] = NEW
            hsh = hashlib.sha256()
            for i in range(r["count"]):
                fo.seek(r["offset"] + i * r["stride"])
                hsh.update(fo.read(r["length"]))
            if hsh.hexdigest() != r["sha256"]:
                bad += 1
                print("SHA MISMATCH", p, r["tensor"])
        os.makedirs(f"{a.dense_out}/{p}", exist_ok=True)
        with open(f"{a.dense_out}/{p}/dense.json", "w") as w:
            json.dump(recs[p], w, indent=1)
            w.write("\n")
assert bad == 0, f"{bad} records differ"

json.dump({"metadata": {"total_size": off}, "weight_map": {n: NEW for n in names}},
          open(f"{a.base}/model.safetensors.index.json", "w"), indent=2)
for s in SMALL:
    if os.path.isfile(f"{a.ckpt}/{s}"):
        shutil.copyfile(f"{a.ckpt}/{s}", f"{a.base}/{s}")
nrec = sum(len(v) for v in recs.values())
print(f"base: {len(names)} tensors, {off / 1e9:.3f} GB -> {out}; {nrec} records re-pointed, sha256 {nrec}/{nrec} equal")
