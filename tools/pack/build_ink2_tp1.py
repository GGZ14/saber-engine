#!/usr/bin/env python3
"""INK41 stage 1: the ONE-CARD (TP=1) layout of an Inkling 2 pack, as a VIEW over the TP=2 pack's bytes.

The TP=2 / EP=2 pack (`ink6-pack/1`) already holds every byte a one-card boot reads: the experts are
WHOLE records (EP: rank r's tier files carry experts 64 r .. 64 r + 63 of every layer, nothing of an
expert is split), the dense tensors are whole in the checkpoint (or the published pack's
`base/dense.safetensors`), the 8-bit trunk's rank halves re-join exactly (the int8 law is per 32-k
block of a row, and every split falls on a block edge), the drafter is shared. So a TP=1 pack is an
INDEX, not new bytes:

  <out>/manifest.json      `ink6-pack/1`, tp 1 / ep 1 / ranks [0], `layout: "tp1-view"`, `blobs`
                           (the TP=2 pack root, RELATIVE to <out>), the TP=2 manifest's sha256, its
                           plan sha256 (the plan the records were built by), experts_per_rank 128,
                           and per format a tier of two PARTS (the TP=2 rank files, first expert
                           0 / 64, each part's v2 header fields + table sha256 pinned)
  <out>/shared/dense.json  every dense tensor WHOLE (count 1): the TP=2 `shared/` records as they are
                           + one whole-tensor extent for each of the 273 split tensors, sha256 of the
                           whole bytes; file names relative to the checkpoint the engine resolves
                           (`<blobs>/base` when present, else the recorded checkpoint: HF3's rule)
  <out>/trunk8.json        `ink41-trunk8-view/1`: every 8-bit trunk tensor WHOLE = its two rank
                           records re-joined, axis 0 (row halves: q/gate/r/sink/k/v proj, the head)
                           or axis 1 (column halves, row by row: o_proj), sha256 of the whole codes /
                           scales, src_sha256 = the whole dense record's sha256

The builder only INDEXES (it reads headers and hashes bytes; it never quantizes or decodes);
`check_ink2_tp1.py` re-derives every view from the checkpoint by its own path (+ corruption controls).

    build_ink2_tp1.py --blobs /mnt/nvme/models/ink2-pack-nv --out /mnt/nvme/models/ink2-pack-nv-tp1
    build_ink2_tp1.py --blobs <published pack> --out <published pack>/tp1          # blobs ".."
"""
import argparse
import hashlib
import json
import os
import struct
import time

ap = argparse.ArgumentParser()
ap.add_argument("--blobs", required=True, help="the TP=2 pack (ink6-pack/1) whose bytes the view reads")
ap.add_argument("--out", required=True, help="the TP=1 pack directory (manifest + index files only)")
a = ap.parse_args()
t0 = time.time()
blobs = os.path.abspath(a.blobs)
out = os.path.abspath(a.out)
rel = os.path.relpath(blobs, out)

bm_raw = open(os.path.join(blobs, "manifest.json"), "rb").read()
bm = json.loads(bm_raw)
assert bm["model"] == "inkling2_loop" and bm["format"] == "ink6-pack/1", "not a native Inkling 2 pack"
assert (bm["tp"], bm["ep"], bm["ranks"]) == (2, 2, [0, 1]), "the view is built over a TP=2 / EP=2 pack"
E2 = bm["experts_per_rank"]
assert E2 == 64

# the checkpoint the engine resolves for these blobs (saber_cli::inkling2::resolve_ckpt's rule)
base = os.path.join(blobs, "base")
if os.path.isfile(os.path.join(base, "model.safetensors.index.json")) and os.path.isfile(os.path.join(base, "config.json")):
    ckpt, ckpt_from = base, "base"
else:
    ckpt, ckpt_from = bm["checkpoint"], "recorded"


def st_header(f, cache={}):
    if f not in cache:
        with open(os.path.join(ckpt, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            cache[f] = (8 + n, json.loads(fh.read(n)))
    return cache[f]


def sha_extent(f, off, length, chunk=64 << 20):
    h = hashlib.sha256()
    with open(os.path.join(ckpt, f), "rb") as fh:
        fh.seek(off)
        left = length
        while left:
            b = fh.read(min(left, chunk))
            assert b, f"{f}: short read"
            h.update(b)
            left -= len(b)
    return h.hexdigest()


# ---- dense: every tensor whole ----
shared = json.load(open(os.path.join(blobs, "shared", "dense.json")))
per = [json.load(open(os.path.join(blobs, f"rank{r}", "dense.json"))) for r in (0, 1)]
assert [x["tensor"] for x in per[0]] == [x["tensor"] for x in per[1]]
dense = []
for x in shared:
    d0, h = st_header(x["file"])
    s, e = h[x["tensor"]]["data_offsets"]
    assert x["count"] == 1 and x["offset"] == d0 + s and x["length"] == e - s, f"{x['tensor']}: a shared record not whole"
    dense.append(dict(x))
split_axis = {}
for x0, x1 in zip(*per):
    name = x0["tensor"]
    assert x0["file"] == x1["file"], f"{name}: halves in two files"
    d0, h = st_header(x0["file"])
    t = h[name]
    s, e = t["data_offsets"]
    start, nb = d0 + s, e - s
    span = lambda x: x["stride"] * (x["count"] - 1) + x["length"]  # noqa: E731
    assert start <= x0["offset"] and x1["offset"] + span(x1) <= start + nb, f"{name}: a half outside its tensor"
    if x0["count"] == 1:  # row halves: rank 1's starts where rank 0's ends, together the tensor
        assert x0["offset"] == start and x0["offset"] + x0["length"] == x1["offset"] and x0["length"] + x1["length"] == nb
        split_axis[name] = 0
    else:  # column halves: per row, rank 0's piece then rank 1's
        assert x0["offset"] == start and x1["offset"] == start + x0["length"] and x0["stride"] == x0["length"] + x1["length"]
        assert x0["count"] * x0["stride"] == nb
        split_axis[name] = 1
    dense.append({"file": x0["file"], "offset": start, "length": nb, "stride": nb, "count": 1,
                  "sha256": sha_extent(x0["file"], start, nb), "tensor": name, "dtype": t["dtype"]})
assert len({x["tensor"] for x in dense}) == len(dense)
whole_sha = {x["tensor"]: x["sha256"] for x in dense}

# ---- experts: one tier a format, two parts (the TP=2 rank files), 128 experts a layer ----
tiers = {}
for kind in bm["tiers"]["0"]:
    t0_, t1_ = bm["tiers"]["0"][kind], bm["tiers"]["1"][kind]
    assert t0_["layers"] == t1_["layers"] and t0_["tag"] == t1_["tag"]
    parts = []
    for r, t in ((0, t0_), (1, t1_)):
        p = os.path.join(blobs, t["file"])
        hdr = open(p, "rb").read(4096)
        ver, hsz, rank, nl, first, ne = struct.unpack_from("<IIIIII", hdr, 8)
        stride = struct.unpack_from("<Q", hdr, 32)[0]
        toff, tbytes, doff, fbytes = struct.unpack_from("<QQQQ", hdr, 152)
        assert hdr[:8] == b"SBRXPRT1" and ver == 2 and rank == r and first == 64 * r and ne == 64
        assert nl == len(t["layers"]) and stride == bm["records"][kind]["stride"] and fbytes == os.path.getsize(p)
        assert hdr[184:216].hex() == t["table_sha256"], f"{t['file']}: header table sha != the TP=2 manifest's"
        parts.append({"file": t["file"], "src_rank": r, "first_expert": first, "experts": ne,
                      "data_offset": doff, "bytes": fbytes, "table_sha256": t["table_sha256"]})
    tiers[kind] = {"parts": parts, "layers": t0_["layers"], "format": t0_["format"], "tag": t0_["tag"],
                   "records": t0_["records"] + t1_["records"], "bytes": t0_["bytes"] + t1_["bytes"]}

# ---- the 8-bit trunk: every tensor whole (two rank records re-joined) ----
t8dir = os.path.join(blobs, "trunk8")
trunk8 = None
if os.path.isfile(os.path.join(t8dir, "manifest.json")):
    t8raw = open(os.path.join(t8dir, "manifest.json"), "rb").read()
    t8 = json.loads(t8raw)
    assert t8["format"] == "ink14-trunk8/1"
    by = {}
    for x in t8["tensors"]:
        by.setdefault(x["tensor"], {})[x["rank"]] = x
    fds = {}

    def rd(x, which):
        p = os.path.join(t8dir, x["file"])
        if p not in fds:
            fds[p] = os.open(p, os.O_RDONLY)
        if which == "codes":
            return os.pread(fds[p], x["rows"] * x["cols"], x["codes_off"])
        return os.pread(fds[p], x["rows"] * x["cols"] // 32 * 2, x["scales_off"])

    views = []
    for name in sorted(by):
        r0, r1 = by[name][0], by[name][1]
        ax = split_axis[name]
        h = {}
        for which, w in (("codes", 1), ("scales", 2)):
            b0, b1 = rd(r0, which), rd(r1, which)
            assert hashlib.sha256(b0).hexdigest() == r0[f"{which}_sha256"]
            assert hashlib.sha256(b1).hexdigest() == r1[f"{which}_sha256"]
            if ax == 0:
                assert r0["cols"] == r1["cols"]
                whole = b0 + b1
            else:
                assert r0["rows"] == r1["rows"] and r0["cols"] % 32 == 0
                n0, n1 = r0["cols"] * w // (1 if which == "codes" else 32), r1["cols"] * w // (1 if which == "codes" else 32)
                whole = b"".join(b0[i * n0:(i + 1) * n0] + b1[i * n1:(i + 1) * n1] for i in range(r0["rows"]))
            h[which] = hashlib.sha256(whole).hexdigest()
        rows = r0["rows"] + r1["rows"] if ax == 0 else r0["rows"]
        cols = r0["cols"] if ax == 0 else r0["cols"] + r1["cols"]
        views.append({"tensor": name, "rows": rows, "cols": cols, "axis": ax,
                      "codes_sha256": h["codes"], "scales_sha256": h["scales"], "src_sha256": whole_sha[name],
                      "parts": [{k: x[k] for k in ("file", "rows", "cols", "codes_off", "scales_off")} for x in (r0, r1)]})
    for fd in fds.values():
        os.close(fd)
    trunk8 = {"format": "ink41-trunk8-view/1", "dir": "trunk8",
              "overlay_manifest_sha256": hashlib.sha256(t8raw).hexdigest(), "quant": t8["quant"],
              "layers_policy": t8["layers_policy"], "skipped_layers": t8["skipped_layers"], "partial": t8["partial"],
              "tensors": views}

draft = None
dm = os.path.join(blobs, "draft", "manifest.json")
if os.path.isfile(dm):
    draft = {"dir": "draft", "manifest_sha256": hashlib.sha256(open(dm, "rb").read()).hexdigest()}

man = {k: bm[k] for k in ("model", "format", "form", "checkpoint", "shard_header_sha256", "layers",
                          "expert_kinds", "records", "geometry", "scale_byte_range")}
man.update({
    "tp": 1, "ep": 1, "ranks": [0], "experts_per_rank": 128,
    "layout": "tp1-view",
    "blobs": rel,
    "blobs_manifest_sha256": hashlib.sha256(bm_raw).hexdigest(),
    "plan_sha256": bm["plan_sha256"],
    "plan_note": "the TP=2 plan the records were built by (the view re-joins its halves)",
    "dense_ckpt": ckpt_from,
    "tiers": {"0": tiers},
    "dense": {"shared": len(dense)},
    "trunk8": "trunk8.json" if trunk8 else None,
    "draft": draft,
    "built": time.strftime("%Y-%m-%d %H:%M:%S"),
})
os.makedirs(os.path.join(out, "shared"), exist_ok=True)


def dump(path, obj):
    tmp = path + ".part"
    with open(tmp, "w") as w:
        json.dump(obj, w, indent=1)
        w.write("\n")
    os.replace(tmp, path)


dump(os.path.join(out, "shared", "dense.json"), dense)
if trunk8:
    dump(os.path.join(out, "trunk8.json"), trunk8)
man["seconds"] = round(time.time() - t0, 1)
dump(os.path.join(out, "manifest.json"), man)
print(f"INK41 tp1 view: {out} -> blobs {rel} ({ckpt_from} checkpoint); dense {len(dense)} whole "
      f"({len(split_axis)} re-joined: {sum(1 for v in split_axis.values() if v == 0)} row / "
      f"{sum(1 for v in split_axis.values() if v == 1)} column halves); experts "
      f"{', '.join(f'{k} {v['records']} records' for k, v in tiers.items())}; trunk8 "
      f"{len(trunk8['tensors']) if trunk8 else 0} whole; {man['seconds']} s")
