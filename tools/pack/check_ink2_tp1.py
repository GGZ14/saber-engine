#!/usr/bin/env python3
"""INK41 stage 1: the independent checker of an Inkling 2 TP=1 VIEW pack (`layout: "tp1-view"`).

Shares no code with build_ink2_tp1.py. It reads the checkpoint through the INK1 checker's own
safetensors walk (`check_inkling2_pack.Ckpt`, by the checkpoint's index), decodes expert records by
the INK6 checker's explicit index walk (`native_raw` / `native_ckpt` / `native_values`) and states the
8-bit law by the INK14 checker's `laws` (scale = the smallest bf16 >= amax / 127 per 32-k block, codes
round-half-even, dequant exact) -- all three written against the TP=2 pack, none of them the view
builder's. Laws:

  (M) the manifest: tp 1 / ep 1 / ranks [0] / 128 experts a rank; `blobs` resolves to a TP=2 pack
      whose manifest sha256, plan sha256, layers, kinds, record layout and geometry are the pinned
      ones; the drafter's and the trunk overlay's manifests are the pinned ones.
  (D) dense: the record set == the TP=2 pack's shared + split names exactly; every record is ONE
      extent; its bytes hash to its sha256 and == the checkpoint tensor WHOLE (the index walk); the
      TP=2 rank halves re-joined == the same bytes.
  (X) experts: per format two parts, part p = the TP=2 rank p file (v2 header: rank p, first expert
      64 p, 64 experts, the layer count, table sha256 == header == the pin == the TP=2 manifest's);
      EVERY record's sha256 == its table entry (3,072 records, ~15.7 GB read); sampled GLOBAL experts
      e (default 0, 63, 64, 127 + 4 more a layer) read THROUGH THE VIEW (part e // 64, slot e % 64) ==
      the checkpoint's expert e byte for byte (codes, scales, globals) and in f32 values bit for bit.
  (T) trunk8: the view set == the overlay's tensors; per tensor the parts re-joined by its axis hash
      to the view's sha256s; src_sha256 == the whole bf16 tensor's sha256 == the dense record's; the
      8-bit law holds on the WHOLE tensor against the checkpoint's bf16 (row chunks).
  CONTROLS (each must FAIL its law, else the checker is blind): the expert parts swapped; expert 64
  read from part 0 (an off-by-one split law); one flipped byte in a record; rank 0's half in place
  of a whole dense record; a whole record one element late; o_proj re-joined on the wrong axis; a
  row-split tensor's parts swapped; one scale +1 bf16 ulp; a wrong blobs manifest pin.

  check_ink2_tp1.py PACK [--sample 8] [--only M,D,X,T,C]
Ends `INK41 TP1 CHECK PASS|FAIL`; exit 0 / 1.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_inkling2_pack as c6  # noqa: E402  (Ckpt + the INK6 record walk)
import check_inkling2_trunk8 as c14  # noqa: E402  (the INK14 8-bit laws)

fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


def control(failed, what):
    """A control: `failed` is True when the law caught the corruption."""
    check(bool(failed), f"CONTROL {what}: {'FIRED' if failed else 'ESCAPED'}")


ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("--sample", type=int, default=8, help="global experts decoded a layer (>= 4: 0, 63, 64, 127 first)")
ap.add_argument("--only", default="M,D,X,T,C", help="stages to run")
a = ap.parse_args()
only = set(a.only.split(","))
pack = os.path.abspath(a.pack)
raw = open(os.path.join(pack, "manifest.json"), "rb").read()
man = json.loads(raw)
blobs = os.path.normpath(os.path.join(pack, man.get("blobs", "")))
braw = open(os.path.join(blobs, "manifest.json"), "rb").read()
bm = json.loads(braw)
basedir = os.path.join(blobs, "base")
ck_dir = basedir if os.path.isfile(os.path.join(basedir, "model.safetensors.index.json")) else bm["checkpoint"]
ck = c6.Ckpt(ck_dir)
# the experts' decode reference: the ORIGINAL checkpoint (a published pack's base/ holds the dense tensors only)
ck_x = ck if ck_dir == bm["checkpoint"] else (
    c6.Ckpt(bm["checkpoint"]) if os.path.isfile(os.path.join(bm["checkpoint"], "model.safetensors.index.json")) else None)
print(f"pack {pack}\nblobs {blobs}\ncheckpoint {ck_dir}\nexpert reference {bm['checkpoint'] if ck_x else 'NONE'}",
      flush=True)

# ---- (M) ----
if "M" in only:
    check(man.get("model") == "inkling2_loop" and man.get("format") == "ink6-pack/1" and man.get("layout") == "tp1-view"
          and (man.get("tp"), man.get("ep"), man.get("ranks"), man.get("experts_per_rank")) == (1, 1, [0], 128),
          "M: a TP=1 / EP=1 view of 128 experts a rank")
    check(hashlib.sha256(braw).hexdigest() == man["blobs_manifest_sha256"] and (bm["tp"], bm["ep"]) == (2, 2),
          "M: blobs = the pinned TP=2 / EP=2 manifest")
    check(all(man[k] == bm[k] for k in ("plan_sha256", "layers", "expert_kinds", "records", "geometry", "checkpoint",
                                        "shard_header_sha256")), "M: plan / layers / kinds / records / geometry == the blobs'")
    if ck_dir == bm["checkpoint"]:
        check(ck.hsha == bm["shard_header_sha256"], "M: the checkpoint's shard headers == the manifest's")
    if man.get("draft"):
        dm = os.path.join(blobs, man["draft"]["dir"], "manifest.json")
        check(hashlib.sha256(open(dm, "rb").read()).hexdigest() == man["draft"]["manifest_sha256"], "M: the drafter manifest pin")
    else:
        check(not os.path.isfile(os.path.join(blobs, "draft", "manifest.json")), "M: no drafter in the blobs (none pinned)")

# ---- (D) ----
dense = json.load(open(os.path.join(pack, "shared", "dense.json")))
byname = {x["tensor"]: x for x in dense}
bshared = json.load(open(os.path.join(blobs, "shared", "dense.json")))
bper = [json.load(open(os.path.join(blobs, f"rank{r}", "dense.json"))) for r in (0, 1)]
cfd = {}


def ext(x, root=ck_dir):
    p = os.path.join(root, x["file"])
    if p not in cfd:
        cfd[p] = os.open(p, os.O_RDONLY)
    return b"".join(os.pread(cfd[p], x["length"], x["offset"] + i * x["stride"]) for i in range(x["count"]))


def rejoin(x0, x1):
    b0, b1 = ext(x0), ext(x1)
    if x0["count"] == 1:
        return b0 + b1
    n0, n1 = x0["length"], x1["length"]
    return b"".join(b0[i * n0:(i + 1) * n0] + b1[i * n1:(i + 1) * n1] for i in range(x0["count"]))


if "D" in only:
    want = {x["tensor"] for x in bshared} | {x["tensor"] for x in bper[0]}
    check(len(byname) == len(dense) and set(byname) == want and {x["tensor"] for x in bper[1]} == {x["tensor"] for x in bper[0]},
          f"D: the record set == the TP=2 shared + split names ({len(dense)})")
    bad_one = bad_sha = bad_whole = bad_join = 0
    for x in dense:
        b = ext(x)
        bad_one += x["count"] != 1
        bad_sha += hashlib.sha256(b).hexdigest() != x["sha256"]
        bad_whole += b != ck.bytes_of(x["tensor"])
    for x0, x1 in zip(*bper):
        bad_join += rejoin(x0, x1) != ck.bytes_of(x0["tensor"])
    check(bad_one == 0, "D: every record one extent")
    check(bad_sha == 0, f"D: {len(dense)} records' bytes == their sha256")
    check(bad_whole == 0, f"D: {len(dense)} records == the checkpoint tensors whole")
    check(bad_join == 0, f"D: {len(bper[0])} TP=2 halves re-joined == the same tensors (the cross law)")

# ---- (X) ----
H, I = man["geometry"]["hidden"], man["geometry"]["inter"]
ptabs = {}


def view_record(kind, slot, e, parts, split=64):
    """The record of GLOBAL expert e of the layer at tier slot `slot`, read through the view."""
    p = parts[e // split]
    path = os.path.join(blobs, p["file"])
    fd = os.open(path, os.O_RDONLY)
    st = man["records"][kind]["stride"]
    b = np.frombuffer(os.pread(fd, st, p["data_offset"] + (slot * p["experts"] + e % split) * st), np.uint8)
    os.close(fd)
    return b


def same_ckpt(buf, kind, L, e):
    rec = man["records"][kind]
    got, want = c6.native_raw(buf, rec, kind, H, I), c6.native_ckpt(ck_x, L, e, kind)
    if not c6.same_raw(got, want):
        return False
    gv, wv = c6.native_values(got, kind, I), c6.native_values(want, kind, I)
    return all(np.array_equal(gv[q].view(np.uint32), wv[q].view(np.uint32)) for q in gv)


if "X" in only or "C" in only:
    for kind, t in man["tiers"]["0"].items():
        st = man["records"][kind]["stride"]
        for pi, p in enumerate(t["parts"]):
            path = os.path.join(blobs, p["file"])
            fd = os.open(path, os.O_RDONLY)
            hdr = os.pread(fd, 4096, 0)
            ver, hsz, rank, nl, first, ne = struct.unpack_from("<IIIIII", hdr, 8)
            toff, tbytes, doff, fbytes = struct.unpack_from("<QQQQ", hdr, 152)
            table = os.pread(fd, tbytes, toff)
            ptabs[(kind, pi)] = (table, doff, ne)
            if "X" in only:
                check(hdr[:8] == b"SBRXPRT1" and ver == 2 and rank == pi == p["src_rank"] and first == 64 * pi == p["first_expert"]
                      and ne == 64 == p["experts"] and nl == len(t["layers"]) and doff == p["data_offset"]
                      and fbytes == os.fstat(fd).st_size == p["bytes"] == doff + nl * ne * st,
                      f"X {kind} part {pi}: v2 header = TP=2 rank {pi}, experts {first}..{first + ne - 1}")
                check(hashlib.sha256(table).digest() == hdr[184:216] and hdr[184:216].hex() == p["table_sha256"]
                      == bm["tiers"][str(pi)][kind]["table_sha256"], f"X {kind} part {pi}: table sha256 == header == pins")
                bad = 0
                for j in range(nl * ne):
                    bad += hashlib.sha256(os.pread(fd, st, doff + j * st)).digest() != table[32 * j:32 * (j + 1)]
                check(bad == 0, f"X {kind} part {pi}: every record's sha256 == its table entry ({nl * ne} records)")
            os.close(fd)
        if "X" in only and ck_x is None:
            print(f"SKIP X {kind}: decode vs the checkpoint (not on this machine; the record sha256s above still bind)")
        elif "X" in only:
            idx = [0, 63, 64, 127] + [int(v) for v in np.linspace(1, 126, max(a.sample - 4, 0)).astype(int)]
            idx = sorted(set(idx))
            bad = n = 0
            for slot, L in enumerate(t["layers"]):
                for e in idx:
                    bad += not same_ckpt(view_record(kind, slot, e, t["parts"]), kind, L, e)
                    n += 1
            check(bad == 0, f"X {kind}: {n} global experts through the view == the checkpoint (bytes + f32 values)")

# ---- (T) ----
tv = json.load(open(os.path.join(pack, man["trunk8"]))) if man.get("trunk8") else None
t8dir = os.path.join(blobs, tv["dir"]) if tv else None
tfd = {}


def t8_part(x, which):
    p = os.path.join(t8dir, x["file"])
    if p not in tfd:
        tfd[p] = os.open(p, os.O_RDONLY)
    if which == "codes":
        return os.pread(tfd[p], x["rows"] * x["cols"], x["codes_off"])
    return os.pread(tfd[p], x["rows"] * x["cols"] // 32 * 2, x["scales_off"])


def t8_whole(v, axis=None, parts=None):
    """(codes int8 [rows, cols], scale bits u16 [rows, cols / 32]) re-joined by `axis`."""
    axis = v["axis"] if axis is None else axis
    parts = v["parts"] if parts is None else parts
    cs = [np.frombuffer(t8_part(x, "codes"), np.int8).reshape(x["rows"], x["cols"]) for x in parts]
    ss = [np.frombuffer(t8_part(x, "scales"), "<u2").reshape(x["rows"], x["cols"] // 32) for x in parts]
    if axis == 0:
        return np.concatenate(cs, 0), np.concatenate(ss, 0)
    return np.concatenate(cs, 1), np.concatenate(ss, 1)


def t8_laws(name, codes, sbits, rows_per=4096):
    """INK14's laws on the whole tensor vs the checkpoint's bf16, in row chunks."""
    w16 = np.frombuffer(ck.bytes_of(name), "<u2").reshape(codes.shape)
    bad = set()
    for r in range(0, codes.shape[0], rows_per):
        w = c14.bf16_to_f64(w16[r:r + rows_per])
        bad |= set(c14.laws(w, codes[r:r + rows_per], sbits[r:r + rows_per]))
    return sorted(bad)


if "T" in only and tv:
    t8m_raw = open(os.path.join(t8dir, "manifest.json"), "rb").read()
    t8m = json.loads(t8m_raw)
    check(hashlib.sha256(t8m_raw).hexdigest() == tv["overlay_manifest_sha256"] and tv["format"] == "ink41-trunk8-view/1",
          "T: the overlay manifest pin")
    check({v["tensor"] for v in tv["tensors"]} == {x["tensor"] for x in t8m["tensors"]}
          and len(t8m["tensors"]) == 2 * len(tv["tensors"]), f"T: the view set == the overlay's ({len(tv['tensors'])} whole)")
    bad_sha = bad_src = bad_law = 0
    for v in tv["tensors"]:
        codes, sbits = t8_whole(v)
        bad_sha += (hashlib.sha256(codes.tobytes()).hexdigest() != v["codes_sha256"]
                    or hashlib.sha256(sbits.tobytes()).hexdigest() != v["scales_sha256"]
                    or codes.shape != (v["rows"], v["cols"]))
        src = hashlib.sha256(ck.bytes_of(v["tensor"])).hexdigest()
        bad_src += src != v["src_sha256"] or src != byname[v["tensor"]]["sha256"]
        e = t8_laws(v["tensor"], codes, sbits)
        if e:
            print(f"  law {v['tensor']}: {e}")
        bad_law += bool(e)
    check(bad_sha == 0, "T: every view's parts re-joined == its codes / scales sha256 + shape")
    check(bad_src == 0, "T: src_sha256 == the whole bf16 tensor's == the dense record's")
    check(bad_law == 0, f"T: the 8-bit law on every WHOLE tensor ({len(tv['tensors'])}) vs the checkpoint")

# ---- (C) controls ----
if "C" in only and ck_x is None:
    print("SKIP C: the expert controls need the original checkpoint")
if "C" in only and ck_x is not None:
    kind = "nvfp4_native"
    t = man["tiers"]["0"][kind]
    L0 = t["layers"][0]
    swapped = list(reversed(t["parts"]))
    control(not same_ckpt(view_record(kind, 0, 0, swapped), kind, L0, 0), "expert parts swapped (expert 0 = rank 1's first)")
    control(not same_ckpt(view_record(kind, 0, 64, t["parts"], split=65), kind, L0, 64), "expert 64 read from part 0 (split law off by one)")
    table, doff, ne = ptabs[(kind, 0)]
    buf = view_record(kind, 0, 5, t["parts"]).copy()
    buf[man["records"][kind]["gate_up_w"][0] + 11] ^= 0x04
    control(hashlib.sha256(buf).digest() != table[32 * 5:32 * 6], "one flipped byte in a record")
    q0 = next(x for x in bper[0] if x["tensor"].endswith("layers.3.self_attn.q_proj.weight"))
    control(ext(q0) != ck.bytes_of(q0["tensor"]), "rank 0's half in place of a whole dense record")
    wq = dict(byname[q0["tensor"]])
    wq["offset"] += 2
    control(hashlib.sha256(ext(wq)).hexdigest() != wq["sha256"], "a whole dense record one element late")
    if tv:
        vo = next(v for v in tv["tensors"] if v["tensor"].endswith("layers.3.self_attn.o_proj.weight"))
        try:
            c, s = t8_whole(vo, axis=0)
            c, s = c.reshape(vo["rows"], vo["cols"]), s.reshape(vo["rows"], vo["cols"] // 32)
            fired = bool(t8_laws(vo["tensor"], c, s))
        except ValueError:
            fired = True
        control(fired, "o_proj re-joined on the wrong axis (rows, not columns)")
        vq = next(v for v in tv["tensors"] if v["tensor"].endswith("layers.3.self_attn.q_proj.weight"))
        c, s = t8_whole(vq, parts=list(reversed(vq["parts"])))
        control(bool(t8_laws(vq["tensor"], c, s)), "q_proj's row halves swapped")
        c, s = t8_whole(vq)
        s = s.copy()
        s[7, 3] += 1
        control(bool(t8_laws(vq["tensor"], c, s)), "one scale +1 bf16 ulp")
    bad_pin = hashlib.sha256(braw + b" ").hexdigest()
    control(bad_pin != man["blobs_manifest_sha256"], "a wrong blobs manifest pin")

print(f"INK41 TP1 CHECK {'PASS' if not fails else 'FAIL'} ({len(fails)} failed)", flush=True)
sys.exit(0 if not fails else 1)
