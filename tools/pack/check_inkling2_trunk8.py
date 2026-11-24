#!/usr/bin/env python3
"""INK14: the independent checker of the 8-bit dense trunk overlay (`ink14-trunk8/1`).

Shares no code with build_inkling2_pack.py: its own safetensors header walk, its own reading of the
base pack's extents, its own bf16 decode (bit arithmetic on u32) and its own statement of the laws.
Per (tensor, rank) record of the overlay:
  (A) binding: the base pack's rank record exists, its extent bytes hash to the record's sha256 and
      to the overlay's `src_sha256`; codes / scales bytes hash to the manifest's sha256s; the record
      shape is the base extent's (rows x cols), cols % 256 == 0.
  (B) the scale law: per 32-k block, scale == the SMALLEST bf16 >= amax / 127 (scale x 127 >= amax and
      the previous bf16 x 127 < amax; scale 0 iff amax 0), computed in f64 from the bf16 source.
  (C) the code law: |code| <= 127 and code = round-half-even(w / scale): |w - code s| <= s / 2, and
      at |w - code s| == s / 2 the code is even.
  (D) the dequant: code x scale is exact in f32 (checked: the f32 product == the f64 product).
CONTROLS (each must FAIL a law, else the checker is blind): one code +1, two rows' codes swapped,
one scale +1 bf16 ulp, the neighbouring tensor's record in this slot, rank 1's record against rank
0's source.
Also reports relRMS of the dequant vs the bf16 weight (per tensor + summary).

  check_inkling2_trunk8.py --overlay /mnt/nvme/saber-packs/<name> [--sample N]
Ends `INK14 TRUNK8 CHECK PASS|FAIL`.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np


def st_headers(ckpt):
    out = {}
    for f in sorted(os.listdir(ckpt)):
        if not f.endswith(".safetensors"):
            continue
        with open(os.path.join(ckpt, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = (f, v["dtype"], v["shape"])
    return out


def bf16_to_f64(u):
    u = u.astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def prev_bf16(bits):
    return np.where(bits > 0, bits - 1, 0).astype(np.uint16)


def laws(w, codes, sbits):
    """w f64 [r, c]; codes int8 [r, c]; sbits u16 [r, c/32] -> list of failed law names."""
    r, c = w.shape
    bad = []
    b = w.reshape(r, c // 32, 32)
    amax = np.abs(b).max(axis=2)
    s = bf16_to_f64(sbits)
    zero = amax == 0
    if np.any(zero != (s == 0)):
        bad.append("scale0")
    nz = ~zero
    if np.any(s[nz] * 127.0 < amax[nz]):
        bad.append("scale_low")
    ps = bf16_to_f64(prev_bf16(sbits))
    if np.any((ps[nz] * 127.0 >= amax[nz]) & (sbits[nz] > 0)):
        bad.append("scale_not_smallest")
    q = codes.astype(np.int64).reshape(r, c // 32, 32)
    if np.abs(q).max() > 127:
        bad.append("code_range")
    d = np.abs(b - q * s[..., None])
    half = s[..., None] / 2.0
    if np.any(d > half):
        bad.append("code_not_nearest")
    tie = (d == half) & (s[..., None] > 0)
    if np.any(tie & (q % 2 != 0)):
        bad.append("code_tie_odd")
    f32 = q.astype(np.float32) * s[..., None].astype(np.float32)
    if not np.array_equal(f32.astype(np.float64), q * s[..., None]):
        bad.append("dequant_inexact")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--sample", type=int, default=0, help="check every Nth record only (0 = all)")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.overlay, "manifest.json")))
    ok = man.get("format") == "ink14-trunk8/1"
    base = man["base_pack"]
    bman_sha = hashlib.sha256(open(os.path.join(base, "manifest.json"), "rb").read()).hexdigest()
    if bman_sha != man["base_manifest_sha256"]:
        print("CHK base manifest sha256 differs", flush=True)
        ok = False
    ckpt = json.load(open(os.path.join(base, "manifest.json")))["checkpoint"]
    hdr = st_headers(ckpt)
    hsize = {}
    for f in {v[0] for v in hdr.values()}:
        with open(os.path.join(ckpt, f), "rb") as fh:
            hsize[f] = 8 + struct.unpack("<Q", fh.read(8))[0]
    dense = {r: {d["tensor"]: d for d in json.load(open(os.path.join(base, f"rank{r}", "dense.json")))}
             for r in (0, 1)}
    fds = {}

    def src(tensor, rank):
        d = dense[rank][tensor]
        f = d["file"]
        if f not in fds:
            fds[f] = os.open(os.path.join(ckpt, f), os.O_RDONLY)
        bb = b"".join(os.pread(fds[f], d["length"], d["offset"] + i * d["stride"]) for i in range(d["count"]))
        return d, bb

    ov = {}

    def rec_bytes(t):
        p = os.path.join(a.overlay, t["file"])
        if p not in ov:
            ov[p] = os.open(p, os.O_RDONLY)
        cb = os.pread(ov[p], t["rows"] * t["cols"], t["codes_off"])
        sb = os.pread(ov[p], t["rows"] * t["cols"] // 32 * 2, t["scales_off"])
        return cb, sb

    skip = man.get("skipped_layers", [])
    want = {(d, r) for r in (0, 1) for d in dense[r]
            if d == "language_model.lm_head.weight" or (".self_attn." in d and d.endswith("_proj.weight")
                                                        and not any(f".layers.{L}." in d for L in skip))}
    extra = sorted(x for x in {(t["tensor"], t["rank"]) for t in man["tensors"]} - want)
    if extra or bool(skip) != bool(man.get("partial", False)):
        print(f"CHK records outside the layer policy {extra[:4]} / partial flag {man.get('partial')}", flush=True)
        ok = False
    have = {(t["tensor"], t["rank"]) for t in man["tensors"]}
    missing = sorted(want - have)
    if missing:
        print(f"CHK missing records {missing[:6]} ({len(missing)})", flush=True)
        ok = False
    n_rec = n_ok = 0
    rel = []
    recs = man["tensors"]
    for i, t in enumerate(recs):
        if a.sample and i % a.sample:
            continue
        n_rec += 1
        d, bb = src(t["tensor"], t["rank"])
        e = []
        if hashlib.sha256(bb).hexdigest() != d["sha256"] or d["sha256"] != t["src_sha256"]:
            e.append("binding")
        cb, sb = rec_bytes(t)
        if hashlib.sha256(cb).hexdigest() != t["codes_sha256"] or hashlib.sha256(sb).hexdigest() != t["scales_sha256"]:
            e.append("record_sha")
        rows, cols = t["rows"], t["cols"]
        if rows * cols * 2 != len(bb) or cols % 256:
            e.append("shape")
        else:
            w = bf16_to_f64(np.frombuffer(bb, "<u2")).reshape(rows, cols)
            codes = np.frombuffer(cb, np.int8).reshape(rows, cols)
            sbits = np.frombuffer(sb, "<u2").reshape(rows, cols // 32)
            e += laws(w, codes, sbits)
            dq = codes.astype(np.float64) * bf16_to_f64(sbits).repeat(32, axis=1)
            rel.append(float(np.sqrt(((dq - w) ** 2).sum() / max((w ** 2).sum(), 1e-30))))
        if e:
            print(f"CHK FAIL {t['tensor']} rank {t['rank']}: {e}", flush=True)
            ok = False
        else:
            n_ok += 1
    print(f"CHK records {n_ok}/{n_rec} pass (of {len(recs)}); relRMS mean {np.mean(rel):.5f} max {np.max(rel):.5f}",
          flush=True)
    # controls on a mid-size record (a q_proj) and its neighbour
    qi = next(i for i, t in enumerate(recs) if t["tensor"].endswith("layers.3.self_attn.q_proj.weight") and t["rank"] == 0)
    t = recs[qi]
    d, bb = src(t["tensor"], 0)
    w = bf16_to_f64(np.frombuffer(bb, "<u2")).reshape(t["rows"], t["cols"])
    cb, sb = rec_bytes(t)
    codes = np.frombuffer(cb, np.int8).reshape(t["rows"], t["cols"]).copy()
    sbits = np.frombuffer(sb, "<u2").reshape(t["rows"], t["cols"] // 32).copy()
    ctl = {}
    c1 = codes.copy()
    c1[3, 17] = np.int8(c1[3, 17] + 1 if c1[3, 17] < 127 else c1[3, 17] - 1)
    ctl["code+1"] = laws(w, c1, sbits)
    c2 = codes.copy()
    c2[[5, 6]] = c2[[6, 5]]
    s2 = sbits.copy()
    ctl["rows_swapped"] = laws(w, c2, s2)
    s3 = sbits.copy()
    s3[2, 4] += 1
    ctl["scale+1ulp"] = laws(w, codes, s3)
    nb = next(x for x in recs if x["tensor"].endswith("layers.4.self_attn.q_proj.weight") and x["rank"] == 0)
    ncb, nsb = rec_bytes(nb)
    ctl["neighbour"] = laws(w, np.frombuffer(ncb, np.int8).reshape(w.shape),
                            np.frombuffer(nsb, "<u2").reshape(sbits.shape))
    r1 = next(x for x in recs if x["tensor"] == t["tensor"] and x["rank"] == 1)
    rcb, rsb = rec_bytes(r1)
    ctl["rank_swap"] = laws(w, np.frombuffer(rcb, np.int8).reshape(w.shape),
                            np.frombuffer(rsb, "<u2").reshape(sbits.shape))
    for k, v in ctl.items():
        fired = bool(v)
        print(f"CHK control {k}: {'FIRED' if fired else 'ESCAPED'} {v}", flush=True)
        ok &= fired
    print(f"INK14 TRUNK8 CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
