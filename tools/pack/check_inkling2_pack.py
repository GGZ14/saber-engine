#!/usr/bin/env python3
"""INK1 stage 3: check an Inkling 2 Mini pack against its checkpoint (docs/INKLING2.md §7).

Host tool, numpy only. Every check reads the PACK bytes and re-derives them from the CHECKPOINT by a
second path:

1. header: SBRXPRT1 v1, rank / layers / first expert / experts / stride / payload / the six
   sub-record pairs / hidden / inter / geometry == the manifest's record; the table's sha256 ==
   the header's; every record's sha256 == its table entry.
2. experts (all, or `--sample N` a layer): the checkpoint dequantized by THIS script's own decoders
   (E4M3 and e2m1 decoded by bit arithmetic, not build_pack's tables), requantized by the plan's
   rule (`build_pack.quantize_mxfp4`, the definition of the record), and the RECORD read backwards
   (`unpermute_w`, nibbles low first, ws transposed back, wref) -- codes, scales and wref equal
   byte for byte.
3. dense: every extent re-read == its sha256; the two ranks' halves of every split record re-joined
   == the checkpoint tensor's bytes; a replicated record == the tensor whole.
4. controls (must FAIL): one flipped byte in a record vs its table sha; the nibble-swapped record vs
   the codes; the neighbouring expert's checkpoint bytes vs the record; rank 1's half in rank 0's slot.

INK6, a NATIVE pack (manifest `ink6-pack/1`): per rank and format the v2 header (version 2, tag 2
NVFP4_E4M3G16 / 3 MXFP8_E8M0G32) + geometry, table and record sha256s; the sampled records decoded by
THIS script's own reading (the record's bytes put back in checkpoint order by an explicit index walk
-- NOT `unpermute_w`: the 16 x 16 fragment tile of e2m1 pairs, `ws` transposed back, the f32 globals)
== the checkpoint's bytes EXACTLY (codes, scales, `weight_scale_2`), and the decoded f32 values ==
this script's arithmetic dequant of the checkpoint bit for bit; controls (must FAIL): a flipped byte
vs the sha, the nibble / byte-swapped record, the neighbouring expert, a scale byte moved, rank 1's
record in rank 0's slot.

  check_inkling2_pack.py PACK [--sample 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pack as bp  # noqa: E402  (the requant rule only)
import flashnext_mx as mx  # noqa: E402  (unpermute_w only)

F32 = np.float32
fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


def e4m3_bits(b):
    """OCP e4m3 by arithmetic (bias 7, subnormals, 0x7F/0xFF NaN)."""
    b = b.astype(np.int32)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * np.exp2(e.astype(np.float64) - 7))
    v = np.where((e == 15) & ((b & 7) == 7), np.nan, v)
    return (s * v).astype(F32)


def e2m1_bits(c):
    c = c.astype(np.int32)
    mag = c & 7
    e = mag >> 1
    m = mag & 1
    v = np.where(e == 0, m * 0.5, (1 + m * 0.5) * np.exp2(e.astype(np.float64) - 1))
    return np.where(c & 8, -v, v).astype(F32)


class Ckpt:
    NP = {"BF16": np.uint16, "F32": np.float32, "U32": np.uint32, "U8": np.uint8}
    ES = {"BF16": 2, "F32": 4, "U32": 4, "U8": 1}

    def __init__(self, d):
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        self.t, self.fds, self.hsha = {}, {}, {}
        for f in sorted(set(idx["weight_map"].values())):
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                raw = fh.read(n)
            self.hsha[f] = hashlib.sha256(raw).hexdigest()
            self.fds[f] = os.open(p, os.O_RDONLY)
            for k, v in json.loads(raw).items():
                if k != "__metadata__":
                    self.t[k] = (f, 8 + n + v["data_offsets"][0], v["dtype"], tuple(v["shape"]))

    def bytes_of(self, name):
        f, off, dt, shape = self.t[name]
        return os.pread(self.fds[f], int(np.prod(shape)) * self.ES[dt], off)

    def read(self, name, index):
        f, off, dt, shape = self.t[name]
        inner = int(np.prod(shape[1:]))
        es = self.ES[dt]
        b = os.pread(self.fds[f], inner * es, off + index * inner * es)
        return np.frombuffer(b, self.NP[dt]).reshape(shape[1:])


def dequant(ck, L, e, proj, kind):
    base = f"language_model.model.layers.{L}.mlp.switch_mlp.{proj}"
    w = ck.read(base + ".weight", e).view(np.uint8)
    s = ck.read(base + ".scales", e)
    n = w.shape[0]
    if kind == "nvfp4_requant_mxfp4":
        gs = np.frombuffer(ck.bytes_of(base + ".weight_scale_2"), np.float32)[e]
        codes = np.stack([w & 0xF, w >> 4], -1).reshape(n, -1)
        return ((e2m1_bits(codes).reshape(n, -1, 16) * e4m3_bits(s)[..., None]).reshape(n, -1) * F32(gs)).astype(F32)
    sc = np.exp2(s.astype(np.float64) - 127).astype(F32)
    return (e4m3_bits(w).reshape(n, -1, 32) * sc[..., None]).reshape(n, -1).astype(F32)


def read_record(buf, rec, H, I):
    def sub(key):
        off, nb = rec[key]
        return buf[off:off + nb]
    out = {}
    for pre, n, k in (("gate_up", 2 * I, H), ("down", H, I)):
        packed = mx.unpermute_w(sub(pre + "_w").reshape(n, k // 2), n, k)
        out[pre] = (np.stack([packed & 0xF, packed >> 4], -1).reshape(n, k),
                    sub(pre + "_ws").reshape(k // 32, n).T, sub(pre + "_wref"))
    return out


def expected(ck, L, e, kind, H, I):
    g, u, d = (dequant(ck, L, e, p, kind) for p in ("gate_proj", "up_proj", "down_proj"))
    return {"gate_up": bp.quantize_mxfp4(np.concatenate([g, u])), "down": bp.quantize_mxfp4(d)}


def native_raw(buf, rec, kind, H, I):
    """The record's bytes put back in checkpoint order by an explicit index walk:
    {pre: (w bytes [N, K/2] or [N, K], scale bytes [N, K/g], globals)}."""
    out = {}
    for pre, n, k in (("gate_up", 2 * I, H), ("down", H, I)):
        def sub(key):
            off, nb = rec[pre + key]
            return buf[off:off + nb]
        if kind == "nvfp4_native":
            pw = sub("_w")
            # permuted byte index of (row n, byte j = codes 2j, 2j+1 at k = 2j): the 16-row tile nt,
            # the 16-k step k/16, the half (k/8)&1, the row in the tile, the byte in the 8-code word
            nn = np.arange(n)[:, None]
            jj = np.arange(k // 2)[None, :]
            kk = 2 * jj
            idx = ((((nn // 16) * (k // 16) + kk // 16) * 2 + (kk // 8) % 2) * 16 + nn % 16) * 4 + (kk % 8) // 2
            out[pre] = (pw[idx], sub("_ws").reshape(k // 16, n).T, sub("_wref").view(np.float32))
        else:
            out[pre] = (sub("_w").reshape(n, k), sub("_ws").reshape(k // 32, n).T, None)
    return out


def native_ckpt(ck, L, e, kind):
    base = f"language_model.model.layers.{L}.mlp.switch_mlp."
    w = {p: ck.read(base + p + ".weight", e).view(np.uint8) for p in ("gate_proj", "up_proj", "down_proj")}
    s = {p: ck.read(base + p + ".scales", e) for p in ("gate_proj", "up_proj", "down_proj")}
    g = None
    if kind == "nvfp4_native":
        g = {p: np.frombuffer(ck.bytes_of(base + p + ".weight_scale_2"), np.float32)[e]
             for p in ("gate_proj", "up_proj", "down_proj")}
    return {"gate_up": (np.concatenate([w["gate_proj"], w["up_proj"]]), np.concatenate([s["gate_proj"], s["up_proj"]]),
                        None if g is None else np.array([g["gate_proj"], g["up_proj"]], np.float32)),
            "down": (w["down_proj"], s["down_proj"], None if g is None else np.array([g["down_proj"]], np.float32))}


def native_values(raw, kind, I):
    """f32 values of {pre: (w, s, g)} by this script's arithmetic decoders."""
    out = {}
    for pre, (w, s, g) in raw.items():
        n = w.shape[0]
        if kind == "nvfp4_native":
            codes = np.stack([w & 0xF, w >> 4], -1).reshape(n, -1)
            gr = np.where(np.arange(n) < (I if pre == "gate_up" else n), g[0], g[-1]).astype(F32)
            out[pre] = ((e2m1_bits(codes).reshape(n, -1, 16) * e4m3_bits(s)[..., None]).reshape(n, -1)
                        * gr[:, None]).astype(F32)
        else:
            sc = np.exp2(s.astype(np.float64) - 127).astype(F32)
            out[pre] = (e4m3_bits(w).reshape(n, -1, 32) * sc[..., None]).reshape(n, -1).astype(F32)
    return out


def same_raw(a, b):
    return all(np.array_equal(a[p][0], b[p][0]) and np.array_equal(a[p][1], b[p][1])
               and (a[p][2] is None or np.array_equal(a[p][2].view(np.uint32), b[p][2].view(np.uint32)))
               for p in ("gate_up", "down"))


def check_native(a, man, ck):
    H, I, E = man["geometry"]["hidden"], man["geometry"]["inter"], man["experts_per_rank"]
    tags = {"nvfp4_native": 2, "mxfp8_native": 3}
    for r in man["ranks"]:
        for kind, t in man["tiers"][str(r)].items():
            rec = man["records"][kind]
            path = os.path.join(a.pack, t["file"])
            fd = os.open(path, os.O_RDONLY)
            hdr = os.pread(fd, 4096, 0)
            ver, hsz, rank, nl, first, ne = struct.unpack_from("<IIIIII", hdr, 8)
            (tag,) = struct.unpack_from("<I", hdr, 256)
            stride, payload = struct.unpack_from("<QQ", hdr, 32)
            pairs = [struct.unpack_from("<QQ", hdr, 48 + 16 * i) for i in range(6)]
            hid, inter = struct.unpack_from("<II", hdr, 144)
            toff, tbytes, doff, fbytes = struct.unpack_from("<QQQQ", hdr, 152)
            want_pairs = [tuple(rec[k]) for k in ("gate_up_w", "gate_up_ws", "gate_up_wref", "down_w", "down_ws", "down_wref")]
            layers = t["layers"]
            check(hdr[:8] == b"SBRXPRT1" and ver == 2 and tag == tags[kind] == t["tag"] and hsz == 4096 and rank == r
                  and nl == len(layers) and ne == E and first == E * r and stride == rec["stride"]
                  and payload == rec["payload"] and pairs == want_pairs and (hid, inter) == (H, I)
                  and fbytes == os.fstat(fd).st_size == doff + nl * ne * stride, f"rank {r} {kind}: v2 header fields")
            table = os.pread(fd, tbytes, toff)
            check(hashlib.sha256(table).digest() == hdr[184:216], f"rank {r} {kind}: table sha256 == header")
            bad_sha = bad_raw = bad_val = checked = 0
            idxs = list(range(E)) if not a.sample else sorted(set(np.linspace(0, E - 1, a.sample).astype(int).tolist()))
            for slot, L in enumerate(layers):
                check(man["expert_kinds"][str(L)] == kind, f"rank {r} layer {L}: kind {kind}") if slot == 0 else None
                for i in range(E):
                    j = slot * E + i
                    buf = np.frombuffer(os.pread(fd, stride, doff + j * stride), np.uint8)
                    bad_sha += hashlib.sha256(buf).digest() != table[32 * j:32 * (j + 1)]
                    if i not in idxs:
                        continue
                    got = native_raw(buf, rec, kind, H, I)
                    want = native_ckpt(ck, L, E * r + i, kind)
                    bad_raw += not same_raw(got, want)
                    gv, wv = native_values(got, kind, I), native_values(want, kind, I)
                    bad_val += not all(np.array_equal(gv[p].view(np.uint32), wv[p].view(np.uint32)) for p in gv)
                    checked += 1
                    if slot == 0 and i == idxs[0]:
                        fl = buf.copy()
                        fl[rec["gate_up_w"][0] + 7] ^= 0x10
                        check(hashlib.sha256(fl).digest() != table[32 * j:32 * (j + 1)],
                              f"rank {r} {kind}: CONTROL flipped byte changes the sha256")
                        sw = buf.copy()
                        o, nb = rec["gate_up_w"]
                        sw[o:o + nb] = (mx.nibble_swap_control(buf[o:o + nb]) if kind == "nvfp4_native"
                                        else buf[o:o + nb].reshape(-1, 4)[:, ::-1].reshape(-1))
                        check(not same_raw(native_raw(sw, rec, kind, H, I), want), f"rank {r} {kind}: CONTROL swapped codes")
                        so, sn = rec["gate_up_ws"]
                        sc = buf.copy()
                        sc[so + 3] ^= 0x01
                        check(not same_raw(native_raw(sc, rec, kind, H, I), want), f"rank {r} {kind}: CONTROL a scale byte moved")
                        check(not same_raw(got, native_ckpt(ck, L, E * r + (i + 1) % E, kind)),
                              f"rank {r} {kind}: CONTROL the neighbouring expert")
                        other = E * (1 - r) + i
                        check(not same_raw(got, native_ckpt(ck, L, other, kind)),
                              f"rank {r} {kind}: CONTROL rank {1 - r}'s expert {other} in this slot")
            os.close(fd)
            check(bad_sha == 0, f"rank {r} {kind}: every record's sha256 == its table entry ({len(layers) * E} records)")
            check(bad_raw == 0 and bad_val == 0,
                  f"rank {r} {kind}: {checked} experts == the checkpoint (bytes exact, f32 values bit for bit)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--sample", type=int, default=0, help="experts checked per layer per rank (0 = all)")
    a = ap.parse_args(argv)
    man = json.load(open(os.path.join(a.pack, "manifest.json")))
    ck = Ckpt(man["checkpoint"])
    check(ck.hsha == man["shard_header_sha256"], "the checkpoint's shard headers == the manifest's")
    if man.get("format") == "ink6-pack/1":
        check_native(a, man, ck)
        check_dense(a, man, ck)
        print("ALL CHECKS PASS" if not fails else f"{len(fails)} CHECKS FAILED")
        return 0 if not fails else 1
    rec = man["record"]
    H, I, E = man["geometry"]["hidden"], man["geometry"]["inter"], man["experts_per_rank"]
    layers = man["layers"]
    for r in man["ranks"]:
        path = os.path.join(a.pack, f"rank{r}", "experts.xtier")
        fd = os.open(path, os.O_RDONLY)
        hdr = os.pread(fd, 4096, 0)
        magic = hdr[:8]
        ver, hsz, rank, nl, first, ne = struct.unpack_from("<IIIIII", hdr, 8)
        stride, payload = struct.unpack_from("<QQ", hdr, 32)
        pairs = [struct.unpack_from("<QQ", hdr, 48 + 16 * i) for i in range(6)]
        hid, inter = struct.unpack_from("<II", hdr, 144)
        toff, tbytes, doff, fbytes = struct.unpack_from("<QQQQ", hdr, 152)
        tsha = hdr[184:216]
        want_pairs = [tuple(rec[k]) for k in ("gate_up_w", "gate_up_ws", "gate_up_wref", "down_w", "down_ws", "down_wref")]
        check(magic == b"SBRXPRT1" and ver == 1 and hsz == 4096 and rank == r and nl == len(layers)
              and ne == E and first == E * r and stride == rec["stride"] and payload == rec["payload"]
              and pairs == want_pairs and (hid, inter) == (H, I) and fbytes == os.fstat(fd).st_size,
              f"rank {r}: header fields")
        table = os.pread(fd, tbytes, toff)
        check(hashlib.sha256(table).digest() == tsha, f"rank {r}: table sha256 == header")
        bad_sha = 0
        bad_codes = 0
        checked = 0
        for slot, L in enumerate(layers):
            kind = man["expert_kinds"][str(L)]
            idxs = range(E) if not a.sample else sorted(set(np.linspace(0, E - 1, a.sample).astype(int).tolist()))
            for i in range(E):
                buf = np.frombuffer(os.pread(fd, stride, doff + (slot * E + i) * stride), np.uint8)
                bad_sha += hashlib.sha256(buf).digest() != table[32 * (slot * E + i):32 * (slot * E + i + 1)]
                if i not in idxs:
                    continue
                got = read_record(buf, rec, H, I)
                want = expected(ck, L, E * r + i, kind, H, I)
                for part in ("gate_up", "down"):
                    c, ws, wref = got[part]
                    wc, we = want[part]
                    ok = (np.array_equal(c, wc) and np.array_equal(ws, we) and np.array_equal(wref, we.max(axis=1)))
                    bad_codes += not ok
                checked += 1
                if i == idxs[0]:
                    # controls on this record
                    flipped = buf.copy()
                    flipped[rec["gate_up_w"][0] + 7] ^= 0x10
                    check(hashlib.sha256(flipped).digest() != table[32 * (slot * E + i):32 * (slot * E + i + 1)],
                          f"rank {r} layer {L}: CONTROL flipped byte changes the sha256")
                    sw = buf.copy()
                    o, nb = rec["gate_up_w"]
                    sw[o:o + nb] = mx.nibble_swap_control(buf[o:o + nb])
                    check(not np.array_equal(read_record(sw, rec, H, I)["gate_up"][0], want["gate_up"][0]),
                          f"rank {r} layer {L}: CONTROL nibble-swapped record != the requant codes")
                    nxt = expected(ck, L, E * r + (i + 1) % E, kind, H, I)
                    check(not np.array_equal(got["gate_up"][0], nxt["gate_up"][0]),
                          f"rank {r} layer {L}: CONTROL the neighbouring expert != this record")
        os.close(fd)
        check(bad_sha == 0, f"rank {r}: every record's sha256 == its table entry ({len(layers) * E} records)")
        check(bad_codes == 0, f"rank {r}: {checked} experts re-derived from the checkpoint == the record (codes, ws, wref)")
    check_dense(a, man, ck)
    print("ALL CHECKS PASS" if not fails else f"{len(fails)} CHECKS FAILED")
    return 0 if not fails else 1


def check_dense(a, man, ck):
    if man.get("dense"):
        shared = json.load(open(os.path.join(a.pack, "shared", "dense.json")))
        per = {r: json.load(open(os.path.join(a.pack, f"rank{r}", "dense.json"))) for r in man["ranks"]}

        def ext_bytes(x):
            fd = ck.fds[x["file"]]
            return b"".join(os.pread(fd, x["length"], x["offset"] + i * x["stride"]) for i in range(x["count"]))
        bad = sum(hashlib.sha256(ext_bytes(x)).hexdigest() != x["sha256"] or ext_bytes(x) != ck.bytes_of(x["tensor"])
                  for x in shared)
        check(bad == 0, f"shared: {len(shared)} replicated records == their tensors whole")
        if sorted(per) == [0, 1]:
            bad = 0
            for x0, x1 in zip(per[0], per[1]):
                full = ck.bytes_of(x0["tensor"])
                b0, b1 = ext_bytes(x0), ext_bytes(x1)
                bad += hashlib.sha256(b0).hexdigest() != x0["sha256"] or hashlib.sha256(b1).hexdigest() != x1["sha256"]
                if x0["count"] == 1:
                    joined = b0 + b1
                else:
                    n = x0["length"]
                    joined = b"".join(b0[i * n:(i + 1) * n] + b1[i * n:(i + 1) * n] for i in range(x0["count"]))
                bad += joined != full
            check(bad == 0, f"split: {len(per[0])} records, rank 0 + rank 1 halves re-joined == the tensor")
            x0, x1 = per[0][0], per[1][0]
            check(ext_bytes(x1) != ext_bytes(x0), "CONTROL: rank 1's half != rank 0's half")


if __name__ == "__main__":
    sys.exit(main())
