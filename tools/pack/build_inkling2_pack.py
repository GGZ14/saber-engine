#!/usr/bin/env python3
"""INK1 stage 3: build the Inkling 2 Mini (`inkling2_loop`) TP=2 / EP=2 pack (docs/INKLING2.md §5, §7).

Host tool, numpy only, CPU only, `nice -n 19`, one process. The record SET is the Rust plan's
(`saber-inkling2-packplan --checkpoint DIR > plan.json`); this script does the bytes.

* Expert tier `rank<r>/experts.xtier` (Flash-Next's SBRXPRT1 v1 header, FN2's MXFP4 record layout,
  `build_flashnext_pack.write_tier_header`): rank r owns experts [64 r, 64 r + 64) of every built
  layer; record i = layer_slot * 64 + (e - 64 r). The source kinds `nvfp4_requant_mxfp4` /
  `mxfp8_requant_mxfp4` = the 27B pack's requant: the EXACT f32 dequant of the checkpoint
  (NVFP4: e2m1 x E4M3(s) x gs, gs MULTIPLIED; MXFP8: e4m3 x 2^(s-127)) -> `build_pack.quantize_mxfp4`
  (the `mse` block exponent) -> low-nibble-first bytes -> `permute_w`; `ws` = the e8m0 transposed
  [K/32, N]; `wref` = the row max of ws. gate_up = gate rows then up rows.
* Verification is part of the build, per record: Saber's dequant of the RECORD bytes
  (`flashnext_mx.saber_dequant`) == the requant's own target values bit for bit; the requant error
  against the exact dequant is recorded per layer (relRMS, max |err| / amax); controls (one expert in
  16): the nibble-swapped record and the neighbouring expert must NOT match.
* Dense `rank<r>/dense.json` + `shared/dense.json`: identity records REFERENCED in place --
  `{tensor, file, offset, length, stride, count, sha256}` extents into the shards (a dim-0 half is one
  extent; a dim-1 half is `count` strided rows).

* INK6, the NATIVE form (plan `form: Native`, `saber-inkling2-packplan --form native`): no rounding.
  NVFP4 layers -> FN144's v2 record (`w` = the checkpoint's low-nibble-first bytes in `permute_w`
  order, `ws` = the E4M3 scales transposed [K/16, N], the `*_wref` slot = the f32 globals
  `weight_scale_2`: gate_up [gate, up], down [down]; FN148's form (b) reads it) in
  `rank<r>/experts.xtier` (v2 header, tag 2); MXFP8 layers -> INK6's record (`w` = the e4m3 bytes
  row-major [N, K] as stored, `ws` = the e8m0 scales transposed [K/32, N]; `ink_mx8_w8a8_*` reads it)
  in `rank<r>/experts_mx8.xtier` (v2 header, tag 3 = MXFP8_E8M0G32, an INK6 tag no saber_mem reader
  knows yet). Verification per record: TWO independent decoders of the RECORD bytes (A: the
  inverse permutation + table decoders; B: the kernel's own address arithmetic -- the FN148 A16
  oracle's offset formula / the e4m3 byte index -- with bit-arithmetic e4m3 / e2m1 / e8m0) == the
  checkpoint's exact f32 dequant (`exact_dequant`) BIT FOR BIT (B on every 4th expert, A on all);
  controls (one expert in 16, all must differ): nibble swap (NVFP4) / byte swap (MXFP8), the
  neighbouring expert, one scale byte -1 / +1, down's global in the gate slot (NVFP4; gate and up
  share one global in this checkpoint, so a gate / up swap would be blind).

* INK14, the 8-bit dense TRUNK overlay (`--trunk8 BASE_PACK`, format `ink14-trunk8/1`, written under
  /mnt/nvme/saber-packs/): every rank record of the attention projections (q, r, sink, gate, k, v,
  the global layers' k / v dynamic-conv `weight_proj`) + o_proj + lm_head, read through the BASE
  pack's own extents (each extent's sha256 checked = the record the overlay is bound to), as int8
  round-to-nearest-even codes per 32-k block + one bf16 scale a block, rounded UP to the smallest
  bf16 >= amax / 127 (so |code| <= 127 never clips; an all-zero block has scale 0). The choice
  (vs e4m3 per 32 at the same 8 bits): int8 RTN keeps 7 magnitude bits on a uniform grid (relRMS
  ~0.3-0.5 % of the block amax) where e4m3 keeps 3 mantissa bits (~2-3 %); the decode is one
  sign-extend + convert + multiply a weight either way (bandwidth-bound at b1), and code x bf16
  scale is EXACT in f32 (7 + 8 bits), so the GEMV's twin is its bf16 parent's on that weight.
  1.0625 B a weight (vs 2): the dense trunk + head 1.61 -> 0.86 GB a rank. Records:
  `rank<r>/trunk8.bin` (codes then scales a tensor, 4 KiB aligned) + `manifest.json`
  (tensor, rank, rows, cols, codes_off / scales_off, codes / scales sha256, src_sha256 =
  the base record's). `--trunk8-layers local` (the default since INK14 h2's bisection) leaves the
  four GLOBAL layers' projections bf16 (a `partial` overlay: the engine keeps a tensor the overlay
  does not carry in bf16). Checked by `check_inkling2_trunk8.py` (independent decode + laws + controls).

  build_inkling2_pack.py --checkpoint DIR --plan plan.json --out PACK [--layers 0,20] [--ranks 0,1]
  build_inkling2_pack.py --checkpoint DIR --trunk8 BASE_PACK --out /mnt/nvme/saber-packs/<name> [--trunk8-layers local|all]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pack as bp  # noqa: E402
import build_flashnext_pack as fx  # noqa: E402
import flashnext_mx as mx  # noqa: E402

F32 = np.float32


class Ckpt:
    """safetensors shards read by header + pread (no third-party loader)."""

    NP = {"BF16": np.uint16, "F32": np.float32, "U32": np.uint32, "U8": np.uint8}
    ES = {"BF16": 2, "F32": 4, "U32": 4, "U8": 1}

    def __init__(self, d):
        self.dir = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        self.files = sorted(set(idx["weight_map"].values()))
        self.t = {}
        self.fds = {}
        self.header_sha = {}
        for f in self.files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                raw = fh.read(n)
            self.header_sha[f] = hashlib.sha256(raw).hexdigest()
            self.fds[f] = os.open(p, os.O_RDONLY)
            for k, v in json.loads(raw).items():
                if k == "__metadata__":
                    continue
                self.t[k] = (f, 8 + n + v["data_offsets"][0], v["dtype"], tuple(v["shape"]))

    def read(self, name, index=None):
        f, off, dt, shape = self.t[name]
        es = self.ES[dt]
        if index is None:
            nb, sh = int(np.prod(shape)) * es, shape
        else:
            inner = int(np.prod(shape[1:]))
            off, nb, sh = off + index * inner * es, inner * es, shape[1:]
        b = os.pread(self.fds[f], nb, off)
        if len(b) != nb:
            raise RuntimeError(f"{name}: short read")
        return np.frombuffer(b, self.NP[dt]).reshape(sh)


def exact_dequant(ck, layer, e, proj, kind):
    base = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}"
    w = ck.read(base + ".weight", e)
    s = ck.read(base + ".scales", e)
    if kind == "nvfp4_requant_mxfp4":
        gs = ck.read(base + ".weight_scale_2")[e]
        b = w.view(np.uint8)
        n = b.shape[0]
        codes = np.empty((n, b.shape[1] * 2), np.uint8)
        codes[:, 0::2] = b & 0xF
        codes[:, 1::2] = b >> 4
        v = bp.FP4_VALUES[codes].reshape(n, -1, 16) * bp.E4M3[s][..., None]
        return (v.reshape(n, -1) * F32(gs)).astype(F32)
    if kind == "mxfp8_requant_mxfp4":
        b = w.view(np.uint8)
        n = b.shape[0]
        sc = np.ldexp(F32(1), s.astype(np.int32) - 127).astype(F32)
        return (bp.E4M3[b].reshape(n, -1, 32) * sc[..., None]).reshape(n, -1).astype(F32)
    raise RuntimeError(f"layer {layer}: source kind {kind} is not built by this script")


def mxfp4_part(x):
    """f32 [N, K] -> (permuted bytes [N, K/2], ws [K/32, N], wref [N], target f32 [N, K])."""
    n, k = x.shape
    codes, e = bp.quantize_mxfp4(x)
    target = (bp.FP4_VALUES[codes].reshape(n, -1, 32)
              * np.ldexp(F32(1), e.astype(np.int32) - 127).astype(F32)[..., None]).reshape(n, k)
    return mx.permute_w(bp.pack_nibbles(codes), n, k), np.ascontiguousarray(e.T), e.max(axis=1), target


def build_expert(ck, layer, e, kind, H, I, rec, stats):
    g = exact_dequant(ck, layer, e, "gate_proj", kind)
    u = exact_dequant(ck, layer, e, "up_proj", kind)
    d = exact_dequant(ck, layer, e, "down_proj", kind)
    if g.shape != (I, H) or u.shape != (I, H) or d.shape != (H, I):
        raise RuntimeError(f"layer {layer} expert {e}: shapes {g.shape} {u.shape} {d.shape}")
    gu = np.concatenate([g, u])
    parts = {}
    parts["gate_up_w"], parts["gate_up_ws"], parts["gate_up_wref"], tgu = mxfp4_part(gu)
    parts["down_w"], parts["down_ws"], parts["down_wref"], td = mxfp4_part(d)
    buf = np.zeros(rec["stride"], np.uint8)
    for key, arr in parts.items():
        off, nb = rec[key]
        b = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
        if b.size != nb:
            raise RuntimeError(f"{key}: {b.size} bytes, the plan says {nb}")
        buf[off:off + nb] = b
    # verification: the RECORD's dequant == the requant target, bit for bit
    def sub(key, shape):
        off, nb = rec[key]
        return buf[off:off + nb].reshape(shape)
    sgu = mx.saber_dequant(sub("gate_up_w", (2 * I, H // 2)), sub("gate_up_ws", (H // 32, 2 * I)), 2 * I, H)
    sd = mx.saber_dequant(sub("down_w", (H, I // 2)), sub("down_ws", (I // 32, H)), H, I)
    if not (mx.bits_equal(sgu, tgu) and mx.bits_equal(sd, td)):
        raise RuntimeError(f"layer {layer} expert {e}: the record's dequant != the requant target")
    st = stats.setdefault(layer, {"err2": 0.0, "ref2": 0.0, "maxrel": 0.0, "n": 0})
    for ref, got in ((gu, tgu), (d, td)):
        diff = (got - ref).astype(np.float64)
        st["err2"] += float((diff * diff).sum())
        st["ref2"] += float((ref.astype(np.float64) ** 2).sum())
        st["maxrel"] = max(st["maxrel"], float(np.abs(diff).max() / max(np.abs(ref).max(), 1e-30)))
    st["n"] += 1
    return buf, sgu


# ------------------------------------------------------------------------------- INK6 native
def e4m3_arith(b):
    """OCP e4m3 by bit arithmetic (decoder B; independent of build_pack's table)."""
    b = b.astype(np.int32)
    e, m = (b >> 3) & 0xF, (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * np.exp2(e.astype(np.float64) - 7))
    v = np.where((e == 15) & ((b & 7) == 7), np.nan, v)
    return np.where(b & 0x80, -v, v).astype(F32)


def e2m1_arith(c):
    c = c.astype(np.int32)
    e, m = (c >> 1) & 3, (c & 1).astype(np.float64)
    v = np.where(e == 0, m * 0.5, (1 + m * 0.5) * np.exp2(e.astype(np.float64) - 1))
    return np.where(c & 8, -v, v).astype(F32)


def native_parts(ck, layer, e, kind):
    """(record parts dict, exact f32 dequant {gate_up, down}) of one expert."""
    base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
    raw = {p: (ck.read(base + p + ".weight", e).view(np.uint8), ck.read(base + p + ".scales", e))
           for p in ("gate_proj", "up_proj", "down_proj")}
    ref = {"gate_up": np.concatenate([exact_dequant(ck, layer, e, "gate_proj", kind.replace("native", "requant_mxfp4")),
                                      exact_dequant(ck, layer, e, "up_proj", kind.replace("native", "requant_mxfp4"))]),
           "down": exact_dequant(ck, layer, e, "down_proj", kind.replace("native", "requant_mxfp4"))}
    w_gu = np.concatenate([raw["gate_proj"][0], raw["up_proj"][0]])
    s_gu = np.concatenate([raw["gate_proj"][1], raw["up_proj"][1]])
    w_dn, s_dn = raw["down_proj"]
    parts = {}
    if kind == "nvfp4_native":
        gs = {p: F32(ck.read(base + p + ".weight_scale_2")[e]) for p in ("gate_proj", "up_proj", "down_proj")}
        for pre, w, sc, g in (("gate_up", w_gu, s_gu, [gs["gate_proj"], gs["up_proj"]]),
                              ("down", w_dn, s_dn, [gs["down_proj"]])):
            n, k = w.shape[0], w.shape[1] * 2
            parts[pre + "_w"] = mx.permute_w(np.ascontiguousarray(w), n, k)
            parts[pre + "_ws"] = np.ascontiguousarray(sc.T)
            parts[pre + "_wref"] = np.array(g, F32)
    elif kind == "mxfp8_native":
        for pre, w, sc in (("gate_up", w_gu, s_gu), ("down", w_dn, s_dn)):
            parts[pre + "_w"] = np.ascontiguousarray(w)
            parts[pre + "_ws"] = np.ascontiguousarray(sc.T)
            parts[pre + "_wref"] = np.zeros(0, np.uint8)
    else:
        raise RuntimeError(f"layer {layer}: kind {kind} is not a native kind")
    return parts, ref


def native_record(parts, rec):
    buf = np.zeros(rec["stride"], np.uint8)
    for key, arr in parts.items():
        off, nb = rec[key]
        b = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
        if b.size != nb:
            raise RuntimeError(f"{key}: {b.size} bytes, the record says {nb}")
        buf[off:off + nb] = b
    return buf


def decode_a(buf, rec, kind, H, I):
    """Decoder A: the inverse permutation + build_pack's tables."""
    out = {}
    for pre, n, k in (("gate_up", 2 * I, H), ("down", H, I)):
        def sub(key):
            off, nb = rec[pre + key]
            return buf[off:off + nb]
        if kind == "nvfp4_native":
            packed = mx.unpermute_w(sub("_w").reshape(n, k // 2), n, k)
            codes = np.empty((n, k), np.uint8)
            codes[:, 0::2], codes[:, 1::2] = packed & 0xF, packed >> 4
            ws = sub("_ws").reshape(k // 16, n).T
            gs = sub("_wref").view(F32)
            g = np.where(np.arange(n) < (I if pre == "gate_up" else n), gs[0], gs[-1]).astype(F32)
            v = bp.FP4_VALUES[codes].reshape(n, -1, 16) * bp.E4M3[ws][..., None]
            out[pre] = (v.reshape(n, k) * g[:, None]).astype(F32)
        else:
            w = sub("_w").reshape(n, k)
            sc = np.ldexp(F32(1), sub("_ws").reshape(k // 32, n).T.astype(np.int32) - 127).astype(F32)
            out[pre] = (bp.E4M3[w].reshape(n, -1, 32) * sc[..., None]).reshape(n, k).astype(F32)
    return out


def decode_b(buf, rec, kind, H, I):
    """Decoder B: the kernels' own address arithmetic over the record bytes, bit-arithmetic decoders."""
    out = {}
    for pre, n, k in (("gate_up", 2 * I, H), ("down", H, I)):
        o_w, _ = rec[pre + "_w"]
        o_s, _ = rec[pre + "_ws"]
        nn = np.arange(n, dtype=np.int64)[:, None]
        kk = np.arange(k, dtype=np.int64)[None, :]
        if kind == "nvfp4_native":
            # fn148_a16_impl: off = ((((n/16 * K/16 + k/16) * 2 + (k/8)&1) * 16 + n%16) * 4 + (k%8)/2
            off = ((((nn >> 4) * (k // 16) + (kk >> 4)) * 2 + ((kk >> 3) & 1)) * 16 + (nn & 15)) * 4 + ((kk & 7) >> 1)
            code = (buf[o_w + off] >> ((kk & 1) * 4).astype(np.uint8)) & 0xF
            s = e4m3_arith(buf[o_s + (kk >> 4) * n + nn])
            o_g, _ = rec[pre + "_wref"]
            gs = buf[o_g:o_g + 8].view(F32) if pre == "gate_up" else buf[o_g:o_g + 4].view(F32)
            g = np.where(nn < (I if pre == "gate_up" else n), gs[0], gs[-1]).astype(F32)
            out[pre] = ((e2m1_arith(code) * s) * g).astype(F32)
        else:
            byte = buf[o_w + nn * k + kk]
            ex = buf[o_s + (kk >> 5) * n + nn].astype(np.int32)
            out[pre] = (e4m3_arith(byte) * np.ldexp(F32(1), ex - 127).astype(F32)).astype(F32)
    return out


def build_native(a, plan, plan_sha, ck, layers, ranks):
    H, I = plan["geometry"]["hidden"], plan["geometry"]["inter"]
    E = plan["experts_per_rank"]
    recs = {"nvfp4_native": plan["records"]["nvfp4"], "mxfp8_native": plan["records"]["mxfp8"]}
    files = {"nvfp4_native": ("experts.xtier", 2), "mxfp8_native": ("experts_mx8.xtier", 3)}
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tiers, checks = {}, {"records": 0, "decoder_a": 0, "decoder_b": 0, "controls": 0, "escaped": 0}
    scale_range = {}
    for r in ranks:
        rd = os.path.join(a.out, f"rank{r}")
        os.makedirs(rd, exist_ok=True)
        tiers[str(r)] = {}
        for kind, (fname, tag) in files.items():
            kl = [L for L in layers if plan["expert_kinds"][L] == kind]
            if not kl:
                continue
            rec = recs[kind]
            t = {"rank": r, "layers": len(kl), "first_expert": E * r, "experts": E, "records": len(kl) * E,
                 "record": {"stride": rec["stride"], "payload": rec["payload"],
                            **{k: tuple(rec[k]) for k in rec if k.startswith(("gate_up_", "down_"))}}}
            table_off, table_bytes, data_off, file_bytes = fx.tier_geometry(t)
            path = os.path.join(rd, fname)
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, file_bytes)
            shas = []
            for slot, L in enumerate(kl):
                for i in range(E):
                    e = E * r + i
                    parts, ref = native_parts(ck, L, e, kind)
                    buf = native_record(parts, rec)
                    sr = scale_range.setdefault(kind, [255, 0])
                    for pre in ("gate_up", "down"):
                        ws = parts[pre + "_ws"]
                        sr[0], sr[1] = min(sr[0], int(ws.min())), max(sr[1], int(ws.max()))
                    da = decode_a(buf, rec, kind, H, I)
                    for pre in ("gate_up", "down"):
                        if not mx.bits_equal(da[pre], ref[pre]):
                            raise RuntimeError(f"layer {L} expert {e} {pre}: decoder A != the exact dequant")
                    checks["decoder_a"] += 1
                    if i % 4 == 0:
                        db = decode_b(buf, rec, kind, H, I)
                        for pre in ("gate_up", "down"):
                            if not mx.bits_equal(db[pre], ref[pre]):
                                raise RuntimeError(f"layer {L} expert {e} {pre}: decoder B != the exact dequant")
                        checks["decoder_b"] += 1
                    if i % 16 == 0:
                        ctl = []
                        off, nb = rec["gate_up_w"]
                        b1 = buf.copy()
                        if kind == "nvfp4_native":
                            b1[off:off + nb] = mx.nibble_swap_control(b1[off:off + nb])
                        else:
                            b1[off:off + nb] = b1[off:off + nb].reshape(-1, 4)[:, ::-1].reshape(-1)
                        ctl.append(b1)
                        ctl.append(native_record(native_parts(ck, L, E * r + (i + 1) % E, kind)[0], rec))
                        b3 = buf.copy()
                        so, sn = rec["gate_up_ws"]
                        j = so + int(np.argmax(b3[so:so + sn]))
                        b3[j] = b3[j] - 1 if kind == "nvfp4_native" else b3[j] + 1
                        ctl.append(b3)
                        if kind == "nvfp4_native":
                            # gate and up share their global in this checkpoint (measured): the control
                            # puts down's global in the gate slot
                            b4 = buf.copy()
                            go, _ = rec["gate_up_wref"]
                            do, _ = rec["down_wref"]
                            b4[go:go + 4] = b4[do:do + 4]
                            ctl.append(b4)
                        for cb in ctl:
                            checks["controls"] += 1
                            checks["escaped"] += int(mx.bits_equal(decode_a(cb, rec, kind, H, I)["gate_up"], ref["gate_up"]))
                    checks["records"] += 1
                    os.pwrite(fd, buf.tobytes(), data_off + (slot * E + i) * rec["stride"])
                    shas.append(hashlib.sha256(buf).digest())
                print(f"rank {r} layer {L} ({kind}): {E} records, {time.time() - t0:.0f}s", flush=True)
            table = b"".join(shas)
            os.pwrite(fd, table, table_off)
            os.close(fd)
            fx.write_tier_header(path, t, plan_sha, hashlib.sha256(table).digest(), table_off, table_bytes,
                                 data_off, file_bytes, H, I)
            fd = os.open(path, os.O_WRONLY)
            os.pwrite(fd, struct.pack("<I", 2), 8)           # v2
            os.pwrite(fd, struct.pack("<I", tag), 256)       # format tag (2 NVFP4_E4M3G16, 3 MXFP8_E8M0G32)
            os.fsync(fd)
            os.close(fd)
            tiers[str(r)][kind] = {"file": f"rank{r}/{fname}", "bytes": file_bytes, "records": len(shas),
                                   "layers": kl, "format": rec["format"], "tag": tag,
                                   "table_sha256": hashlib.sha256(table).hexdigest()}
    if checks["escaped"]:
        raise SystemExit(f"{checks['escaped']} of {checks['controls']} controls matched: the verification is blind")
    return tiers, checks, recs, scale_range


def dense_extents(ck, rec, rank):
    f, off, dt, shape = ck.t[rec["tensor"]]
    es = ck.ES[dt]
    if list(shape) != rec["shape"] or dt != rec["dtype"]:
        raise RuntimeError(f"{rec['tensor']}: checkpoint {dt} {shape}, plan {rec['dtype']} {rec['shape']}")
    total = int(np.prod(shape)) * es
    if rec["placement"] == "shared":
        ext = dict(file=f, offset=off, length=total, stride=total, count=1)
    elif rec["split_dim"] == 0:
        half = total // 2
        ext = dict(file=f, offset=off + rank * half, length=half, stride=half, count=1)
    else:
        rows = int(np.prod(shape[:rec["split_dim"]]))
        row = total // rows
        ext = dict(file=f, offset=off + rank * (row // 2), length=row // 2, stride=row, count=rows)
    h = hashlib.sha256()
    for i in range(ext["count"]):
        b = os.pread(ck.fds[f], ext["length"], ext["offset"] + i * ext["stride"])
        if len(b) != ext["length"]:
            raise RuntimeError(f"{rec['tensor']}: short read")
        h.update(b)
    ext["sha256"] = h.hexdigest()
    ext["tensor"] = rec["tensor"]
    ext["dtype"] = dt
    return ext


def E_of(plan):
    return plan["experts_per_rank"]


def write_dense(a, plan, ck, ranks):
    dense = {}
    if not a.no_dense:
        shared = [dense_extents(ck, d, 0) for d in plan["dense"] if d["placement"] == "shared"]
        os.makedirs(os.path.join(a.out, "shared"), exist_ok=True)
        json.dump(shared, open(os.path.join(a.out, "shared", "dense.json"), "w"), indent=0)
        dense["shared"] = len(shared)
        for r in ranks:
            per = [dense_extents(ck, d, r) for d in plan["dense"] if d["placement"] == "rank"]
            os.makedirs(os.path.join(a.out, f"rank{r}"), exist_ok=True)
            json.dump(per, open(os.path.join(a.out, f"rank{r}", "dense.json"), "w"), indent=0)
            dense[f"rank{r}"] = len(per)
    return dense


# ---------------------------------------------------------------------------------------- INK14
TRUNK8_FORMAT = "ink14-trunk8/1"
TRUNK8_PROJ = ("q_proj", "r_proj", "sink_proj", "gate_proj", "k_proj", "v_proj", "o_proj",
               "k_sconv.weight_proj", "v_sconv.weight_proj")


def trunk8_selected(name, skip=()):
    if name == "language_model.lm_head.weight":
        return True
    if any(f".layers.{L}." in name for L in skip):
        return False
    if ".self_attn." not in name or not name.endswith(".weight"):
        return False
    tail = name.split(".self_attn.", 1)[1][: -len(".weight")]
    return tail in TRUNK8_PROJ


def bf16_f32(u16):
    return (u16.astype(np.uint32) << 16).view(F32)


def quant_trunk8(w):
    """f32 [rows, cols] (bf16 values) -> (codes int8 [rows, cols], scale bits u16 [rows, cols/32])."""
    rows, cols = w.shape
    b = w.reshape(rows, cols // 32, 32).astype(np.float64)
    amax = np.abs(b).max(axis=2)
    x = amax / 127.0
    bits = (x.astype(F32).view(np.uint32) >> 16).astype(np.uint32)
    for _ in range(2):   # smallest bf16 >= amax / 127
        low = bf16_f32(bits.astype(np.uint16)).astype(np.float64) * 127.0 < amax
        bits = bits + low
    bits = np.where(amax == 0, 0, bits).astype(np.uint16)
    sc = bf16_f32(bits).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(sc[..., None] > 0, np.rint(b / np.where(sc > 0, sc, 1.0)[..., None]), 0.0)
    if np.abs(q).max() > 127:
        raise RuntimeError("trunk8: a code past 127 (the scale law broke)")
    return q.astype(np.int8).reshape(rows, cols), bits


def build_trunk8(a, ck):
    base = a.trunk8
    out = a.out
    rp = os.path.realpath(out)
    if not rp.startswith("/mnt/nvme/saber-packs/"):
        raise SystemExit("--trunk8: the overlay goes under /mnt/nvme/saber-packs/ (not the root disk)")
    bman = json.load(open(os.path.join(base, "manifest.json")))
    if bman.get("model") != "inkling2_loop":
        raise SystemExit("--trunk8: the base is not an inkling2_loop pack")
    os.makedirs(out, exist_ok=True)
    # INK14 h2's bisection: the 4 GLOBAL layers' projections (5, 11, 17, 23: 8 of the 48 visits)
    # carry the 8-bit arm's first-token KL defect (p06); `--trunk8-layers local` keeps them bf16
    skip = () if a.trunk8_layers == "all" else (5, 11, 17, 23)
    t0 = time.time()
    recs, err = [], {}
    for r in (0, 1):
        os.makedirs(os.path.join(out, f"rank{r}"), exist_ok=True)
        dense = json.load(open(os.path.join(base, f"rank{r}", "dense.json")))
        path = os.path.join(out, f"rank{r}", "trunk8.bin")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        off = 0
        for d in dense:
            if not trunk8_selected(d["tensor"], skip):
                continue
            if d["dtype"] != "BF16":
                raise SystemExit(f"{d['tensor']}: {d['dtype']}, not BF16")
            fdc = ck.fds[d["file"]]
            h = hashlib.sha256()
            parts = []
            for i in range(d["count"]):
                bb = os.pread(fdc, d["length"], d["offset"] + i * d["stride"])
                if len(bb) != d["length"]:
                    raise RuntimeError(f"{d['tensor']}: short read")
                h.update(bb)
                parts.append(bb)
            if h.hexdigest() != d["sha256"]:
                raise SystemExit(f"{d['tensor']} rank {r}: the base extent's sha256 does not match its record")
            raw = np.frombuffer(b"".join(parts), np.uint16)
            shape = ck.t[d["tensor"]][3]
            if d["count"] == 1:
                rows = shape[0] // 2 if len(shape) > 1 else 1
            else:
                rows = d["count"]
            cols = raw.size // rows
            if cols % 256:
                raise SystemExit(f"{d['tensor']}: K {cols} is not a multiple of 256")
            w = bf16_f32(raw).reshape(rows, cols)
            codes, sbits = quant_trunk8(w)
            deq = codes.astype(F32) * bf16_f32(sbits.reshape(rows, cols // 32)).repeat(32, axis=1)
            e = (deq.astype(np.float64) - w) ** 2
            err[f"{d['tensor']}@{r}"] = float((e.sum() / max((w.astype(np.float64) ** 2).sum(), 1e-30)) ** 0.5)
            cb, sb = codes.tobytes(), sbits.astype("<u2").tobytes()
            co = off
            os.pwrite(fd, cb, co)
            so = co + ((len(cb) + 4095) // 4096) * 4096
            os.pwrite(fd, sb, so)
            off = so + ((len(sb) + 4095) // 4096) * 4096
            recs.append({"tensor": d["tensor"], "rank": r, "rows": rows, "cols": cols,
                         "file": f"rank{r}/trunk8.bin", "codes_off": co, "scales_off": so,
                         "codes_sha256": hashlib.sha256(cb).hexdigest(),
                         "scales_sha256": hashlib.sha256(sb).hexdigest(), "src_sha256": d["sha256"]})
        os.ftruncate(fd, off)
        os.close(fd)
        print(f"trunk8 rank {r}: {sum(1 for x in recs if x['rank'] == r)} tensors, {off / 1e9:.3f} GB, "
              f"{time.time() - t0:.0f}s", flush=True)
    rel = np.array(list(err.values()))
    man = {"model": "inkling2_loop", "format": TRUNK8_FORMAT, "base_pack": os.path.realpath(base),
           "base_manifest_sha256": hashlib.sha256(open(os.path.join(base, "manifest.json"), "rb").read()).hexdigest(),
           "checkpoint": a.checkpoint, "quant": "int8 RTNE per 32-k block, bf16 scale = smallest bf16 >= amax/127",
           "layers_policy": a.trunk8_layers, "skipped_layers": list(skip), "partial": bool(skip),
           "tensors": recs, "relrms": err,
           "relrms_summary": {"mean": float(rel.mean()), "max": float(rel.max())},
           "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({"tensors": len(recs), "relrms": man["relrms_summary"]}), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", help="comma list (a fixture); default every layer")
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--trunk8", metavar="BASE_PACK", help="INK14: build the 8-bit dense trunk overlay of BASE_PACK")
    ap.add_argument("--trunk8-layers", choices=("all", "local"), default="local",
                    help="INK14: all = every layer; local = the 20 local layers (the global 5 / 11 / 17 / 23 stay bf16)")
    a = ap.parse_args(argv)
    if a.trunk8:
        return build_trunk8(a, Ckpt(a.checkpoint))
    # The full pack is a model artefact beside its checkpoint (the user's approval, 2026-11-21:
    # /mnt/nvme/models/ink2-pack); nothing else of this tool goes on the model drive.
    # INK6 (the user's word 2026-11-21): the native pack at /mnt/nvme/models/ink2-pack-nv.
    if a.out.startswith("/mnt/nvme") and os.path.realpath(a.out) not in ("/mnt/nvme/models/ink2-pack",
                                                                          "/mnt/nvme/models/ink2-pack-nv"):
        raise SystemExit("on /mnt/nvme only /mnt/nvme/models/ink2-pack(-nv); write elsewhere (not the model drive)")
    plan_bytes = open(a.plan, "rb").read()
    plan = json.loads(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).digest()
    if plan.get("model") != "inkling2_loop":
        raise SystemExit("plan: not an inkling2_loop plan")
    ck = Ckpt(a.checkpoint)
    H, I = plan["geometry"]["hidden"], plan["geometry"]["inter"]
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(plan["layers"]))
    ranks = [int(x) for x in a.ranks.split(",")]
    if plan.get("form") == "Native":
        t0 = time.time()
        tiers, checks, recs, srange = build_native(a, plan, plan_sha, ck, layers, ranks)
        dense = write_dense(a, plan, ck, ranks)
        manifest = {
            "model": "inkling2_loop", "format": "ink6-pack/1", "form": "Native", "tp": plan["tp"], "ep": plan["ep"],
            "checkpoint": a.checkpoint, "shard_header_sha256": ck.header_sha, "plan_sha256": plan_sha.hex(),
            "layers": layers, "ranks": ranks, "expert_kinds": {str(L): plan["expert_kinds"][L] for L in layers},
            "records": {"nvfp4_native": recs["nvfp4_native"], "mxfp8_native": recs["mxfp8_native"]},
            "geometry": plan["geometry"], "experts_per_rank": E_of(plan), "tiers": tiers, "dense": dense,
            "checks": checks, "scale_byte_range": srange,
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
        }
        json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
        print(json.dumps({"checks": checks, "tiers": tiers, "scale_byte_range": srange}, indent=1))
        return 0
    rec = plan["records"]["mxfp4"]
    E = plan["experts_per_rank"]
    os.makedirs(a.out, exist_ok=True)
    stats, controls, escaped = {}, 0, 0
    t0 = time.time()
    tiers = {}
    for r in ranks:
        rd = os.path.join(a.out, f"rank{r}")
        os.makedirs(rd, exist_ok=True)
        t = {"rank": r, "layers": len(layers), "first_expert": E * r, "experts": E,
             "records": len(layers) * E, "record": {"stride": rec["stride"], "payload": rec["payload"],
                                                   **{k: tuple(rec[k]) for k in rec if k.startswith(("gate_up_", "down_"))}}}
        table_off, table_bytes, data_off, file_bytes = fx.tier_geometry(t)
        path = os.path.join(rd, "experts.xtier")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, file_bytes)
        shas = []
        for slot, L in enumerate(layers):
            kind = plan["expert_kinds"][L]
            for i in range(E):
                e = E * r + i
                buf, sgu = build_expert(ck, L, e, kind, H, I, rec, stats)
                os.pwrite(fd, buf.tobytes(), data_off + (slot * E + i) * rec["stride"])
                shas.append(hashlib.sha256(buf).digest())
                if i % 16 == 0:
                    off, nb = rec["gate_up_w"]
                    soff, sn = rec["gate_up_ws"]
                    swapped = mx.saber_dequant(mx.nibble_swap_control(buf[off:off + nb].reshape(2 * I, H // 2)),
                                               buf[soff:soff + sn].reshape(H // 32, 2 * I), 2 * I, H)
                    nb_, _ = build_expert(ck, L, E * r + (i + 1) % E, kind, H, I, rec, {})
                    other = mx.saber_dequant(nb_[off:off + nb].reshape(2 * I, H // 2),
                                             nb_[soff:soff + sn].reshape(H // 32, 2 * I), 2 * I, H)
                    for ctl in (swapped, other):
                        controls += 1
                        escaped += int(mx.bits_equal(sgu, ctl))
            print(f"rank {r} layer {L} ({kind}): {E} records, {time.time()-t0:.0f}s", flush=True)
        table = b"".join(shas)
        os.pwrite(fd, table, table_off)
        os.close(fd)
        fx.write_tier_header(path, t, plan_sha, hashlib.sha256(table).digest(), table_off, table_bytes,
                             data_off, file_bytes, H, I)
        tiers[r] = {"file": f"rank{r}/experts.xtier", "bytes": file_bytes, "records": len(shas),
                    "table_sha256": hashlib.sha256(table).hexdigest()}
    dense = write_dense(a, plan, ck, ranks)
    if escaped:
        raise SystemExit(f"{escaped} of {controls} controls matched the record: the verification is blind")
    err = {str(L): {"relrms": (s["err2"] / s["ref2"]) ** 0.5, "max_rel_to_amax": s["maxrel"], "experts": s["n"]}
           for L, s in sorted(stats.items())}
    manifest = {
        "model": "inkling2_loop", "format": "ink1-pack/1", "tp": plan["tp"], "ep": plan["ep"],
        "checkpoint": a.checkpoint, "shard_header_sha256": ck.header_sha,
        "plan_sha256": plan_sha.hex(), "layers": layers, "ranks": ranks,
        "expert_kinds": {str(L): plan["expert_kinds"][L] for L in layers},
        "record": rec, "geometry": plan["geometry"], "experts_per_rank": E, "tiers": tiers, "dense": dense,
        "requant_error": err, "controls": {"run": controls, "escaped": escaped},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({"requant_error": err, "controls": manifest["controls"], "tiers": tiers}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
