#!/usr/bin/env python3
"""INK48: the Saber drafter pack of the public Inkling-2-Mini DFlash2 drafter (`incoai/Inkling-2-Mini-DFlash2`),
built from its public `model.safetensors` + `config.json` ONLY (clean-room: no other engine's code).

Layout (`<out>/manifest.json`, one file a record, the Inkling drafter pack's convention -- build_ink2_draft.py):
  format     "ink2-dflash2/1"
  placement  the drafter lives WHOLE on rank 0 at both TP=2 and TP=1 (Inkling's drafter law: rank 0 drafts, the
             verify is the target's), so the one rank-0 shard serves both layouts (`ranks: {"tp2": [0], "tp1": [0]}`)
  config     the checkpoint's config.json (the engine checks the INK47 geometry from it)
  records    name -> {file, shape, dtype, sha256, form}
    form "bf16": every one of the 81 tensors, the checkpoint's OWN bf16 bytes
    form "w8":   `<name>.w8` (I8 codes [N, K]) + `<name>.w8s` (BF16 scales [N, K / 32]) for fc and the five layers'
                 q / k / v / o / gate / up / down = INK14's record law (the engine's `quant_w8`): a 32-k block's scale
                 = the smallest bf16 >= amax / 127, code = round-half-even(w / scale) clamped to +-127, 0 when the
                 scale is 0. The 8-bit layer form runs on the Inkling drafter / target 8-bit GEMVs + GEMMs
                 (`ink_dr_gemv_sk_w8`, `ink_dr4_mm_w8`, the target's w8 GEMM) -- no new kernel.
  iu8        INK50: the W8A8 drafter form (`--ink-draft-layers iu8`) over the SAME w8 records (no new bytes):
             {law: IU8_LAW, records: {name: {n, k, waves}}} for every w8 weight -- n / k = the record's shape, waves =
             `ink_df2_i8_w8`'s waves a workgroup = the largest divisor of K / 128 at or below 16 (numerics: each wave
             sums one contiguous run of 128-k groups); every K % 128 == 0 (the kernel's load law). The engine refuses
             the iu8 arm on a pack without this section or with any entry that differs from its own derivation.
Checker: check_ink2_dflash2.py (re-reads the source; flipped-byte control).

  build_ink2_dflash2.py --src /mnt/nvme/models/ink48-work --out /mnt/nvme/models/ink48-work/pack
"""
import argparse
import hashlib
import json
import os
import struct
import time

import numpy as np

FORMAT = "ink2-dflash2/1"
IU8_LAW = "ink50-iu8/1: ink_act_q8_b32_v2 + ink_df2_i8_w8 on the w8 records"
W8 = [f"layers.{i}.{s}.weight" for i in range(5) for s in (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")] + ["fc.weight"]


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return n, h


def tensors(path):
    n, h = header(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for k in sorted(h):
        v = h[k]
        a, b = v["data_offsets"]
        yield k, v["dtype"], v["shape"], mm[8 + n + a: 8 + n + b]


def quant_w8(raw, shape):
    """INK14's record law on bf16 bytes [N, K] -> (int8 codes bytes, bf16 scale bytes)."""
    u = np.frombuffer(raw.tobytes() if hasattr(raw, "tobytes") else raw, dtype=np.uint16)
    w = (u.astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
    t = np.abs(w).max(axis=1) / np.float32(127.0)
    s16 = (t.view(np.uint32) >> 16).astype(np.uint32)
    s = (s16 << 16).view(np.float32)
    s16 = np.where(s < t, s16 + 1, s16).astype(np.uint16)
    s = (s16.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s[:, None] == 0, np.float32(0), np.rint(w / s[:, None]))
    q = np.clip(q, -127, 127).astype(np.int8)
    assert w.size == int(np.prod(shape))
    return q.tobytes(), s16.tobytes()


def iu8_waves(k):
    """INK50: ink_df2_i8_w8's waves a workgroup (inkling2_df2::iu8_waves)."""
    g = max(k // 128, 1)
    return next(d for d in range(16, 0, -1) if g % d == 0)


def iu8_section(shapes):
    """INK50: the iu8 form's declaration over the w8 records {name: [N, K]}."""
    recs = {}
    for k, (n, kk) in sorted(shapes.items()):
        assert kk % 128 == 0, (k, kk)
        recs[k] = {"n": n, "k": kk, "waves": iu8_waves(kk)}
    return {"law": IU8_LAW, "records": recs}


def put(out, recs, name, data, shape, dtype, form):
    f = name + ".bin"
    with open(os.path.join(out, f), "wb") as fh:
        fh.write(data)
    recs[name] = {"file": f, "shape": list(shape), "dtype": dtype, "form": form,
                  "sha256": hashlib.sha256(data).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    t0 = time.time()
    recs = {}
    w8_shapes = {}
    st = os.path.join(a.src, "model.safetensors")
    for k, dt, shape, raw in tensors(st):
        assert dt == "BF16", (k, dt)
        b = raw.tobytes()
        put(a.out, recs, k, b, shape, "BF16", "bf16")
        if k in W8:
            assert len(shape) == 2 and shape[1] % 32 == 0, (k, shape)
            codes, sc = quant_w8(b, shape)
            w8_shapes[k] = list(shape)
            put(a.out, recs, k + ".w8", codes, shape, "I8", "w8")
            put(a.out, recs, k + ".w8s", sc, [shape[0], shape[1] // 32], "BF16", "w8")
    with open(st, "rb") as fh:
        src_sha = hashlib.sha256(fh.read()).hexdigest()
    man = {"format": FORMAT, "source": {"file": "model.safetensors", "sha256": src_sha},
           "ranks": {"tp2": [0], "tp1": [0]}, "config": cfg, "records": recs,
           "iu8": iu8_section(w8_shapes)}
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    n8 = sum(1 for r in recs.values() if r["form"] == "w8")
    print(f"INK50 iu8 section: {len(man['iu8']['records'])} records")
    print(f"INK48 PACK {a.out}: {len(recs)} records ({n8} w8), source {src_sha[:16]}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
