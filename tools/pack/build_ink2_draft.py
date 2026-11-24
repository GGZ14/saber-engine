#!/usr/bin/env python3
"""INK12: the Inkling 2 block drafter's records for an `inkling2_loop` pack (`<pack>/draft/`).

Every drafter tensor of `mtp.safetensors` is written as its checkpoint's OWN bf16 bytes (no
requant: the head is bf16, never 4-bit), plus the Markov head folded into the TARGET's lm_head once
in f64 and stored bf16 (docs/INKLING2.md section 6):
  fold_w = bf16(lm_head (f64) @ hidden_out.weight (f64))   [201024, 256]
  fold_b = bf16(lm_head (f64) @ hidden_out.bias (f64))     [201024]
and two derived slices of rnn_proj.weight [768, 5120] (bytes copied, not computed):
  rnn_lo = columns 0..512 (the [state ; w1] part), rnn_a = 512..2816 (the drafter output part),
  rnn_b = 2816..5120 (the f_rnn part).
`manifest.json`: name -> {file, shape, dtype, sha256}; the checker is check_ink2_draft.py.

  build_ink2_draft.py --model /mnt/nvme/models/Inkling-2-Mini-nvfp4 --pack /mnt/nvme/models/ink2-pack-nv
"""
import argparse
import hashlib
import json
import os
import struct
import time

import numpy as np


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return n, h


def raw(path, n, h, name):
    v = h[name]
    a, b = v["data_offsets"]
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return np.asarray(mm[8 + n + a: 8 + n + b]), v["dtype"], v["shape"]


def bf16_of_f64(x):
    f = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    return (((f + 0x7FFF + ((f >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    a = ap.parse_args()
    out = os.path.join(a.pack, "draft")
    os.makedirs(out, exist_ok=True)
    mtp = os.path.join(a.model, "mtp.safetensors")
    n, h = header(mtp)
    man = {"format": "ink12-draft/1", "source": "mtp.safetensors", "records": {}}

    def put(name, data, shape, dtype="BF16"):
        fn = name + ".bin"
        b = np.ascontiguousarray(data).tobytes()
        with open(os.path.join(out, fn), "wb") as fh:
            fh.write(b)
        man["records"][name] = {"file": fn, "shape": list(shape), "dtype": dtype,
                                "sha256": hashlib.sha256(b).hexdigest()}

    for name in sorted(h):
        b, dt, shape = raw(mtp, n, h, name)
        assert dt == "BF16", (name, dt)
        put(name, b, shape)
    wr = raw(mtp, n, h, "markov_head.rnn_proj.weight")[0].view(np.uint16).reshape(768, 5120)
    put("rnn_lo", wr[:, :512], (768, 512))
    put("rnn_a", wr[:, 512:2816], (768, 2304))
    put("rnn_b", wr[:, 2816:], (768, 2304))
    # the fold, f64, against the target's lm_head
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    hp = os.path.join(a.model, idx["language_model.lm_head.weight"])
    hn, hh = header(hp)
    head = raw(hp, hn, hh, "language_model.lm_head.weight")[0].view(np.uint16).reshape(201024, 2304)
    proj = (raw(mtp, n, h, "markov_head.hidden_out.weight")[0].view(np.uint16).astype(np.uint32) << 16
            ).view(np.float32).reshape(2304, 256).astype(np.float64)
    bias = (raw(mtp, n, h, "markov_head.hidden_out.bias")[0].view(np.uint16).astype(np.uint32) << 16
            ).view(np.float32).astype(np.float64)
    t0 = time.time()
    fw = np.empty((201024, 256), np.uint16)
    fb = np.empty((201024,), np.uint16)
    for s in range(0, 201024, 8192):
        hd = (head[s:s + 8192].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        fw[s:s + 8192] = bf16_of_f64(hd @ proj)
        fb[s:s + 8192] = bf16_of_f64(hd @ bias)
    put("fold_w", fw, (201024, 256))
    put("fold_b", fb, (201024,))
    man["fold_s"] = round(time.time() - t0, 1)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"INK12 draft pack: {len(man['records'])} records in {out} (fold {man['fold_s']} s)")


if __name__ == "__main__":
    main()
