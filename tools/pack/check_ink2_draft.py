#!/usr/bin/env python3
"""INK12: independent checker of `<pack>/draft/` (build_ink2_draft.py's records).

Own index walk (never the builder's code): every record's bytes hash to the manifest's sha256; every
checkpoint tensor's record == the checkpoint's bytes; the rnn slices == the columns of the
checkpoint's rnn_proj.weight; `--sample N` fold rows re-derived in f64 (torch-free: per-row dot in
math.fsum order is not needed -- numpy f64 matmul, then RNE to bf16) must be BYTE-equal. Controls: a
flipped byte in a copy of each kind of record must be caught (sha, identity, fold) or the check
FAILS.

  check_ink2_draft.py --model DIR --pack DIR [--sample 4096]
"""
import argparse
import hashlib
import json
import os
import struct

import numpy as np


def tensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return {k: mm[8 + n + v["data_offsets"][0]: 8 + n + v["data_offsets"][1]] for k, v in h.items()}


def rne(x):
    f = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    return (((f + 0x7FFF + ((f >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def f32(b):
    return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--sample", type=int, default=4096)
    a = ap.parse_args()
    d = os.path.join(a.pack, "draft")
    man = json.load(open(os.path.join(d, "manifest.json")))
    recs = man["records"]
    ck = tensors(os.path.join(a.model, "mtp.safetensors"))
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    head = tensors(os.path.join(a.model, idx["language_model.lm_head.weight"]))["language_model.lm_head.weight"]
    head = np.frombuffer(head, np.uint16).reshape(201024, 2304)
    rng = np.random.default_rng(12)
    rows = np.sort(rng.choice(201024, a.sample, replace=False))
    proj = f32(bytes(ck["markov_head.hidden_out.weight"])).reshape(2304, 256).astype(np.float64)
    bias = f32(bytes(ck["markov_head.hidden_out.bias"])).astype(np.float64)
    wr = np.frombuffer(bytes(ck["markov_head.rnn_proj.weight"]), np.uint16).reshape(768, 5120)
    want = {"rnn_lo": wr[:, :512], "rnn_a": wr[:, 512:2816], "rnn_b": wr[:, 2816:]}

    def check(name, b):
        r = recs[name]
        if hashlib.sha256(b).hexdigest() != r["sha256"]:
            return "sha"
        if name in ck and bytes(ck[name]) != b:
            return "identity"
        if name in want and np.ascontiguousarray(want[name]).tobytes() != b:
            return "slice"
        if name in ("fold_w", "fold_b"):
            hd = (head[rows].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            if name == "fold_w":
                got = np.frombuffer(b, np.uint16).reshape(201024, 256)[rows]
                ref = rne(hd @ proj)
            else:
                got = np.frombuffer(b, np.uint16)[rows]
                ref = rne(hd @ bias)
            if not np.array_equal(got, ref):
                return f"fold ({int(np.sum(got != ref))} of {got.size} differ)"
        return None

    fails = 0
    missing = [k for k in ck if k not in recs] + [k for k in ("rnn_lo", "rnn_a", "rnn_b", "fold_w", "fold_b") if k not in recs]
    if missing:
        print("MISSING", missing)
        fails += 1
    data = {}
    for name in sorted(recs):
        b = open(os.path.join(d, recs[name]["file"]), "rb").read()
        data[name] = b
        e = check(name, b)
        if e:
            print("FAIL", name, e)
            fails += 1
    # controls: one flipped byte a record kind must be caught
    ctl = 0
    for name in ("layers.0.self_attn.q_proj.weight", "rnn_a", "fold_w", "fold_b", "mask_embed"):
        b = bytearray(data[name])
        pos = int(rows[3]) * (512 if name == "fold_w" else 2) if name.startswith("fold") else len(b) // 3
        b[pos] ^= 0x40
        e = check(name, bytes(b))
        # a corrupted record re-hashed (an attacker / a builder bug that rewrote the manifest)
        recs_sha = recs[name]["sha256"]
        recs[name]["sha256"] = hashlib.sha256(bytes(b)).hexdigest()
        e2 = check(name, bytes(b))
        recs[name]["sha256"] = recs_sha
        caught = e is not None and e2 is not None
        print(f"CONTROL {name}: {'caught' if caught else 'ESCAPED'} ({e} / {e2})")
        ctl += 0 if caught else 1
    ok = fails == 0 and ctl == 0
    print(f"INK12 draft check: {len(recs)} records, {fails} failures, {ctl} controls escaped, fold rows {a.sample}: "
          + ("ALL CHECKS PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
