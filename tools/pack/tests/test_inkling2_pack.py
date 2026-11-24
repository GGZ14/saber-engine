#!/usr/bin/env python3
"""INK1: unit tests for the Inkling 2 pack builder + checker (numpy only, no checkpoint, no GPU).

    python3 -m unittest tools/pack/tests/test_inkling2_pack.py -v

A tiny synthetic checkpoint in this checkpoint's layout (layer 0 NVFP4 g16 with a MULTIPLIED f32
global, layer 1 MXFP8 g32; 4 experts per rank; H 64, I 32) is built, packed and checked; the checker
must pass, and must fail on a corrupted record. The dequant laws are pinned on hand-made bytes.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_inkling2_pack as bip  # noqa: E402
import check_inkling2_pack as cip  # noqa: E402

H, I, EX = 64, 32, 8  # 8 experts = 4 a rank


def record_layout(h, i):
    """qwen4_exp::pack_plan::ExpertRecordLayout::new (sub-records 256-aligned, stride 4 KiB)."""
    sizes = [2 * i * h // 2, (h // 32) * 2 * i, 2 * i, h * i // 2, (i // 32) * h, h]
    keys = ["gate_up_w", "gate_up_ws", "gate_up_wref", "down_w", "down_ws", "down_wref"]
    off, out = 0, {}
    for k, s in zip(keys, sizes):
        out[k] = [off, s]
        off = -(-(off + s) // 256) * 256
    end = out["down_wref"][0] + out["down_wref"][1]
    out.update(format="MXFP4_E8M0", payload=sum(sizes), stride=-(-end // 4096) * 4096)
    return out


def write_st(path, tensors):
    hdr, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def synth(d, rng):
    t = {}
    p = "language_model.model.layers"
    for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
        b = f"{p}.0.mlp.switch_mlp.{proj}"
        t[b + ".weight"] = ("U32", rng.integers(0, 2**32, (EX, n, k // 8), dtype=np.uint32))
        # E4M3 scales: normal values with small exponents (no NaN 0x7F/0xFF)
        t[b + ".scales"] = ("U8", rng.integers(0x30, 0x48, (EX, n, k // 16), dtype=np.uint8))
        t[b + ".weight_scale_2"] = ("F32", (rng.random(EX).astype(np.float32) * 1e-3 + 1e-4))
        b = f"{p}.1.mlp.switch_mlp.{proj}"
        e4 = rng.integers(0, 0x7E, (EX, n, k), dtype=np.uint8) | (rng.integers(0, 2, (EX, n, k), dtype=np.uint8) << 7)
        t[b + ".weight"] = ("U32", e4.view(np.uint32).reshape(EX, n, k // 4))
        t[b + ".scales"] = ("U8", rng.integers(115, 125, (EX, n, k // 32), dtype=np.uint8))
    t["language_model.model.norm.weight"] = ("BF16", rng.integers(0, 2**16, (H,), dtype=np.uint16))
    t[f"{p}.0.self_attn.q_proj.weight"] = ("BF16", rng.integers(0, 2**16, (8, H), dtype=np.uint16))
    t[f"{p}.0.self_attn.o_proj.weight"] = ("BF16", rng.integers(0, 2**16, (H, 8), dtype=np.uint16))
    write_st(os.path.join(d, "model-00001-of-00001.safetensors"), t)
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    plan = {
        "model": "inkling2_loop", "tp": 2, "ep": 2, "experts_per_rank": EX // 2, "layers": 2,
        "expert_kinds": ["nvfp4_requant_mxfp4", "mxfp8_requant_mxfp4"],
        "geometry": {"hidden": H, "inter": I}, "records": {"mxfp4": record_layout(H, I)},
        "dense": [
            {"tensor": "language_model.model.norm.weight", "dtype": "BF16", "shape": [H], "placement": "shared", "split_dim": None},
            {"tensor": f"{p}.0.self_attn.q_proj.weight", "dtype": "BF16", "shape": [8, H], "placement": "rank", "split_dim": 0},
            {"tensor": f"{p}.0.self_attn.o_proj.weight", "dtype": "BF16", "shape": [H, 8], "placement": "rank", "split_dim": 1},
        ],
    }
    pp = os.path.join(d, "plan.json")
    json.dump(plan, open(pp, "w"))
    return pp


class DequantLaws(unittest.TestCase):
    def test_nvfp4_low_nibble_first_and_the_global_multiplies(self):
        # byte 0x71: low nibble 1 (0.5) is element 0, high nibble 7 (6.0) element 1; E4M3 0x38 = 1.0
        w = np.zeros((1, 2), np.uint32)
        w.view(np.uint8)[0, 0] = 0x71
        s = np.full((1, 1), 0x38, np.uint8)
        codes = np.stack([w.view(np.uint8) & 0xF, w.view(np.uint8) >> 4], -1).reshape(1, -1)
        v = cip.e2m1_bits(codes) * cip.e4m3_bits(s)[0, 0] * np.float32(0.25)
        self.assertEqual(v[0, 0], 0.125)
        self.assertEqual(v[0, 1], 1.5)

    def test_the_independent_decoders_equal_build_packs_tables(self):
        import build_pack as bp
        b = np.arange(256, dtype=np.uint8)
        ok = np.isnan(bp.E4M3) == np.isnan(cip.e4m3_bits(b))
        self.assertTrue(ok.all())
        fin = ~np.isnan(bp.E4M3)
        self.assertTrue(np.array_equal(bp.E4M3[fin].view(np.uint32), cip.e4m3_bits(b)[fin].view(np.uint32)))
        self.assertTrue(np.array_equal(bp.FP4_VALUES.view(np.uint32), cip.e2m1_bits(np.arange(16)).view(np.uint32)))


class BuildAndCheck(unittest.TestCase):
    def test_a_synthetic_pack_round_trips_and_a_corrupt_record_fails(self):
        rng = np.random.default_rng(1120)
        with tempfile.TemporaryDirectory() as d:
            plan = synth(d, rng)
            out = os.path.join(d, "pack")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bip.main(["--checkpoint", d, "--plan", plan, "--out", out]), 0)
            cip.fails.clear()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cip.main([out])
            self.assertEqual(rc, 0, buf.getvalue())
            self.assertIn("ALL CHECKS PASS", buf.getvalue())
            # corrupt one code byte of rank 0's second record (not the table)
            p = os.path.join(out, "rank0", "experts.xtier")
            hdr = open(p, "rb").read(4096)
            doff, = struct.unpack_from("<Q", hdr, 152 + 16)
            stride, = struct.unpack_from("<Q", hdr, 32)
            with open(p, "r+b") as f:
                f.seek(doff + stride + 3)
                b = f.read(1)
                f.seek(doff + stride + 3)
                f.write(bytes([b[0] ^ 0x22]))
            cip.fails.clear()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cip.main([out])
            self.assertEqual(rc, 1)
            self.assertIn("FAIL rank 0: every record's sha256", buf.getvalue())
            self.assertIn("re-derived from the checkpoint == the record", buf.getvalue())
            self.assertIn("FAIL rank 0: 8 experts re-derived", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
