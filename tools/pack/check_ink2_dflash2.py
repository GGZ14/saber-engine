#!/usr/bin/env python3
"""INK48: check a DFlash2 drafter pack (build_ink2_dflash2.py's output) against its SOURCE, re-reading the record
files themselves (not the builder's memory): the format tag + placement; every record file's sha256 == the
manifest's; every "bf16" record's bytes == the source tensor's bytes; every "w8" pair == INK14's record law
re-derived from the source (build_ink2_dflash2.quant_w8); all 81 source tensors present; no extra records.
CONTROL: one flipped byte in a copy of the first `.w8` record must be caught. numpy only.
INK50: the `iu8` section (the W8A8 form's declaration) == build_ink2_dflash2.iu8_section re-derived from the
SOURCE's w8 tensors (law, record set, n / k / waves); CONTROL 2: a copy of the manifest with one entry's `waves`
changed must be caught.

    check_ink2_dflash2.py --pack DIR --src DIR
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_ink2_dflash2 as B  # noqa: E402


def check(pack, src, files, man=None):
    man = man or json.load(open(os.path.join(pack, "manifest.json")))
    bad = []
    if man.get("format") != B.FORMAT:
        bad.append("format")
    if man.get("ranks") != {"tp2": [0], "tp1": [0]}:
        bad.append("placement")
    recs = man["records"]
    seen = set()
    w8_shapes = {}
    for k, dt, shape, raw in B.tensors(os.path.join(src, "model.safetensors")):
        seen.add(k)
        b = raw.tobytes()
        r = recs.get(k)
        if r is None or r["shape"] != shape or r["form"] != "bf16":
            bad.append(f"{k}: missing / shape / form")
            continue
        fb = open(files(r["file"]), "rb").read()
        if fb != b or hashlib.sha256(fb).hexdigest() != r["sha256"]:
            bad.append(f"{k}: bytes")
        if k in B.W8:
            w8_shapes[k] = list(shape)
            codes, sc = B.quant_w8(b, shape)
            for n, want in ((k + ".w8", codes), (k + ".w8s", sc)):
                seen.add(n)
                rr = recs.get(n)
                if rr is None:
                    bad.append(f"{n}: missing")
                    continue
                got = open(files(rr["file"]), "rb").read()
                if got != want or hashlib.sha256(got).hexdigest() != rr["sha256"]:
                    bad.append(f"{n}: not the record law")
    if man.get("iu8") != B.iu8_section(w8_shapes):
        bad.append("iu8 section")
    extra = set(recs) - seen
    if extra:
        bad.append(f"extra records {sorted(extra)[:4]}")
    return len(recs), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    n, bad = check(a.pack, a.src, lambda f: os.path.join(a.pack, f))
    print(f"CHECK records {n}: {'PASS' if not bad else 'FAIL ' + '; '.join(bad[:6])}")
    # control: a flipped byte in a copy of the first w8 record
    man = json.load(open(os.path.join(a.pack, "manifest.json")))
    first = sorted(r["file"] for r in man["records"].values() if r["form"] == "w8")[0]
    tmp = tempfile.mkdtemp(prefix="ink48chk-")
    try:
        cp = os.path.join(tmp, first)
        shutil.copy(os.path.join(a.pack, first), cp)
        with open(cp, "r+b") as fh:
            fh.seek(12345)
            v = fh.read(1)[0]
            fh.seek(12345)
            fh.write(bytes([v ^ 0x01]))
        _, cbad = check(a.pack, a.src, lambda f: cp if f == first else os.path.join(a.pack, f))
    finally:
        shutil.rmtree(tmp)
    print(f"CHECK control (flipped byte in {first}): {'CAUGHT' if cbad else 'MISSED'}")
    # control 2: one iu8 entry's wave count changed (a different summation order)
    m2 = json.loads(json.dumps(man))
    k0 = sorted(m2["iu8"]["records"])[0]
    m2["iu8"]["records"][k0]["waves"] += 1
    _, ibad = check(a.pack, a.src, lambda f: os.path.join(a.pack, f), man=m2)
    print(f"CHECK control (iu8 waves of {k0} changed): {'CAUGHT' if ibad else 'MISSED'}")
    sys.exit(0 if (not bad and cbad and ibad) else 1)


if __name__ == "__main__":
    main()
