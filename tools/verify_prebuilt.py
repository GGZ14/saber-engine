#!/usr/bin/env python3
"""Check that the code objects in prebuilt/ are what the sources in this repo compile to.

    tools/verify_prebuilt.py --table                 # which compiler built each shipped object
    tools/verify_prebuilt.py --rebuilt DIR           # compare a rebuild (saber-kbuild --out DIR)

Why not compare the files: two builds of the SAME source with the SAME compiler differ in the
dynamic symbol-table sections (.dynstr / .dynsym / .gnu.hash / .hash), so a whole-file hash can
never match. The machine code does match: this compares .text, .rodata and
.AMDGPU.gpr_maximums (the code, the kernel descriptors and the register limits) and ignores the
symbol tables, .dynamic (whose only variable entry is DT_STRSZ, the size of the symbol-name string
table, which moves by a byte between builds), the notes and .comment.

A shipped object is compared only with a rebuild by the SAME compiler (the producer string each
object records); an object built by a different compiler is reported, not compared, because
different compilers legitimately emit different code. Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import os
import re
import struct
import sys

CODE_SECTIONS = (".text", ".rodata", ".AMDGPU.gpr_maximums")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def elf_of(data: bytes) -> bytes:
    """The gfx1201 ELF inside a clang offload bundle (or the ELF itself)."""
    if data[:4] == b"\x7fELF":
        return data
    if data[:24] != b"__CLANG_OFFLOAD_BUNDLE__":
        raise ValueError("neither an ELF nor an offload bundle")
    (n,) = struct.unpack_from("<Q", data, 24)
    o = 32
    for _ in range(n):
        off, size, idl = struct.unpack_from("<QQQ", data, o)
        o += 24
        ident = data[o : o + idl].decode()
        o += idl
        if "gfx1201" in ident:
            return data[off : off + size]
    raise ValueError("no gfx1201 entry")


def sections(path: str) -> dict[str, str]:
    e = elf_of(open(path, "rb").read())
    (shoff,) = struct.unpack_from("<Q", e, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", e, 0x3A)
    raw = [struct.unpack_from("<IIQQQQ", e, shoff + i * shentsize) for i in range(shnum)]
    stro = raw[shstrndx][4]
    out = {}
    for name, typ, _flags, _addr, off, size in raw:
        if typ == 8:  # NOBITS
            continue
        n = e[stro + name : e.index(b"\0", stro + name)].decode()
        if n in CODE_SECTIONS:
            out[n] = hashlib.sha256(e[off : off + size]).hexdigest()
    return out


def producer(path: str) -> str:
    data = open(path, "rb").read()
    m = re.search(rb"AMD clang version [^\0\n]+", data)
    return m.group(0).decode() if m else "unknown"


def short(p: str) -> str:
    m = re.match(r"AMD clang version (\S+) \(\S+ ([0-9a-f]{8})[0-9a-f]*(\+PATCHED)?", p)
    return f"clang {m.group(1)} {m.group(2)}{' +PATCHED' if m.group(3) else ''}" if m else p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prebuilt", default=os.path.join(REPO, "prebuilt", "gfx1201"))
    ap.add_argument("--rebuilt", help="directory of rebuilt .hsaco files (saber-kbuild --out)")
    ap.add_argument("--table", action="store_true", help="list the compiler of every shipped object")
    a = ap.parse_args()

    shipped = sorted(glob.glob(os.path.join(a.prebuilt, "*.hsaco")))
    if not shipped:
        print(f"no code objects under {a.prebuilt}", file=sys.stderr)
        return 2
    if a.table or not a.rebuilt:
        groups = collections.defaultdict(list)
        for p in shipped:
            groups[producer(p)].append(os.path.basename(p)[: -len(".hsaco")])
        for prod, stems in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(stems):3d} objects  {prod}")
            if a.table:
                print("     " + ", ".join(sorted(stems)))
        if not a.rebuilt:
            return 0

    ok = bad = skipped = missing = 0
    for p in shipped:
        stem = os.path.basename(p)[: -len(".hsaco")]
        q = os.path.join(a.rebuilt, stem + ".hsaco")
        if not os.path.exists(q):
            missing += 1
            continue
        if producer(p) != producer(q):
            skipped += 1  # built by a different compiler: not comparable
            continue
        s, r = sections(p), sections(q)
        if s == r:
            ok += 1
        else:
            bad += 1
            print(f"DIFFERS  {stem}: {[k for k in CODE_SECTIONS if s.get(k) != r.get(k)]}")
    print(f"code identical: {ok}   differs: {bad}   different compiler (not compared): {skipped}   not rebuilt: {missing}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
