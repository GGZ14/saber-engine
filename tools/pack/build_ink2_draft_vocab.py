#!/usr/bin/env python3
"""INK18: the drafter's frequency-ranked vocabulary (`--ink-draft-vocab N`).

Counts tokens over local text groups with the checkpoint's own tokenizer, normalises each group's
counts to sum 1, sums the groups, and writes `<pack>/draft/vocab_rank.u32`: a permutation of the
token ids 0..VOCAB-1 (u32 LE) -- every added (special) token first, then by summed frequency
(descending), then the never-seen ids in id order. The engine drafts over the first N ids of it.
A draft over a subset is lossless under the drawn-row walk (the target decides every committed
token); a subset only moves acceptance. Provenance: `vocab_rank.json` beside it.

    build_ink2_draft_vocab.py --tokenizer TOK.json --out PACK/draft \
        --group 'code=/repo/crates/**/*.rs,/usr/lib/python3.12/**/*.py' --group 'chat=/corp/*.txt'
(runs where `tokenizers` is importable, e.g. the dev image)
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import struct

VOCAB = 200_058
ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--group", action="append", required=True, help="NAME=GLOB[,GLOB...]")
ap.add_argument("--max-file-chars", type=int, default=4_000_000)
a = ap.parse_args()

from tokenizers import Tokenizer  # noqa: E402

tk = Tokenizer.from_file(a.tokenizer)
tj = json.load(open(a.tokenizer))
special = sorted(t["id"] for t in tj.get("added_tokens", []) if t["id"] < VOCAB)
score = collections.defaultdict(float)
prov = {}
for g in a.group:
    name, pats = g.split("=", 1)
    files = sorted({f for p in pats.split(",") for f in glob.glob(p, recursive=True) if os.path.isfile(f)})
    cnt = collections.Counter()
    chars = 0
    for f in files:
        try:
            t = open(f, encoding="utf-8", errors="strict").read()[: a.max_file_chars]
        except (UnicodeDecodeError, OSError):
            continue
        chars += len(t)
        for i in range(0, len(t), 200_000):
            cnt.update(tk.encode(t[i : i + 200_000], add_special_tokens=False).ids)
    tot = sum(cnt.values())
    for k, v in cnt.items():
        if k < VOCAB:
            score[k] += v / tot
    prov[name] = {"patterns": pats, "files": len(files), "chars": chars, "tokens": tot, "distinct": len(cnt)}
    print(f"VOCAB group {name}: files {len(files)} chars {chars} tokens {tot} distinct {len(cnt)}")
sp = set(special)
seen = sorted((k for k in score if k not in sp), key=lambda k: (-score[k], k))
rest = [k for k in range(VOCAB) if k not in sp and k not in score]
order = special + seen + rest
assert sorted(order) == list(range(VOCAB))
blob = struct.pack(f"<{VOCAB}I", *order)
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "vocab_rank.u32"), "wb").write(blob)
json.dump({"schema": "ink18-draft-vocab/1", "vocab": VOCAB, "special": len(special), "seen": len(seen),
           "sha256": hashlib.sha256(blob).hexdigest(), "groups": prov},
          open(os.path.join(a.out, "vocab_rank.json"), "w"), indent=1)
print(f"VOCAB wrote {a.out}/vocab_rank.u32: {len(special)} special + {len(seen)} seen + {len(rest)} unseen")
