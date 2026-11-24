#!/usr/bin/env python3
"""MACFN8: build THE served MACaQ Flash-Next pack -- one directory the launcher points at.

The pack = the served MXFP4 `-trunk8` pack's members (the same absolute symlinks the
`-trunk8-macaq-shq` gate pack used: dense / trunk8 / hc8 / PLE / MTP / profile / vision / the MXFP4 tp2
and EP expert tiers, so `--fn-experts-fmt mxfp4` on this very pack is the rollback too) + the MACaQ
members as HARDLINKS (same XFS volume: no byte is copied, and the pack survives a prune of the build
directories under /mnt/nvme/models/macfn{3,5}-pack):

  rank<r>/experts.macaq.xtier   MACFN3's routed int4 tier (tag 3, tp2 half records)
  rank<r>/experts.tsh(.json)    the layer-shared gate_up t (+ its sha256 json)
  rank<r>/experts.shq.xtier     MACFN5's shared-expert int4 records (expert 512)
  macaq/routed-plan.json        provenance (the routed tier's builder plan; its sha256 is NOT in the header)
  macaq/shq-plan.json           the shared tier's plan (its sha256 IS in the shq header: m5check's T)
  macaq/shq-stats.json          the shq builder's stats
  macaq/MANIFEST.json           every member: kind (link / hardlink / file), target, size, inode; the xtier
                                headers' table sha256; the small files' sha256 (tools/capture/macfn8/m8check.py)

Usage: python3 build_macaq_served_pack.py --out /mnt/nvme/saber-packs/qwen38-flashnext-macaq-tp2-trunk8
       [--base .../qwen38-flashnext-quark-ep2-trunk8-macaq-shq] [--routed /mnt/nvme/models/macfn3-pack/full]
       [--shq /mnt/nvme/models/macfn5-pack/full]
Refuses an existing --out (never overwrites a pack)."""
import argparse, hashlib, json, os, struct, sys, time

RANK_LINKS = ["aliases.json", "dense.json", "dense.pack", "experts.tp2.xtier", "experts.xtier", "mtp_fp8.json",
              "mtp_fp8.pack", "mtp_x4.json", "mtp_x4.pack", "trunk8.json", "trunk8.pack"]
TOP_LINKS = ["experts_tp2.json", "kv_scales.json", "manifest.json", "plan.json", "ple", "profile", "shared", "vision"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def xtier_table_sha(p):
    b = open(p, "rb").read(4096)
    assert b[:8] == b"SBRXPRT1", p
    return b[184:216].hex(), struct.unpack_from("<Q", b, 176)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2-trunk8-macaq-shq")
    ap.add_argument("--routed", default="/mnt/nvme/models/macfn3-pack/full")
    ap.add_argument("--shq", default="/mnt/nvme/models/macfn5-pack/full")
    a = ap.parse_args()
    if os.path.lexists(a.out):
        sys.exit(f"{a.out} exists: refusing (a pack is never overwritten)")
    if os.stat(os.path.dirname(a.out.rstrip("/"))).st_dev != os.stat(a.routed).st_dev:
        sys.exit("the MACaQ members must be on the pack's volume (hardlinks)")
    man = {"pack": a.out, "built": time.strftime("%F %T %Z"), "law": "MACFN8 served MACaQ pack", "members": {}}
    os.makedirs(os.path.join(a.out, "macaq"))
    for n in TOP_LINKS:
        t = os.readlink(os.path.join(a.base, n))
        os.symlink(t, os.path.join(a.out, n))
        man["members"][n] = {"kind": "link", "target": t}
    for r in (0, 1):
        d = os.path.join(a.out, f"rank{r}")
        os.makedirs(d)
        for n in RANK_LINKS:
            t = os.readlink(os.path.join(a.base, f"rank{r}", n))
            os.symlink(t, os.path.join(d, n))
            man["members"][f"rank{r}/{n}"] = {"kind": "link", "target": t}
        for src, n in ((a.routed, "experts.macaq.xtier"), (a.routed, "experts.tsh"), (a.routed, "experts.tsh.json"),
                       (a.shq, "experts.shq.xtier")):
            s = os.path.join(src, f"rank{r}", n)
            os.link(s, os.path.join(d, n))
            st = os.stat(s)
            m = {"kind": "hardlink", "source": s, "bytes": st.st_size, "inode": st.st_ino}
            if n.endswith(".xtier"):
                m["table_sha256"], fb = xtier_table_sha(s)
                assert fb == st.st_size, (s, fb, st.st_size)
            else:
                m["sha256"] = sha(s)
            man["members"][f"rank{r}/{n}"] = m
    for src, n in ((os.path.join(a.routed, "plan.json"), "routed-plan.json"),
                   (os.path.join(a.shq, "plan.json"), "shq-plan.json"), (os.path.join(a.shq, "stats.json"), "shq-stats.json")):
        dst = os.path.join(a.out, "macaq", n)
        open(dst, "wb").write(open(src, "rb").read())
        man["members"][f"macaq/{n}"] = {"kind": "file", "source": src, "sha256": sha(dst)}
    json.dump(man, open(os.path.join(a.out, "macaq", "MANIFEST.json"), "w"), indent=1, sort_keys=True)
    print(f"MACFN8 served pack {a.out}: {len(man['members'])} members "
          f"({sum(m['kind'] == 'hardlink' for m in man['members'].values())} hardlinks, 0 bytes copied)")


if __name__ == "__main__":
    main()
