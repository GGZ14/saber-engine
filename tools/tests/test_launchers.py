#!/usr/bin/env python3
"""Tests for the 27B launcher's print mode (``SABER_PRINT_CMD=1 tools/serve_27b.sh``: the engine argv
for each weights / drafter / TP choice and the refusals) and for ``tools/fetch_packs.sh``'s W4A8-iu8
targets against a stand-in Hugging Face CLI (the download layout, the flat pack folders, the checks).

    python3 -m unittest discover -s tools/tests -v

No network, no GPU, no container: the print mode exits before any of them, and the fetch runs a fake
``hf`` that writes small pack files into the staging folder it is pointed at.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SERVE = os.path.join(REPO, "tools", "serve_27b.sh")
FETCH = os.path.join(REPO, "tools", "fetch_packs.sh")


def serve(*args, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("WEIGHTS", "DRAFTER", "TP", "TIER", "DRAFT", "MODEL", "PORT")}
    e.update(SABER_PRINT_CMD="1", SABER_CONTAINER="true", SABER_PACKS="/nonexistent/saber-packs")
    e.update(env)
    return subprocess.run(["bash", SERVE, "--no-build", *args], env=e, capture_output=True, text=True)


def read(path):
    with open(path) as fh:
        return fh.read()


def flags(out):
    """The argv lines as {flag: value} (a switch maps to None)."""
    words = out.split()
    d, i = {}, 2  # binary, subcommand
    while i < len(words):
        k = words[i]
        if i + 1 < len(words) and not words[i + 1].startswith("--"):
            d[k] = words[i + 1]
            i += 2
        else:
            d[k] = None
            i += 1
    return d


class ServePrintTest(unittest.TestCase):
    def boot(self, *args, **env):
        r = serve(*args, **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        return flags(r.stdout)

    def test_the_default_is_the_w4a8_pack_and_drafter(self):
        f = self.boot()
        self.assertEqual(f["--pack"], "/packs/qwen38-27b-macaq-w4a8")
        self.assertEqual(f["--engine-draft-pack"], "/packs/qwen38-dflash2-drafter-w4a8")
        self.assertEqual(f["--served-model-name"], "Qwen3.8-27B-MACaQ-W4A8-SABER")
        self.assertEqual(f["--engine-decode-widths"], "1,5,8")
        self.assertEqual((f["--engine-tp"], f["--engine-max-ctx"]), ("2", "262144"))
        self.assertEqual(f["--tokenizer"], "/work/.cache/quark27b.tokenizer.json")
        # WEIGHTS / DRAFTER in the environment name the same boot
        self.assertEqual(self.boot(WEIGHTS="macaq-w4a8", DRAFTER="w4a8"), f)

    def test_one_card(self):
        f = self.boot("--tp", "1")
        self.assertEqual(f["--pack"], "/packs/qwen38-27b-macaq-w4a8-tp1")
        self.assertEqual(f["--engine-draft-pack"], "/packs/qwen38-dflash2-drafter-w4a8-tp1")
        self.assertEqual((f["--engine-tp"], f["--engine-max-ctx"], f["--engine-max-seqs"]), ("1", "131072", "1"))

    def test_the_fp8_drafter_beside_the_w4a8_target(self):
        for tp, sfx in (("2", ""), ("1", "-tp1")):
            f = self.boot("--tp", tp, "--drafter", "fp8")
            self.assertEqual(f["--pack"], f"/packs/qwen38-27b-macaq-w4a8{sfx}")
            self.assertEqual(f["--engine-draft-pack"], f"/packs/qwen38-dflash2-drafter-fp8{sfx}")
            self.assertEqual(f["--served-model-name"], "Qwen3.8-27B-MACaQ-W4A8-SABER")

    def test_the_mxfp4_pack_stays_selectable(self):
        for name in ("macaq", "macaq-mxfp4"):
            for tp, sfx in (("2", ""), ("1", "-tp1")):
                f = self.boot("--weights", name, "--tp", tp)
                self.assertEqual(f["--pack"], f"/packs/qwen38-27b-macaq7-bc3{sfx}")
                self.assertEqual(f["--engine-draft-pack"], f"/packs/qwen38-dflash2-drafter-quarkhead{sfx}")
                self.assertEqual(f["--served-model-name"], "Qwen3.8-27B-MACaQ-MXFP4-SABER")
                self.assertEqual(f["--engine-decode-widths"], "1,5,8")
            # its drafter is the FP8 one, so --drafter fp8 names the same boot
            self.assertEqual(self.boot("--weights", name, "--drafter", "fp8"), self.boot("--weights", name))

    def test_the_other_weights(self):
        f = self.boot("--weights", "quark")
        self.assertEqual(f["--pack"], "/packs/qwen38-27b-quark-awq-mxfp4")
        self.assertEqual((f["--served-model-name"], f["--engine-decode-widths"]), ("Qwen3.8", "1,8"))
        f = self.boot("--weights", "nvfp4", "--tp", "1")
        self.assertEqual(f["--pack"], "/packs/qwen38-27b-nvfp4-devfinal-tp1-fp8head")
        self.assertEqual(f["--engine-draft-pack"], "/packs/qwen38-dflash2-drafter-tp1-fp8head")

    def test_overrides(self):
        f = self.boot(TIER="qwen38-27b-macaq-w4a8/tp2", DRAFT="qwen38-dflash2-drafter-w4a8/tp2", MODEL="m", PORT="8100")
        self.assertEqual(f["--pack"], "/packs/qwen38-27b-macaq-w4a8/tp2")
        self.assertEqual(f["--engine-draft-pack"], "/packs/qwen38-dflash2-drafter-w4a8/tp2")
        self.assertEqual((f["--served-model-name"], f["--port"]), ("m", "8100"))

    def test_refusals(self):
        for args, msg in (
            (["--weights", "macaq", "--drafter", "w4a8"], "paired with --weights macaq-w4a8"),
            (["--weights", "nvfp4", "--drafter", "fp8"], "no drafter choice"),
            (["--drafter", "int4"], "w4a8 (the default) or fp8"),
            (["--weights", "bogus"], "macaq-w4a8 (the default)"),
            (["--tp", "3"], "TP=2 (two cards) or TP=1"),
        ):
            r = serve(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(msg, r.stderr, args)
            self.assertEqual(r.stdout, "", args)


FAKE_HF = r"""#!/usr/bin/env bash
# a stand-in `hf download REPO --local-dir DIR --include PAT...`: logs the call, writes what the
# patterns name (tpN/* -> a pack folder, else the named files)
set -eu
echo "$*" >> "$FAKE_HF_LOG"
[ "$1" = download ] || exit 3
repo=$2; shift 2
dir=""; pats=()
while [ $# -gt 0 ]; do
  case "$1" in --local-dir) dir=$2; shift 2 ;; --include) pats+=("$2"); shift 2 ;; *) exit 4 ;; esac
done
mkdir -p "$dir/.cache/huggingface"
for p in "${pats[@]}"; do
  case "$p" in
    tp2/*) mkdir -p "$dir/tp2"; for r in 0 1; do printf 'SBRPACK1%s' "$repo" > "$dir/tp2/rank$r.pack"; done; echo '{}' > "$dir/tp2/manifest.json" ;;
    tp1/*) mkdir -p "$dir/tp1"; printf 'SBRPACK1%s' "$repo" > "$dir/tp1/rank0.pack"; echo '{}' > "$dir/tp1/manifest.json" ;;
    *) printf '%s' "$repo" > "$dir/$p" ;;
  esac
done
"""


class FetchTest(unittest.TestCase):
    """`tools/fetch_packs.sh 27b-macaq-w4a8[-tp1]` with a fake `hf` (no network)."""

    def run_fetch(self, tmp, target, **env):
        bin_dir = os.path.join(tmp, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        hf = os.path.join(bin_dir, "hf")
        with open(hf, "w") as fh:
            fh.write(FAKE_HF)
        os.chmod(hf, os.stat(hf).st_mode | stat.S_IEXEC)
        # a copy of the repo's tools/ so the script's .cache/ lands in the scratch dir
        repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "tools"), exist_ok=True)
        dst = os.path.join(repo, "tools", "fetch_packs.sh")
        with open(FETCH) as src, open(dst, "w") as out:
            out.write(src.read())
        e = {k: v for k, v in os.environ.items() if k not in ("DRAFTER", "SABER_HF_OWNER")}
        e.update(PATH=bin_dir + os.pathsep + e.get("PATH", ""), SABER_PACKS=os.path.join(tmp, "packs"),
                 FAKE_HF_LOG=os.path.join(tmp, "hf.log"))
        e.update(env)
        r = subprocess.run(["bash", dst, target], env=e, capture_output=True, text=True)
        log = read(e["FAKE_HF_LOG"]).splitlines() if os.path.exists(e["FAKE_HF_LOG"]) else []
        return r, log, os.path.join(tmp, "packs"), repo

    def test_default_tp2(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, log, packs, repo = self.run_fetch(tmp, "27b-macaq-w4a8")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ready: tools/serve_27b.sh\n", r.stdout)
            self.assertEqual(sorted(os.listdir(packs)), ["qwen38-27b-macaq-w4a8", "qwen38-dflash2-drafter-w4a8"])
            for n in ("qwen38-27b-macaq-w4a8", "qwen38-dflash2-drafter-w4a8"):
                self.assertEqual(sorted(os.listdir(os.path.join(packs, n))), ["manifest.json", "rank0.pack", "rank1.pack"])
            self.assertTrue(read(os.path.join(packs, "qwen38-dflash2-drafter-w4a8", "rank0.pack"))
                            .endswith("Launch80/Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER"))
            self.assertEqual(log[0].split()[1], "Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER")
            self.assertIn("--include tp2/*", log[0])
            self.assertTrue(os.path.isfile(os.path.join(repo, ".cache", "quark27b.tokenizer.json")))
            self.assertTrue(os.path.isfile(os.path.join(repo, ".cache", "template_corpus.jinja")))
            # a second run keeps what is there and downloads only the tokenizer again
            r, log2, _, _ = self.run_fetch(tmp, "27b-macaq-w4a8")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("already there", r.stdout)
            self.assertEqual(len(log2) - len(log), 1)

    def test_all_fetches_the_default_27b(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, log, packs, _ = self.run_fetch(tmp, "all")
            # the Flash-Next half fails on the fake's empty repo; the 27B half ran first
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("27B MACaQ W4A8 packs ready", r.stdout)
            self.assertEqual([l.split()[1] for l in log[:2]],
                             ["Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER", "Launch80/Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER"])

    def test_tp1_with_the_fp8_drafter(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, log, packs, _ = self.run_fetch(tmp, "27b-macaq-w4a8-tp1", DRAFTER="fp8")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ready: tools/serve_27b.sh --tp 1 --drafter fp8", r.stdout)
            self.assertEqual(sorted(os.listdir(packs)), ["qwen38-27b-macaq-w4a8-tp1", "qwen38-dflash2-drafter-fp8-tp1"])
            self.assertEqual(log[1].split()[1], "Launch80/Qwen3.8-27B-DFlash2-Drafter-FP8-SABER")
            self.assertIn("--include tp1/*", log[1])

    def test_a_nested_folder_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "packs", "qwen38-27b-macaq-w4a8", "tp2")
            os.makedirs(nested)
            r, log, _, _ = self.run_fetch(tmp, "27b-macaq-w4a8")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("holds no rank0.pack", r.stderr)
            self.assertEqual(log, [])
            self.assertTrue(os.path.isdir(nested))

    def test_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, _, _, _ = self.run_fetch(tmp, "bogus")
            self.assertEqual(r.returncode, 2)
            self.assertIn("27b-macaq-w4a8 = the default 27B", r.stderr)


if __name__ == "__main__":
    unittest.main()
