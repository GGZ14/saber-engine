#!/usr/bin/env python3
"""Unit tests for ``tools/saber-setup`` (the terminal wizard): the shared model catalog, the
shared config.toml (read, write back only the owned keys, migrate setup.json) and ``--print``.

    python3 -m unittest discover -s tools/tests -v

No terminal, no network, no GPU: everything runs against temporary files.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
SCRIPT = os.path.join(TOOLS, "saber-setup")


def load_module():
    loader = importlib.machinery.SourceFileLoader("saber_setup", SCRIPT)
    spec = importlib.util.spec_from_loader("saber_setup", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


S = load_module()


class Paths:
    """Point the module's config paths at a temporary dir for one test."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def __enter__(self):
        self.saved = (S.CONF_DIR, S.CONF, S.CONSOLE_CONF)
        S.CONF_DIR = self.dir
        S.CONF = os.path.join(self.dir, "setup.json")
        S.CONSOLE_CONF = os.path.join(self.dir, "config.toml")
        return self

    def __exit__(self, *a):
        S.CONF_DIR, S.CONF, S.CONSOLE_CONF = self.saved
        self.tmp.cleanup()


class Catalog(unittest.TestCase):
    def test_models_come_from_the_shared_catalog(self):
        self.assertEqual(sorted(S.MODELS), ["27b", "flashnext"])  # the mock has no launcher
        with open(S.CATALOG) as f:
            cat = json.load(f)
        self.assertEqual(cat["version"], 1)
        m = S.MODELS["27b"]
        for k in ("title", "launcher", "container", "packs", "extra_files", "download_gb", "note"):
            self.assertIn(k, m)
        self.assertEqual(m["launcher"], "tools/serve_27b.sh")
        self.assertEqual(S.MODELS["flashnext"]["packs"], ["qwen38-flashnext-quark-trunk8"])


class SharedConfig(unittest.TestCase):
    def test_migrates_setup_json_into_config_toml(self):
        with Paths() as p:
            legacy = S.default_cfg("27b")
            legacy.update(ctx27="131072", seqs="4", vision27="lazy", prefix=True, port="8100")
            with open(S.CONF, "w") as f:
                json.dump(legacy, f)
            self.assertIsNone(S.read_console())
            cfg = S.load_cfg()
            self.assertEqual(cfg["ctx27"], "131072")
            S.save_cfg(cfg)
            with open(S.CONSOLE_CONF, "rb") as f:
                doc = tomllib.load(f)
            self.assertEqual(doc["version"], S.CONSOLE_CONFIG_VERSION)
            e = doc["engine"]
            self.assertEqual(e["preset"], "qwen27b")
            self.assertEqual(e["runtime"], "container")
            self.assertEqual(e["port"], 8100)
            self.assertEqual(
                e["flags"],
                {"--engine-max-ctx": 131072, "--engine-max-seqs": 4, "--vision": "lazy"},
            )
            self.assertEqual(doc["setup"], {"model": "27b", "completed": True, "step": "done"})
            mode = stat.S_IMODE(os.stat(S.CONSOLE_CONF).st_mode)
            self.assertEqual(mode, 0o600)
            # from now on config.toml is the source, whatever setup.json says
            with open(S.CONF, "w") as f:
                json.dump(S.default_cfg("flashnext"), f)
            self.assertEqual(S.load_cfg()["model"], "27b")
            self.assertEqual(S.load_cfg()["seqs"], "4")

    def test_round_trip_both_models(self):
        for model, changes in [
            ("27b", dict(ctx27="65536", seqs="2", vision27="on", prefix=False, port="8081")),
            ("flashnext", dict(visionfn="off", thinking=False, ptree=False, routes=True, ctxfn="131072")),
            ("flashnext", {}),
            ("27b", {}),
        ]:
            cfg = S.default_cfg(model)
            cfg["model"] = model
            cfg.update(changes)
            doc, notes = S.console_from_cfg(cfg)
            self.assertEqual(notes, [])
            text = S.dump_toml(doc)
            back = S.cfg_from_console(tomllib.loads(text))
            for k in [o["key"] for o in S.options_for(model)] + ["model"]:
                self.assertEqual(str(back[k]), str(cfg[k]), f"{model} {k}: {text}")
            if not changes:
                # the recommended settings are the presets' own (prefix caching is the engine's default)
                flags = tomllib.loads(text)["engine"].get("flags", {})
                self.assertEqual(flags, {}, text)

    def test_tp1_round_trip_and_back_to_tp2(self):
        cfg = S.default_cfg("27b")
        S.set_tp(cfg, "1")
        cfg["gpu1"] = "1"
        doc, notes = S.console_from_cfg(cfg)
        self.assertEqual(notes, [])
        text = S.dump_toml(doc)
        flags = tomllib.loads(text)["engine"]["flags"]
        self.assertEqual(flags["--engine-tp"], 1)
        self.assertEqual(flags["--engine-device"], 1)
        self.assertIs(flags["--engine-peer-device"], False)
        self.assertEqual(flags["--engine-max-ctx"], int(S.TP1_CTX_DEFAULT))
        self.assertEqual(flags["--engine-max-seqs"], int(S.TP1_SEQS_DEFAULT))
        self.assertTrue(flags["--pack"].endswith("-tp1"), flags)
        self.assertTrue(flags["--engine-draft-pack"].endswith("-tp1"), flags)
        back = S.cfg_from_console(tomllib.loads(text))
        self.assertEqual((back["tp"], back["gpu1"], back["ctx27"], back["seqs"]),
                         ("1", "1", S.TP1_CTX_DEFAULT, S.TP1_SEQS_DEFAULT))
        # back to TP=2: the single-card overrides go, the preset's boot remains
        S.set_tp(back, "2")
        doc2, _ = S.console_from_cfg(back, tomllib.loads(text))
        self.assertEqual(doc2["engine"].get("flags", {}), {})

    def test_writes_back_only_the_keys_it_owns(self):
        with Paths():
            existing = (
                'version = 4\n'
                '[engine]\npreset = "flashnext"\nport = 8080\nagent_defaults = false\n'
                '[engine.paths]\nnvme = "/data/nvme"\n'
                '[engine.flags]\n"--fn-kv-dtype" = "fp8"\n"--fn-thinking" = "off"\n'
                '[usage]\nraw_retention_days = 30\n'
                '[[proxy.keys]]\nlabel = "ops"\nkey = "sk-saber-0123456789"\n'
                '[console]\nadmin_token = "a-very-long-admin-token"\n'
                '[setup]\ncompleted = false\nstep = "profile"\nprofile = "Balanced"\n'
            )
            with open(S.CONSOLE_CONF, "w") as f:
                f.write(existing)
            cfg = S.load_cfg()
            self.assertEqual(cfg["model"], "flashnext")
            self.assertFalse(cfg["thinking"])
            self.assertEqual(cfg["packs"], "/data/nvme/saber-packs")
            cfg["thinking"] = True
            cfg["routes"] = True
            S.save_cfg(cfg)
            with open(S.CONSOLE_CONF, "rb") as f:
                doc = tomllib.load(f)
            self.assertTrue(os.path.exists(S.CONSOLE_CONF + ".prev"))
            e = doc["engine"]
            self.assertFalse(e["agent_defaults"])
            self.assertEqual(e["flags"]["--fn-kv-dtype"], "fp8")  # not ours: kept
            self.assertNotIn("--fn-thinking", e["flags"])  # back to the preset's
            self.assertEqual(e["flags"]["--fn-route-record"], "/data/nvme/saber-kv/routes")
            self.assertEqual(doc["usage"]["raw_retention_days"], 30)
            self.assertEqual(doc["proxy"]["keys"], [{"label": "ops", "key": "sk-saber-0123456789"}])
            self.assertEqual(doc["console"]["admin_token"], "a-very-long-admin-token")
            self.assertEqual(doc["setup"]["profile"], "Balanced")
            self.assertTrue(doc["setup"]["completed"])
            # switching the model to the other subcommand drops the other flag set
            cfg["model"] = "27b"
            S.save_cfg(cfg)
            with open(S.CONSOLE_CONF, "rb") as f:
                doc = tomllib.load(f)
            self.assertEqual(doc["engine"]["preset"], "qwen27b")
            self.assertNotIn("--fn-kv-dtype", doc["engine"].get("flags", {}))

    def test_toml_writer_shapes(self):
        doc = {"version": 4, "a": {"b": {"c": [1, 2]}, "s": "x\"y", "f": 1.5, "t": True},
               "k": [{"label": "a", "key": "b"}], "e": {}}
        back = tomllib.loads(S.dump_toml(doc))
        self.assertEqual(back, doc)

    def test_print_reads_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, SABER_SETUP_DIR=d)
            r = subprocess.run([sys.executable, "-I", SCRIPT, "--print", "--model", "flashnext"],
                               capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Launch command:", r.stdout)
            self.assertIn("Settings from defaults", r.stdout)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()
