# Models: the registry behind `saber models` and `saber serve --model`

`saber models` lists, inspects, downloads and removes models by name; `saber serve --model <name>` serves
one. Both read one registry: **`models.toml`** at the repo root (compiled into the `saber` binary) merged
with **your own file**, `~/.config/saber/models.toml` (`$XDG_CONFIG_HOME/saber/models.toml`). The registry
is data only: adding a model is adding a `[[model]]` block, with no code change.

```bash
saber models                         # = saber models list
saber models list --json             # for scripts
saber models info <name> [--json]
saber models pull [<name>] [--tp 1|2] [--drafter <name>|none] [--dry-run] [--yes]
saber models rm <name> [--tp 1|2] [--dry-run] [--yes]
saber serve [--model <name>] [--tp 1|2] [--device N] [--drafter <name>|none] [--dry-run] [--print-argv]
            [--yes] [--packs-dir DIR] [--runtime container|direct] [--image REF] [engine flags...]
```

`--packs-dir DIR` works on every `saber models` command. `saber serve --model <name> --help` lists the
serve options.

## The schema

```toml
schema = 1                                # optional; this saber reads schema 1
default = "Qwen3.8-27B-MACaQ-W4A8-SABER"  # the model `saber serve` uses without --model (a target)
endpoint = "https://huggingface.co"       # where `repo`s live (HF_ENDPOINT overrides this default)

[[model]]
name = "Qwen3.8-27B-MACaQ-W4A8-SABER"     # unique; letters, digits, - _ . +
aliases = ["w4a8", "default", "27b"]      # also unique across the registry; the first is the short handle
kind = "target"                           # "target" (served) or "drafter" (served beside a target)
description = "..."                       # shown by `info`
repo = "Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER"   # owner/name on the endpoint
revision = "main"                         # optional: a branch, tag or commit
endpoint = "file:///srv/mirror"           # optional: this entry only (an https:// URL or a mirror directory)
served_id = "Qwen3.8-27B-MACaQ-W4A8-SABER"       # targets: the model id clients send (--served-model-name)
drafter = "Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER"  # targets, optional: a drafter entry's name or alias
requires_tag = "macaq_iu8/1"              # optional: the pack format tag the engine must load
license = "Apache-2.0 (...)"              # shown by `info`
files = ["tokenizer.json", "qwen-fixed-v22.3.jinja"]   # targets: files at the repo root every TP needs
tokenizer = "tokenizer.json"              # targets: one of `files`
chat_template = "qwen-fixed-v22.3.jinja"  # targets: one of `files`
engine_args = ["--engine-rectangular-widths"]   # optional: engine flags after the shapes
default_tp = 2                            # optional: the TP when no GPU is visible (else the largest)

[model.tp.2]                              # one table a TP degree (1 and/or 2)
path = "tp2"                              # the pack's folder inside the repo
bytes = 20543143513                       # its size (shown offline; a pull lists the exact sizes)
legacy_dirs = ["qwen38-27b-macaq-w4a8"]   # optional: older flat folder names under the packs root
max_ctx = 262144                          # targets: --engine-max-ctx (tokens a sequence)
max_seqs = 8                              # targets: --engine-max-seqs
prefill_rows = "96,128,...,2048"          # targets, optional: --engine-prefill-rows
decode_batches = "1,2,3,4,8"              # targets: --engine-decode-batches (none above max_seqs)
decode_widths = "1,5,8"                   # targets: --engine-decode-widths (1 without a drafter)
draft_block = 8                           # targets, optional: --engine-draft-block (default 8)
budget_mib = 30500                        # targets, optional: --engine-budget-mib
tail_args = ["--vision", "lazy"]          # optional: engine flags after --port, at this TP only
```

Every entry is checked when the registry loads, and all problems are reported together, each with the
file to fix: duplicate names or aliases, a `repo` that is not `owner/name`, a TP other than 1 or 2, a
`path` with `..`, a target without `served_id`, `tokenizer` or `chat_template` (or with one not listed in
`files`), a `drafter` that names no drafter entry or lacks one of the target's TPs, decode batches larger
than `max_seqs`, a `default` that is not a target, an unknown field (a typo), and a malformed
`requires_tag`. A TOML syntax error names the line.

## Adding your own model

Put a block in `~/.config/saber/models.toml`:

```toml
default = "my-27b"          # optional: make it what `saber serve` serves

[[model]]
name = "my-27b"
aliases = ["mine"]
kind = "target"
repo = "me/my-27b-saber-packs"
served_id = "My-27B"
drafter = "drafter-w4a8"    # reuse a shipped drafter by its alias
requires_tag = "macaq_iu8/1"
files = ["tokenizer.json", "chat_template.jinja"]
tokenizer = "tokenizer.json"
chat_template = "chat_template.jinja"
engine_args = ["--engine-rectangular-widths"]

[model.tp.2]
path = "tp2"
bytes = 20000000000
max_ctx = 131072
max_seqs = 4
decode_batches = "1,2,4"
decode_widths = "1,5,8"
```

Then `saber models info mine`, `saber models pull mine --dry-run`, `saber serve --model mine --dry-run`.
An entry in your file with the same `name` as a shipped one replaces it whole (to change one field, copy
the shipped block and edit it); an alias in your file takes that alias from a shipped entry. The repo
must hold, under each TP's `path`, a Saber pack: `manifest.json` and `rank0.pack` (+ `rank1.pack` at
TP=2), the layout `tools/build_packs.sh` writes.

## Cache layout

```
~/.cache/saber/packs/                       # $SABER_PACKS or --packs-dir, else $XDG_CACHE_HOME/saber/packs
  Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER/
    tokenizer.json  qwen-fixed-v22.3.jinja  # the entry's `files`
    tp2/manifest.json  tp2/rank0.pack  tp2/rank1.pack  tp2/.saber-pull.json
    tp1/...
  Launch80/Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER/
    tp2/...
```

The cache is a partial mirror of each Hugging Face repo, so a cache directory can itself be another
machine's mirror (`endpoint = "file:///path/to/that/cache"`).

- A pull downloads into `<file>.part` and renames when complete; an interrupted pull resumes where it
  stopped when you run the same command again. Before downloading it lists the repo, adds up what is
  missing, and refuses (with the fix) if the disk does not have it plus 2 GB.
- After downloading it checks every file's sha256 (the Hub's LFS hash, or the repo's `SHA256SUMS`), its
  size, every rank file's `SBRPACK1` header, the manifest's `tp_size`, and the `requires_tag` format tag,
  then writes `.saber-pull.json`. A file that fails its hash is deleted, so the next pull fetches it afresh.
- A pack found under one of the entry's `legacy_dirs` directly under the packs root (what
  `tools/fetch_packs.sh` made) is served in place and shown as "downloaded (older layout)". `saber models rm`
  never deletes those; it prints the `rm -r` to run if you want to.
- A complete folder put there by hand (`hf download --local-dir`) is served as "not verified"; `saber models
  pull` then checks its hashes without downloading it again.

## TP handling

A pack is cut for one tensor-parallel degree, so each TP has its own folder. Without `--tp`, `pull` and
`serve` read the GPUs from `/sys/class/kfd` (no GPU is opened): two or more supported (gfx12) GPUs pick
TP=2, one picks TP=1; with no visible GPU (a download-only machine) `pull` takes `default_tp`. `--tp 2` on
a one-GPU machine, or a TP the entry has no pack for, is refused with the command that works. At TP=1,
`--device N` picks the GPU (the container sees only that one, as device 0).

## Drafter sharing

Drafters are their own entries (`kind = "drafter"`) in their own repos, named by targets. A drafter is
downloaded once and shared: `pull` fetches it only if it is not already there, at the target's TP.
`saber models rm <target>` also removes the target's drafter unless another downloaded target names it (it
then says which); removing a drafter directly warns when a downloaded target uses it. `--drafter <name>`
on `pull` and `serve` uses another drafter entry for one run, `--drafter none` serves without one (verify
width 1).

## What `serve --model` runs

It composes the engine's whole command line from the entry (`--engine real`, the TP and devices, the pack,
prebuilt kernels, tokenizer and chat template, the shapes, the drafter, `--served-model-name`, `--host
0.0.0.0 --port 8080`); any engine flag you add replaces the entry's value in place or is appended (an
`--engine-max-seqs` below the decode ladder trims the ladder). The command is checked by the engine's own
argument parser before anything starts, so a misspelled flag is caught with a suggestion. On a host it runs
in the dev image (`podman`, or `docker`) with `/dev/kfd` and `/dev/dri`, host networking, and every path the
command names mounted read-only at the same path, so the engine command reads the same in and out of the
container; inside a container (or with `--runtime direct`) it runs directly. It refuses to start when the
port is taken, another saber container holds the GPUs, the image is not built or no supported GPU is found,
each with the fix. `--dry-run` prints all of this (the engine and container commands and every problem a
real run would stop at) and changes nothing; `--print-argv` prints just the engine argv, one word a line.

## Environment

The front door reads `HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `SABER_PACKS`, `HF_ENDPOINT`, and for
your own Hugging Face login `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or the token file under `HF_HOME`
(`~/.cache/huggingface/token`, what `hf auth login` writes). The token is passed to `curl` on its standard
input, never on a command line, in a file, or in any output; public repos need none. The engine command it
runs is fully explicit and reads no environment variable. Nothing here needs sudo, and only `pull` (and
`serve` of a model that is not downloaded, after you say yes) uses the network.
