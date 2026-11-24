# Saber Console (`saber ui`)

The console is a browser dashboard that runs in front of the engine. It starts, watches,
reconfigures and stops the engine for you, so after the first launch everything (setup, settings,
monitoring, usage history, API keys) happens in a browser tab. It is additive: `saber serve`,
`saber engine` and the `tools/serve_*.sh` launchers keep working headless exactly as before.

```
browser ──HTTP/SSE──> saber ui (:8090, crates/saber-console)
                        ├─ config store    ~/.config/saber/config.toml (typed, versioned, 0600)
                        ├─ flag schema     read from the CLI's own clap structs (never drifts)
                        ├─ supervisor      spawn / drain / stop / restart, log ring, exit 86
                        ├─ metrics ring    engine /metrics every 5 s, 24 h in memory, + GPU sysfs
                        ├─ usage store     SQLite (~/.local/share/saber/usage.db), rollups
                        ├─ /v1 proxy       OpenAI API with per-label keys and usage capture
                        └─ audit log       who applied / started / stopped what, when
engine child: the same `saber engine --fn-serve` / `saber serve` command line the launchers run
```

"Apply" never edits a running engine: it saves the config, drains, stops, relaunches with the new
command line and waits for `/health`; a config that does not reach ready is rolled back to the last
working one automatically.

## Quick start

```bash
tools/serve_console.sh                  # build the release binary, then the console on http://127.0.0.1:8090/
tools/serve_console.sh --no-build       # reuse the last build
tools/serve_flashnext.sh --ui --no-build   # the same from a launcher: first run seeds config.toml
tools/serve_27b.sh --ui --no-build         #   with that launcher's preset, autostart on
tools/serve_console.sh --mock           # GPU-free: the console over the mock engine (development, demos)
```

Open `http://127.0.0.1:8090/`. From another machine use an SSH tunnel
(`ssh -L 8090:127.0.0.1:8090 server`) or bind the LAN with a token (see Security). The console runs
on the host (it needs no ROCm: the container-built binary runs there) and starts the engine in the
dev image the way the launchers do. Ctrl-C (or SIGTERM) stops the console and the engine it
started. The plain binary works too:

```bash
.cache/target-container/release/saber ui                    # 127.0.0.1:8090, ~/.config/saber/config.toml
saber ui --port 8091 --config /path/config.toml
saber ui --engine mock --mock-port 8099                      # override the stored engine with the mock (not saved)
saber ui --gen-admin-token                                   # make an admin token, store its hash, print it ONCE
saber ui --host 0.0.0.0                                      # LAN bind: refuses to start without a token
saber ui --dump-schema                                       # the flag schema the Settings page is built from
```

The console refuses to start an engine while another Saber server or a GPU lock file holds the
cards (the launchers' guard) and never changes the GPU power cap. A one-card (TP=1) engine is
refused only by what holds its card: a Saber container whose `HIP_VISIBLE_DEVICES` names that card
(or is unset), or the shared or that card's lock file.

The console's presets are the launchers' default boots: `tools/serve_flashnext.sh`, and
`tools/serve_27b.sh` with its default packs (the MACaQ W4A8-iu8 target and drafter) at TP=2 or,
with `--tp 1`, on one card (a test runs the launchers in their `SABER_PRINT_CMD=1` mode and
compares the command lines). The TP is a choice of
its own (the wizard's Model step, the Settings Engine panel, `tools/saber-setup --tp 1`); another
27B pack is a flag override on the Settings page. If your clone is not at `~/saber`, the console
finds it from the binary's location (`<repo>/.cache/target-container/release/saber`); otherwise set
`[engine.paths] repo` in `config.toml`.

```bash
tools/serve_27b.sh --ui --tp 1 --no-build    # first run: seed config.toml with the 27B on one card
DEVICE=1 tools/serve_console.sh --preset qwen27b --tp 1  # the same on card 1
```

## First run: the setup wizard

While `config.toml` is absent or the setup is unfinished, the console opens the wizard:

1. **System check**: GPUs (gfx arch, VRAM, PCIe link, P2P pool), ROCm / KFD access, the container
   runtime and image, disk free on the pack and KV volumes, a conflicting server.
2. **Model**: the catalog (`tools/models.json`), size and fit for this machine; packs download
   with resume, sha256 verification and progress (the list survives a console restart: a running
   download comes back paused, resume continues from its `.part` files). Flash-Next has two
   entries (CAT1 / CAT2): **MACaQ** (`flashnext`, what `tools/serve_flashnext.sh` boots, marked
   default; `Launch80/Qwen3.8-Flash-Next-MACaQ-SABER`) and **MXFP4** (`flashnext-mxfp4`, the
   rollback: its own pack + `--fn-experts-fmt mxfp4`); both read the same 83-shard subset of AMD's
   Quark checkpoint (revision b2091c38) in place under the storage root (210.6 / 213.7 GB in all).
   The 27B asks **how many
   cards**: two (TP=2, each card holds half of the model: the full 262,144-token context and 8
   sequences) or one (TP=1, the whole model on one GPU with the single-card `-tp1` packs: one
   conversation at a time with up to ~128k tokens of context by default, other requests queue;
   2 or 4 sequences serve several at once; the other card stays free), and which card. The console
   counts the supported GPUs and recommends TP=2 with two, TP=1 with one; with one GPU TP=2 is
   refused. Flash-Next runs on two cards only (no single-card pack exists).
3. **Profile**: Balanced / Max throughput / Lowest latency / Long context / Low power, with the
   memory budget computed before anything boots. At TP=1 the same five profiles are sized for
   one card's KV pool; a profile never changes the TP.
4. **Network and security**: ports, LAN exposure (needs an admin token), an API key.
5. **Review and launch**: the exact command line, the boot log streaming, then a ready screen with
   curl and OpenAI-SDK snippets and a link to the playground.

"I'll configure manually" skips it; Settings > Setup runs it again. `tools/saber-setup` (the text
wizard) reads and writes the same `config.toml`, the TP field included (`tools/saber-setup --tp 1`).

## Pages

| Page | What |
|---|---|
| Overview | status banner, live tiles (tok/s, running / waiting, TTFT, update p99, acceptance), alerts, lifetime totals, the log tail |
| Throughput | generation / prompt tok/s, TTFT, step time, ITL, e2e and queue quantiles, TTFT and step heatmaps, finish reasons |
| Speculative | acceptance rate and acceptance per draft position |
| Memory | KV pool, elastic lend / reclaim, expert-tier hit rate by path, tier moves and bytes, PLE reads, the byte ledger |
| Hardware | per-GPU power, temperature, clocks, VRAM (sysfs hwmon), host load |
| Requests | the recent request log and its detail drawer |
| Usage | persistent history: 24 h / 7 d / 30 d averages, trends, monthly totals with a calendar heatmap, top models and keys, cached-prefix savings, an energy / cost estimate, the request table with CSV / JSON export |
| Playground | chat with streaming and a thinking toggle, through the console's `/v1` proxy |
| Logs | live tail, filter, download |
| Settings | every engine flag from the schema (Basic / Advanced / Diagnostic), validation, the memory estimator, profiles, apply with diff, history and rollback, the model manager |
| Security | the admin token, API keys per label, the audit log |

A global time range, auto-refresh and light / dark / system themes apply to every page. Dashboard
panels can be rearranged: drag a panel's grip (the dots at its top) onto another panel, or focus
the grip and use the arrow keys; the order is kept per page in this browser ("Reset the panel
layout" at the bottom of the page forgets it).

## Settings, apply and rollback

- Every change is validated (ranges, cross-flag rules such as tp/ep or a KV tier without a KV
  dir) and shows the diff and the new command line before it is applied.
- **Parallelism** (the Engine panel, the 27B): Two cards (TP=2) or One card (TP=1) and the card.
  It is `engine.tp` / `engine.device` in `config.toml`, not a flag override: TP=1 composes exactly
  `tools/serve_27b.sh --tp 1`'s boot (`--engine-tp 1 --engine-device 0`, no peer, the `-tp1` target
  and drafter packs, `--engine-max-ctx 131072 --engine-max-seqs 1 --engine-decode-batches 1`: one conversation at a time) in
  the container `saber-27b-tp1-serve` with `HIP_VISIBLE_DEVICES=<card>` (inside it the card is
  device 0). Switching moves a target / drafter pack that names one of the catalog's packs to the
  same pack cut for the other TP (a pack is cut for one TP; the validator refuses a mismatch by
  name). Flash-Next shows TP=2 only. A `--engine-tp` override that disagrees with the choice is
  refused; TP=2 on a machine with one supported GPU, or a card that does not exist, too.
- The memory estimate is labelled per card at TP=2 and "one card holds the whole model" at TP=1
  (the unsplit weights of the `-tp1` packs, every KV head, each sequence's whole GDN state), and
  uses the engine's own ledger law (`lean`: budget 31,744 MiB, KV headroom 848 MiB, in place of
  `--engine-budget-mib` / `--engine-kv-headroom-mib`; `--engine-ledger today` keeps the flags).
- **KV cache: context vs sequences.** The estimate follows the engine's own 27B ledger (fitted to
  five TP=1 boots of the MACaQ MXFP4 `-tp1` pair within 0.1 %, and within 1 % of the NVFP4 pair's measured
  168,768 tokens) and shows, beside the figure, what fewer sequences or a shorter context would
  give, computed by the same law. Measured at TP=1 (one R9700, MACaQ MXFP4, 2026-11-14):

  | context | sequences (batches) | KV tokens |
  |---|---|---|
  | 65,536 | 4 (1,2,4) | 127,712 (1 full-length session at a time) |
  | 32,768 | 4 | 127,904 |
  | 16,384 | 4 | 147,552 |
  | 65,536 | 2 (1,2) | 171,248 |
  | 65,536 | 1 (1) | 213,824 |

  | 131,072 | 1 (1), **the TP=1 default** since 2026-11-14 | 192,832 (3,860 blocks = ~62k tokens spare) |
  | 163,840 | 1 | 182,336 (1,156 blocks spare) |
  | 176,128 | 1 | 178,400 (142 blocks spare: the last value that boots) |
  | 180,224 | 1 | refused at boot (the ledger's 11,114 blocks < the 11,265 one sequence needs) |

  Sequences are the lever: each one holds its own model state (the GDN snapshots) on the card.
  A shorter context frees only the drafter's KV pages, which share the pool with the target's
  blocks until the context is small (16k), so halving 64k to 32k gives almost nothing. For one
  long conversation at a time pick the Long context profile (1 sequence); for several users keep
  the sequences and lower the context.
- **The boot check.** The engine refuses to start when one sequence at the max context does
  not fit the ledger's KV blocks (`max_ctx / 16 + 1`); the estimate computes the same check, so a
  context past the ceiling is a field error naming it (at TP=1, MACaQ MXFP4, 1 sequence: ~177k; the
  tip shows the ceiling for each sequence count). The TP=1 default is one conversation of up to
  131,072 tokens with ~62k tokens spare for the prefix cache; the Settings Engine panel's
  Sequences control serves several at once (2 or 4, which share the KV).
- **Decode batches never exceed the sequences**: the engine refuses a batch larger than its
  sequence count, so with fewer sequences than the preset's ladder (1,2,4 at TP=1, 1,2,3,4,8 at
  TP=2) the console trims the ladder (2 sequences -> 1,2; 1 -> 1). Batches you set yourself are
  kept, and one above the sequences is a field error.
- On/off flags whose engine default is on (`--enable-prefix-caching`) are toggles: off writes
  `--enable-prefix-caching off` (a bare `false` in `config.toml` means the same), on needs no flag.
- Apply = save, drain (running + waiting requests reach 0, `engine.drain_timeout_s`), stop,
  start, wait for ready (`console.ready_timeout_s`, default 900 s). A field that does not reach the
  command line saves without a restart.
- Every applied config is in `<data dir>/history/` (default `~/.local/share/saber/console/`); the
  newest one that reached ready is never pruned. A config that exits during load or times out is
  rolled back automatically (`console.auto_rollback`, on by default); any history entry can be
  rolled back by hand.
- Profiles save, duplicate, rename, import and export as TOML.
- Credentials are not config: a config save, an apply or a rollback keeps the current admin
  token and API keys (a rollback never brings a revoked key back). They change on the Security page.

## Usage history

The `/v1` proxy records one row per chat / completion request (time, endpoint, model, key label,
prompt / completion / cached tokens, TTFT at the first token frame, latency, decode tok/s, finish
reason, status, image count, thinking); bodies are not stored unless "Store content" is on.
Requests that reach the engine directly (port 8080) are picked up from its `/v1/requests` ring
and counted once. Writes are batched on a writer thread, never on the request path. Rows are kept
`usage.raw_retention_days` (90); the minute / hour / day rollups forever.

Days and months are cut at a fixed UTC offset (`usage.utc_offset_minutes`), chosen on the Usage
page (a list of offsets, or "use this browser's"); a change applies at once to new rows, days
already totalled keep their cut.

## Security

- **Guarded on the LAN by default.** The first `saber ui` start with no admin token and no host
  chosen makes a token, stores only its hash in the config, keeps a copy for you in
  `~/.config/saber/admin-token` (mode 0600), listens on the LAN (`0.0.0.0:8090`) and prints a
  sign-in link: `http://<this-machine>:8090/#token=...`. Opening the link signs that browser in
  and removes the token from the address bar (a fragment is never sent to any server). While the
  token file still holds the console's token, the link is printed on every start.
- **Remembered browsers.** Signing in (the link, or typing the token) sets a signed `HttpOnly`,
  `SameSite=Strict` cookie that lasts 30 days, so a new tab or a restart stays signed in. The
  cookie is a signature, never the token; it is checked against a per-install secret
  (`<data dir>/session.key`) and the admin credential, so **rotating or removing the admin token
  signs every browser out**. Sign out forgets this browser at once.
- **`--no-auth`** runs an open console (no token, even if one is stored). Beyond the loopback it
  serves only private-network clients (loopback, RFC 1918, link-local) on local hostnames (an IP
  literal, `localhost`, a name without a dot, `.local`, `.lan`, `.home.arpa`, `.internal`), refuses
  a cross-site `Origin`, and prints a warning at start. It is never the default and is not saved.
- **Loopback.** `--host 127.0.0.1` (or `HOST=127.0.0.1`, or `console.host`) keeps the console on
  this machine; there it answers only a `Host` that names the loopback (a DNS-rebinding guard)
  and needs no token. A host you choose that is not a loopback address still refuses to start
  without a token you made (`saber ui --gen-admin-token`).
- **The admin token** guards every `/api/*` route and the console's `/metrics` (only the SPA's
  static files, which include the sign-in screen, are served without it). It is stored as a
  salted SHA-256 hash (`console.admin_token_hash`), never as text: make one with
  `saber ui --gen-admin-token` (printed once) or the Security page's Rotate (shown once, the old one
  stops working at once). `--admin-token <secret>` on the command line is kept in memory only and
  wins over the config. A `config.toml` written by an older build that still holds the token or
  keys as text is rewritten with hashes when the console starts.
- **Sign-in.** The browser asks for the token once per tab and keeps it in `sessionStorage`
  (closing the tab forgets it; "Sign out" in the header forgets it now). Calls send it as
  `Authorization: Bearer`; an `EventSource` sends `?token=`.
- **CSRF.** Every mutating call needs the `X-Saber-Console: 1` header, which a cross-site form or
  `fetch` cannot send without a CORS preflight the console never answers.
- **Rate limit.** An address that presents a wrong token or key 10 times in 60 s is refused (429,
  `Retry-After`) for 60 s, even with the right one. A call with no credential (a signed-out page)
  is refused but not counted.
- **API keys per label** (Security page): each `/v1` key has a label that rides its usage rows; the
  key is shown once and stored as a hash. Revoke refuses it at once; delete removes a revoked key.
  With no keys, `/v1` accepts the admin token, or is open when the console has no token (loopback).
- **Audit log** (`<data dir>/audit.jsonl`, 0600, shown on the Security page): every config save,
  apply and rollback (including the console's own auto-rollbacks), engine start / stop / restart,
  token and key change and wizard step, with who (`admin`, `local` for an open loopback console,
  `console`), the client address, the time and the reply's status. Never a secret.
- **Engine auth.** The engine's own keys (`--api-key LABEL=KEY`, `--api-key-file`) are optional;
  the console sends `proxy.engine_key` (or the engine flag's key labelled `console`, else its first
  key). With engine auth on, the engine keeps a request's `x-saber-console-ref` (the id that lets
  the usage store count a proxied request once) only from the key labelled **`console`**, so a
  client cannot spoof it; label the console's engine key `console`.
- Files: `config.toml`, the history entries and the audit log are owner-only (0600).

## Prometheus and Grafana

The engine's `/metrics` (`:8080`) carries the live families (`saber:*` requests, tokens,
histograms of TTFT / ITL / step / e2e / queue time, speculative acceptance per position, KV;
`saber_xtier_*`, `saber_ple_*`, `saber_ledger_*` on Flash-Next). The console's `/metrics` (`:8090`,
admin token when one is set) exports the long-term usage totals (`saber_console_tokens_total`,
`saber_console_requests_total`, `saber_console_request_errors_total`,
`saber_console_gpu_energy_joules_total`). Import [`tools/console/grafana-saber.json`](../tools/console/grafana-saber.json)
into Grafana (Dashboards > Import, pick your Prometheus): overview stats, throughput and latency
with heatmaps, speculative decoding, the memory plane and the console's usage totals. A test
(`saber-console` `grafana_dashboard`) fails if the dashboard queries a family no exporter writes.

## Proxy overhead

Measured on a two-card Flash-Next boot started by the console, alternating requests straight to
the engine (`:8080`) and through the console's proxy (`:8090/v1`, a per-label key): single-stream
and 8-way throughput, time to first token and prefill showed no measurable difference. The gap
between streamed updates is about 1.5 ms (~7 %) longer at p99 through the proxy, one more hop for
every SSE frame. Latency-sensitive clients can keep calling the engine on `:8080` directly; their
requests still reach the usage history through the poller.

## HTTP API

Every route needs the admin token when one is set; every non-GET needs `X-Saber-Console: 1`.

| Route | What |
|---|---|
| `GET /api/status` | engine state, uptime, model, command |
| `GET/PUT /api/config` | the config (secrets redacted as `(set)`) / save it |
| `POST /api/config/validate`, `POST /api/config/apply`, `GET /api/apply` | check / apply (drain-and-restart) / the apply job |
| `GET /api/history`, `GET /api/history/{id}`, `POST /api/history/{id}/rollback` | config history |
| `POST /api/config/tp` | `{config, tp, device?}`: the draft switched to that TP (the packs moved), not saved |
| `GET/POST /api/profiles`, `PUT/DELETE /api/profiles/{name}`, `POST .../duplicate`, `GET .../export`, `POST /api/profiles/import`, `POST /api/profiles/compose` | profiles |
| `GET /api/schema`, `GET /api/command` | the flag schema / the composed command line |
| `POST /api/engine/start`, `/stop`, `/restart` | engine control |
| `GET /api/logs`, `/api/logs/stream` (SSE), `/api/logs/download` | the engine log ring |
| `GET /api/metrics`, `/api/metrics/latest`, `/api/metrics/stream` (SSE), `/api/series`, `/api/hw` | the metrics ring and hardware samples |
| `GET /api/usage/status`, `/summary?range=`, `/requests`, `/requests/{uid}`, `/monthly`, `/export?format=csv\|json` | usage history |
| `GET /metrics` | the console's Prometheus export |
| `GET /api/packs`, `DELETE /api/packs/{name}`, `GET /api/catalog`, `GET /api/system` | model manager, catalog, system check |
| `GET/POST /api/downloads`, `/api/downloads/stream` (SSE), `POST /api/downloads/{id}/pause\|resume\|cancel`, `POST /api/downloads/clear` | downloads |
| `GET /api/setup`, `POST /api/setup/{goto,model,tp,profile,network,keygen,launch,finish,skip,rerun}` | the wizard (`model` takes an optional `tp`; `tp` = `{tp, device?}`) |
| `GET /api/auth` | who this session is, the token's source, the keys (labels only) |
| `POST /api/auth/admin-token` (rotate, returns the token once), `DELETE /api/auth/admin-token` (loopback only) | the admin token |
| `GET/POST /api/auth/keys`, `POST /api/auth/keys/{label}/revoke`, `DELETE /api/auth/keys/{label}` | API keys (a new key is returned once) |
| `GET /api/audit?limit=` | the audit log, newest first |
| `ANY /v1/*` | the OpenAI API, proxied to the engine (API key or admin token, per the rules above) |

## Files

| Path | What |
|---|---|
| `~/.config/saber/config.toml` | the config (0600; `.prev` = the previous save). Version 5 added `engine.tp` / `engine.device`; a version-4 file is migrated when read (its `--engine-tp 1` overrides from `tools/saber-setup --tp 1` become `tp = 1`, and a prefix-caching `false` becomes `"off"`) |
| `~/.local/share/saber/console/history/` | applied configs |
| `~/.local/share/saber/console/profiles.toml` | user profiles |
| `~/.local/share/saber/console/audit.jsonl` | the audit log |
| `~/.local/share/saber/console/downloads.json` | the download list |
| `~/.local/share/saber/usage.db` | the usage history (SQLite, WAL) |

## Troubleshooting

| You see | Cause |
|---|---|
| `... is not a loopback address ... refusing to start` | a LAN bind without a token: `saber ui --gen-admin-token` first |
| the sign-in screen, then "not this console's admin token" | the token was rotated, or `--admin-token` differs from the config's |
| 429 "too many failed authentication attempts" | 10 wrong tokens / keys from this address in a minute: wait 60 s |
| "the engine is not running" (503) on `/v1` | start it (Overview or `POST /api/engine/start`) |
| start refused: another server holds the cards | stop the other server first (the console never stops it for you) |
| an apply rolled back | the engine exited during load or missed `ready_timeout_s`; the job and the history entry name why |
