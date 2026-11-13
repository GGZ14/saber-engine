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
SABER_CONSOLE=1 tools/serve_flashnext.sh --no-build   # the same from a launcher: first run seeds config.toml
SABER_CONSOLE=1 tools/serve_27b.sh --no-build         #   with that launcher's preset, autostart on
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
cards (the launchers' guard) and never changes the GPU power cap.

The console's presets are the launchers' default boots: `tools/serve_flashnext.sh` and
`tools/serve_27b.sh` at TP=2 with the MACaQ pack (a test runs both launchers in their
`SABER_PRINT_CMD=1` mode and compares the command lines). Another 27B pack, or the single-card
(TP=1) form, is a set of flag overrides on the Settings page; `tools/saber-setup --tp 1` writes
them for you. If your clone is not at `~/saber`, the console finds it from the binary's location
(`<repo>/.cache/target-container/release/saber`); otherwise set `[engine.paths] repo` in
`config.toml`.

## First run: the setup wizard

While `config.toml` is absent or the setup is unfinished, the console opens the wizard:

1. **System check**: GPUs (gfx arch, VRAM, PCIe link, P2P pool), ROCm / KFD access, the container
   runtime and image, disk free on the pack and KV volumes, a conflicting server.
2. **Model**: the catalog (`tools/models.json`), size and fit for this machine; packs download
   with resume, sha256 verification and progress (the list survives a console restart: a running
   download comes back paused, resume continues from its `.part` files).
3. **Profile**: Balanced / Max throughput / Lowest latency / Long context / Low power, with the
   memory budget computed before anything boots.
4. **Network and security**: ports, LAN exposure (needs an admin token), an API key.
5. **Review and launch**: the exact command line, the boot log streaming, then a ready screen with
   curl and OpenAI-SDK snippets and a link to the playground.

"I'll configure manually" skips it; Settings > Setup runs it again. `tools/saber-setup` (the text
wizard) reads and writes the same `config.toml`.

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

- **Loopback by default.** `saber ui` binds 127.0.0.1 and then only answers a `Host` that names
  the loopback (a DNS-rebinding guard). Any other bind refuses to start without an admin token.
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
| `GET/POST /api/profiles`, `PUT/DELETE /api/profiles/{name}`, `POST .../duplicate`, `GET .../export`, `POST /api/profiles/import`, `POST /api/profiles/compose` | profiles |
| `GET /api/schema`, `GET /api/command` | the flag schema / the composed command line |
| `POST /api/engine/start`, `/stop`, `/restart` | engine control |
| `GET /api/logs`, `/api/logs/stream` (SSE), `/api/logs/download` | the engine log ring |
| `GET /api/metrics`, `/api/metrics/latest`, `/api/metrics/stream` (SSE), `/api/series`, `/api/hw` | the metrics ring and hardware samples |
| `GET /api/usage/status`, `/summary?range=`, `/requests`, `/requests/{uid}`, `/monthly`, `/export?format=csv\|json` | usage history |
| `GET /metrics` | the console's Prometheus export |
| `GET /api/packs`, `DELETE /api/packs/{name}`, `GET /api/catalog`, `GET /api/system` | model manager, catalog, system check |
| `GET/POST /api/downloads`, `/api/downloads/stream` (SSE), `POST /api/downloads/{id}/pause\|resume\|cancel`, `POST /api/downloads/clear` | downloads |
| `GET /api/setup`, `POST /api/setup/{goto,model,profile,network,keygen,launch,finish,skip,rerun}` | the wizard |
| `GET /api/auth` | who this session is, the token's source, the keys (labels only) |
| `POST /api/auth/admin-token` (rotate, returns the token once), `DELETE /api/auth/admin-token` (loopback only) | the admin token |
| `GET/POST /api/auth/keys`, `POST /api/auth/keys/{label}/revoke`, `DELETE /api/auth/keys/{label}` | API keys (a new key is returned once) |
| `GET /api/audit?limit=` | the audit log, newest first |
| `ANY /v1/*` | the OpenAI API, proxied to the engine (API key or admin token, per the rules above) |

## Files

| Path | What |
|---|---|
| `~/.config/saber/config.toml` | the config (0600; `.prev` = the previous save) |
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
