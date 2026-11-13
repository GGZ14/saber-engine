<script>
  // One request's detail: timings, token counts, where it was seen, and (only with the
  // store-content setting on) the bodies.
  import { api, compact, ms, int, pct } from './api.js';
  let { uid } = $props();
  let r = $state(null);
  let err = $state('');
  $effect(() => {
    r = null;
    err = '';
    api(`/api/usage/requests/${encodeURIComponent(uid)}`).then((x) => (r = x)).catch((e) => (err = e.message));
  });
  const src = (s) => ((s & 1) ? ((s & 2) ? 'console proxy, merged with the engine record' : 'console proxy') : 'engine (direct, found by the poller)');
</script>

{#if err}<p class="note">{err}</p>
{:else if !r}<p class="note">Loading…</p>
{:else}
  <dl class="kv">
    <dt>Request</dt><dd class="mono">{r.uid}</dd>
    <dt>Time</dt><dd>{new Date(r.ts_ms).toLocaleString()}</dd>
    <dt>Endpoint</dt><dd class="mono">{r.endpoint}</dd>
    <dt>Model</dt><dd>{r.model ?? '–'}</dd>
    <dt>API key</dt><dd>{r.key ?? '–'}</dd>
    <dt>Status</dt><dd>{r.status ?? '–'}{r.status === 499 ? ' (client went away)' : ''}</dd>
    <dt>Finish</dt><dd>{r.finish_reason ?? '–'}</dd>
    <dt>Stream</dt><dd>{r.stream == null ? '–' : r.stream ? 'yes' : 'no'}</dd>
    <dt>Thinking</dt><dd>{r.thinking == null ? '–' : r.thinking ? 'on' : 'off'}</dd>
    <dt>Images</dt><dd>{int(r.images)}</dd>
    <dt>Prompt tokens</dt><dd>{int(r.prompt_tokens)}</dd>
    <dt>Cached prefix</dt><dd>{int(r.cached_tokens)}{r.prompt_tokens && r.cached_tokens != null ? ` (${pct(r.cached_tokens / r.prompt_tokens)})` : ''}</dd>
    <dt>Output tokens</dt><dd>{int(r.completion_tokens)}</dd>
    <dt>Time to first token</dt><dd>{ms(r.ttft_ms)}</dd>
    <dt>Decode</dt><dd>{r.decode_tps == null ? '–' : `${compact(r.decode_tps)} tok/s`}</dd>
    <dt>Total latency</dt><dd>{ms(r.latency_ms)}</dd>
    <dt>Recorded by</dt><dd>{src(r.source)}</dd>
    {#if r.engine_id != null}<dt>Engine id</dt><dd class="mono">{r.engine_boot}:{r.engine_id}</dd>{/if}
    {#if r.error}<dt>Error</dt><dd>{r.error}</dd>{/if}
  </dl>
  {#if r.request_body || r.response_body}
    <h3 class="h">Request body</h3>
    <pre class="tail body">{r.request_body ?? '(not stored)'}</pre>
    <h3 class="h">Response</h3>
    <pre class="tail body">{r.response_body ?? '(not stored)'}</pre>
  {:else}
    <p class="note">Bodies are not stored (Usage, settings: "Store request and response content", off by default).</p>
  {/if}
{/if}

<style>
  .mono { font-family: var(--mono); font-size: var(--fs-xs); }
  .h { margin-top: 14px; margin-bottom: 4px; }
  .body { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 320px; overflow: auto; }
</style>
