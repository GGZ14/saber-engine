<script>
  // The console's credentials and the audit log. The admin token and every /v1 key are
  // stored as salted hashes: a new one is shown ONCE, here, when it is made. Keys are per label
  // (their label rides each usage row) and are revoked (refused at once) before they are deleted.
  import { api, when, setToken } from '../lib/api.js';
  import { copyText } from '../lib/setup.svelte.js';
  import { ui, guarded } from '../lib/store.svelte.js';

  let info = $state(null);
  let audit = $state([]);
  let err = $state('');
  let shown = $state(null); // { kind: 'admin' | 'key', label, secret }
  let newLabel = $state('');
  let copied = $state(false);
  let busy = $state(false);

  async function reload() {
    try {
      info = await guarded(() => api('/api/auth'));
      const a = await guarded(() => api('/api/audit?limit=200'));
      audit = a?.entries ?? [];
      err = '';
    } catch (e) { err = e.message; }
  }
  $effect(() => { ui.tick; reload(); });

  async function act(fn) {
    busy = true;
    err = '';
    try { await fn(); } catch (e) { err = e.message; } finally { busy = false; reload(); }
  }
  function rotate() {
    if (info?.required && !confirm('Make a new admin token? The current one stops working at once (this tab switches to the new one).')) return;
    act(async () => {
      const r = await api('/api/auth/admin-token', { method: 'POST' });
      setToken(r.token);
      shown = { kind: 'admin', label: 'admin', secret: r.token };
    });
  }
  function removeToken() {
    if (!confirm('Remove the admin token? Anyone who can reach this console on this machine can then use it without one.')) return;
    act(async () => {
      await api('/api/auth/admin-token', { method: 'DELETE' });
      setToken('');
    });
  }
  function createKey() {
    const label = newLabel.trim();
    if (!label) return;
    act(async () => {
      const r = await api('/api/auth/keys', { method: 'POST', body: { label } });
      shown = { kind: 'key', label: r.label, secret: r.key };
      newLabel = '';
    });
  }
  function revoke(label) {
    if (!confirm(`Revoke the key "${label}"? Clients using it are refused at once.`)) return;
    act(() => api(`/api/auth/keys/${encodeURIComponent(label)}/revoke`, { method: 'POST' }));
  }
  function del(label) {
    act(() => api(`/api/auth/keys/${encodeURIComponent(label)}`, { method: 'DELETE' }));
  }
  async function copy() {
    if (shown && (await copyText(shown.secret))) { copied = true; setTimeout(() => (copied = false), 1600); }
  }
  const tone = (s) => (s >= 200 && s < 300 ? 'good' : s === 0 ? 'neutral' : 'critical');
</script>

<div class="page">
  <h1 class="title">Security</h1>
  {#if err}<p class="err" role="alert">{err}</p>{/if}

  {#if shown}
    <section class="panel once" aria-live="polite">
      <h2>{shown.kind === 'admin' ? 'New admin token' : `New API key "${shown.label}"`}</h2>
      <p>Copy it now: it is shown <strong>once</strong>. The console keeps only a salted hash of it.</p>
      <div class="row">
        <input type="text" readonly value={shown.secret} aria-label="The new secret" spellcheck="false" />
        <button class="btn small" onclick={copy}>{copied ? 'Copied' : 'Copy'}</button>
        <button class="btn small" onclick={() => (shown = null)}>Done</button>
      </div>
      {#if shown.kind === 'key'}<p class="note">Clients send it as <code>Authorization: Bearer &lt;key&gt;</code> to <code>{location.origin}/v1</code>.</p>{/if}
    </section>
  {/if}

  <section class="panel">
    <div class="c-head"><h2>Admin token</h2></div>
    {#if info}
      <dl class="kv">
        <dt>Status</dt>
        <dd>{info.required ? 'required for every /api call' : 'none (this console answers loopback only)'}</dd>
        <dt>Source</dt>
        <dd>{info.source === 'flag' ? 'the --admin-token flag (change it on the command line)' : info.source === 'config' ? 'config.toml (salted hash)' : '–'}</dd>
        <dt>This session</dt><dd>{info.who ?? '–'}</dd>
        <dt>Failed sign-ins</dt>
        <dd>{info.rate_limit.max_fails} failures in {info.rate_limit.window_s} s lock an address out for {info.rate_limit.lockout_s} s</dd>
      </dl>
      <div class="row">
        <button class="btn primary" disabled={busy || info.source === 'flag'} onclick={rotate}>{info.required ? 'Rotate the token' : 'Set a token'}</button>
        {#if info.required && info.loopback && info.source !== 'flag'}
          <button class="btn" disabled={busy} onclick={removeToken}>Remove the token</button>
        {/if}
      </div>
    {/if}
  </section>

  <section class="panel">
    <div class="c-head">
      <div>
        <h2>API keys</h2>
        <div class="legend"><span class="muted">For the OpenAI-compatible API at /v1. Each key's label is recorded with its requests on the Usage page.</span></div>
      </div>
    </div>
    {#if info}
      {#if info.proxy_open}
        <p class="note">No keys and no admin token: the /v1 API is open to this machine. Make a key to require one.</p>
      {:else if !info.keys.length}
        <p class="note">No keys: /v1 accepts the admin token.</p>
      {/if}
      {#if info.keys.length}
        <div class="scroll">
          <table>
            <thead><tr><th>Label</th><th>Made</th><th>State</th><th></th></tr></thead>
            <tbody>
              {#each info.keys as k}
                <tr>
                  <td>{k.label}</td>
                  <td>{k.created_ms ? when(k.created_ms) : '–'}</td>
                  <td><span class="st st-{k.revoked ? 'critical' : 'good'}">{k.revoked ? 'revoked' : 'active'}</span></td>
                  <td class="acts">
                    {#if !k.revoked}<button class="btn small" disabled={busy} onclick={() => revoke(k.label)}>Revoke</button>
                    {:else}<button class="btn small" disabled={busy} onclick={() => del(k.label)}>Delete</button>{/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
      <form class="row" onsubmit={(e) => { e.preventDefault(); createKey(); }}>
        <input type="text" bind:value={newLabel} placeholder="label, e.g. laptop" aria-label="New key label" maxlength="64" pattern="[A-Za-z0-9._\-]+" />
        <button class="btn primary" type="submit" disabled={busy || !newLabel.trim()}>Make a key</button>
      </form>
      <p class="note">The engine's own key (the console sends it to the engine) {info.engine_key_set ? 'is set' : 'is not set: engine auth is off'}. With engine auth on, label the console's engine key <code>console</code> so the engine trusts the console's request ids.</p>
    {/if}
  </section>

  <section class="panel">
    <div class="c-head">
      <div>
        <h2>Audit log</h2>
        <div class="legend"><span class="muted">Every config save and apply, rollback, engine start / stop / restart, and credential change. {info?.audit_path ?? ''}</span></div>
      </div>
    </div>
    {#if audit.length}
      <div class="scroll">
        <table>
          <thead><tr><th>When</th><th class="l">Who</th><th class="l">Action</th><th>Status</th><th class="l">From</th><th class="l">Detail</th></tr></thead>
          <tbody>
            {#each audit as e}
              <tr>
                <td>{when(e.ts_ms)}</td>
                <td class="l">{e.who}</td>
                <td class="l">{e.action}</td>
                <td><span class="st st-{tone(e.status)}">{e.status || '–'}</span></td>
                <td class="l">{e.ip ?? 'console'}</td>
                <td class="l detail" title={e.detail}>{e.detail}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <p class="empty">Nothing recorded yet.</p>
    {/if}
  </section>
</div>

<style>
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
  .row input { flex: 1 1 220px; min-width: 0; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 0 0 6px; font-size: var(--fs-sm); }
  .kv dt { color: var(--muted); }
  .kv dd { margin: 0; }
  .once { border-color: var(--accent); }
  .once h2 { margin: 0 0 4px; }
  .once p { margin: 0; }
  .err { color: var(--critical); margin: 0; }
  .st { font-size: var(--fs-xs); font-weight: 600; }
  .st-good { color: var(--good, inherit); }
  .st-critical { color: var(--critical); }
  .st-neutral { color: var(--muted); }
  .acts { text-align: right; }
  .detail { max-width: 320px; overflow: hidden; text-overflow: ellipsis; }
  code { font-size: 0.92em; }
</style>
