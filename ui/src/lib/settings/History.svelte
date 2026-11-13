<script>
  // Config history: every applied config (the newest 50 kept, in the console data dir) with its
  // outcome; view one, or roll back to it in one click (through the same apply flow, so a
  // rollback that fails to reach ready is itself rolled back).
  import { onMount } from 'svelte';
  import { api, when } from '../api.js';
  import { S, pollJob } from '../settings.svelte.js';
  import { guarded, ui } from '../store.svelte.js';
  import Drawer from '../Drawer.svelte';

  let { onprogress } = $props();
  let data = $state(null);
  let err = $state('');
  let open = $state(null);

  async function load() {
    try { data = await guarded(() => api('/api/history')); err = ''; } catch (e) { err = e.message; }
  }
  onMount(load);
  $effect(() => { S.job?.state; load(); });

  async function view(m) {
    try { open = await api(`/api/history/${m.id}`); } catch (e) { err = e.message; }
  }
  async function rollback(m) {
    if (!confirm(`Roll back to history #${m.id}? The engine restarts on that config if its command line differs.`)) return;
    try {
      S.job = await api(`/api/history/${m.id}/rollback`, { method: 'POST', body: {} });
      open = null;
      onprogress?.();
      await pollJob();
    } catch (e) {
      err = e.message;
    }
  }
  const cls = { ready: 'good', failed: 'critical', pending: 'warning', saved: 'neutral' };
</script>

<div class="page">
  <section class="panel">
    <div class="c-head">
      <div>
        <h2>Config history</h2>
        <div class="legend"><span class="muted">{data?.dir ?? ''} · the newest 50 kept, and always the newest that reached ready</span></div>
      </div>
      <button class="btn small" onclick={load}>Refresh</button>
    </div>
    {#if err}<p class="banner" role="alert">{err}</p>{/if}
    {#if data && !data.entries.length}
      <p class="empty">Nothing applied yet. Every apply is recorded here.</p>
    {:else if data}
      <div class="scroll">
        <table>
          <thead><tr><th class="l">#</th><th class="l">When</th><th class="l">Origin</th><th class="l">Outcome</th><th class="l">Note</th><th class="l"></th></tr></thead>
          <tbody>
            {#each data.entries as m (m.id)}
              <tr>
                <td class="l">{m.id}</td>
                <td class="l">{when(m.ts_ms)}</td>
                <td class="l">{m.origin}</td>
                <td class="l"><span class="status-pill st-{cls[m.outcome] ?? 'neutral'}"><i></i>{m.outcome}</span></td>
                <td class="l note-c">{m.note}{#if m.message}<div class="muted">{m.message}</div>{/if}</td>
                <td class="l acts">
                  <button class="btn small" onclick={() => view(m)}>View</button>
                  <button class="btn small" onclick={() => rollback(m)} disabled={S.job?.state === 'running'}>Roll back</button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
</div>

<Drawer open={!!open} title={open ? `History #${open.meta.id} (${open.meta.outcome})` : ''} onclose={() => (open = null)}>
  {#if open}
    <dl class="kv">
      <dt>When</dt><dd>{when(open.meta.ts_ms)}</dd>
      <dt>Origin</dt><dd>{open.meta.origin}</dd>
      {#if open.meta.message}<dt>Message</dt><dd>{open.meta.message}</dd>{/if}
    </dl>
    <h3 class="h">Command line</h3>
    <pre class="tail w">{open.meta.command}</pre>
    <h3 class="h">config.toml</h3>
    <pre class="tail w">{open.toml}</pre>
    <button class="btn primary" onclick={() => rollback(open.meta)} disabled={S.job?.state === 'running' || ui.status == null}>Roll back to this</button>
  {/if}
</Drawer>

<style>
  .note-c { white-space: normal; min-width: 220px; max-width: 420px; overflow-wrap: anywhere; }
  .w { white-space: pre-wrap; overflow-wrap: anywhere; }
  .h { margin: 14px 0 6px; }
  .banner { margin: 8px 0; padding: 8px 10px; border-radius: 5px; font-size: var(--fs-sm); background: color-mix(in srgb, var(--critical) 14%, transparent); }
</style>
