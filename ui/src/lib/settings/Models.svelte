<script>
  // The model manager: the catalog's models with Download (resumable, verified; one at a time)
  // and the packs installed under the configured packs dir, their size, which the config
  // serves, a switch that puts the pick into the DRAFT (`--pack` / `--engine-draft-pack`) to
  // review and apply, and Delete behind a type-the-name confirmation (never the active pack).
  import { onMount, onDestroy } from 'svelte';
  import { api, bytes, when } from '../api.js';
  import { S, setFlag } from '../settings.svelte.js';
  import { guarded } from '../store.svelte.js';
  import { W, loadCatalog, followDownloads, gb } from '../setup.svelte.js';
  import DownloadRow from '../DownloadRow.svelte';

  let unfollow = null;
  onMount(() => { loadCatalog(); unfollow = followDownloads(); });
  onDestroy(() => unfollow?.());
  // a finished download shows up as an installed pack
  $effect(() => { W.catalog; load(); });

  let del = $state(null);
  let delName = $state('');
  let delErr = $state('');
  let deleting = $state(false);
  function askDelete(p) { del = p; delName = ''; delErr = ''; }
  async function doDelete() {
    deleting = true;
    try {
      const r = await guarded(() => api(`/api/packs/${encodeURIComponent(del.name)}`, { method: 'DELETE', body: { confirm: delName } }));
      msg = `Deleted ${r.deleted} (${bytes(r.freed_bytes)} freed).`;
      del = null;
      load();
      loadCatalog();
    } catch (e) { delErr = e.message; } finally { deleting = false; }
  }

  let { onreview } = $props();
  let data = $state(null);
  let err = $state('');
  let msg = $state('');

  async function load() {
    try { data = await guarded(() => api('/api/packs')); err = ''; } catch (e) { err = e.message; }
  }
  onMount(load);

  const preset = $derived(S.draft?.engine?.preset);
  const gpu = $derived(preset && preset !== 'mock');
  const draftFlag = $derived(preset === 'qwen27b' ? '--engine-draft-pack' : null);

  function use(p, as) {
    const flag = as === 'drafter' ? draftFlag : '--pack';
    setFlag(flag, p.arg);
    msg = `${p.name} set as the ${as} in the draft (${flag} ${p.arg}): review and apply.`;
    onreview?.();
  }
  const draftPick = (flag) => S.draft?.engine?.flags?.[flag];
</script>

<div class="page">
  <section class="panel">
    <div class="c-head">
      <div>
        <h2>Download models</h2>
        <div class="legend"><span class="muted">{W.catalog ? `catalog: ${W.catalog.source} · ${gb(W.catalog.free_bytes)} free at ${W.catalog.roots.packs}` : 'Loading the catalog…'}</span></div>
      </div>
      <button class="btn small" onclick={loadCatalog}>Refresh</button>
    </div>
    {#if W.error}<p class="banner b-critical" role="alert">{W.error}</p>{/if}
    {#each (W.catalog?.models ?? []).filter((m) => m.components.length) as m (m.id)}
      <div class="cat">
        <div class="cat-h">
          <strong>{m.name}</strong> <span class="badge">{m.kind}</span>
          <span class="badge fit-{m.fit.verdict}">{m.fit.verdict === 'fits' ? 'fits this machine' : m.fit.verdict === 'tight' ? 'fits, with a caveat' : 'does not fit'}</span>
          {#if m.installed}<span class="badge on">installed</span>{/if}
        </div>
        {#if m.fit.reasons.length}<div class="muted e">{m.fit.reasons.join(' · ')}</div>{/if}
        {#each m.components as c (c.id)}<DownloadRow model={m} comp={c} />{/each}
      </div>
    {:else}
      {#if W.catalog}<p class="empty">The catalog lists no downloadable model.</p>{/if}
    {/each}
  </section>

  <section class="panel">
    <div class="c-head">
      <div>
        <h2>Installed packs</h2>
        <div class="legend"><span class="muted">{data?.dir ?? ''}</span></div>
      </div>
      <button class="btn small" onclick={load}>Rescan</button>
    </div>
    {#if err}<p class="banner b-critical" role="alert">{err}</p>{/if}
    {#if msg}<p class="banner b-good" role="status">{msg}</p>{/if}
    {#if !gpu}
      <p class="note">The mock engine serves no pack; switching here applies to the GPU presets.</p>
    {/if}
    {#if data && !data.packs.length}
      <p class="empty">No packs under {data.dir}.</p>
    {:else if data}
      <div class="scroll">
        <table>
          <thead>
            <tr><th class="l">Pack</th><th class="l"></th><th class="l">Role</th><th>Size</th><th>Weights / card</th><th>TP</th><th class="l">Model</th><th class="l">Built</th></tr>
          </thead>
          <tbody>
            {#each data.packs as p (p.name)}
              <tr class:active={p.active_as}>
                <td class="l">
                  <strong>{p.name}</strong>
                  {#if p.active_as}<span class="badge on">active {p.active_as}</span>{/if}
                  {#if draftPick('--pack') === p.arg || (draftFlag && draftPick(draftFlag) === p.arg)}<span class="badge">in draft</span>{/if}
                  {#if p.error}<div class="muted e">{p.error}</div>{/if}
                </td>
                <td class="l acts">
                  {#if p.role === 'drafter'}
                    <button class="btn small" disabled={!draftFlag || p.active_as === 'drafter'} onclick={() => use(p, 'drafter')}>Use as drafter</button>
                  {:else}
                    <button class="btn small" disabled={!gpu || p.active_as === 'target'} onclick={() => use(p, 'target')}>Use</button>
                  {/if}
                  <button class="btn small danger" disabled={!!p.active_as} title={p.active_as ? 'The engine is configured to serve this pack: switch first' : 'Delete this pack from disk'} onclick={() => askDelete(p)}>Delete</button>
                </td>
                <td class="l">{p.role}{p.has_vision ? ' · vision' : ''}</td>
                <td title={p.referenced_bytes ? `+ ${bytes(p.referenced_bytes)} of checkpoint referenced in place` : ''}>
                  {bytes(p.size_bytes)}{#if p.referenced_bytes}<div class="muted e">+{bytes(p.referenced_bytes)} referenced</div>{/if}
                </td>
                <td>{p.rank_weight_bytes != null ? bytes(p.rank_weight_bytes) : '–'}</td>
                <td>{p.tp_size ?? '–'}</td>
                <td class="l">{p.model ?? '–'}</td>
                <td class="l">{p.built_unix ? when(p.built_unix * 1000) : '–'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
</div>

{#if del}
  <div class="modal-bg" role="presentation" onclick={(e) => { if (e.target === e.currentTarget) del = null; }}>
    <div class="modal panel" role="dialog" aria-modal="true" aria-labelledby="del-h">
      <h2 id="del-h">Delete {del.name}?</h2>
      <p>This removes <strong>{bytes(del.size_bytes)}</strong> from <code>{del.path}</code>. It cannot be undone; you can download it again from the catalog.</p>
      <label class="confirm">Type the pack's name to confirm
        <input type="text" bind:value={delName} placeholder={del.name} spellcheck="false" autocomplete="off" />
      </label>
      {#if delErr}<p class="banner b-critical" role="alert">{delErr}</p>{/if}
      <div class="mact">
        <button class="btn" onclick={() => (del = null)}>Keep it</button>
        <button class="btn danger-fill" disabled={delName !== del.name || deleting} onclick={doDelete}>{deleting ? 'Deleting…' : 'Delete'}</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .cat { padding: 8px 0 4px; border-top: 1px solid var(--grid); }
  .cat:first-of-type { border-top: 0; }
  .cat-h { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .fit-fits { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good-text); }
  .fit-tight { background: color-mix(in srgb, var(--warning) 22%, transparent); }
  .fit-no-fit { background: color-mix(in srgb, var(--critical) 14%, transparent); color: var(--critical); }
  .acts { display: flex; gap: 6px; }
  .danger { color: var(--critical); }
  .danger-fill { background: var(--critical); border-color: var(--critical); color: #fff; }
  .modal-bg { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.45); display: grid; place-items: center; z-index: 20; padding: 16px; }
  .modal { max-width: 460px; width: 100%; display: grid; gap: 10px; }
  .modal p { margin: 0; font-size: var(--fs-sm); overflow-wrap: anywhere; }
  .confirm { display: grid; gap: 4px; font-size: var(--fs-sm); }
  .mact { display: flex; justify-content: flex-end; gap: 8px; }
  .badge { font-size: 0.6875rem; padding: 1px 6px; border-radius: 9px; background: var(--surface-2); color: var(--ink-2); margin-left: 6px; }
  .badge.on { background: var(--accent-fill); color: var(--accent); font-weight: 600; }
  tr.active td { background: var(--accent-fill); }
  .e { font-size: var(--fs-xs); }
  .banner { margin: 8px 0; padding: 8px 10px; border-radius: 5px; font-size: var(--fs-sm); }
  .b-good { background: color-mix(in srgb, var(--good) 14%, transparent); }
  .b-critical { background: color-mix(in srgb, var(--critical) 14%, transparent); }
</style>
