<script>
  // Profiles: the built-in five (overrides over the stock preset, each documented) and the
  // user's own (save the draft, rename, duplicate, delete, import / export TOML). Using a profile
  // replaces the DRAFT only; review and apply it like any other change.
  import { onMount } from 'svelte';
  import { api, apiText, saveText } from '../api.js';
  import { S, sectionOf, replaceDraft, postText } from '../settings.svelte.js';
  import { guarded } from '../store.svelte.js';

  let { onreview } = $props();
  let data = $state(null);
  let err = $state('');
  let msg = $state('');
  let newName = $state('');
  let newDesc = $state('');
  let renaming = $state(null);
  let renameTo = $state('');
  let importText = $state('');

  const section = $derived(sectionOf(S.draft?.engine?.preset));
  const builtins = $derived(data?.builtin?.[section] ?? []);
  const users = $derived(data?.user ?? []);

  async function load() {
    try { data = await guarded(() => api('/api/profiles')); err = ''; } catch (e) { err = e.message; }
  }
  onMount(load);

  async function run(fn, ok) {
    err = ''; msg = '';
    try { await fn(); msg = ok; await load(); } catch (e) { err = e.message; }
  }
  async function use(p, builtin) {
    await run(async () => {
      const r = await api('/api/profiles/compose', { method: 'POST', body: { config: S.draft, name: p.name, builtin } });
      replaceDraft(r.config);
      onreview?.();
    }, `“${p.name}” loaded into the draft: review the changes and apply.`);
  }
  const saveNew = () => run(() => api('/api/profiles', { method: 'POST', body: { name: newName.trim(), description: newDesc.trim(), config: S.draft } }).then(() => { newName = ''; newDesc = ''; }), 'Saved.');
  const rename = (p) => run(() => api(`/api/profiles/${encodeURIComponent(p.name)}`, { method: 'PUT', body: { name: renameTo.trim() } }).then(() => { renaming = null; }), 'Renamed.');
  const dup = (p) => run(() => api(`/api/profiles/${encodeURIComponent(p.name)}/duplicate`, { method: 'POST', body: {} }), 'Duplicated.');
  async function del(p) {
    if (!confirm(`Delete the profile “${p.name}”?`)) return;
    await run(() => api(`/api/profiles/${encodeURIComponent(p.name)}`, { method: 'DELETE' }), 'Deleted.');
  }
  async function exp(p) {
    try { saveText(`saber-profile-${p.name.replace(/[^a-z0-9-]/gi, '_')}.toml`, await apiText(`/api/profiles/${encodeURIComponent(p.name)}/export`), 'application/toml'); } catch (e) { err = e.message; }
  }
  const imp = () => run(async () => {
    const r = await postText('/api/profiles/import', importText);
    importText = '';
    msg = `Imported ${r.imported.join(', ')}.`;
  }, '');
  async function file(e) {
    const f = e.currentTarget.files?.[0];
    if (f) importText = await f.text();
  }
  const flagsLine = (p) => {
    const e = Object.entries(p.flags || {});
    return e.length ? e.map(([k, v]) => `${k} ${v === true ? '' : v === false ? '(removed)' : Array.isArray(v) ? v.join(',') : v}`.trim()).join('  ') : 'the stock preset, no overrides';
  };
</script>

<div class="page">
  {#if err}<p class="banner b-critical" role="alert">{err}</p>{/if}
  {#if msg}<p class="banner b-good" role="status">{msg}</p>{/if}

  <section class="panel">
    <div class="c-head">
      <div>
        <h2>Built-in profiles</h2>
        <div class="legend"><span class="muted">For <code>saber {section}</code>. A built-in keeps your model and server flags ({(data?.sticky ?? []).slice(0, 4).join(', ')}, …) and replaces the other overrides. No profile changes the GPU power cap.</span></div>
      </div>
    </div>
    <div class="cards">
      {#each builtins as p (p.name)}
        <article class="card">
          <h3>{p.name}</h3>
          <p>{p.description}</p>
          <code class="flags">{flagsLine(p)}</code>
          <div class="acts">
            <button class="btn small" onclick={() => use(p, true)}>Use</button>
            <button class="btn small" onclick={() => exp(p)}>Export</button>
          </div>
        </article>
      {/each}
    </div>
  </section>

  <section class="panel">
    <h2>Your profiles</h2>
    {#if !users.length}<p class="empty">None yet. Save the current draft below, or import a TOML file.</p>{/if}
    {#if users.length}
      <div class="scroll">
        <table>
          <thead><tr><th class="l">Name</th><th class="l">For</th><th class="l">Overrides</th><th class="l">Actions</th></tr></thead>
          <tbody>
            {#each users as p (p.name)}
              <tr>
                <td class="l">
                  {#if renaming === p.name}
                    <form class="inline" onsubmit={(e) => { e.preventDefault(); rename(p); }}>
                      <input type="text" bind:value={renameTo} aria-label="New name" />
                      <button class="btn small" type="submit">Save</button>
                      <button class="btn small" type="button" onclick={() => (renaming = null)}>Cancel</button>
                    </form>
                  {:else}
                    <strong>{p.name}</strong>{#if p.description}<div class="muted d">{p.description}</div>{/if}
                  {/if}
                </td>
                <td class="l"><code>{p.section}</code></td>
                <td class="l ov"><code>{flagsLine(p)}</code></td>
                <td class="l acts">
                  <button class="btn small" onclick={() => use(p, false)} disabled={p.section !== section} title={p.section !== section ? `for saber ${p.section}` : ''}>Use</button>
                  <button class="btn small" onclick={() => { renaming = p.name; renameTo = p.name; }}>Rename</button>
                  <button class="btn small" onclick={() => dup(p)}>Duplicate</button>
                  <button class="btn small" onclick={() => exp(p)}>Export</button>
                  <button class="btn small" onclick={() => del(p)}>Delete</button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  <div class="grid2">
    <section class="panel">
      <h2>Save the draft as a profile</h2>
      <form class="col" onsubmit={(e) => { e.preventDefault(); saveNew(); }}>
        <label>Name <input type="text" bind:value={newName} maxlength="64" required /></label>
        <label>Description <input type="text" bind:value={newDesc} /></label>
        <button class="btn primary" type="submit" disabled={!newName.trim()}>Save profile</button>
      </form>
    </section>
    <section class="panel">
      <h2>Import TOML</h2>
      <div class="col">
        <input type="file" accept=".toml,text/plain,application/toml" onchange={file} aria-label="Profile file" />
        <textarea rows="5" bind:value={importText} placeholder={'name = "my profile"\nsection = "engine"\n[flags]\n"--fn-max-ctx" = 65536'} aria-label="Profile TOML"></textarea>
        <button class="btn" onclick={imp} disabled={!importText.trim()}>Import</button>
      </div>
    </section>
  </div>
</div>

<style>
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; margin-top: 8px; }
  .card { border: 1px solid var(--ring); border-radius: var(--radius); padding: 12px; display: grid; gap: 6px; align-content: start; background: var(--surface); }
  .card h3 { color: var(--ink); text-transform: none; letter-spacing: 0; font-size: var(--fs-md); }
  .card p { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
  .flags { font-size: var(--fs-xs); color: var(--ink-2); background: var(--code-bg); padding: 4px 6px; border-radius: 4px; overflow-wrap: anywhere; }
  .acts { display: flex; gap: 6px; flex-wrap: wrap; }
  .ov code { font-size: var(--fs-xs); white-space: normal; overflow-wrap: anywhere; }
  td.ov { white-space: normal; min-width: 180px; max-width: 360px; }
  td.acts { min-width: 150px; }
  .d { font-size: var(--fs-xs); white-space: normal; }
  .col { display: grid; gap: 8px; }
  .col label { display: grid; gap: 4px; font-size: var(--fs-sm); color: var(--ink-2); }
  .inline { display: flex; gap: 4px; flex-wrap: wrap; }
  .banner { margin: 0; padding: 8px 10px; border-radius: 5px; font-size: var(--fs-sm); }
  .b-good { background: color-mix(in srgb, var(--good) 14%, transparent); }
  .b-critical { background: color-mix(in srgb, var(--critical) 14%, transparent); }
</style>
