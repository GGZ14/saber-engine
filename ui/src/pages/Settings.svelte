<script>
  // Settings: the full configuration UI, generated from the flag schema (/api/schema). Grouped,
  // searchable, typed controls with inline help, default / preset / overridden badges and a reset
  // per flag, the Basic / Advanced / Diagnostic tiers (Basic by default), restart markers, the
  // server's validation at each field, the live memory estimate, profiles, the model manager,
  // the config history, and the review-and-apply drawer (diff + command line + progress).
  import { onMount } from 'svelte';
  import { S, TIERS, setTier, sectionOf, loadSettings, discard, setField, issuesFor, scheduleCheck } from '../lib/settings.svelte.js';
  import { ui } from '../lib/store.svelte.js';
  import { api } from '../lib/api.js';
  import { loadSetup } from '../lib/setup.svelte.js';
  import FlagRow from '../lib/settings/FlagRow.svelte';
  import Estimate from '../lib/settings/Estimate.svelte';
  import ApplyPanel from '../lib/settings/ApplyPanel.svelte';
  import Profiles from '../lib/settings/Profiles.svelte';
  import Models from '../lib/settings/Models.svelte';
  import History from '../lib/settings/History.svelte';
  import Drawer from '../lib/Drawer.svelte';

  const TABS = [
    { id: 'flags', label: 'Engine flags' },
    { id: 'profiles', label: 'Profiles' },
    { id: 'models', label: 'Models' },
    { id: 'history', label: 'History' },
  ];
  let tab = $state('flags');
  let q = $state('');
  let extraText = $state('');

  onMount(() => { loadSettings(); });
  $effect(() => { if (S.draft) extraText = (S.draft.engine.extra_args ?? []).join(' '); });

  const section = $derived(sectionOf(S.draft?.engine?.preset));
  const rank = (t) => TIERS.indexOf(t);
  const portFlag = $derived(section === 'engine' ? '--fn-serve' : '--port');
  const inSection = $derived((S.schema?.flags ?? []).filter((f) => (f.section === section || f.section === 'global') && f.name !== portFlag));
  const needle = $derived(q.trim().toLowerCase());
  const matches = (f) => !needle || f.name.toLowerCase().includes(needle) || f.help.toLowerCase().includes(needle) || f.group.toLowerCase().includes(needle);
  const shown = $derived(inSection.filter((f) => rank(f.tier) <= rank(S.tier) && matches(f)));
  const hiddenMatches = $derived(needle ? inSection.filter((f) => rank(f.tier) > rank(S.tier) && matches(f)).length : 0);
  const groups = $derived.by(() => {
    const order = S.schema?.groups ?? [];
    const by = new Map();
    for (const f of shown) {
      if (!by.has(f.group)) by.set(f.group, []);
      by.get(f.group).push(f);
    }
    return [...by.entries()]
      .sort((a, b) => (order.indexOf(a[0]) + 999) % 999 - (order.indexOf(b[0]) + 999) % 999)
      .map(([g, flags]) => ({ g, flags, n: flags.filter((f) => S.draft?.engine?.flags?.[f.name] !== undefined).length }));
  });
  const tierCounts = $derived(TIERS.map((t) => inSection.filter((f) => rank(f.tier) <= rank(t)).length));
  const overrides = $derived(Object.keys(S.draft?.engine?.flags ?? {}).length);
  const report = $derived(S.check?.report);
  const changed = $derived(S.check && !S.check.unchanged);
  const running = $derived(S.job?.state === 'running');
  const orphanErrors = $derived.by(() => {
    if (!report) return [];
    const shownNames = new Set(shown.map((f) => f.name));
    return [...report.errors, ...report.warnings].filter((i) => i.field !== 'engine.port' && i.field !== 'config' && !shownNames.has(i.field));
  });

  function review() { S.applyOpen = true; }
  async function rerunSetup() {
    try {
      await api('/api/setup/rerun', { method: 'POST', body: {} });
      await loadSetup();
      location.hash = '#setup';
    } catch (e) { S.error = e.message; }
  }
  function setExtra() {
    setField('engine.extra_args', extraText.trim() ? extraText.trim().split(/\s+/) : []);
  }
  function setPreset(e) {
    const p = e.currentTarget.value;
    S.draft.engine.preset = p;
    S.draft.engine.runtime = p === 'mock' ? 'direct' : 'container';
    scheduleCheck(0);
  }
</script>

<div class="page">
  <div class="title-row">
    <h1 class="title">Settings</h1>
    <div class="seg" role="tablist" aria-label="Settings sections">
      {#each TABS as t}
        <button role="tab" aria-selected={tab === t.id} class:on={tab === t.id} onclick={() => (tab = t.id)}>{t.label}</button>
      {/each}
    </div>
    <button class="btn small" onclick={rerunSetup} title="Walk through the setup wizard again (your current settings are the starting point)">Run setup again</button>
  </div>
  {#if S.error}<p class="banner b-critical" role="alert">{S.error}</p>{/if}
  {#if S.job && S.job.state !== 'running' && !S.applyOpen && S.job.state !== 'done'}
    <p class="banner b-{S.job.state === 'rolled-back' ? 'warning' : 'critical'}" role="status">
      <strong>{S.job.state === 'rolled-back' ? 'Auto-rolled back:' : 'Apply failed:'}</strong> {S.job.message}
      <button class="link" onclick={review}>details</button>
    </p>
  {/if}

  {#if !S.draft}
    <p class="empty">Loading the configuration…</p>
  {:else if tab === 'flags'}
    <div class="layout">
      <div class="main">
        <section class="panel">
          <div class="c-head">
            <div>
              <h2>Engine</h2>
              <div class="legend"><span class="muted">{S.path}</span></div>
            </div>
          </div>
          <div class="general">
            <label>Preset
              <select value={S.draft.engine.preset} onchange={setPreset}>
                <option value="flashnext">Flash-Next (saber engine, TP=2)</option>
                <option value="qwen27b">Qwen3.8-27B (saber serve, TP=2)</option>
                <option value="mock">Mock engine (no GPU)</option>
              </select>
            </label>
            <label>Runtime
              <select value={S.draft.engine.runtime} onchange={(e) => setField('engine.runtime', e.currentTarget.value)}>
                <option value="container">container (dev image)</option>
                <option value="direct">direct child</option>
              </select>
            </label>
            <label>Engine port
              <input type="number" min="1" max="65535" value={S.draft.engine.port} onchange={(e) => setField('engine.port', Number(e.currentTarget.value))} />
            </label>
            <label>Drain timeout (s)
              <input type="number" min="0" value={S.draft.engine.drain_timeout_s} onchange={(e) => setField('engine.drain_timeout_s', Number(e.currentTarget.value))} />
            </label>
            <label>Ready timeout (s)
              <input type="number" min="5" value={S.draft.console.ready_timeout_s} onchange={(e) => setField('console.ready_timeout_s', Number(e.currentTarget.value))} />
            </label>
            <label class="chk"><input type="checkbox" checked={S.draft.console.auto_rollback} onchange={(e) => setField('console.auto_rollback', e.currentTarget.checked)} /> Auto-rollback when a config fails to reach ready</label>
            {#if S.draft.engine.preset === 'flashnext'}
              <label class="chk"><input type="checkbox" checked={S.draft.engine.agent_defaults} onchange={(e) => setField('engine.agent_defaults', e.currentTarget.checked)} /> Agent-load defaults (lazy KV, slab, prefill yield, coupled drafting)</label>
            {/if}
            <label class="chk"><input type="checkbox" checked={S.draft.engine.autostart} onchange={(e) => setField('engine.autostart', e.currentTarget.checked)} /> Start the engine with the console</label>
            <label class="wide">Extra arguments (verbatim, space-separated)
              <input type="text" bind:value={extraText} onchange={setExtra} spellcheck="false" placeholder="--fn-route-trace /tmp/x" />
            </label>
          </div>
          {#each [...issuesFor('engine.port'), ...issuesFor('config')] as i}
            <div class="issue {i.level}" role={i.level === 'error' ? 'alert' : undefined}><code>{i.field}</code> {i.message}</div>
          {/each}
        </section>

        <section class="panel filters-bar">
          <input type="search" placeholder="Search {inSection.length} flags…" bind:value={q} aria-label="Search flags" />
          <div class="seg" role="radiogroup" aria-label="Flag tier">
            {#each TIERS as t, i}
              <button role="radio" aria-checked={S.tier === t} class:on={S.tier === t} onclick={() => setTier(t)}>{t[0].toUpperCase() + t.slice(1)} <span class="muted">{tierCounts[i]}</span></button>
            {/each}
          </div>
          <span class="muted cnt">{shown.length} shown · {overrides} overridden</span>
          {#if hiddenMatches}
            <button class="link" onclick={() => setTier('diagnostic')}>{hiddenMatches} more match in the hidden tiers</button>
          {/if}
        </section>

        {#if orphanErrors.length}
          <section class="panel">
            <h3>Issues on flags not shown in this tier</h3>
            {#each orphanErrors as i}<div class="issue {i.level}"><code>{i.field}</code> {i.message}</div>{/each}
          </section>
        {/if}

        {#each groups as grp (grp.g)}
          <details class="panel grp" open={S.tier === 'basic' || !!needle || grp.n > 0}>
            <summary>
              <h2>{grp.g}</h2>
              <span class="muted">{grp.flags.length} flags{grp.n ? ` · ${grp.n} overridden` : ''}</span>
            </summary>
            {#each grp.flags as f (f.section + f.name)}
              <FlagRow flag={f} />
            {/each}
          </details>
        {:else}
          <p class="empty">No flags match “{q}”.</p>
        {/each}
      </div>
      <aside class="side">
        <Estimate />
      </aside>
    </div>
  {:else if tab === 'profiles'}
    <Profiles onreview={review} />
  {:else if tab === 'models'}
    <div class="layout">
      <div class="main"><Models onreview={review} /></div>
      <aside class="side"><Estimate /></aside>
    </div>
  {:else if tab === 'history'}
    <History onprogress={review} />
  {/if}
</div>

{#if S.draft && (changed || running)}
  <div class="applybar" role="region" aria-label="Pending changes">
    <span>
      {#if running}<strong>Applying…</strong>
      {:else}<strong>{S.check.diff.length} change{S.check.diff.length === 1 ? '' : 's'}</strong>
        <span class="muted">· {S.check.restart_required ? 'restart required' : 'no restart'}{report?.errors?.length ? ` · ${report.errors.length} error${report.errors.length === 1 ? '' : 's'}` : ''}</span>
      {/if}
    </span>
    <span class="acts">
      {#if !running}<button class="btn" onclick={discard}>Discard</button>{/if}
      <button class="btn primary" onclick={review}>{running ? 'Show progress' : 'Review and apply'}</button>
    </span>
  </div>
{/if}

<Drawer open={S.applyOpen} title="Review and apply" onclose={() => (S.applyOpen = false)}>
  <ApplyPanel />
</Drawer>

<style>
  .title-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; align-items: start; }
  .main { display: grid; gap: 14px; min-width: 0; }
  .side { position: sticky; top: 104px; display: grid; gap: 14px; }
  @media (max-width: 960px) {
    .layout { grid-template-columns: minmax(0, 1fr); }
    .side { position: static; order: -1; }
    .filters-bar { position: static; }
  }
  .general { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px 14px; }
  .general label { display: grid; gap: 4px; font-size: var(--fs-sm); color: var(--ink-2); align-content: start; }
  .general label.chk { display: flex; align-items: center; gap: 6px; grid-column: span 1; }
  .general label.wide { grid-column: 1 / -1; }
  .filters-bar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; position: sticky; top: 96px; z-index: 2; }
  .filters-bar input[type="search"] { flex: 1 1 220px; }
  .cnt { font-size: var(--fs-sm); }
  .grp summary { display: flex; align-items: baseline; gap: 10px; cursor: pointer; list-style: none; }
  .grp summary::-webkit-details-marker { display: none; }
  .grp summary::before { content: '▸'; color: var(--muted); font-size: var(--fs-sm); }
  .grp[open] summary::before { content: '▾'; }
  .grp[open] summary { margin-bottom: 4px; }
  .issue { font-size: var(--fs-xs); padding: 4px 8px; border-radius: 4px; margin-top: 8px; }
  .issue.error { color: var(--critical); background: color-mix(in srgb, var(--critical) 10%, transparent); }
  .issue.warning { color: var(--ink-2); background: color-mix(in srgb, var(--warning) 16%, transparent); }
  .link { border: 0; background: none; color: var(--accent); cursor: pointer; padding: 0; font-size: var(--fs-sm); }
  .banner { margin: 0; padding: 8px 10px; border-radius: 5px; font-size: var(--fs-sm); }
  .b-warning { background: color-mix(in srgb, var(--warning) 22%, transparent); }
  .b-critical { background: color-mix(in srgb, var(--critical) 14%, transparent); }
  .applybar {
    position: sticky; bottom: 0; z-index: 6; display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;
    margin: 0 auto; max-width: 1280px; padding: 10px 16px; background: var(--surface); border-top: 1px solid var(--ring); box-shadow: 0 -4px 14px rgba(0, 0, 0, 0.08);
  }
  .applybar .acts { display: flex; gap: 8px; }
</style>
