<script>
  // One flag: its typed control, where its value comes from (default / preset / overridden),
  // the restart marker, inline help, a reset, and the server's errors and warnings for it.
  import { S, setFlag, issuesFor, effectiveOf, valueText, parseValue } from '../settings.svelte.js';

  let { flag } = $props();
  let more = $state(false);

  const override = $derived(S.draft?.engine?.flags?.[flag.name]);
  const overridden = $derived(override !== undefined);
  const eff = $derived(effectiveOf(flag.name));
  const issues = $derived(issuesFor(flag.name));
  const hasError = $derived(issues.some((i) => i.level === 'error'));
  const id = $derived(`f-${flag.section}-${flag.name.replace(/[^a-z0-9]/gi, '')}`);

  // what the engine gets when there is no override: the preset's / agent default's value, else
  // clap's default
  const inherited = $derived.by(() => {
    if (eff && !overridden) {
      const vs = eff.values.map((v) => (v == null ? 'on' : v));
      return { text: vs.join(', '), src: eff.source === 'agent-default' ? 'agent default' : eff.source };
    }
    if (flag.default != null && flag.default !== '') return { text: flag.default, src: 'default' };
    return null;
  });
  const badge = $derived(overridden ? 'overridden' : eff ? (eff.source === 'agent-default' ? 'agent default' : eff.source) : 'default');

  // bool: effective on/off
  const boolOn = $derived(overridden ? override === true : !!eff);
  // an on/off flag whose `false` means `off` (the engine default is on): no "remove" choice
  const onOff = $derived(!!S.schema?.on_off_flags?.includes(flag.name));
  const enumValue = $derived.by(() => {
    if (!overridden) return '';
    if (typeof override === 'boolean') return onOff ? (override ? 'on' : 'off') : override ? '' : '__off';
    return String(override);
  });

  function onText(e) {
    const v = parseValue(flag, e.currentTarget.value);
    setFlag(flag.name, v);
  }
  function onBool(e) {
    setFlag(flag.name, e.currentTarget.checked);
  }
  function onEnum(e) {
    const v = e.currentTarget.value;
    if (v === '') setFlag(flag.name, undefined);
    else if (v === '__off') setFlag(flag.name, false);
    else setFlag(flag.name, v);
  }
  const summary = $derived(flag.help.split(/(?<=[.;])\s/)[0]);
  const hasMore = $derived((flag.long_help || flag.help).length > summary.length + 2);
  const step = $derived(flag.kind === 'float' ? 'any' : '1');
  const inputType = $derived(['int', 'uint', 'float'].includes(flag.kind) && !flag.multiple ? 'number' : 'text');
</script>

<div class="row" class:err={hasError} class:ovr={overridden}>
  <div class="head">
    <label class="name" for={id}><code>{flag.name}</code></label>
    <span class="badge b-{badge.replace(' ', '-')}">{badge}</span>
    {#if flag.tier !== 'basic'}<span class="badge tier">{flag.tier}</span>{/if}
    {#if flag.requires_restart}
      <span class="restart" title="Changing this restarts the engine (drain, stop, start)">
        <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true"><path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        restart
      </span>
    {/if}
    {#if overridden}
      <button class="btn small reset" onclick={() => setFlag(flag.name, undefined)} aria-label="Reset {flag.name} to its default">Reset</button>
    {/if}
  </div>
  <div class="ctl">
    {#if flag.kind === 'bool'}
      <label class="switch">
        <input {id} type="checkbox" checked={boolOn} onchange={onBool} />
        <span>{boolOn ? 'on' : 'off'}</span>
      </label>
    {:else if flag.kind === 'enum'}
      <select {id} value={enumValue} onchange={onEnum}>
        <option value="">{inherited ? `${inherited.text} (${inherited.src})` : '(engine default)'}</option>
        {#each flag.choices as c}<option value={c}>{c}</option>{/each}
        {#if !onOff}<option value="__off">(remove the flag)</option>{/if}
      </select>
    {:else}
      <input
        {id}
        type={inputType}
        {step}
        min={flag.min ?? undefined}
        max={flag.max ?? undefined}
        value={overridden ? valueText(override) : ''}
        placeholder={inherited ? `${inherited.text} (${inherited.src})` : flag.multiple ? 'comma-separated values' : '(engine default)'}
        onchange={onText}
        aria-invalid={hasError}
        spellcheck="false"
        autocomplete="off"
      />
      {#if flag.min != null || flag.max != null}<span class="muted rng">{flag.min ?? ''}…{flag.max ?? ''}</span>{/if}
    {/if}
  </div>
  <div class="help">
    {summary}
    {#if hasMore}
      <button class="link" onclick={() => (more = !more)} aria-expanded={more}>{more ? 'less' : 'more'}</button>
    {/if}
    {#if more}<p class="long">{flag.long_help || flag.help}</p>{/if}
  </div>
  {#each issues as i}
    <div class="issue {i.level}" role={i.level === 'error' ? 'alert' : undefined}>{i.message}</div>
  {/each}
</div>

<style>
  .row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 320px); gap: 4px 16px; padding: 10px 0; border-bottom: 1px solid var(--grid); }
  .row:last-child { border-bottom: 0; }
  .head { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; min-width: 0; }
  .name code { font-size: var(--fs-sm); font-weight: 600; color: var(--ink); overflow-wrap: anywhere; }
  .ctl { display: flex; align-items: center; gap: 6px; grid-row: span 2; justify-content: flex-end; }
  .ctl input:not([type="checkbox"]), .ctl select { width: 100%; }
  .help { font-size: var(--fs-xs); color: var(--ink-2); min-width: 0; overflow-wrap: anywhere; }
  .long { margin: 6px 0 0; white-space: pre-wrap; color: var(--ink-2); }
  .badge { font-size: 0.6875rem; padding: 1px 6px; border-radius: 9px; background: var(--surface-2); color: var(--ink-2); white-space: nowrap; }
  .b-overridden { background: var(--accent-fill); color: var(--accent); font-weight: 600; }
  .tier { background: transparent; border: 1px solid var(--grid); }
  .restart { display: inline-flex; align-items: center; gap: 3px; font-size: 0.6875rem; color: var(--muted); }
  .reset { padding: 1px 8px; font-size: var(--fs-xs); }
  .rng { font-size: var(--fs-xs); white-space: nowrap; }
  .switch { display: inline-flex; align-items: center; gap: 6px; font-size: var(--fs-sm); cursor: pointer; }
  .link { border: 0; background: none; color: var(--accent); cursor: pointer; padding: 0 2px; font-size: inherit; }
  .issue { grid-column: 1 / -1; font-size: var(--fs-xs); padding: 4px 8px; border-radius: 4px; }
  .issue.error { color: var(--critical); background: color-mix(in srgb, var(--critical) 10%, transparent); }
  .issue.warning { color: var(--ink-2); background: color-mix(in srgb, var(--warning) 16%, transparent); }
  .err .name code { color: var(--critical); }
  @media (max-width: 640px) {
    .row { grid-template-columns: minmax(0, 1fr); }
    .ctl { grid-row: auto; justify-content: flex-start; }
  }
</style>
