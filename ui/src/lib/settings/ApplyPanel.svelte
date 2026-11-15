<script>
  // Review and apply: the diff against the running (else the saved) config, the exact engine
  // command line, the blocking errors and the warnings, then the drain-and-restart's progress
  // (validate, save, drain, stop, start, loading, ready, and the auto-rollback when it happens).
  import { onDestroy } from 'svelte';
  import { S, startApply, pollJob } from '../settings.svelte.js';
  import { ui } from '../store.svelte.js';

  let note = $state('');
  let busy = $state(false);
  let err = $state('');
  let timer = null;
  let copied = $state(false);

  const c = $derived(S.check);
  const errors = $derived(c?.report?.errors ?? []);
  const warnings = $derived(c?.report?.warnings ?? []);
  const job = $derived(S.job);
  const running = $derived(job?.state === 'running');

  function poll() {
    clearInterval(timer);
    timer = setInterval(async () => {
      try {
        const r = await pollJob();
        if (!r.job || r.job.state !== 'running') clearInterval(timer);
      } catch (e) {}
    }, 700);
  }
  async function apply() {
    busy = true;
    err = '';
    try {
      await startApply(note.trim());
      poll();
    } catch (e) {
      err = e.message;
    } finally {
      busy = false;
    }
  }
  async function copy() {
    try { await navigator.clipboard.writeText(c.command_line); copied = true; setTimeout(() => (copied = false), 1500); } catch (e) {}
  }
  onDestroy(() => clearInterval(timer));
  $effect(() => { if (S.job?.state === 'running') poll(); });

  const icon = { pending: '○', running: '◐', done: '●', failed: '✕', skipped: '–' };
  const banner = $derived(
    !job || job.state === 'running' ? null
      : job.state === 'done' ? 'good'
      : job.state === 'rolled-back' ? 'warning' : 'critical',
  );
</script>

<div class="apply">
  {#if job}
    <section aria-labelledby="ap-prog">
      <h3 id="ap-prog">Progress{job.origin !== 'apply' ? ` (${job.origin})` : ''}</h3>
      <ol class="stepper">
        {#each job.steps as s (s.id)}
          <li class="s-{s.state}">
            <span class="dot" aria-hidden="true">{icon[s.state]}</span>
            <span class="lbl">{s.label}</span>
            <span class="st">{s.state}</span>
            {#if s.message}<span class="msg">{s.message}</span>{/if}
          </li>
        {/each}
      </ol>
      {#if banner}
        <p class="banner b-{banner}" role="status">
          {#if job.state === 'rolled-back'}<strong>Auto-rolled back.</strong>{:else if job.state === 'failed'}<strong>Apply failed.</strong>{:else}<strong>Applied.</strong>{/if}
          {job.message}
        </p>
      {/if}
    </section>
  {/if}

  {#if !running}
    <section aria-labelledby="ap-diff">
      <h3 id="ap-diff">Changes vs the {c?.diff_base ?? 'saved'} config</h3>
      {#if !c}
        <p class="muted">Checking…</p>
      {:else if c.unchanged}
        <p class="muted">No changes.</p>
      {:else}
        <div class="scroll">
          <table>
            <thead><tr><th class="l">Field</th><th class="l">Was</th><th class="l">Now</th></tr></thead>
            <tbody>
              {#each c.diff as d}
                <tr>
                  <td class="l"><code>{d.field}</code>{#if d.kind === 'config'} <span class="muted">config</span>{/if}</td>
                  <td class="l old">{d.old ?? '(absent)'}</td>
                  <td class="l new">{d.new ?? '(removed)'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
      <p class="restart-note">
        {#if c?.restart_required}
          <strong>Restart required:</strong> the engine's command line changes, so applying drains running requests, stops and restarts it.
        {:else if ui.status?.state && !['stopped', 'crashed', 'fatal-86'].includes(ui.status.state)}
          No restart: nothing that reaches the engine's command line changed.
        {:else}
          The engine is not running: applying saves the config and starts it.
        {/if}
      </p>
    </section>

    <section aria-labelledby="ap-cmd">
      <div class="c-head"><h3 id="ap-cmd">Command line</h3><button class="btn small" onclick={copy} disabled={!c}>{copied ? 'Copied' : 'Copy'}</button></div>
      <pre class="tail cmd">{c?.command_line ?? ''}</pre>
    </section>

    {#if errors.length}
      <section class="errs" role="alert">
        <h3>Fix before applying ({errors.length})</h3>
        <ul>{#each errors as e}<li><code>{e.field}</code> {e.message}</li>{/each}</ul>
      </section>
    {/if}
    {#if warnings.length}
      <section class="warns">
        <h3>Warnings ({warnings.length})</h3>
        <ul>{#each warnings as w}<li><code>{w.field}</code> {w.message}</li>{/each}</ul>
      </section>
    {/if}

    <div class="go">
      <input type="text" bind:value={note} placeholder="note for the history (optional)" aria-label="Note" />
      <button class="btn primary" onclick={apply} disabled={busy || !c || errors.length > 0 || running}>
        {c?.restart_required ? 'Apply and restart' : 'Apply'}
      </button>
    </div>
    {#if err}<p class="banner b-critical" role="alert">{err}</p>{/if}
  {/if}
</div>

<style>
  .apply { display: grid; gap: 16px; }
  h3 { margin-bottom: 6px; }
  .stepper { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }
  .stepper li { display: grid; grid-template-columns: 18px 80px 70px minmax(0, 1fr); gap: 6px; align-items: baseline; font-size: var(--fs-sm); }
  .dot { text-align: center; }
  .st { font-size: var(--fs-xs); color: var(--muted); }
  .msg { font-size: var(--fs-xs); color: var(--ink-2); overflow-wrap: anywhere; }
  .s-done .dot { color: var(--good); }
  .s-running .dot { color: var(--accent); }
  .s-failed .dot, .s-failed .lbl { color: var(--critical); }
  .s-skipped { color: var(--muted); }
  .banner { margin: 10px 0 0; padding: 8px 10px; border-radius: 5px; font-size: var(--fs-sm); }
  .b-good { background: color-mix(in srgb, var(--good) 14%, transparent); }
  .b-warning { background: color-mix(in srgb, var(--warning) 22%, transparent); }
  .b-critical { background: color-mix(in srgb, var(--critical) 14%, transparent); color: var(--ink); }
  .old { color: var(--muted); text-decoration: line-through; white-space: normal; overflow-wrap: anywhere; }
  .new { white-space: normal; overflow-wrap: anywhere; }
  td.l, th.l { text-align: left; }
  .cmd { white-space: pre-wrap; overflow-wrap: anywhere; min-height: 3em; }
  .restart-note { font-size: var(--fs-sm); color: var(--ink-2); margin: 8px 0 0; }
  .errs ul, .warns ul { margin: 0; padding-left: 18px; font-size: var(--fs-sm); }
  .errs { color: var(--critical); }
  .go { display: flex; gap: 8px; flex-wrap: wrap; }
  .go input { flex: 1 1 200px; }
</style>
