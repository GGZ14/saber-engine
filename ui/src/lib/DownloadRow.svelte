<script>
  // One catalog component's download: installed / not yet / in progress (bar, speed, ETA) with
  // Download, Pause, Resume, Cancel and Retry, and the failure in plain English.
  import { jobFor, startDownload, dlAction, ACTIVE, eta, rate, gb } from './setup.svelte.js';

  let { model, comp } = $props();
  const job = $derived(jobFor(model.id, comp.id));
  const live = $derived(job && ACTIVE.includes(job.state));
  const pct = $derived(job?.total_bytes ? Math.min(100, (job.done_bytes / job.total_bytes) * 100) : 0);
  const STATE = {
    queued: 'Queued: one download runs at a time',
    listing: 'Asking the server for the file list…',
    downloading: 'Downloading',
    verifying: 'Checking the files (sha256)…',
    installing: 'Putting it in place…',
    paused: 'Paused',
  };
</script>

<div class="dl">
  <div class="top">
    <div class="name">
      <strong>{comp.label || comp.id}</strong>
      <span class="muted">{gb(comp.approx_bytes)} · {comp.source}</span>
    </div>
    <div class="acts">
      {#if comp.installed}
        <span class="ok" role="status"><svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8.5 6.5 12 13 4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>Installed</span>
      {:else if live}
        {#if job.state === 'paused'}
          <button class="btn small primary" onclick={() => dlAction(job.id, 'resume')}>Resume</button>
        {:else}
          <button class="btn small" onclick={() => dlAction(job.id, 'pause')}>Pause</button>
        {/if}
        <button class="btn small" onclick={() => dlAction(job.id, 'cancel')} title="Stop and delete what was downloaded">Cancel</button>
      {:else if job?.state === 'failed'}
        <button class="btn small primary" onclick={() => dlAction(job.id, 'resume')}>Retry</button>
        <button class="btn small" onclick={() => dlAction(job.id, 'cancel')} title="Delete the partial download">Discard</button>
      {:else}
        <button class="btn small primary" onclick={() => startDownload(model.id, comp.id)}>{comp.partial ? 'Resume download' : 'Download'}</button>
      {/if}
    </div>
  </div>
  {#if live}
    <div class="meter" role="progressbar" aria-label="{comp.label} download" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(pct)}>
      <span style:width="{pct}%" class:paused={job.state === 'paused'}></span>
    </div>
    <div class="info">
      <span>{STATE[job.state] ?? job.state}{job.state === 'downloading' && job.current_file ? ` ${job.current_file}` : ''}</span>
      <span class="muted">
        {gb(job.done_bytes)}{job.total_bytes ? ` of ${gb(job.total_bytes)}` : ''}
        {#if job.files_total} · file {Math.min(job.files_done + 1, job.files_total)} of {job.files_total}{/if}
        {#if job.speed_bps} · {rate(job.speed_bps)}{/if}
        {#if job.eta_s && job.state === 'downloading'} · {eta(job.eta_s)}{/if}
      </span>
    </div>
  {:else if job?.state === 'failed'}
    <p class="err" role="alert">{job.error}</p>
  {:else if job?.state === 'cancelled' && !comp.installed}
    <p class="muted info">Cancelled; the partial download was deleted.</p>
  {:else if !comp.installed && comp.partial}
    <p class="muted info">A partial download is on disk: it resumes where it stopped.</p>
  {/if}
</div>

<style>
  .dl { padding: 10px 0; border-top: 1px solid var(--grid); display: grid; gap: 6px; }
  .dl:first-child { border-top: 0; }
  .top { display: flex; justify-content: space-between; align-items: center; gap: 8px 12px; flex-wrap: wrap; }
  .name { display: grid; gap: 1px; min-width: 0; }
  .name .muted { font-size: var(--fs-xs); overflow-wrap: anywhere; }
  .acts { display: flex; gap: 6px; flex-wrap: wrap; }
  .ok { display: inline-flex; align-items: center; gap: 4px; color: var(--good-text); font-size: var(--fs-sm); font-weight: 600; }
  .meter > span { transition: width 0.4s ease; }
  .meter > span.paused { background: var(--deemph); }
  .info { display: flex; justify-content: space-between; gap: 4px 12px; flex-wrap: wrap; font-size: var(--fs-xs); margin: 0; }
  .err { margin: 0; font-size: var(--fs-sm); color: var(--critical); }
</style>
