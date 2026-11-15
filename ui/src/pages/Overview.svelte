<script>
  import { sortable, resetLayout } from '../lib/sortable.js';
  import { api, AuthError, compact, duration, pct as fpct, int } from '../lib/api.js';
  import { ui, liveS, fetchCols, refreshStatus, guarded } from '../lib/store.svelte.js';
  import Sparkline from '../lib/Sparkline.svelte';
  import LineChart from '../lib/LineChart.svelte';

  const SPARK_S = 900;
  const FIELDS = ['gen_tps', 'prompt_tps', 'running', 'waiting', 'accept_rate', 'accepted_per_draft', 'req_rate',
    'ttft_p50_ms', 'ttft_p99_ms', 'step_p99_ms'];

  let cols = $state(empty());
  let busy = $state('');
  let actionError = $state('');
  let logs = $state([]);
  let life = $state(null);
  let day = $state(null);
  let hw = $state(null);
  let lastSeq = 0;

  function empty() {
    const c = { t: [], up: [] };
    for (const f of FIELDS) c[f] = [];
    return c;
  }

  async function reload() {
    const seconds = Math.max(liveS(), SPARK_S);
    try { cols = (await fetchCols(seconds)) ?? cols; } catch (e) {}
    try {
      const got = await guarded(() => api(`/api/logs?since=${lastSeq}&limit=200`));
      if (got?.length) {
        lastSeq = got[got.length - 1].seq;
        logs = [...logs, ...got].slice(-200);
      }
    } catch (e) {}
    try { life = (await guarded(() => api('/api/usage/summary?range=all')))?.totals ?? null; } catch (e) { life = null; }
    try { day = (await guarded(() => api('/api/usage/summary?range=24h')))?.totals ?? null; } catch (e) { day = null; }
    try { hw = await guarded(() => api('/api/hw')); } catch (e) {}
  }

  $effect(() => {
    ui.tick; ui.range;
    reload();
  });

  async function act(what) {
    busy = what;
    actionError = '';
    try {
      await api(`/api/engine/${what}`, { method: 'POST' });
    } catch (e) {
      if (e instanceof AuthError) ui.needToken = true;
      else actionError = e.message;
    } finally {
      busy = '';
      refreshStatus();
    }
  }

  function lastOf(arr) {
    for (let i = (arr?.length ?? 0) - 1; i >= 0; i--) if (arr[i] != null) return arr[i];
    return null;
  }
  function windowOf(seconds) {
    const end = cols.t.length ? cols.t[cols.t.length - 1] : 0;
    let i = 0;
    while (i < cols.t.length && cols.t[i] < end - seconds) i++;
    const out = { t: cols.t.slice(i) };
    for (const f of FIELDS) out[f] = (cols[f] ?? []).slice(i);
    return out;
  }
  function mean(arr) {
    const v = arr.filter((x) => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }

  const status = $derived(ui.status);
  const spark = $derived(windowOf(SPARK_S));
  const chart = $derived(windowOf(liveS()));
  const genNow = $derived(lastOf(cols.gen_tps));
  const promptNow = $derived(lastOf(cols.prompt_tps));
  const runningNow = $derived(lastOf(cols.running));
  const waitingNow = $derived(lastOf(cols.waiting));
  const acceptNow = $derived(lastOf(spark.accept_rate));
  const apdNow = $derived(lastOf(spark.accepted_per_draft));
  const genMean = $derived(mean(spark.gen_tps));
  const promptMean = $derived(mean(spark.prompt_tps));
  const ttftP50 = $derived(lastOf(spark.ttft_p50_ms));
  const ttftP99 = $derived(lastOf(spark.ttft_p99_ms));
  const stepP99 = $derived(lastOf(spark.step_p99_ms));
  const stepMean = $derived(mean(spark.step_p99_ms));

  const STATES = {
    ready: { label: 'Ready', tone: 'good', icon: 'check' },
    loading: { label: 'Loading', tone: 'warning', icon: 'ring' },
    draining: { label: 'Draining', tone: 'warning', icon: 'ring' },
    stopping: { label: 'Stopping', tone: 'warning', icon: 'ring' },
    stopped: { label: 'Stopped', tone: 'neutral', icon: 'square' },
    crashed: { label: 'Crashed', tone: 'serious', icon: 'bang' },
    'fatal-86': { label: 'Rank failure', tone: 'critical', icon: 'bang' },
  };
  const st = $derived(status ? STATES[status.state] ?? STATES.stopped : null);
  const alive = $derived(status && ['loading', 'ready', 'draining', 'stopping'].includes(status.state));
  const presetName = $derived(
    status ? { flashnext: 'Flash-Next', qwen27b: 'Qwen3.8-27B', mock: 'Mock engine' }[status.preset] ?? status.preset : '',
  );
  const uptime = $derived.by(() => {
    if (!status) return '–';
    if (status.state === 'ready' && status.ready_ms) return duration(status.now_ms - status.ready_ms);
    if (alive && status.started_ms) return duration(status.now_ms - status.started_ms);
    return '–';
  });
  const alerts = $derived.by(() => {
    const a = [];
    if (status?.state === 'crashed') a.push({ tone: 'serious', text: status.message ?? 'The engine crashed.' });
    if (status?.state === 'fatal-86') a.push({ tone: 'critical', text: status.message ?? 'A rank failed (exit 86).' });
    for (const g of hw?.gpus ?? []) {
      if (g.temp_junction_c != null && g.temp_junction_c >= 95) a.push({ tone: 'critical', text: `GPU ${g.card}: junction ${g.temp_junction_c.toFixed(0)} °C` });
      if (g.vram_used != null && g.vram_total && g.vram_used / g.vram_total > 0.985) a.push({ tone: 'warning', text: `GPU ${g.card}: VRAM ${fpct(g.vram_used / g.vram_total)} used` });
    }
    if (day?.requests >= 20 && day.error_rate > 0.05) a.push({ tone: 'warning', text: `Error rate ${fpct(day.error_rate)} over the last 24 h (${day.errors} of ${day.requests})` });
    if (waitingNow >= 8) a.push({ tone: 'warning', text: `${waitingNow} requests waiting` });
    return a;
  });
  const tableRows = $derived.by(() => {
    const rows = [];
    for (let i = chart.t.length - 1; i >= 0 && rows.length < 12; i--) {
      rows.push({
        time: new Date(chart.t[i] * 1000).toLocaleTimeString(),
        gen: chart.gen_tps[i], prompt: chart.prompt_tps[i], running: chart.running[i],
        waiting: chart.waiting[i], accept: chart.accept_rate[i],
      });
    }
    return rows;
  });
  const pct = (v) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`);
  const msv = (v) => (v == null || !Number.isFinite(v) ? '–' : v >= 1000 ? (v / 1000).toFixed(2) : v >= 100 ? Math.round(v).toString() : v.toFixed(1));
  const msu = (v) => (v != null && v >= 1000 ? 's' : 'ms');
</script>

<div class="page" use:sortable={'overview'}>
  <section class="panel banner tone-{st?.tone ?? 'neutral'}" aria-live="polite">
    <div class="b-main">
      <div class="pill tone-{st?.tone ?? 'neutral'}">
        {#if st?.icon === 'check'}
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
        {:else if st?.icon === 'ring'}
          <svg viewBox="0 0 16 16" class="spin" aria-hidden="true"><circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="24 12" /></svg>
        {:else if st?.icon === 'bang'}
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3.5v6M8 12.2v.3" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" /></svg>
        {:else}
          <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" /></svg>
        {/if}
        <span>{st?.label ?? 'Connecting'}</span>
      </div>
      <div class="b-text">
        <div class="model">{status?.model ?? (alive ? 'Loading model…' : 'No engine running')}</div>
        <dl class="facts">
          <div><dt>Engine</dt><dd>{presetName || '–'}</dd></div>
          <div><dt>Port</dt><dd>{status?.port ?? '–'}</dd></div>
          <div><dt>{status?.state === 'ready' ? 'Uptime' : 'Elapsed'}</dt><dd>{uptime}</dd></div>
          <div><dt>Runtime</dt><dd>{status ? (status.container ?? (status.runtime === 'direct' ? `pid ${status.pid ?? '–'}` : status.runtime)) : '–'}</dd></div>
          {#if status?.exit_code != null}<div><dt>Exit code</dt><dd>{status.exit_code}</dd></div>{/if}
        </dl>
      </div>
      <div class="actions">
        <button class="btn primary" disabled={!status || alive || !!busy} onclick={() => act('start')}>{busy === 'start' ? 'Starting…' : 'Start'}</button>
        <button class="btn" disabled={!alive || !!busy} onclick={() => act('restart')} title="Waits for running requests to finish, then restarts">Restart</button>
        <button class="btn" disabled={!alive || !!busy} onclick={() => act('stop')}>{busy === 'stop' ? 'Stopping…' : 'Stop'}</button>
      </div>
    </div>
    {#if actionError || status?.message}
      <p class="msg">{actionError || status?.message}</p>
    {/if}
    {#if status?.command?.length}
      <details class="cmd">
        <summary>Command line</summary>
        <pre>{status.command.join(' ')}</pre>
      </details>
    {/if}
  </section>

  {#if alerts.length}
    <section class="panel alerts" aria-label="Alerts">
      <h2>Alerts</h2>
      <ul>
        {#each alerts as a}
          <li class="status-pill st-{a.tone === 'serious' ? 'critical' : a.tone}"><i></i><span class="atext">{a.tone === 'critical' ? 'Critical' : a.tone === 'serious' ? 'Serious' : 'Warning'}: {a.text}</span></li>
        {/each}
      </ul>
    </section>
  {/if}

  <section class="tiles" aria-label="Live figures">
    <article class="panel tile">
      <h3>Generation</h3>
      <div class="val">{compact(genNow)} <span class="unit">tok/s</span></div>
      <div class="sub">15-min mean {compact(genMean)}</div>
      <Sparkline t={spark.t} v={spark.gen_tps} theme={ui.themeKey} label="Generation tokens per second, last 15 minutes" />
    </article>
    <article class="panel tile">
      <h3>Prompt processing</h3>
      <div class="val">{compact(promptNow)} <span class="unit">tok/s</span></div>
      <div class="sub">15-min mean {compact(promptMean)}</div>
      <Sparkline t={spark.t} v={spark.prompt_tps} theme={ui.themeKey} label="Prompt tokens per second, last 15 minutes" />
    </article>
    <article class="panel tile">
      <h3>Requests</h3>
      <div class="val">{runningNow ?? '–'} <span class="unit">running</span> <span class="sep">·</span> {waitingNow ?? '–'} <span class="unit">waiting</span></div>
      <div class="sub">running, last 15 min</div>
      <Sparkline t={spark.t} v={spark.running} theme={ui.themeKey} label="Running requests, last 15 minutes" />
    </article>
    <article class="panel tile">
      <h3>Draft acceptance</h3>
      <div class="val">{pct(acceptNow)}</div>
      <div class="sub">{apdNow == null ? 'no drafts in the last interval' : `${apdNow.toFixed(2)} accepted per draft`}</div>
      <Sparkline t={spark.t} v={spark.accept_rate} theme={ui.themeKey} label="Draft acceptance rate, last 15 minutes" />
    </article>
    <article class="panel tile">
      <h3>Time to first token</h3>
      <div class="val">{msv(ttftP50)} <span class="unit">{msu(ttftP50)} p50</span></div>
      <div class="sub">{ttftP99 == null ? 'no requests in the last 15 min' : `p99 ${msv(ttftP99)} ${msu(ttftP99)}, latest interval with requests`}</div>
      <Sparkline t={spark.t} v={spark.ttft_p50_ms} theme={ui.themeKey} label="Time to first token p50, last 15 minutes" />
    </article>
    <article class="panel tile">
      <h3>Step time</h3>
      <div class="val">{msv(stepP99)} <span class="unit">{msu(stepP99)} p99</span></div>
      <div class="sub">{stepP99 == null ? 'no decode steps in the last 15 min' : `15-min mean of p99 ${msv(stepMean)} ${msu(stepMean)}`}</div>
      <Sparkline t={spark.t} v={spark.step_p99_ms} theme={ui.themeKey} label="Engine step time p99, last 15 minutes" />
    </article>
  </section>

  <section class="panel lifetime" aria-label="Lifetime totals">
    <div class="c-head">
      <h2>All time</h2>
      <a class="muted small" href="#usage">Usage history</a>
    </div>
    {#if life}
      <dl class="totals">
        <div><dt>Tokens served</dt><dd>{compact((life.prompt_tokens ?? 0) + (life.completion_tokens ?? 0))}</dd></div>
        <div><dt>Prompt tokens</dt><dd>{compact(life.prompt_tokens)}</dd></div>
        <div><dt>Generated tokens</dt><dd>{compact(life.completion_tokens)}</dd></div>
        <div><dt>Requests</dt><dd>{int(life.requests)}</dd></div>
        <div><dt>Cached prefix</dt><dd>{fpct(life.cached_ratio)}</dd></div>
      </dl>
    {:else}
      <p class="note">The usage history is off or empty.</p>
    {/if}
  </section>

  <section class="panel chart">
    <div class="c-head">
      <div>
        <h2>Throughput</h2>
        <div class="legend">
          <span><i style="background: var(--accent)"></i>Generation</span>
          <span><i style="background: var(--series-2)"></i>Prompt processing</span>
          <span class="muted">tokens per second, {cols.interval_s ?? 5} s intervals</span>
        </div>
      </div>
    </div>
    <LineChart
      t={chart.t}
      series={[
        { label: 'Generation', values: chart.gen_tps, color: '--accent' },
        { label: 'Prompt processing', values: chart.prompt_tps, color: '--series-2' },
      ]}
      unit="tok/s"
      theme={ui.themeKey}
      label="Generation and prompt tokens per second over the selected range"
    />
    <details class="tbl">
      <summary>Table view</summary>
      <div class="scroll">
        <table>
          <thead><tr><th>Time</th><th>Generation tok/s</th><th>Prompt tok/s</th><th>Running</th><th>Waiting</th><th>Acceptance</th></tr></thead>
          <tbody>
            {#each tableRows as r}
              <tr><td>{r.time}</td><td>{compact(r.gen)}</td><td>{compact(r.prompt)}</td><td>{r.running ?? '–'}</td><td>{r.waiting ?? '–'}</td><td>{pct(r.accept)}</td></tr>
            {/each}
          </tbody>
        </table>
      </div>
    </details>
  </section>

  <section class="panel logs">
    <div class="c-head"><h2>Engine output</h2><a class="muted small" href="#logs">All logs</a></div>
    <pre class="tail">{#each logs.slice(-14) as l}<span class="ln s-{l.stream}">{l.text}</span>{'\n'}{/each}</pre>
  </section>
  <p class="layout-reset"><button class="linkish" onclick={() => { resetLayout('overview'); location.reload(); }}>Reset the panel layout</button></p>
</div>

<style>
  .banner { border-left: 4px solid var(--stripe, var(--axis)); }
  .banner.tone-good { --stripe: var(--good); }
  .banner.tone-warning { --stripe: var(--warning); }
  .banner.tone-serious { --stripe: var(--serious); }
  .banner.tone-critical { --stripe: var(--critical); }
  .b-main { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px 4px 8px; border-radius: 999px;
    font-size: var(--fs-sm); font-weight: 600; color: var(--on-status); background: var(--muted);
  }
  .pill svg { width: 14px; height: 14px; }
  .pill.tone-good { background: var(--good); }
  .pill.tone-warning { background: var(--warning); color: #0b0b0b; }
  .pill.tone-serious { background: var(--serious); color: #0b0b0b; }
  .pill.tone-critical { background: var(--critical); }
  .spin { animation: spin 1.1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .b-text { flex: 1 1 320px; min-width: 0; }
  .model { font-size: var(--fs-lg); font-weight: 600; overflow-wrap: anywhere; }
  .facts { display: flex; flex-wrap: wrap; gap: 4px 20px; margin: 2px 0 0; }
  .facts div { display: flex; gap: 6px; font-size: var(--fs-sm); }
  .facts dt { color: var(--muted); }
  .facts dd { margin: 0; color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .msg {
    margin: 10px 0 0; font-family: var(--mono); font-size: var(--fs-sm); color: var(--ink);
    background: var(--code-bg); padding: 6px 8px; border-radius: 4px; overflow-wrap: anywhere;
  }
  .cmd { margin-top: 10px; font-size: var(--fs-sm); }
  .cmd summary { cursor: pointer; color: var(--ink-2); }
  .cmd pre {
    margin: 6px 0 0; padding: 8px; background: var(--code-bg); border-radius: 4px;
    font: var(--fs-xs) / 1.5 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; color: var(--ink-2);
  }
  .alerts ul { list-style: none; margin: 8px 0 0; padding: 0; display: grid; gap: 6px; }
  .atext { font-weight: 500; color: var(--ink); font-size: var(--fs-sm); }
  .small { font-size: var(--fs-sm); }
  .totals { display: flex; flex-wrap: wrap; gap: 10px 36px; margin: 0; }
  .totals div { display: grid; }
  .totals dt { font-size: var(--fs-xs); color: var(--muted); }
  .totals dd { margin: 0; font-size: var(--fs-lg); font-weight: 600; }
  .logs h2 { margin-bottom: 0; }
</style>
