<script>
  // The persistent usage history: average rates (24 h / 7 d / 30 d), trends over the global
  // range, monthly totals with a calendar heatmap, month-over-month, top models / keys, cached
  // prefix savings, an energy / cost estimate, the request history with filters + export, and
  // the store's settings. Every long-range figure is read from the rollups.
  import { api, apiText, saveText, compact, ms, pct, int, AuthError } from '../lib/api.js';
  import { ui, rangeS, guarded, save } from '../lib/store.svelte.js';
  import LineChart from '../lib/LineChart.svelte';
  import BarChart from '../lib/BarChart.svelte';
  import Calendar from '../lib/Calendar.svelte';
  import Panel from '../lib/Panel.svelte';
  import Tile from '../lib/Tile.svelte';
  import RequestTable from '../lib/RequestTable.svelte';
  import RequestDetail from '../lib/RequestDetail.svelte';
  import Drawer from '../lib/Drawer.svelte';

  let status = $state(null);
  let periods = $state({});
  let sum = $state(null);
  let monthly = $state(null);
  let hist = $state({ total: 0, rows: [] });
  let err = $state('');
  let open = $state(null);
  let f = $state({ model: '', key: '', status: '', q: '', min_latency_ms: '' });
  let sort = $state('ts_ms');
  let asc = $state(false);
  let offset = $state(0);
  const PAGE = 25;
  let rate = $state(loadRate());
  let calMetric = $state('tokens');
  let settings = $state(null);
  let saveMsg = $state('');

  function loadRate() {
    try { return localStorage.getItem('saber-kwh-rate') || ''; } catch (e) { return ''; }
  }

  function qs(extra = {}) {
    const p = new URLSearchParams();
    p.set('from', String(Date.now() - rangeS() * 1000));
    for (const [k, v] of Object.entries({ ...f, ...extra })) if (v !== '' && v != null) p.set(k, String(v));
    return p.toString();
  }

  async function reload() {
    try {
      status = await guarded(() => api('/api/usage/status'));
      if (!status?.enabled) return;
      if (!settings) settings = { ...status.settings };
      const [d, w, m] = await Promise.all(['24h', '7d', '30d'].map((r) => api(`/api/usage/summary?range=${r}`)));
      periods = { '24h': d.totals, '7d': w.totals, '30d': m.totals };
      sum = await api(`/api/usage/summary?range=${rangeS()}s`);
      monthly = await api('/api/usage/monthly?months=12&days=371');
      err = '';
    } catch (e) {
      if (!(e instanceof AuthError)) err = e.message;
    }
  }
  async function reloadHist() {
    try {
      hist = (await guarded(() => api(`/api/usage/requests?${qs({ sort, asc, limit: PAGE, offset })}`))) ?? hist;
    } catch (e) {}
  }
  $effect(() => { ui.tick; ui.range; reload(); });
  $effect(() => { ui.tick; ui.range; f.model; f.key; f.status; f.q; f.min_latency_ms; sort; asc; offset; reloadHist(); });

  function onsort(k) {
    if (sort === k) asc = !asc;
    else { sort = k; asc = false; }
    offset = 0;
  }
  async function exportAs(fmt) {
    try {
      const text = await apiText(`/api/usage/export?${qs({ format: fmt, sort, asc })}`);
      saveText(`saber-usage.${fmt}`, text, fmt === 'json' ? 'application/json' : 'text/csv');
    } catch (e) { err = e.message; }
  }
  async function saveSettings() {
    saveMsg = '';
    try {
      const cur = await api('/api/config');
      const cfg = cur.config;
      cfg.usage = {
        ...cfg.usage,
        raw_retention_days: Math.max(1, Math.round(+settings.raw_retention_days || 90)),
        store_content: !!settings.store_content,
        utc_offset_minutes: Math.round(+settings.utc_offset_minutes || 0),
      };
      await api('/api/config', { method: 'PUT', body: cfg });
      saveMsg = 'Saved. Retention, the content setting and the day boundary apply now (days already totalled keep their cut).';
      reload();
    } catch (e) { saveMsg = e.message; }
  }
  // the zone offsets in use (minutes east of UTC), and the browser's own
  const TZ = [-720, -660, -600, -570, -540, -480, -420, -360, -300, -240, -210, -180, -120, -60, 0, 60, 120, 180, 210, 240, 270, 300, 330, 345, 360, 390, 420, 480, 525, 540, 570, 600, 630, 660, 720, 765, 780, 840];
  const browserOffset = -new Date().getTimezoneOffset();
  function tzOptions(cur) {
    const set = new Set([...TZ, browserOffset, Math.round(+cur || 0)]);
    return [...set].sort((a, b) => a - b);
  }
  function tzLabel(m) {
    m = Math.round(+m || 0);
    const sign = m < 0 ? '−' : '+';
    const a = Math.abs(m);
    return `UTC${sign}${String(Math.floor(a / 60)).padStart(2, '0')}:${String(a % 60).padStart(2, '0')}`;
  }
  function setRate(v) {
    rate = v;
    try { localStorage.setItem('saber-kwh-rate', v); } catch (e) {}
  }

  const series = $derived(sum?.series ?? []);
  const st = $derived(series.map((p) => p.t / 1000));
  const totals = $derived(sum?.totals ?? null);
  const stepLabel = $derived(sum ? (sum.step_ms >= 86400000 ? `${sum.step_ms / 86400000} d` : sum.step_ms >= 3600000 ? `${sum.step_ms / 3600000} h` : `${sum.step_ms / 60000} min`) : '');
  const months = $derived(monthly?.months ?? []);
  const mLabel = (m) => {
    const [y, mo] = m.split('-').map(Number);
    return new Date(y, mo - 1, 1).toLocaleString([], { month: 'short', year: '2-digit' });
  };
  const cur = $derived(months[months.length - 1] ?? null);
  const prev = $derived(months[months.length - 2] ?? null);
  const delta = (a, b) => (a == null || b == null || b === 0 ? null : (a - b) / b);
  const mom = $derived(cur && prev ? [
    { k: 'Requests', a: cur.requests, b: prev.requests, f: int },
    { k: 'Prompt tokens', a: cur.prompt_tokens, b: prev.prompt_tokens, f: compact },
    { k: 'Output tokens', a: cur.completion_tokens, b: prev.completion_tokens, f: compact },
    { k: 'Cached prefix share', a: cur.cached_ratio, b: prev.cached_ratio, f: pct },
    { k: 'Decode tok/s, mean', a: cur.decode_tps_mean, b: prev.decode_tps_mean, f: compact },
    { k: 'TTFT p50', a: cur.ttft_p50_ms, b: prev.ttft_p50_ms, f: ms, lowerBetter: true },
    { k: 'Error rate', a: cur.error_rate, b: prev.error_rate, f: pct, lowerBetter: true },
    { k: 'GPU energy', a: cur.energy_kwh, b: prev.energy_kwh, f: (v) => (v == null ? '–' : `${v.toFixed(2)} kWh`), lowerBetter: true },
  ] : []);
  const top = (list) => (list ?? []).slice(0, 6);
  const topModels = $derived(top(sum?.by_model));
  const topKeys = $derived(top(sum?.by_key));
  const tpsHist = $derived(totals?.decode_tps_hist ?? []);
  const kwhRange = $derived(months.reduce((s, m) => s + (m.energy_kwh ?? 0), 0));
  const rateN = $derived(parseFloat(rate));
  const cost = (kwh) => (Number.isFinite(rateN) && rateN > 0 && kwh != null ? `$${(kwh * rateN).toFixed(2)}` : null);
  const tokPerKwh = $derived(cur?.energy_kwh ? (cur.completion_tokens + cur.prompt_tokens) / cur.energy_kwh : null);
  const calValue = $derived(calMetric === 'tokens' ? (d) => d.prompt_tokens + d.completion_tokens : calMetric === 'requests' ? (d) => d.requests : (d) => d.energy_kwh ?? 0);
</script>

<div class="page">
  <h1 class="title">Usage</h1>
  {#if status && !status.enabled}
    <section class="panel"><p class="empty">The usage history is off (<code>usage.enabled = false</code> in the console config, or the database did not open; see Logs).</p></section>
  {:else}
    {#if err}<p class="note">{err}</p>{/if}

    <section class="tiles">
      {#each ['24h', '7d', '30d'] as p}
        {@const t = periods[p]}
        <article class="panel tile">
          <h3>Last {p === '24h' ? '24 hours' : p === '7d' ? '7 days' : '30 days'}</h3>
          <div class="val">{compact(t?.decode_tps_mean)} <span class="unit">tok/s decode, mean</span></div>
          <div class="sub">p50 {compact(t?.decode_tps_p50)} · p95 {compact(t?.decode_tps_p95)} tok/s</div>
          <dl class="mini">
            <div><dt>Prompt</dt><dd>{compact(t?.prompt_tps_mean)} tok/s</dd></div>
            <div><dt>TTFT p50</dt><dd>{ms(t?.ttft_p50_ms)}</dd></div>
            <div><dt>Requests</dt><dd>{int(t?.requests)}</dd></div>
          </dl>
        </article>
      {/each}
    </section>

    <div class="grid2">
      <Panel title="Tokens over the range" legend={[{ label: 'Prompt', color: '--accent' }, { label: 'Output', color: '--series-2' }]} note="per {stepLabel}">
        {#if series.length}
          <BarChart categories={series.map((p) => new Date(p.t).toLocaleString([], sum.step_ms >= 86400000 ? { month: 'short', day: 'numeric' } : { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }))} series={[{ label: 'Prompt', values: series.map((p) => p.prompt_tokens), color: '--accent' }, { label: 'Output', values: series.map((p) => p.completion_tokens), color: '--series-2' }]} theme={ui.themeKey} label="Prompt and output tokens per interval" />
        {:else}<p class="empty">No requests in this range.</p>{/if}
      </Panel>
      <Panel title="Decode speed and TTFT" legend={[{ label: 'Decode tok/s, mean', color: '--accent' }]} note="per {stepLabel}">
        {#if series.length}
          <LineChart t={st} series={[{ label: 'Decode tok/s', values: series.map((p) => p.decode_tps_mean), color: '--accent' }]} unit="tok/s" height={150} theme={ui.themeKey} label="Mean decode tokens per second" />
          <LineChart t={st} series={[{ label: 'TTFT p50', values: series.map((p) => p.ttft_p50_ms), color: '--series-2' }]} fmt={ms} height={130} theme={ui.themeKey} label="Time to first token p50" />
          <div class="legend"><span><i style="background: var(--series-2)"></i>TTFT p50 (lower chart)</span></div>
        {:else}<p class="empty">No requests in this range.</p>{/if}
      </Panel>
    </div>

    <div class="grid2">
      <Panel title="Decode speed distribution" note="requests by tokens per second, over the range">
        {#if tpsHist.length}
          <BarChart categories={tpsHist.map((b) => `${compact(b.lo)}–${compact(b.hi)}`)} series={[{ label: 'Requests', values: tpsHist.map((b) => b.count), color: '--accent' }]} fmt={(v) => compact(v, 0)} theme={ui.themeKey} label="Distribution of decode speed" />
        {:else}<p class="empty">No decode samples in this range.</p>{/if}
      </Panel>
      <Panel title="Range at a glance">
        {#if totals}
          <dl class="kv">
            <dt>Requests</dt><dd>{int(totals.requests)} ({int(totals.errors)} errors, {pct(totals.error_rate)}; {int(totals.aborted)} client hang-ups)</dd>
            <dt>Prompt tokens</dt><dd>{compact(totals.prompt_tokens)}</dd>
            <dt>Output tokens</dt><dd>{compact(totals.completion_tokens)}</dd>
            <dt>Cached-prefix savings</dt><dd>{compact(totals.cached_tokens)} prompt tokens served from the cache ({pct(totals.cached_ratio)} of prompt tokens)</dd>
            <dt>TTFT</dt><dd>p50 {ms(totals.ttft_p50_ms)} · p95 {ms(totals.ttft_p95_ms)} · p99 {ms(totals.ttft_p99_ms)}</dd>
            <dt>Mean latency</dt><dd>{ms(totals.latency_mean_ms)}</dd>
          </dl>
          <h3 class="h">Top models</h3>
          <table><thead><tr><th>Model</th><th>Requests</th><th>Tokens</th><th>Cached</th></tr></thead>
            <tbody>{#each topModels as m}<tr><td class="trunc">{m.name || '(none)'}</td><td>{int(m.requests)}</td><td>{compact(m.prompt_tokens + m.completion_tokens)}</td><td>{pct(m.cached_ratio)}</td></tr>{/each}</tbody></table>
          <h3 class="h">Top API keys</h3>
          <table><thead><tr><th>Key label</th><th>Requests</th><th>Tokens</th><th>Errors</th></tr></thead>
            <tbody>{#each topKeys as m}<tr><td>{m.name || '(no key)'}</td><td>{int(m.requests)}</td><td>{compact(m.prompt_tokens + m.completion_tokens)}</td><td>{int(m.errors)}</td></tr>{/each}</tbody></table>
        {/if}
      </Panel>
    </div>

    <Panel title="Monthly tokens" legend={[{ label: 'Prompt (in)', color: '--accent' }, { label: 'Output (out)', color: '--series-2' }]} note="last 12 months, days cut at {tzLabel(monthly?.utc_offset_minutes ?? 0)}">
      {#if months.length}
        <BarChart categories={months.map((m) => mLabel(m.month))} series={[{ label: 'Prompt (in)', values: months.map((m) => m.prompt_tokens), color: '--accent' }, { label: 'Output (out)', values: months.map((m) => m.completion_tokens), color: '--series-2' }]} theme={ui.themeKey} label="Prompt and output tokens per month" />
      {/if}
      {#snippet table()}
        <table><thead><tr><th>Month</th><th>Requests</th><th>Prompt</th><th>Output</th><th>Cached</th><th>Error rate</th><th>Energy</th><th>Cost</th></tr></thead>
          <tbody>{#each months as m}<tr><td>{m.month}</td><td>{int(m.requests)}</td><td>{compact(m.prompt_tokens)}</td><td>{compact(m.completion_tokens)}</td><td>{pct(m.cached_ratio)}</td><td>{pct(m.error_rate)}</td><td>{m.energy_kwh == null ? '–' : `${m.energy_kwh.toFixed(2)} kWh`}</td><td>{cost(m.energy_kwh) ?? '–'}</td></tr>{/each}</tbody></table>
      {/snippet}
    </Panel>

    <div class="grid2">
      <Panel title="Daily activity" note="last 53 weeks">
        {#snippet actions()}
          <div class="seg" role="radiogroup" aria-label="Calendar measure">
            {#each [['tokens', 'Tokens'], ['requests', 'Requests'], ['energy', 'Energy']] as [k, l]}
              <button role="radio" aria-checked={calMetric === k} class:on={calMetric === k} onclick={() => (calMetric = k)}>{l}</button>
            {/each}
          </div>
        {/snippet}
        <Calendar days={monthly?.days ?? []} value={calValue} fmt={calMetric === 'energy' ? (v) => `${v.toFixed(2)} kWh` : (v) => compact(v)} what={calMetric === 'energy' ? '' : calMetric} label="Calendar heatmap of daily {calMetric}" />
      </Panel>
      <Panel title="Month over month" note={cur ? `${mLabel(cur.month)} so far vs ${prev ? mLabel(prev.month) : '–'}` : ''}>
        {#if mom.length}
          <table>
            <thead><tr><th>Measure</th><th>This month</th><th>Last month</th><th>Change</th></tr></thead>
            <tbody>
              {#each mom as r}
                {@const d = delta(r.a, r.b)}
                <tr><td>{r.k}</td><td>{r.f(r.a)}</td><td>{r.f(r.b)}</td>
                  <td class={d == null ? '' : (d > 0) !== !!r.lowerBetter ? 'up' : 'down'}>{d == null ? '–' : `${d > 0 ? '▲' : d < 0 ? '▼' : ''} ${Math.abs(d * 100).toFixed(1)}%`}</td></tr>
              {/each}
            </tbody>
          </table>
        {/if}
        <h3 class="h">Energy estimate</h3>
        <p class="note">From the GPU power the console samples (sysfs), whether or not requests run: 12 months {kwhRange.toFixed(2)} kWh{tokPerKwh ? `, ${compact(tokPerKwh)} tokens per kWh this month` : ''}.</p>
        <label class="rate">Electricity rate <span class="muted">$ per kWh</span>
          <input type="number" min="0" step="0.0001" placeholder="0.1324" value={rate} oninput={(e) => setRate(e.currentTarget.value)} aria-label="Electricity rate in dollars per kWh" />
        </label>
        {#if cost(cur?.energy_kwh)}<p class="note">This month {cost(cur.energy_kwh)} · last 12 months {cost(kwhRange)}</p>{/if}
      </Panel>
    </div>

    <section class="panel">
      <div class="c-head">
        <div>
          <h2>Request history</h2>
          <div class="legend"><span class="muted">{hist.total.toLocaleString()} matching in the range · raw rows are kept {status?.settings?.raw_retention_days ?? 90} days</span></div>
        </div>
        <div class="filters">
          <button class="btn small" onclick={() => exportAs('csv')}>Export CSV</button>
          <button class="btn small" onclick={() => exportAs('json')}>Export JSON</button>
        </div>
      </div>
      <div class="filters fl">
        <label>Model <select bind:value={f.model} onchange={() => (offset = 0)}><option value="">all</option>{#each sum?.by_model ?? [] as m}{#if m.name}<option value={m.name}>{m.name}</option>{/if}{/each}</select></label>
        <label>Key <select bind:value={f.key} onchange={() => (offset = 0)}><option value="">all</option>{#each sum?.by_key ?? [] as m}{#if m.name}<option value={m.name}>{m.name}</option>{/if}{/each}</select></label>
        <label>Status <select bind:value={f.status} onchange={() => (offset = 0)}><option value="">all</option><option value="ok">ok</option><option value="error">errors</option><option value="aborted">client hang-ups</option></select></label>
        <label>Latency ≥ <input type="number" min="0" step="100" placeholder="ms" style="width: 90px" bind:value={f.min_latency_ms} /></label>
        <label><input type="search" placeholder="search id, model, finish…" bind:value={f.q} /></label>
      </div>
      {#if hist.rows.length}
        <RequestTable rows={hist.rows} {sort} {asc} {onsort} onopen={(r) => (open = r.uid)} />
        <div class="filters pager">
          <button class="btn small" disabled={offset === 0} onclick={() => (offset = Math.max(0, offset - PAGE))}>Newer</button>
          <span class="muted">{offset + 1}–{Math.min(offset + PAGE, hist.total)} of {hist.total.toLocaleString()}</span>
          <button class="btn small" disabled={offset + PAGE >= hist.total} onclick={() => (offset += PAGE)}>Older</button>
        </div>
      {:else}<p class="empty">No requests match.</p>{/if}
    </section>

    {#if settings}
      <section class="panel">
        <h2>Settings</h2>
        <form class="set" onsubmit={(e) => { e.preventDefault(); saveSettings(); }}>
          <label>Keep per-request rows <input type="number" min="1" style="width: 80px" bind:value={settings.raw_retention_days} /> days <span class="muted">(rollups are kept forever)</span></label>
          <label><input type="checkbox" bind:checked={settings.store_content} /> Store request and response content <span class="muted">(off by default; bodies stay on this machine, in the database)</span></label>
          <label>Days and months are cut at
            <select bind:value={settings.utc_offset_minutes} aria-label="Time zone offset for day and month totals">
              {#each tzOptions(settings.utc_offset_minutes) as o}<option value={o}>{tzLabel(o)}</option>{/each}
            </select>
            <button class="btn small" type="button" onclick={() => (settings.utc_offset_minutes = browserOffset)} disabled={settings.utc_offset_minutes === browserOffset}>Use this browser's ({tzLabel(browserOffset)})</button>
          </label>
          <p class="note">A fixed offset (no daylight-saving switch). A change applies at once to new rows; days already totalled keep their cut.</p>
          <div><button class="btn primary" type="submit">Save settings</button></div>
          {#if saveMsg}<p class="note">{saveMsg}</p>{/if}
          <p class="note">Database {status?.path ?? '–'} · {int(status?.rows)} rows · writer {int(status?.writer?.rows_written)} written, {int(status?.writer?.rows_merged)} merged, {int(status?.writer?.rows_pruned)} pruned{status?.writer?.last_error ? ` · last error: ${status.writer.last_error}` : ''}</p>
        </form>
      </section>
    {/if}
  {/if}
</div>

<Drawer open={!!open} title="Request detail" onclose={() => (open = null)}>
  {#if open}<RequestDetail uid={open} />{/if}
</Drawer>

<style>
  .mini { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 6px 0 0; font-size: var(--fs-sm); }
  .mini div { display: flex; gap: 6px; }
  .mini dt { color: var(--muted); }
  .mini dd { margin: 0; font-variant-numeric: tabular-nums; }
  .h { margin: 14px 0 4px; }
  .trunc { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
  .up { color: var(--good-text); }
  .down { color: var(--critical); }
  .rate { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: var(--fs-sm); }
  .rate input { width: 110px; }
  .fl { margin-bottom: 8px; }
  .pager { justify-content: flex-end; margin-top: 8px; }
  .set { display: grid; gap: 10px; margin-top: 8px; font-size: var(--fs-sm); }
  .set label { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
</style>
