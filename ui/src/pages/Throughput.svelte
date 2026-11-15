<script>
  import { sortable, resetLayout } from '../lib/sortable.js';
  import { compact, ms, last, meanOf, total, groupBy, seriesColor } from '../lib/api.js';
  import { ui, liveS, rangeS, fetchCols, fetchSeries } from '../lib/store.svelte.js';
  import LineChart from '../lib/LineChart.svelte';
  import BarChart from '../lib/BarChart.svelte';
  import Heatmap from '../lib/Heatmap.svelte';
  import Panel from '../lib/Panel.svelte';
  import Tile from '../lib/Tile.svelte';
  import { heat, secLabel } from '../lib/hist.js';

  let cols = $state({ t: [] });
  let win = $state({ t: [], series: {} });

  async function reload() {
    try { cols = (await fetchCols(liveS())) ?? cols; } catch (e) {}
    try {
      win = (await fetchSeries(['time_to_first_token_seconds_bucket', 'step_time_seconds_bucket', 'request_success_total'], liveS(), 240)) ?? win;
    } catch (e) {}
  }
  $effect(() => { ui.tick; ui.range; reload(); });

  const c = (k) => cols[k] ?? [];
  const ttftHeat = $derived(heat(win, 'time_to_first_token_seconds_bucket', secLabel));
  const stepHeat = $derived(heat(win, 'step_time_seconds_bucket', secLabel));
  const reasons = $derived.by(() => {
    const g = groupBy(win, 'request_success_total', 'finished_reason');
    const dt = win.interval_s || 5;
    return Object.entries(g)
      .map(([k, v]) => ({ k: k || 'unknown', v: (total(v) ?? 0) * dt }))
      .sort((a, b) => b.v - a.v);
  });
  const clamped = $derived(rangeS() > liveS());
</script>

<div class="page" use:sortable={'throughput'}>
  <h1 class="title">Throughput and latency</h1>
  {#if clamped}<p class="note">Live metrics are kept for 24 h in memory; showing the last 24 h. Longer ranges are on the Usage page.</p>{/if}
  <section class="tiles">
    <Tile title="Generation" value={compact(last(c('gen_tps')))} unit="tok/s" sub="mean {compact(meanOf(c('gen_tps')))} over the range" spark={{ t: cols.t, v: c('gen_tps') }} theme={ui.themeKey} />
    <Tile title="Prompt processing" value={compact(last(c('prompt_tps')))} unit="tok/s" sub="mean {compact(meanOf(c('prompt_tps')))} over the range" spark={{ t: cols.t, v: c('prompt_tps') }} theme={ui.themeKey} />
    <Tile title="Finished requests" value={compact(last(c('req_rate')))} unit="req/s" sub="mean {compact(meanOf(c('req_rate')))}" spark={{ t: cols.t, v: c('req_rate') }} theme={ui.themeKey} />
    <Tile title="Time to first token" value={ms(last(c('ttft_p50_ms')))} unit="p50" sub="p99 {ms(last(c('ttft_p99_ms')))}, latest interval with requests" spark={{ t: cols.t, v: c('ttft_p50_ms') }} theme={ui.themeKey} />
    <Tile title="Inter-token latency" value={ms(last(c('itl_p50_ms')))} unit="p50" sub="p99 {ms(last(c('itl_p99_ms')))}" spark={{ t: cols.t, v: c('itl_p50_ms') }} theme={ui.themeKey} />
    <Tile title="Step time" value={ms(last(c('step_p99_ms')))} unit="p99" sub="p50 {ms(last(c('step_p50_ms')))}" spark={{ t: cols.t, v: c('step_p99_ms') }} theme={ui.themeKey} />
  </section>

  <Panel title="Tokens per second" legend={[{ label: 'Generation', color: '--accent' }, { label: 'Prompt processing', color: '--series-2' }]} note="{cols.interval_s ?? 5} s intervals">
    <LineChart t={cols.t} series={[{ label: 'Generation', values: c('gen_tps'), color: '--accent' }, { label: 'Prompt processing', values: c('prompt_tps'), color: '--series-2' }]} unit="tok/s" theme={ui.themeKey} label="Generation and prompt tokens per second" />
  </Panel>

  <div class="grid2">
    <Panel title="Time to first token" legend={[{ label: 'p50', color: '--accent' }, { label: 'p95', color: '--series-2' }, { label: 'p99', color: '--series-3' }]}>
      <LineChart t={cols.t} series={[{ label: 'p50', values: c('ttft_p50_ms'), color: '--accent' }, { label: 'p95', values: c('ttft_p95_ms'), color: '--series-2' }, { label: 'p99', values: c('ttft_p99_ms'), color: '--series-3' }]} fmt={ms} theme={ui.themeKey} label="Time to first token quantiles" />
    </Panel>
    <Panel title="Engine step time" legend={[{ label: 'p50', color: '--accent' }, { label: 'p99', color: '--series-2' }]}>
      <LineChart t={cols.t} series={[{ label: 'p50', values: c('step_p50_ms'), color: '--accent' }, { label: 'p99', values: c('step_p99_ms'), color: '--series-2' }]} fmt={ms} theme={ui.themeKey} label="Step time quantiles" />
    </Panel>
    <Panel title="Inter-token latency" legend={[{ label: 'p50', color: '--accent' }, { label: 'p99', color: '--series-2' }]}>
      <LineChart t={cols.t} series={[{ label: 'p50', values: c('itl_p50_ms'), color: '--accent' }, { label: 'p99', values: c('itl_p99_ms'), color: '--series-2' }]} fmt={ms} theme={ui.themeKey} label="Inter-token latency quantiles" />
    </Panel>
    <Panel title="Request latency and queue wait" legend={[{ label: 'End-to-end p50', color: '--accent' }, { label: 'End-to-end p99', color: '--series-2' }, { label: 'Queue wait p99', color: '--series-3' }]}>
      <LineChart t={cols.t} series={[{ label: 'End-to-end p50', values: c('e2e_p50_ms'), color: '--accent' }, { label: 'End-to-end p99', values: c('e2e_p99_ms'), color: '--series-2' }, { label: 'Queue wait p99', values: c('queue_p99_ms'), color: '--series-3' }]} fmt={ms} theme={ui.themeKey} label="End-to-end latency and queue wait" />
    </Panel>
  </div>

  <div class="grid2">
    <Panel title="Time to first token, distribution" note="requests per second in each latency bucket">
      {#if ttftHeat.rows.length}
        <Heatmap t={ttftHeat.t} rows={ttftHeat.rows} cells={ttftHeat.cells} theme={ui.themeKey} label="Heatmap of time to first token over time" />
      {:else}<p class="empty">No requests in this range.</p>{/if}
    </Panel>
    <Panel title="Step time, distribution" note="steps per second in each bucket">
      {#if stepHeat.rows.length}
        <Heatmap t={stepHeat.t} rows={stepHeat.rows} cells={stepHeat.cells} theme={ui.themeKey} label="Heatmap of engine step time over time" />
      {:else}<p class="empty">No engine steps in this range.</p>{/if}
    </Panel>
  </div>

  <Panel title="Requests finished, by reason" note="over the range">
    {#if reasons.length}
      <BarChart categories={reasons.map((r) => r.k)} series={[{ label: 'Requests', values: reasons.map((r) => r.v), color: '--accent' }]} fmt={(v) => compact(v, 0)} theme={ui.themeKey} label="Finished requests by reason" />
    {:else}<p class="empty">No finished requests in this range.</p>{/if}
    {#snippet table()}
      <table><thead><tr><th>Reason</th><th>Requests</th></tr></thead><tbody>{#each reasons as r}<tr><td>{r.k}</td><td>{compact(r.v, 0)}</td></tr>{/each}</tbody></table>
    {/snippet}
  </Panel>
  <p class="layout-reset"><button class="linkish" onclick={() => { resetLayout('throughput'); location.reload(); }}>Reset the panel layout</button></p>
</div>
