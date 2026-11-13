<script>
  import { sortable, resetLayout } from '../lib/sortable.js';
  import { compact, pct, last, meanOf, total, groupBy } from '../lib/api.js';
  import { ui, liveS, fetchCols, fetchSeries } from '../lib/store.svelte.js';
  import LineChart from '../lib/LineChart.svelte';
  import BarChart from '../lib/BarChart.svelte';
  import Panel from '../lib/Panel.svelte';
  import Tile from '../lib/Tile.svelte';

  let cols = $state({ t: [] });
  let win = $state({ t: [], series: {} });
  async function reload() {
    try { cols = (await fetchCols(liveS())) ?? cols; } catch (e) {}
    try {
      win = (await fetchSeries(['spec_decode_', 'saber_fn_mtp_'], liveS(), 240)) ?? win;
    } catch (e) {}
  }
  $effect(() => { ui.tick; ui.range; reload(); });
  const c = (k) => cols[k] ?? [];

  // acceptance per draft position over the range: accepted at k / drafts (the engine's
  // per-position counters), or Flash-Next's MTP accepted / evaluated per position
  const perPos = $derived.by(() => {
    const drafts = total(groupBy(win, 'spec_decode_num_drafts_total', null)[''] ?? []);
    const acc = groupBy(win, 'spec_decode_num_accepted_tokens_per_pos_total', 'position');
    const mtpA = groupBy(win, 'saber_fn_mtp_accepted_per_pos_total', 'pos');
    const mtpE = groupBy(win, 'saber_fn_mtp_evaluated_per_pos_total', 'pos');
    const rows = [];
    if (Object.keys(acc).length && drafts) {
      for (const [p, v] of Object.entries(acc)) rows.push({ p: +p, rate: (total(v) ?? 0) / drafts, src: 'engine' });
    } else if (Object.keys(mtpA).length) {
      for (const [p, v] of Object.entries(mtpA)) {
        const e = total(mtpE[p] ?? []);
        rows.push({ p: +p, rate: e ? (total(v) ?? 0) / e : null, src: 'mtp' });
      }
    }
    return rows.filter((r) => Number.isFinite(r.p)).sort((a, b) => a.p - b.p);
  });
  const draftsPerS = $derived(groupBy(win, 'spec_decode_num_drafts_total', null)[''] ?? []);
</script>

<div class="page" use:sortable={'spec'}>
  <h1 class="title">Speculative decoding</h1>
  <section class="tiles">
    <Tile title="Draft acceptance" value={pct(last(c('accept_rate')))} sub="mean {pct(meanOf(c('accept_rate')))} over the range" spark={{ t: cols.t, v: c('accept_rate') }} theme={ui.themeKey} />
    <Tile title="Accepted per draft" value={compact(last(c('accepted_per_draft')))} unit="tokens" sub="mean {compact(meanOf(c('accepted_per_draft')))}" spark={{ t: cols.t, v: c('accepted_per_draft') }} theme={ui.themeKey} />
    <Tile title="Drafts" value={compact(last(draftsPerS))} unit="/s" sub="mean {compact(meanOf(draftsPerS))}" spark={{ t: win.t, v: draftsPerS }} theme={ui.themeKey} />
  </section>

  <Panel title="Acceptance per draft position" note={perPos[0]?.src === 'mtp' ? 'MTP accepted / evaluated at each depth, over the range' : 'accepted at each position / drafts, over the range'}>
    {#if perPos.length}
      <BarChart categories={perPos.map((r) => `pos ${r.p}`)} series={[{ label: 'Acceptance', values: perPos.map((r) => r.rate), color: '--accent' }]} fmt={(v) => pct(v, 0)} theme={ui.themeKey} label="Acceptance rate at each draft position" />
    {:else}<p class="empty">No drafts in this range (speculative decoding off, or the engine idle).</p>{/if}
    {#snippet table()}
      <table><thead><tr><th>Position</th><th>Acceptance</th></tr></thead><tbody>{#each perPos as r}<tr><td>{r.p}</td><td>{pct(r.rate)}</td></tr>{/each}</tbody></table>
    {/snippet}
  </Panel>

  <div class="grid2">
    <Panel title="Acceptance rate">
      <LineChart t={cols.t} series={[{ label: 'Acceptance', values: c('accept_rate').map((v) => (v == null ? null : v * 100)), color: '--accent' }]} fmt={(v) => `${v.toFixed(0)}%`} ymax={100} theme={ui.themeKey} label="Draft acceptance rate over time" />
    </Panel>
    <Panel title="Accepted tokens per draft">
      <LineChart t={cols.t} series={[{ label: 'Accepted per draft', values: c('accepted_per_draft'), color: '--accent' }]} unit="tokens" theme={ui.themeKey} label="Accepted tokens per draft over time" />
    </Panel>
  </div>
  <p class="layout-reset"><button class="linkish" onclick={() => { resetLayout('spec'); location.reload(); }}>Reset the panel layout</button></p>
</div>
