<script>
  // The memory plane, from the engine's /metrics families: the KV pool, the elastic lend /
  // reclaim controller, the expert tier's hit rate by path, tier moves and bytes, PLE reads and
  // the byte ledger. A family the engine does not export shows as absent, never as zero.
  import { sortable, resetLayout } from '../lib/sortable.js';
  import { compact, bytes, pct, last, total, groupBy, seriesColor, ratioCols } from '../lib/api.js';
  import { ui, liveS, fetchSeries } from '../lib/store.svelte.js';
  import LineChart from '../lib/LineChart.svelte';
  import BarChart from '../lib/BarChart.svelte';
  import Panel from '../lib/Panel.svelte';
  import Tile from '../lib/Tile.svelte';

  let win = $state({ t: [], series: {} });
  async function reload() {
    try {
      win = (await fetchSeries(['kv_', 'gpu_cache_usage_perc', 'admission_refused_kv', 'saber_fn_kv_', 'saber_xtier_', 'saber_ple_', 'saber_ledger_', 'saber_fn_ptree_'], liveS(), 300)) ?? win;
    } catch (e) {}
  }
  $effect(() => { ui.tick; ui.range; reload(); });

  const one = (name) => groupBy(win, name, null)[''] ?? null;
  const dt = $derived(win.interval_s || 5);
  const has = (prefix) => Object.keys(win.series).some((k) => k.startsWith(prefix));

  // KV pool
  const kvTotal = $derived.by(() => {
    const v = one('kv_blocks_total');
    return v && (last(v) ?? 0) > 0 ? v : null;
  });
  const kvFree = $derived(one('kv_blocks_free'));
  const kvUsed = $derived(kvTotal && kvFree ? kvTotal.map((t, i) => (t == null || kvFree[i] == null || !t ? null : ((t - kvFree[i]) / t) * 100)) : null);
  const cacheUsage = $derived(one('gpu_cache_usage_perc'));
  const kvPct = $derived(kvUsed ?? (cacheUsage ? cacheUsage.map((v) => (v == null ? null : v * 100)) : null));
  const kvHot = $derived(groupBy(win, 'saber_fn_kv_hot_pages', 'kind'));
  const kvRes = $derived(one('kv_reserved_tokens'));

  // elastic controller
  const lent = $derived(groupBy(win, 'saber_xtier_elastic_lent_slots', 'rank'));
  const elasticMoves = $derived(Object.entries(groupBy(win, 'saber_xtier_elastic_moves_total', 'kind')).map(([k, v]) => ({ k, v: (total(v) ?? 0) * dt })).filter((x) => x.k !== 'tick'));

  // expert tier by path: share of routed experts each path served
  const routed = $derived(groupBy(win, 'saber_xtier_routed_total', 'path'));
  const PATHS = ['resident', 'prefetched', 'ram', 'nvme'];
  const routedSum = $derived(win.t.map((_, i) => PATHS.reduce((s, p) => s + (routed[p]?.[i] ?? 0), 0)));
  const share = $derived(Object.fromEntries(PATHS.filter((p) => routed[p]).map((p) => [p, ratioCols(routed[p], routedSum, 100)])));
  const hitRange = $derived.by(() => {
    const all = PATHS.reduce((s, p) => s + (total(routed[p]) ?? 0), 0);
    return all ? ((total(routed.resident) ?? 0) + (total(routed.prefetched) ?? 0)) / all : null;
  });

  // tier moves and bytes
  const moves = $derived(Object.entries(groupBy(win, 'saber_xtier_moves_total', 'kind')).map(([k, v]) => ({ k, v: (total(v) ?? 0) * dt })).sort((a, b) => b.v - a.v));
  const tierBytes = $derived(groupBy(win, 'saber_xtier_bytes_total', 'dir'));
  const records = $derived(groupBy(win, 'saber_xtier_records', 'tier'));

  // PLE
  const pleRows = $derived(groupBy(win, 'saber_ple_rows_total', 'path'));
  const pleBytes = $derived(one('saber_ple_bytes_read_total'));

  // the byte ledger (latest values)
  const ledger = $derived.by(() => {
    const cap = groupBy(win, 'saber_ledger_capacity_bytes', 'pool');
    const res = groupBy(win, 'saber_ledger_reserved_bytes', 'pool');
    const wr = groupBy(win, 'saber_ledger_written_bytes_total', 'pool');
    return Object.keys(cap).sort().map((p) => ({ pool: p, cap: last(cap[p]), res: last(res[p] ?? []), wr: last(wr[p] ?? []) }));
  });
  const legend = (keys) => keys.map((k, i) => ({ label: k, color: seriesColor(i) }));
  const lines = (g, keys, f = (v) => v) => keys.map((k, i) => ({ label: k, values: (g[k] ?? []).map((v) => (v == null ? null : f(v))), color: seriesColor(i) }));
</script>

<div class="page" use:sortable={'memory'}>
  <h1 class="title">Memory plane</h1>
  <section class="tiles four">
    <Tile title="KV pool in use" value={kvPct ? `${(last(kvPct) ?? 0).toFixed(1)}%` : '–'} sub={kvTotal ? `${compact(last(kvTotal))} blocks` : 'not exported by this engine'} spark={kvPct ? { t: win.t, v: kvPct } : null} theme={ui.themeKey} />
    <Tile title="Expert hits in VRAM" value={pct(hitRange)} sub={hitRange == null ? 'no expert tier (or idle)' : 'resident + prefetched, over the range'} spark={share.resident ? { t: win.t, v: share.resident } : null} theme={ui.themeKey} />
    <Tile title="Elastic slots lent" value={Object.keys(lent).length ? compact(Object.values(lent).reduce((s, v) => s + (last(v) ?? 0), 0)) : '–'} sub={Object.keys(lent).length ? 'expert slots lent to the KV pool, all ranks' : 'no elastic controller'} />
    <Tile title="PLE reads" value={pleBytes ? bytes(last(pleBytes)) : '–'} unit={pleBytes ? '/s' : ''} sub={pleBytes ? 'from the drive tier' : 'no PLE tier'} spark={pleBytes ? { t: win.t, v: pleBytes } : null} theme={ui.themeKey} />
  </section>

  <div class="grid2">
    <Panel title="KV cache" legend={kvRes ? [{ label: 'Blocks in use', color: '--accent' }] : []} note="percent of the pool">
      {#if kvPct}
        <LineChart t={win.t} series={[{ label: 'In use', values: kvPct, color: '--accent' }]} fmt={(v) => `${v.toFixed(0)}%`} ymax={100} theme={ui.themeKey} label="KV pool use over time" />
      {:else if Object.keys(kvHot).length}
        <LineChart t={win.t} series={lines(kvHot, Object.keys(kvHot))} unit="pages" theme={ui.themeKey} label="KV hot pages by kind" />
      {:else}<p class="empty">The engine exports no KV pool gauge.</p>{/if}
    </Panel>
    <Panel title="Elastic lend and reclaim" note="moves over the range">
      {#if elasticMoves.length}
        <BarChart categories={elasticMoves.map((m) => m.k)} series={[{ label: 'Moves', values: elasticMoves.map((m) => m.v), color: '--accent' }]} fmt={(v) => compact(v, 0)} theme={ui.themeKey} label="Elastic controller moves by kind" />
        {#if Object.keys(lent).length}
          <LineChart t={win.t} series={[{ label: 'Slots lent, all ranks', values: win.t.map((_, i) => Object.values(lent).reduce((a, v) => (v[i] == null ? a : (a ?? 0) + v[i]), null)), color: '--accent' }]} unit="slots" theme={ui.themeKey} height={140} label="Expert slots lent to the KV pool, all ranks" />
          <p class="note">Slots lent to the KV pool, all ranks.</p>
        {/if}
      {:else}<p class="empty">No elastic controller on this engine.</p>{/if}
    </Panel>
  </div>

  <Panel title="Expert cache, routed experts by path" legend={legend(Object.keys(share))} note="share of routed experts each path served">
    {#if Object.keys(share).length}
      <LineChart t={win.t} series={lines(share, Object.keys(share))} fmt={(v) => `${v.toFixed(0)}%`} ymax={100} theme={ui.themeKey} label="Share of routed experts by path" />
    {:else}<p class="empty">No expert tier on this engine.</p>{/if}
    {#snippet table()}
      <table><thead><tr><th>Path</th><th>Experts routed over the range</th></tr></thead><tbody>{#each PATHS.filter((p) => routed[p]) as p}<tr><td>{p}</td><td>{compact((total(routed[p]) ?? 0) * dt, 0)}</td></tr>{/each}</tbody></table>
    {/snippet}
  </Panel>

  <div class="grid2">
    <Panel title="Tier moves" note="over the range">
      {#if moves.length}
        <BarChart categories={moves.map((m) => m.k)} series={[{ label: 'Moves', values: moves.map((m) => m.v), color: '--accent' }]} fmt={(v) => compact(v, 0)} theme={ui.themeKey} label="Expert tier moves by kind" />
      {:else}<p class="empty">No tier moves exported.</p>{/if}
    </Panel>
    <Panel title="Tier traffic" legend={legend(Object.keys(tierBytes))} note="bytes per second">
      {#if Object.keys(tierBytes).length}
        <LineChart t={win.t} series={lines(tierBytes, Object.keys(tierBytes))} fmt={(v) => `${bytes(v, 0)}/s`} theme={ui.themeKey} label="Expert tier bytes per second by direction" />
      {:else}<p class="empty">No tier traffic exported.</p>{/if}
    </Panel>
    <Panel title="PLE rows" legend={legend(Object.keys(pleRows))} note="rows per second by path">
      {#if Object.keys(pleRows).length}
        <LineChart t={win.t} series={lines(pleRows, Object.keys(pleRows))} unit="rows/s" theme={ui.themeKey} label="PLE rows per second by path" />
      {:else}<p class="empty">No PLE tier on this engine.</p>{/if}
    </Panel>
    <Panel title="Expert records by tier" legend={legend(Object.keys(records))}>
      {#if Object.keys(records).length}
        <LineChart t={win.t} series={lines(records, Object.keys(records))} unit="records" theme={ui.themeKey} label="Expert records by tier" />
      {:else}<p class="empty">No expert tier on this engine.</p>{/if}
    </Panel>
  </div>

  <Panel title="Byte ledger" note="latest">
    {#if ledger.length}
      <div class="scroll">
        <table>
          <thead><tr><th>Pool</th><th>Capacity</th><th>Reserved</th><th class="l">Use</th><th>Written</th></tr></thead>
          <tbody>
            {#each ledger as l}
              <tr>
                <td>{l.pool}</td><td>{bytes(l.cap)}</td><td>{bytes(l.res)}</td>
                <td class="l"><div class="meter" style="width: 120px" title={pct(l.cap ? l.res / l.cap : null)}><span style:width="{l.cap ? Math.min(100, (l.res / l.cap) * 100) : 0}%"></span></div></td>
                <td>{l.wr == null ? '–' : `${bytes(l.wr)}/s`}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}<p class="empty">The engine exports no ledger.</p>{/if}
  </Panel>
  {#if !has('saber_') && !has('kv_')}
    <p class="note">This engine exports none of the memory-plane families (the mock engine does not). They appear with Flash-Next or the 27B engine.</p>
  {/if}
  <p class="layout-reset"><button class="linkish" onclick={() => { resetLayout('memory'); location.reload(); }}>Reset the panel layout</button></p>
</div>
