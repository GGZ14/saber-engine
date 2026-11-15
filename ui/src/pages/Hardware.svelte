<script>
  // GPUs and host, read by the console from sysfs / /proc (the engine exports none of this).
  // Read-only: the power cap is shown, never changed.
  import { sortable, resetLayout } from '../lib/sortable.js';
  import { api, bytes, compact, pct, groupBy, seriesColor, last } from '../lib/api.js';
  import { ui, liveS, fetchSeries, guarded } from '../lib/store.svelte.js';
  import LineChart from '../lib/LineChart.svelte';
  import Panel from '../lib/Panel.svelte';
  import Tile from '../lib/Tile.svelte';

  let hw = $state(null);
  let win = $state({ t: [], series: {} });
  async function reload() {
    try { hw = await guarded(() => api('/api/hw')); } catch (e) {}
    try { win = (await fetchSeries(['hw_'], liveS(), 400)) ?? win; } catch (e) {}
  }
  $effect(() => { ui.tick; ui.range; reload(); });

  const by = (name) => groupBy(win, name, 'gpu');
  const gpuKeys = $derived(Object.keys(by('hw_gpu_power_watts')).sort());
  const lines = (name, f = (v) => v) => {
    const g = by(name);
    return Object.keys(g).sort().map((k, i) => ({ label: `GPU ${k}`, values: g[k].map((v) => (v == null ? null : f(v))), color: seriesColor(i) }));
  };
  const legend = $derived(gpuKeys.map((k, i) => ({ label: `GPU ${k}`, color: seriesColor(i) })));
  const host = $derived(hw?.host ?? {});
  const one = (n) => groupBy(win, n, null)[''] ?? [];
  const memUsedPct = $derived(one('hw_host_mem_total_bytes').map((t, i) => {
    const a = one('hw_host_mem_available_bytes')[i];
    return t && a != null ? ((t - a) / t) * 100 : null;
  }));
  const tone = (c) => (c == null ? '' : c >= 95 ? 'hot' : c >= 85 ? 'warm' : '');
</script>

<div class="page" use:sortable={'hardware'}>
  <h1 class="title">Hardware</h1>
  {#if hw?.gpus?.length}
    <section class="gpus">
      {#each hw.gpus as g}
        <article class="panel gpu">
          <div class="c-head">
            <h2>GPU {g.card}</h2>
            <span class="muted mono">{g.bdf}</span>
          </div>
          <dl class="kv">
            <dt>Power</dt><dd>{g.power_w == null ? '–' : `${g.power_w.toFixed(0)} W`}{g.power_cap_w ? ` of ${g.power_cap_w.toFixed(0)} W cap` : ''}</dd>
            <dt>Temperature</dt><dd>edge {g.temp_edge_c?.toFixed(0) ?? '–'} °C · <span class={tone(g.temp_junction_c)}>junction {g.temp_junction_c?.toFixed(0) ?? '–'} °C{#if tone(g.temp_junction_c) === 'hot'} (hot){/if}</span> · mem {g.temp_mem_c?.toFixed(0) ?? '–'} °C</dd>
            <dt>Clocks</dt><dd>shader {g.sclk_mhz?.toFixed(0) ?? '–'} MHz · memory {g.mclk_mhz?.toFixed(0) ?? '–'} MHz</dd>
            <dt>Busy</dt><dd>{g.busy_pct == null ? '–' : `${g.busy_pct.toFixed(0)}%`}</dd>
            {#if g.fan_rpm != null}<dt>Fan</dt><dd>{compact(g.fan_rpm)} rpm</dd>{/if}
            <dt>VRAM</dt><dd>{bytes(g.vram_used)} of {bytes(g.vram_total)} ({pct(g.vram_total ? g.vram_used / g.vram_total : null)})</dd>
          </dl>
          <div class="meter" role="img" aria-label="VRAM in use"><span style:width="{g.vram_total ? Math.min(100, (g.vram_used / g.vram_total) * 100) : 0}%"></span></div>
        </article>
      {/each}
    </section>
  {:else}
    <section class="panel"><p class="empty">No AMD GPU found in /sys/class/drm.</p></section>
  {/if}

  <div class="grid2">
    <Panel title="GPU power" {legend} note="watts">
      <LineChart t={win.t} series={lines('hw_gpu_power_watts')} unit="W" theme={ui.themeKey} label="Power per GPU" />
    </Panel>
    <Panel title="GPU junction temperature" {legend} note="°C">
      <LineChart t={win.t} series={lines('hw_gpu_temp_junction_celsius')} unit="°C" theme={ui.themeKey} label="Junction temperature per GPU" />
    </Panel>
    <Panel title="Shader clock" {legend} note="MHz">
      <LineChart t={win.t} series={lines('hw_gpu_sclk_mhz')} unit="MHz" theme={ui.themeKey} label="Shader clock per GPU" />
    </Panel>
    <Panel title="VRAM in use" {legend}>
      <LineChart t={win.t} series={lines('hw_gpu_vram_used_bytes')} fmt={(v) => bytes(v, 1)} theme={ui.themeKey} label="VRAM in use per GPU" />
    </Panel>
    <Panel title="GPU busy" {legend} note="percent">
      <LineChart t={win.t} series={lines('hw_gpu_busy_percent')} fmt={(v) => `${v.toFixed(0)}%`} ymax={100} theme={ui.themeKey} label="GPU busy percent" />
    </Panel>
    <Panel title="Memory clock" {legend} note="MHz">
      <LineChart t={win.t} series={lines('hw_gpu_mclk_mhz')} unit="MHz" theme={ui.themeKey} label="Memory clock per GPU" />
    </Panel>
  </div>

  <h2 class="sect">Host</h2>
  <section class="tiles">
    <Tile title="Load" value={host.load1?.toFixed(2) ?? '–'} unit="1 min" sub="5 min {host.load5?.toFixed(2) ?? '–'} · 15 min {host.load15?.toFixed(2) ?? '–'}{host.cpus ? ` · ${host.cpus} CPUs` : ''}" spark={{ t: win.t, v: one('hw_host_load1') }} theme={ui.themeKey} />
    <Tile title="CPU busy" value={host.cpu_busy_pct == null ? '–' : `${host.cpu_busy_pct.toFixed(0)}%`} sub="all cores" spark={{ t: win.t, v: one('hw_host_cpu_busy_percent') }} theme={ui.themeKey} />
    <Tile title="Memory available" value={bytes(host.mem_available)} sub="of {bytes(host.mem_total)}{host.swap_total ? ` · swap free ${bytes(host.swap_free)}` : ''}" spark={{ t: win.t, v: one('hw_host_mem_available_bytes') }} theme={ui.themeKey} />
  </section>
  <div class="grid2">
    <Panel title="Host load" legend={[{ label: '1 min', color: '--accent' }, { label: '15 min', color: '--series-2' }]}>
      <LineChart t={win.t} series={[{ label: '1 min', values: one('hw_host_load1'), color: '--accent' }, { label: '15 min', values: one('hw_host_load15'), color: '--series-2' }]} theme={ui.themeKey} label="Host load average" />
    </Panel>
    <Panel title="Host memory in use" note="percent of RAM">
      <LineChart t={win.t} series={[{ label: 'In use', values: memUsedPct, color: '--accent' }]} fmt={(v) => `${v.toFixed(0)}%`} ymax={100} theme={ui.themeKey} label="Host memory in use" />
    </Panel>
  </div>
  <p class="note">Host DRAM bandwidth is not shown: this host has no unprivileged memory-bandwidth counter. Last sample {hw?.t ? new Date(hw.t * 1000).toLocaleTimeString() : '–'}{last(one('hw_host_load1')) == null ? '' : ''}.</p>
  <p class="layout-reset"><button class="linkish" onclick={() => { resetLayout('hardware'); location.reload(); }}>Reset the panel layout</button></p>
</div>

<style>
  .gpus { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
  .mono { font-family: var(--mono); font-size: var(--fs-xs); }
  .sect { margin-top: 6px; }
  .hot { color: var(--critical); font-weight: 600; }
  .warm { color: var(--ink); font-weight: 600; }
</style>
