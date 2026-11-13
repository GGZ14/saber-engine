<script>
  // A time-series panel: one y-axis, thin lines, a recessive grid, a crosshair with a tooltip.
  // `series` = [{ label, values, color: '--css-var' }]; nulls are gaps (engine down).
  // `fmt` formats a value for the axis and the tooltip (default: compact + unit); `ymax` pins
  // the top (100 for a percentage).
  import uPlot from 'uplot';
  import { onDestroy, untrack } from 'svelte';
  import { compact } from './api.js';

  let { t = [], series = [], unit = '', theme = '', height = 220, label = '', fmt = null, ymax = null } = $props();
  const fv = (v) => (fmt ? fmt(v) : `${compact(v)} ${unit}`.trim());
  const fa = (v) => (fmt ? fmt(v) : compact(v, 0));
  let el;
  let plot;
  let width = $state(0);
  let tip = $state(null);

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmtTime = (s) =>
    new Date(s * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  function build() {
    plot?.destroy();
    plot = null;
    if (!el || width < 50) return;
    const muted = css('--muted');
    const grid = css('--grid');
    const axis = css('--axis');
    const font = `12px ${css('--font')}`;
    const ax = {
      stroke: muted,
      font,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: axis, width: 1, size: 4 },
    };
    const ser = untrack(() => series);
    plot = new uPlot(
      {
        width,
        height,
        padding: [10, 12, 0, 0],
        legend: { show: false },
        select: { show: false },
        cursor: {
          drag: { x: false, y: false },
          points: { size: 8, width: 2, fill: css('--surface') },
        },
        scales: {
          x: { time: true, range: (u, lo, hi) => [lo, hi] },
          y: { range: (u, lo, hi) => [0, ymax ?? Math.max(fmt ? 1e-9 : 1, (hi ?? 1) * 1.15)] },
        },
        axes: [
          { ...ax, space: 70 },
          {
            ...ax,
            // wide enough for the longest tick label (formatted bytes / durations run long)
            size: (u, vals) => Math.max(44, 14 + 7 * Math.max(0, ...(vals ?? []).map((v) => String(v ?? '').length))),
            values: (u, vals) => vals.map(fa),
          },
        ],
        series: [
          {},
          ...ser.map((s, i) => ({
            label: s.label,
            stroke: css(s.color),
            width: 2,
            fill: i === 0 && ser.length === 1 ? css('--accent-fill') : undefined,
            points: { show: false },
          })),
        ],
        hooks: {
          setCursor: [
            (u) => {
              const i = u.cursor.idx;
              if (i == null || u.cursor.left < 0) {
                tip = null;
                return;
              }
              tip = {
                left: u.over.offsetLeft + u.cursor.left,
                top: u.over.offsetTop + u.cursor.top,
                time: fmtTime(u.data[0][i]),
                rows: ser.map((s, k) => ({ label: s.label, color: s.color, v: u.data[k + 1][i] })),
                flip: u.cursor.left > u.bbox.width / devicePixelRatio - 170,
              };
            },
          ],
        },
      },
      untrack(() => [t, ...series.map((s) => s.values)]),
      el,
    );
  }

  $effect(() => {
    theme; width; series.length;
    build();
  });
  $effect(() => {
    const d = [t, ...series.map((s) => s.values)];
    if (plot) plot.setData(d);
  });
  onDestroy(() => plot?.destroy());
</script>

<div class="wrap" role="img" aria-label={label}>
  <div class="plot" bind:this={el} bind:clientWidth={width}></div>
  {#if tip}
    <div class="tip" style:left="{tip.left}px" style:top="{Math.max(0, tip.top - 10)}px" class:flip={tip.flip}>
      <div class="tt">{tip.time}</div>
      {#each tip.rows as r}
        <div class="tr">
          <span class="sw" style:background="var({r.color})"></span>
          <span class="tl">{r.label}</span>
          <span class="tv">{r.v == null ? 'no data' : fv(r.v)}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .wrap { position: relative; }
  .plot { width: 100%; }
  .tip {
    position: absolute;
    pointer-events: none;
    background: var(--surface);
    border: 1px solid var(--ring);
    border-radius: var(--radius);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    padding: 6px 8px;
    font-size: var(--fs-xs);
    min-width: 150px;
    transform: translateX(8px);
    z-index: 2;
  }
  .tip.flip { transform: translateX(calc(-100% - 8px)); }
  .tt { color: var(--muted); margin-bottom: 2px; font-variant-numeric: tabular-nums; }
  .tr { display: flex; align-items: center; gap: 6px; }
  .sw { width: 10px; height: 3px; border-radius: 2px; flex: none; }
  .tl { color: var(--ink-2); flex: 1; }
  .tv { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
</style>
