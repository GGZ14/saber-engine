<script>
  // Vertical bars by category, one or more series side by side (fixed categorical order).
  // Thin bars, 4px rounded data-ends anchored to the baseline, a 2px gap between neighbours,
  // a recessive grid, a hover tooltip per bar group.
  import { compact } from './api.js';

  let { categories = [], series = [], fmt = null, height = 200, label = '', theme = '' } = $props();
  let width = $state(0);
  let tip = $state(null);
  const f = (v) => (v == null ? 'no data' : fmt ? fmt(v) : compact(v));

  const PAD = { l: 56, r: 8, t: 10, b: 26 };
  // a "nice" top: 1, 2, 2.5 or 5 x 10^k above the largest bar, so the four ticks are round
  const max = $derived.by(() => {
    let m = 0;
    for (const s of series) for (const v of s.values) if (v != null && v > m) m = v;
    if (m <= 0) return 1;
    const step = m / 4;
    const p = 10 ** Math.floor(Math.log10(step));
    const n = [1, 2, 2.5, 5, 10].find((k) => k * p >= step) * p;
    return n * 4;
  });
  const ticks = $derived([0, 0.25, 0.5, 0.75, 1].map((k) => k * max));
  const geo = $derived.by(() => {
    const iw = Math.max(10, width - PAD.l - PAD.r);
    const ih = height - PAD.t - PAD.b;
    const n = Math.max(1, categories.length);
    const band = iw / n;
    const ns = Math.max(1, series.length);
    const groupW = Math.min(band * 0.72, 28 * ns + 2 * (ns - 1));
    const bw = Math.max(2, (groupW - 2 * (ns - 1)) / ns);
    return { iw, ih, band, bw, groupW };
  });
  const y = (v) => PAD.t + geo.ih - (v / max) * geo.ih;
  // a bar with rounded top corners only
  function bar(x, v) {
    const top = y(v);
    const h = PAD.t + geo.ih - top;
    if (h <= 0.5) return '';
    const r = Math.min(4, geo.bw / 2, h);
    const b = PAD.t + geo.ih;
    return `M${x},${b}V${top + r}Q${x},${top} ${x + r},${top}H${x + geo.bw - r}Q${x + geo.bw},${top} ${x + geo.bw},${top + r}V${b}Z`;
  }
  // thin the category labels so they never collide
  const labelEvery = $derived(Math.max(1, Math.ceil((categories.length * 54) / Math.max(1, geo.iw))));
</script>

<div class="wrap" bind:clientWidth={width} role="img" aria-label={label}>
  {#if width > 0}
    <svg {width} {height} data-theme-key={theme}>
      {#each ticks as tk}
        <line x1={PAD.l} x2={PAD.l + geo.iw} y1={y(tk)} y2={y(tk)} class="grid" />
        <text x={PAD.l - 6} y={y(tk) + 4} class="ax" text-anchor="end">{fmt ? fmt(tk) : compact(tk, 0)}</text>
      {/each}
      {#each categories as c, i}
        {@const gx = PAD.l + i * geo.band + (geo.band - geo.groupW) / 2}
        {#each series as s, k}
          {#if s.values[i] != null}
            <path d={bar(gx + k * (geo.bw + 2), s.values[i])} style:fill="var({s.color})" />
          {/if}
        {/each}
        <rect
          x={PAD.l + i * geo.band} y={PAD.t} width={geo.band} height={geo.ih}
          class="hit" role="presentation"
          onmouseenter={() => (tip = { i, x: PAD.l + i * geo.band + geo.band / 2 })}
          onmouseleave={() => (tip = null)}
        />
        {#if i % labelEvery === 0}
          <text x={PAD.l + i * geo.band + geo.band / 2} y={height - 8} class="ax" text-anchor="middle">{c}</text>
        {/if}
      {/each}
      <line x1={PAD.l} x2={PAD.l + geo.iw} y1={PAD.t + geo.ih} y2={PAD.t + geo.ih} class="base" />
    </svg>
    {#if tip}
      <div class="tip" style:left="{tip.x}px" class:flip={tip.x > width - 170}>
        <div class="tt">{categories[tip.i]}</div>
        {#each series as s}
          <div class="tr"><span class="sw" style:background="var({s.color})"></span><span class="tl">{s.label}</span><span class="tv">{f(s.values[tip.i])}</span></div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  .wrap { position: relative; width: 100%; }
  svg { display: block; overflow: visible; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .base { stroke: var(--axis); stroke-width: 1; }
  .ax { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .hit { fill: transparent; }
  .hit:hover { fill: var(--ring); opacity: 0.35; }
  .tip {
    position: absolute; top: 4px; pointer-events: none; z-index: 2;
    background: var(--surface); border: 1px solid var(--ring); border-radius: var(--radius);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12); padding: 6px 8px; font-size: var(--fs-xs);
    min-width: 140px; transform: translateX(8px);
  }
  .tip.flip { transform: translateX(calc(-100% - 8px)); }
  .tt { color: var(--muted); margin-bottom: 2px; }
  .tr { display: flex; align-items: center; gap: 6px; }
  .sw { width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .tl { color: var(--ink-2); flex: 1; }
  .tv { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
</style>
