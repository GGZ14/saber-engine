<script>
  // A calendar heatmap: one cell a day (weeks as columns, Monday on top), the one-hue
  // sequential ramp by the day's value, month labels, a hover tooltip per day.
  import { compact } from './api.js';

  let { days = [], value = (d) => d.v, fmt = (v) => compact(v), label = '', what = 'tokens' } = $props();
  let tip = $state(null);
  let scroller;
  const CELL = 12;
  const GAP = 2;

  const grid = $derived.by(() => {
    if (!days.length) return { cols: [], months: [], max: 0 };
    const first = new Date(days[0].day + 'T00:00:00');
    const lead = (first.getDay() + 6) % 7; // Monday = 0
    const cols = [];
    let max = 0;
    days.forEach((d, i) => {
      const k = i + lead;
      const c = Math.floor(k / 7);
      (cols[c] ??= [])[k % 7] = d;
      const v = value(d) ?? 0;
      if (v > max) max = v;
    });
    const months = [];
    let lastM = -1;
    cols.forEach((col, c) => {
      const d = col.find(Boolean);
      if (!d) return;
      const m = new Date(d.day + 'T00:00:00').getMonth();
      if (m !== lastM) {
        months.push({ c, label: new Date(d.day + 'T00:00:00').toLocaleString([], { month: 'short' }) });
        lastM = m;
      }
    });
    return { cols, months, max };
  });
  // the newest week is the one that matters: keep the strip scrolled to its end
  $effect(() => {
    grid.cols.length;
    if (scroller) queueMicrotask(() => { scroller.scrollLeft = scroller.scrollWidth; });
  });
  function level(v) {
    if (!v || v <= 0 || grid.max <= 0) return 0;
    return 1 + Math.min(6, Math.floor((v / grid.max) * 6.999));
  }
</script>

<div class="cal" role="img" aria-label={label}>
  <div class="scroll" bind:this={scroller}>
    <svg width={grid.cols.length * (CELL + GAP) + 24} height={7 * (CELL + GAP) + 18}>
      {#each grid.months as m}
        <text x={24 + m.c * (CELL + GAP)} y="10" class="ax">{m.label}</text>
      {/each}
      {#each ['Mon', '', 'Wed', '', 'Fri', '', ''] as w, r}
        {#if w}<text x="0" y={18 + r * (CELL + GAP) + 10} class="ax">{w}</text>{/if}
      {/each}
      {#each grid.cols as col, c}
        {#each col as d, r}
          {#if d}
            <rect
              x={24 + c * (CELL + GAP)} y={16 + r * (CELL + GAP)} width={CELL} height={CELL} rx="2"
              style:fill="var(--seq-{level(value(d))})"
              role="presentation"
              onmouseenter={() => (tip = { d, x: Math.max(0, 24 + c * (CELL + GAP) - (scroller?.scrollLeft ?? 0) - 60), y: 16 + r * (CELL + GAP) })}
              onmouseleave={() => (tip = null)}
            />
          {/if}
        {/each}
      {/each}
    </svg>
  </div>
  {#if tip}
    <div class="tip" style:left="{tip.x}px" style:top="{tip.y - 34}px">
      <b>{fmt(value(tip.d) ?? 0)}</b> {what} <span class="muted">{new Date(tip.d.day + 'T00:00:00').toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}</span>
    </div>
  {/if}
  <div class="scale" aria-hidden="true">
    <span>less</span>{#each [0, 1, 2, 3, 4, 5, 6, 7] as i}<i style:background="var(--seq-{i})"></i>{/each}<span>more</span>
  </div>
</div>

<style>
  .cal { position: relative; }
  .scroll { overflow-x: auto; }
  svg { display: block; }
  .ax { fill: var(--muted); font-size: 10px; }
  rect:hover { stroke: var(--ink); stroke-width: 1; }
  .tip {
    position: absolute; pointer-events: none; z-index: 2; white-space: nowrap; background: var(--surface);
    border: 1px solid var(--ring); border-radius: var(--radius); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    padding: 4px 8px; font-size: var(--fs-xs);
  }
  .muted { color: var(--muted); }
  .scale { display: flex; align-items: center; gap: 2px; font-size: var(--fs-xs); color: var(--muted); margin-top: 6px; }
  .scale i { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  .scale span { margin: 0 4px; }
</style>
