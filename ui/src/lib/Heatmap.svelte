<script>
  // A latency distribution over time: columns = time intervals, rows = histogram buckets
  // (low at the bottom), cell = requests in that bucket in that interval, on the one-hue
  // sequential ramp (near-zero recedes to the surface). Hover names the cell.
  import { onDestroy } from 'svelte';
  import { compact } from './api.js';

  let { t = [], rows = [], cells = [], theme = '', height = 200, label = '', unit = '' } = $props();
  let canvas;
  let width = $state(0);
  let tip = $state(null);
  const PAD = { l: 64, r: 8, t: 6, b: 22 };
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmtTime = (s) => new Date(s * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const max = $derived.by(() => {
    let m = 0;
    for (const c of cells) for (const v of c) if (v > m) m = v;
    return m;
  });
  function level(v) {
    if (!v || v <= 0 || max <= 0) return 0;
    return 1 + Math.min(6, Math.floor((Math.log1p(v) / Math.log1p(max)) * 6.999));
  }

  function draw() {
    if (!canvas || width < 50) return;
    const dpr = devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    const ramp = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => css(`--seq-${i}`));
    const iw = width - PAD.l - PAD.r;
    const ih = height - PAD.t - PAD.b;
    const nc = Math.max(1, cells.length);
    const nr = Math.max(1, rows.length);
    const cw = iw / nc;
    const ch = ih / nr;
    for (let c = 0; c < cells.length; c++) {
      for (let r = 0; r < rows.length; r++) {
        ctx.fillStyle = ramp[level(cells[c][r])];
        // a 1px surface gap between cells when they are big enough to show it
        const g = cw > 4 ? 1 : 0;
        ctx.fillRect(PAD.l + c * cw, PAD.t + (nr - 1 - r) * ch, Math.max(1, cw - g), Math.max(1, ch - 1));
      }
    }
    ctx.fillStyle = css('--muted');
    ctx.font = `11px ${css('--font')}`;
    ctx.textAlign = 'right';
    const every = Math.max(1, Math.ceil((nr * 14) / ih));
    for (let r = 0; r < rows.length; r += every) ctx.fillText(rows[r].label, PAD.l - 6, PAD.t + (nr - 1 - r) * ch + ch / 2 + 4);
    ctx.textAlign = 'center';
    const ticks = Math.max(2, Math.floor(iw / 90));
    for (let k = 0; k <= ticks; k++) {
      const c = Math.min(cells.length - 1, Math.round((k / ticks) * (cells.length - 1)));
      if (c >= 0 && t[c] != null) ctx.fillText(fmtTime(t[c]), PAD.l + c * cw + cw / 2, height - 6);
    }
  }

  $effect(() => {
    theme; width; cells; rows;
    draw();
  });
  onDestroy(() => {});

  function move(e) {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const iw = width - PAD.l - PAD.r;
    const ih = height - PAD.t - PAD.b;
    const c = Math.floor(((x - PAD.l) / iw) * cells.length);
    const rr = rows.length - 1 - Math.floor(((y - PAD.t) / ih) * rows.length);
    if (c < 0 || c >= cells.length || rr < 0 || rr >= rows.length) { tip = null; return; }
    tip = { x, y, time: fmtTime(t[c]), row: rows[rr].label, v: cells[c][rr] ?? 0, flip: x > width - 170 };
  }
</script>

<div class="wrap" bind:clientWidth={width} role="img" aria-label={label}>
  <canvas bind:this={canvas} style:width="{width}px" style:height="{height}px" onmousemove={move} onmouseleave={() => (tip = null)}></canvas>
  {#if tip}
    <div class="tip" style:left="{tip.x}px" style:top="{Math.max(0, tip.y - 40)}px" class:flip={tip.flip}>
      <div class="tt">{tip.time}</div>
      <div><span class="muted">≤ {tip.row}{unit}</span> <b>{compact(tip.v)}</b> requests/s</div>
    </div>
  {/if}
  <div class="scale" aria-hidden="true">
    <span>fewer</span>{#each [1, 2, 3, 4, 5, 6, 7] as i}<i style:background="var(--seq-{i})"></i>{/each}<span>more</span>
  </div>
</div>

<style>
  .wrap { position: relative; width: 100%; }
  canvas { display: block; }
  .tip {
    position: absolute; pointer-events: none; z-index: 2; background: var(--surface);
    border: 1px solid var(--ring); border-radius: var(--radius); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    padding: 6px 8px; font-size: var(--fs-xs); white-space: nowrap; transform: translateX(10px);
  }
  .tip.flip { transform: translateX(calc(-100% - 10px)); }
  .tt { color: var(--muted); }
  .muted { color: var(--muted); }
  .scale { display: flex; align-items: center; gap: 2px; justify-content: flex-end; font-size: var(--fs-xs); color: var(--muted); margin-top: 4px; }
  .scale i { width: 14px; height: 8px; border-radius: 2px; display: inline-block; }
  .scale span { margin: 0 4px; }
</style>
