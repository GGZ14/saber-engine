<script>
  // A stat tile's trend: one series, soft area, no axes, the newest point emphasised.
  import uPlot from 'uplot';
  import { onDestroy, untrack } from 'svelte';

  let { t = [], v = [], theme = '', label = '' } = $props();
  let el;
  let plot;
  let width = $state(0);

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function build() {
    plot?.destroy();
    plot = null;
    if (!el || width < 10) return;
    const stroke = css('--accent') || '#2a78d6';
    const fill = css('--accent-fill') || 'rgba(42, 120, 214, 0.12)';
    const surface = css('--surface') || '#fcfcfb';
    plot = new uPlot(
      {
        width,
        height: 40,
        padding: [6, 6, 4, 2],
        legend: { show: false },
        cursor: { show: false },
        select: { show: false },
        scales: { x: { time: true, range: (u, lo, hi) => [lo, hi] }, y: { range: (u, lo, hi) => [Math.min(0, lo ?? 0), Math.max(1e-9, (hi ?? 1) * 1.1)] } },
        axes: [{ show: false }, { show: false }],
        series: [
          {},
          {
            stroke,
            fill,
            width: 2,
            spanGaps: false,
            points: { show: false },
          },
        ],
        hooks: {
          draw: [
            (u) => {
              // endpoint marker: the newest non-null value, ringed with the surface
              const d = u.data[1];
              for (let i = d.length - 1; i >= 0; i--) {
                if (d[i] != null) {
                  const x = u.valToPos(u.data[0][i], 'x', true);
                  const y = u.valToPos(d[i], 'y', true);
                  const ctx = u.ctx;
                  ctx.save();
                  const r = 3.5 * devicePixelRatio;
                  ctx.beginPath();
                  ctx.arc(x, y, r + 2 * devicePixelRatio, 0, 2 * Math.PI);
                  ctx.fillStyle = surface;
                  ctx.fill();
                  ctx.beginPath();
                  ctx.arc(x, y, r, 0, 2 * Math.PI);
                  ctx.fillStyle = stroke;
                  ctx.fill();
                  ctx.restore();
                  break;
                }
              }
            },
          ],
        },
      },
      untrack(() => [t, v]),
      el,
    );
  }

  $effect(() => {
    theme; width;
    build();
  });
  $effect(() => {
    if (plot) plot.setData([t, v]);
  });
  onDestroy(() => plot?.destroy());
</script>

<div class="spark" bind:this={el} bind:clientWidth={width} role="img" aria-label={label}></div>

<style>
  .spark { width: 100%; height: 40px; }
</style>
