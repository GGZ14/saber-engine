// Histogram bucket series (cumulative `_bucket{le}` rates from /api/series) -> heatmap cells.
import { parseKey } from './api.js';

/** -> { t, rows: [{le, label}], cells: [col][row] } (per-bucket rates, +Inf dropped). */
export function heat(win, family, toLabel) {
  const by = [];
  for (const [k, vals] of Object.entries(win?.series ?? {})) {
    const p = parseKey(k);
    if (p.name !== family || p.labels.le == null || p.labels.le === '+Inf') continue;
    const le = parseFloat(p.labels.le);
    let e = by.find((b) => b.le === le);
    if (!e) by.push((e = { le, vals: vals.map(() => 0) }));
    vals.forEach((v, i) => { if (v != null) e.vals[i] += v; });
  }
  by.sort((a, b) => a.le - b.le);
  const t = win?.t ?? [];
  const cells = t.map((_, i) => by.map((b, r) => Math.max(0, b.vals[i] - (r ? by[r - 1].vals[i] : 0))));
  // drop the rows no interval ever reached, keeping the range contiguous
  let lo = 0;
  let hi = by.length - 1;
  const any = (r) => cells.some((c) => c[r] > 0);
  while (lo < hi && !any(lo)) lo++;
  while (hi > lo && !any(hi)) hi--;
  if (!cells.some((c) => c.some((v) => v > 0))) return { t, rows: [], cells: [] };
  return {
    t,
    rows: by.slice(lo, hi + 1).map((b) => ({ le: b.le, label: toLabel(b.le) })),
    cells: cells.map((c) => c.slice(lo, hi + 1)),
  };
}

export const secLabel = (le) => (le >= 1 ? `${+le.toFixed(2)} s` : `${+(le * 1000).toFixed(le < 0.01 ? 1 : 0)} ms`);
