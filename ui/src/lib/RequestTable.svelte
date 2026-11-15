<script>
  // The request log table: sortable headers, a row click opens the detail drawer.
  import { compact, ms, when, int } from './api.js';
  let { rows = [], sort = 'ts_ms', asc = false, onsort = null, onopen = null } = $props();
  const COLS = [
    { k: 'ts_ms', label: 'Time', l: true },
    { k: 'model', label: 'Model', l: true },
    { k: 'key', label: 'Key', l: true },
    { k: 'prompt_tokens', label: 'Prompt' },
    { k: 'completion_tokens', label: 'Output' },
    { k: 'cached_tokens', label: 'Cached' },
    { k: 'ttft_ms', label: 'TTFT' },
    { k: 'decode_tps', label: 'tok/s' },
    { k: 'latency_ms', label: 'Latency' },
    { k: null, label: 'Finish', l: true },
    { k: 'status', label: 'Status' },
  ];
  const tone = (s) => (s == null ? 'neutral' : s === 499 ? 'warning' : s >= 400 ? 'critical' : 'good');
  const src = (r) => ((r.source & 1) ? ((r.source & 2) ? 'proxy + engine' : 'proxy') : 'direct to engine');
</script>

<div class="scroll">
  <table>
    <thead>
      <tr>
        {#each COLS as c}
          <th class:l={c.l} aria-sort={sort === c.k ? (asc ? 'ascending' : 'descending') : undefined}>
            {#if c.k && onsort}
              <button class="sort" onclick={() => onsort(c.k)}>{c.label}{sort === c.k ? (asc ? ' ▲' : ' ▼') : ''}</button>
            {:else}{c.label}{/if}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as r (r.uid)}
        <tr class="click" onclick={() => onopen?.(r)} tabindex="0" onkeydown={(e) => e.key === 'Enter' && onopen?.(r)} title={src(r)}>
          <td>{when(r.ts_ms)}</td>
          <td class="l trunc">{r.model ?? '–'}</td>
          <td class="l">{r.key ?? '–'}</td>
          <td>{int(r.prompt_tokens)}</td>
          <td>{int(r.completion_tokens)}</td>
          <td>{int(r.cached_tokens)}</td>
          <td>{ms(r.ttft_ms)}</td>
          <td>{compact(r.decode_tps)}</td>
          <td>{ms(r.latency_ms)}</td>
          <td class="l">{r.finish_reason ?? '–'}</td>
          <td><span class="status-pill st-{tone(r.status)}"><i></i>{r.status ?? '–'}</span></td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .trunc { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
</style>
