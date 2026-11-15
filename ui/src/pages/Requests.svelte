<script>
  import { api } from '../lib/api.js';
  import { ui, rangeS, guarded } from '../lib/store.svelte.js';
  import RequestTable from '../lib/RequestTable.svelte';
  import RequestDetail from '../lib/RequestDetail.svelte';
  import Drawer from '../lib/Drawer.svelte';

  const PAGE = 50;
  let data = $state({ total: 0, rows: [] });
  let sort = $state('ts_ms');
  let asc = $state(false);
  let offset = $state(0);
  let open = $state(null);
  let err = $state('');

  async function reload() {
    const from = Date.now() - rangeS() * 1000;
    try {
      data = (await guarded(() => api(`/api/usage/requests?from=${from}&sort=${sort}&asc=${asc}&limit=${PAGE}&offset=${offset}`))) ?? data;
      err = '';
    } catch (e) { err = e.message; }
  }
  $effect(() => { ui.tick; ui.range; sort; asc; offset; reload(); });
  function onsort(k) {
    if (sort === k) asc = !asc;
    else { sort = k; asc = false; }
    offset = 0;
  }
</script>

<div class="page">
  <div class="c-head">
    <h1 class="title">Requests</h1>
    <a class="btn small" href="#playground">Open the playground</a>
  </div>
  <section class="panel">
    <div class="c-head">
      <div>
        <h2>Recent requests</h2>
        <div class="legend"><span class="muted">{data.total.toLocaleString()} in the range · proxied and direct-to-engine traffic · click a row for its detail</span></div>
      </div>
      <div class="filters">
        <button class="btn small" disabled={offset === 0} onclick={() => (offset = Math.max(0, offset - PAGE))}>Newer</button>
        <span class="muted">{data.total ? `${offset + 1}–${Math.min(offset + PAGE, data.total)}` : '0'}</span>
        <button class="btn small" disabled={offset + PAGE >= data.total} onclick={() => (offset += PAGE)}>Older</button>
      </div>
    </div>
    {#if err}<p class="note">{err}</p>
    {:else if data.rows.length}
      <RequestTable rows={data.rows} {sort} {asc} {onsort} onopen={(r) => (open = r.uid)} />
    {:else}<p class="empty">No requests in this range. Send one from the playground, or through the console's /v1 endpoint.</p>{/if}
  </section>
</div>

<Drawer open={!!open} title="Request detail" onclose={() => (open = null)}>
  {#if open}<RequestDetail uid={open} />{/if}
</Drawer>
