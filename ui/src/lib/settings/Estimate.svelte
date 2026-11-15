<script>
  // The live memory-budget estimate for the draft (from /api/config/validate). Always labelled
  // an ESTIMATE: the engine's own ledger at boot is the truth and refuses what does not fit.
  import { S } from '../settings.svelte.js';
  import { bytes, int } from '../api.js';
  import KvTip from '../KvTip.svelte';

  const e = $derived(S.check?.estimate ?? null);
  const parts = $derived.by(() => {
    if (!e?.budget_bytes) return [];
    const b = e.budget_bytes;
    const rows = [
      { k: 'Weights', v: e.weights_bytes, c: '--series-1' },
      { k: 'Drafter', v: e.drafter_bytes, c: '--series-7' },
      { k: 'GDN state', v: e.state_bytes, c: '--series-4' },
      { k: 'Reserved', v: e.reserved_bytes, c: '--deemph' },
      { k: e.law === 'flashnext' ? 'KV hot pool' : 'KV pool', v: e.kv_bytes > 0 ? e.kv_bytes : 0, c: '--series-3' },
    ].filter((r) => r.v);
    return rows.map((r) => ({ ...r, pct: Math.max(0, Math.min(100, (r.v / b) * 100)) }));
  });
  // TP=1: the one card holds everything; TP=2: each figure is one of the two cards
  const card = $derived(e?.tp === 1 ? 'on the card' : 'per card');
  const law = $derived(e?.law === 'flashnext' ? 'Flash-Next (tiered KV)' : e?.law === 'dense27b' ? '27B (KV in VRAM)' : e?.law === 'mock' ? 'Mock engine' : '');
</script>

<section class="panel est" aria-labelledby="est-h">
  <div class="c-head">
    <div>
      <h2 id="est-h">Memory budget</h2>
      <div class="legend"><span class="muted">{law}</span></div>
    </div>
    <span class="tag" title="Computed from the pack manifest and the model's config.json; the engine's boot ledger is the truth">Estimate</span>
  </div>
  {#if !e}
    <p class="empty">Checking the draft…</p>
  {:else}
    {#if e.label}<p class="lbl">{e.label}</p>{/if}
    <div class="big">
      <div>
        <div class="val">{e.seqs_at_ctx ?? '–'}<span class="unit"> / {e.requested_seqs} seqs</span></div>
        <div class="sub">fit at the full {int(e.max_ctx)}-token context</div>
      </div>
      <div>
        <div class="val">{e.kv_capacity_tokens != null ? int(e.kv_capacity_tokens) : '–'}</div>
        <div class="sub">KV capacity, tokens{e.law === 'flashnext' ? ' (admission slab)' : ''}</div>
      </div>
    </div>
    {#if !e.fits}
      <p class="bad" role="alert">The estimate says this config does not fit.</p>
    {/if}
    {#if parts.length}
      <div class="bar" role="img" aria-label={e.tp === 1 ? 'The one card\'s budget split' : 'Per-card budget split'}>
        {#each parts as p}<span style:width="{p.pct}%" style:background="var({p.c})" title="{p.k}: {bytes(p.v)}"></span>{/each}
      </div>
      <div class="legend">
        {#each parts as p}<span><i style:background="var({p.c})"></i>{p.k}</span>{/each}
      </div>
    {/if}
    <dl class="kv">
      {#if e.vram_bytes}<dt>VRAM {card}</dt><dd>{bytes(e.vram_bytes)}</dd>{/if}
      {#if e.budget_bytes}<dt>Budget {card}</dt><dd>{bytes(e.budget_bytes)}</dd>{/if}
      {#if e.weights_bytes != null}<dt>Weights {card}</dt><dd>{bytes(e.weights_bytes)}</dd>{/if}
      {#if e.drafter_bytes != null}<dt>Drafter {card}</dt><dd>{bytes(e.drafter_bytes)}</dd>{/if}
      {#if e.state_bytes != null}<dt>GDN state</dt><dd>{bytes(e.state_bytes)}</dd>{/if}
      {#if e.kv_bytes != null}<dt>{e.law === 'flashnext' ? 'KV hot pool' : 'KV pool'}</dt><dd>{bytes(e.kv_bytes)}</dd>{/if}
      {#if e.draft_kv_bytes}<dt>Drafter KV</dt><dd>{bytes(e.draft_kv_bytes)}</dd>{/if}
      {#if e.kv_ram_bytes != null}<dt>KV in host RAM / card</dt><dd>{bytes(e.kv_ram_bytes)}</dd>{/if}
      {#if e.kv_bytes_per_token != null}<dt>KV per token {card}</dt><dd>{bytes(e.kv_bytes_per_token)}</dd>{/if}
      {#if e.expert_slots != null}<dt>Expert slots in VRAM</dt><dd>{int(e.expert_slots)}{e.expert_records ? ` of ${int(e.expert_records)}` : ''}</dd>{/if}
    </dl>
    {#each e.warnings as w}<p class="warn">{w}</p>{/each}
    <KvTip estimate={e} />
    <details class="notes">
      <summary>How this is estimated</summary>
      <ul>{#each e.notes as n}<li>{n}</li>{/each}</ul>
    </details>
  {/if}
</section>

<style>
  .lbl { margin: 0 0 6px; font-size: var(--fs-xs); color: var(--ink-2); }
  .tag { font-size: var(--fs-xs); font-weight: 600; padding: 2px 8px; border-radius: 9px; border: 1px dashed var(--axis); color: var(--ink-2); }
  .big { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 4px 0 10px; }
  .bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--surface-2); gap: 1px; }
  .bar span { display: block; height: 100%; }
  .kv { margin-top: 10px; }
  .bad { color: var(--critical); font-size: var(--fs-sm); font-weight: 600; margin: 0 0 8px; }
  .warn { font-size: var(--fs-xs); color: var(--ink-2); background: color-mix(in srgb, var(--warning) 16%, transparent); padding: 4px 8px; border-radius: 4px; margin: 8px 0 0; }
  .notes { margin-top: 10px; font-size: var(--fs-xs); color: var(--muted); }
  .notes summary { cursor: pointer; }
  .notes ul { margin: 6px 0 0; padding-left: 18px; }
</style>
