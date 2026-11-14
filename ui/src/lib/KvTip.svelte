<script>
  // The "KV vs context vs sequences" tip: what the same config would get with fewer sequences or
  // a shorter context, from the estimator's own law (/api/config/validate `estimate.alternatives`).
  // Every number here is the server's; the sentences only order and compare them.
  import { int } from './api.js';

  let { estimate } = $props();
  const alts = $derived(estimate?.alternatives ?? []);
  const base = $derived(estimate?.kv_capacity_tokens ?? 0);
  const k = (t) => (t >= 10000 ? `~${Math.round(t / 1000)}k` : `~${int(t)}`);
  const pct = (t) => {
    const p = base ? Math.round(((t - base) / base) * 100) : 0;
    return `${p >= 0 ? '+' : ''}${p} %`;
  };
  const seqs = $derived(alts.filter((a) => a.kind === 'seqs'));
  const ctx = $derived(alts.filter((a) => a.kind === 'ctx'));
  const best = (xs) => xs.reduce((m, a) => Math.max(m, a.kv_capacity_tokens), 0);
  const seqsWin = $derived(seqs.length && best(seqs) > best(ctx));
</script>

{#if estimate?.law === 'dense27b' && alts.length && base}
  <div class="tip" role="note">
    <p class="h">KV cache: context vs sequences</p>
    <p>
      This config gets about <strong>{k(base)} tokens</strong> of KV cache:
      {estimate.seqs_at_ctx} of {estimate.requested_seqs} sequence{estimate.requested_seqs === 1 ? '' : 's'} fit at the full {int(estimate.max_ctx)}-token context at once{estimate.spare_blocks != null ? `, with ${k(estimate.spare_blocks * 16)} tokens spare for cached prefixes` : ''}.
      {#if estimate.ctx_ceiling}Max context at {estimate.requested_seqs} sequence{estimate.requested_seqs === 1 ? '' : 's'}: ~{int(estimate.ctx_ceiling)} tokens.{/if}
    </p>
    <ul>
      {#if seqs.length}
        <li><span class="lbl">Other sequence counts:</span> {#each seqs as a, i}{i ? '; ' : ''}{a.max_seqs} at {int(a.max_ctx)} → {k(a.kv_capacity_tokens)} ({pct(a.kv_capacity_tokens)}{a.ctx_ceiling ? `, max context ~${int(a.ctx_ceiling)}` : ''}){/each}</li>
      {/if}
      {#if ctx.length}
        <li><span class="lbl">Shorter context:</span> {#each ctx as a, i}{i ? '; ' : ''}{int(a.max_ctx)} → {k(a.kv_capacity_tokens)} ({pct(a.kv_capacity_tokens)}){/each}</li>
      {/if}
    </ul>
    {#if seqsWin}
      <p>The sequence count is the lever: every sequence holds its own model state on the card{estimate.tp === 1 ? ', and one card has little to spare' : ''}. A shorter context frees only the drafter's share, which helps once the context gets small.</p>
    {:else}
      <p>A shorter context frees more than a different sequence count here.</p>
    {/if}
    <p class="sugg">One long conversation at a time: 1 sequence (the <strong>Long context</strong> profile for the longest). Several people or agents at once: 2 or 4 sequences, which share the KV.</p>
  </div>
{/if}

<style>
  .tip { margin-top: 10px; padding: 8px 10px; border-radius: 6px; background: var(--surface-2); font-size: var(--fs-xs); color: var(--ink-2); display: grid; gap: 4px; }
  .tip p { margin: 0; }
  .tip .h { font-weight: 600; color: var(--ink); font-size: var(--fs-sm); }
  .tip ul { margin: 0; padding-left: 16px; }
  .lbl { font-weight: 600; }
  .sugg { color: var(--ink); }
</style>
