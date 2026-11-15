<script>
  // A detail drawer from the right (full width on a phone); Escape or the backdrop closes it.
  let { open = false, title = '', onclose, children } = $props();
  function key(e) { if (open && e.key === 'Escape') onclose?.(); }
</script>

<svelte:window onkeydown={key} />
{#if open}
  <div class="backdrop" role="presentation" onclick={() => onclose?.()}></div>
  <div class="drawer" role="dialog" aria-modal="true" aria-label={title} tabindex="-1">
    <header>
      <h2>{title}</h2>
      <button class="btn small" onclick={() => onclose?.()} aria-label="Close">Close</button>
    </header>
    <div class="body">{@render children()}</div>
  </div>
{/if}

<style>
  .backdrop { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.3); z-index: 20; }
  .drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(560px, 100vw); z-index: 21;
    background: var(--surface); border-left: 1px solid var(--ring); box-shadow: -8px 0 24px rgba(0, 0, 0, 0.15);
    display: flex; flex-direction: column;
  }
  header { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 12px 16px; border-bottom: 1px solid var(--ring); }
  header h2 { overflow-wrap: anywhere; }
  .body { overflow: auto; padding: 14px 16px; }
</style>
