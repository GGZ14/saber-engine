<script>
  // A chart panel: title, legend (always present for two or more series), the body, and an
  // optional table view.
  let { title, legend = [], note = '', children, table = null, actions = null } = $props();
</script>

<section class="panel">
  <div class="c-head">
    <div>
      <h2>{title}</h2>
      {#if legend.length > 1 || note}
        <div class="legend">
          {#if legend.length > 1}
            {#each legend as l}<span><i style:background="var({l.color})"></i>{l.label}</span>{/each}
          {/if}
          {#if note}<span class="muted">{note}</span>{/if}
        </div>
      {/if}
    </div>
    {#if actions}{@render actions()}{/if}
  </div>
  {@render children()}
  {#if table}
    <details class="tbl">
      <summary>Table view</summary>
      <div class="scroll">{@render table()}</div>
    </details>
  {/if}
</section>
