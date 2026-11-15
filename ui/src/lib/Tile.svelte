<script>
  // A stat tile: the headline figure, its unit, one line of context, an optional trend.
  import Sparkline from './Sparkline.svelte';
  let { title, value = '–', unit = '', sub = '', spark = null, theme = '', children } = $props();
</script>

<article class="panel tile">
  <h3>{title}</h3>
  <div class="val">{value} {#if unit}<span class="unit">{unit}</span>{/if}</div>
  {#if sub}<div class="sub">{sub}</div>{/if}
  {#if children}{@render children()}{/if}
  {#if spark}<Sparkline t={spark.t} v={spark.v} {theme} label="{title} trend" />{/if}
</article>
