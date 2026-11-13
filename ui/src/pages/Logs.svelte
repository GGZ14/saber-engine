<script>
  // The engine's output and the console's own notes: a live tail with a text filter, a stream
  // filter, pause, and a download of the whole ring.
  import { onMount, onDestroy } from 'svelte';
  import { api, apiText, saveText } from '../lib/api.js';
  import { guarded } from '../lib/store.svelte.js';

  const KEEP = 5000;
  let lines = $state([]);
  let q = $state('');
  let streams = $state({ out: true, err: true, console: true });
  let paused = $state(false);
  let follow = $state(true);
  let pre;
  let lastSeq = 0;
  let timer;

  async function poll() {
    if (paused) return;
    try {
      const got = await guarded(() => api(`/api/logs?since=${lastSeq}&limit=2000`));
      if (got?.length) {
        lastSeq = got[got.length - 1].seq;
        lines = [...lines, ...got].slice(-KEEP);
      }
    } catch (e) {}
  }
  onMount(() => { poll(); timer = setInterval(poll, 1500); });
  onDestroy(() => clearInterval(timer));

  const shown = $derived.by(() => {
    const needle = q.trim().toLowerCase();
    return lines.filter((l) => streams[l.stream] !== false && (!needle || l.text.toLowerCase().includes(needle)));
  });
  $effect(() => {
    shown.length;
    if (follow && pre) queueMicrotask(() => { pre.scrollTop = pre.scrollHeight; });
  });
  async function download() {
    try { saveText('saber-engine.log', await apiText('/api/logs/download')); } catch (e) {}
  }
  const ts = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
</script>

<div class="page">
  <h1 class="title">Logs</h1>
  <section class="panel">
    <div class="filters bar">
      <input type="search" placeholder="filter lines…" bind:value={q} aria-label="Filter log lines" />
      {#each ['out', 'err', 'console'] as s}
        <label><input type="checkbox" bind:checked={streams[s]} /> {s === 'out' ? 'stdout' : s === 'err' ? 'stderr' : 'console'}</label>
      {/each}
      <label><input type="checkbox" bind:checked={follow} /> follow</label>
      <button class="btn small" onclick={() => (paused = !paused)}>{paused ? 'Resume' : 'Pause'}</button>
      <button class="btn small" onclick={() => { lines = []; }}>Clear view</button>
      <button class="btn small" onclick={download}>Download</button>
      <span class="muted">{shown.length.toLocaleString()} of {lines.length.toLocaleString()} lines</span>
    </div>
    <pre class="tail big" bind:this={pre}>{#each shown as l (l.seq)}<span class="t">{ts(l.t_ms)} </span><span class="ln s-{l.stream}">{l.text}</span>{'\n'}{/each}</pre>
  </section>
</div>

<style>
  .bar { margin-bottom: 8px; }
  .big { height: min(70vh, 720px); overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
  .t { color: var(--muted); }
</style>
