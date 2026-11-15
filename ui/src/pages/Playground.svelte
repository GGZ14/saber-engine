<script>
  // Chat with the engine through the console's /v1 proxy (streamed, with the thinking toggle).
  // Requests made here are recorded in the usage history like any other proxied request.
  import { onMount } from 'svelte';
  import { getToken, compact, ms } from '../lib/api.js';
  import { ui } from '../lib/store.svelte.js';

  const KEY = 'saber-playground-key';
  let models = $state([]);
  let model = $state('');
  let system = $state('');
  let input = $state('');
  let thinking = $state(true);
  let temperature = $state(0.7);
  let maxTokens = $state(1024);
  let apiKey = $state(load());
  let msgs = $state([]);
  let running = $state(false);
  let err = $state('');
  let ctl = null;

  function load() { try { return localStorage.getItem(KEY) || ''; } catch (e) { return ''; } }
  function saveKey(v) { apiKey = v; try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (e) {} }
  function headers() {
    const h = { 'Content-Type': 'application/json', Accept: 'text/event-stream' };
    const k = apiKey || getToken();
    if (k) h.Authorization = `Bearer ${k}`;
    return h;
  }

  async function loadModels() {
    try {
      const r = await fetch('/v1/models', { headers: headers() });
      if (!r.ok) throw new Error(r.status === 401 ? 'API key required (set it below)' : `${r.status}`);
      const j = await r.json();
      models = (j.data ?? []).map((m) => m.id);
      if (!model && models.length) model = models[0];
      err = '';
    } catch (e) {
      models = [];
      err = ui.status?.state === 'ready' ? `Could not list models: ${e.message}` : 'The engine is not running. Start it on the Overview page.';
    }
  }
  onMount(loadModels);
  $effect(() => { ui.status?.state; if (!models.length) loadModels(); });

  async function send() {
    const text = input.trim();
    if (!text || running) return;
    input = '';
    err = '';
    msgs.push({ role: 'user', content: text });
    const out = $state({ role: 'assistant', content: '', reasoning: '', stats: null, showReasoning: false });
    msgs.push(out);
    const history = msgs.slice(0, -1).map((m) => ({ role: m.role, content: m.content }));
    if (system.trim()) history.unshift({ role: 'system', content: system.trim() });
    const body = {
      model: model || undefined,
      messages: history,
      stream: true,
      stream_options: { include_usage: true },
      temperature: +temperature,
      max_tokens: +maxTokens,
      chat_template_kwargs: { enable_thinking: thinking },
    };
    running = true;
    ctl = new AbortController();
    const t0 = performance.now();
    let first = null;
    let lastT = null;
    let usage = null;
    let finish = null;
    try {
      const r = await fetch('/v1/chat/completions', { method: 'POST', headers: headers(), body: JSON.stringify(body), signal: ctl.signal });
      if (!r.ok || !r.body) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error?.message ?? `${r.status} ${r.statusText}`);
      }
      const rd = r.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { value, done } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim();
          buf = buf.slice(i + 1);
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') continue;
          let ev;
          try { ev = JSON.parse(data); } catch (e) { continue; }
          if (ev.usage) usage = ev.usage;
          if (ev.error) throw new Error(ev.error.message ?? 'engine error');
          for (const c of ev.choices ?? []) {
            const d = c.delta ?? {};
            const piece = d.content ?? '';
            const think = d.reasoning ?? d.reasoning_content ?? '';
            if (piece || think) {
              const now = performance.now();
              first ??= now;
              lastT = now;
            }
            out.content += piece;
            out.reasoning += think;
            if (c.finish_reason) finish = c.finish_reason;
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') err = e.message;
      else finish = 'stopped';
    } finally {
      running = false;
      ctl = null;
      const n = usage?.completion_tokens;
      out.stats = {
        ttft: first == null ? null : first - t0,
        total: performance.now() - t0,
        tps: n > 1 && first != null && lastT > first ? (n - 1) / ((lastT - first) / 1000) : null,
        prompt: usage?.prompt_tokens,
        completion: n,
        cached: usage?.prompt_tokens_details?.cached_tokens,
        finish,
      };
    }
  }
  function stop() { ctl?.abort(); }
  function clear() { msgs = []; err = ''; }
  function key(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }
</script>

<div class="page">
  <h1 class="title">Playground</h1>
  <div class="layout">
    <section class="panel chat">
      {#if !msgs.length}
        <p class="empty">Send a message to the engine. Replies stream through the console's <code>/v1</code> proxy and are recorded in the usage history.</p>
      {/if}
      <div class="msgs" aria-live="polite">
        {#each msgs as m}
          <div class="msg {m.role}">
            <div class="who">{m.role === 'user' ? 'You' : model || 'Assistant'}</div>
            {#if m.reasoning}
              <details class="think" bind:open={m.showReasoning}>
                <summary>Thinking ({m.reasoning.length.toLocaleString()} chars)</summary>
                <div class="text r">{m.reasoning}</div>
              </details>
            {/if}
            <div class="text">{m.content}{#if running && m === msgs[msgs.length - 1] && m.role === 'assistant'}<span class="caret">▍</span>{/if}</div>
            {#if m.stats}
              <div class="stats">TTFT {ms(m.stats.ttft)} · {m.stats.tps == null ? '–' : `${compact(m.stats.tps)} tok/s`} · {m.stats.prompt ?? '–'} in{m.stats.cached ? ` (${m.stats.cached} cached)` : ''} / {m.stats.completion ?? '–'} out · {m.stats.finish ?? '–'}</div>
            {/if}
          </div>
        {/each}
      </div>
      {#if err}<p class="note errline">{err}</p>{/if}
      <form class="compose" onsubmit={(e) => { e.preventDefault(); send(); }}>
        <textarea rows="3" placeholder="Message (Enter to send, Shift+Enter for a new line)" bind:value={input} onkeydown={key} aria-label="Message"></textarea>
        <div class="acts">
          {#if running}<button class="btn" type="button" onclick={stop}>Stop</button>
          {:else}<button class="btn primary" type="submit" disabled={!input.trim()}>Send</button>{/if}
          <button class="btn" type="button" onclick={clear} disabled={running || !msgs.length}>Clear</button>
        </div>
      </form>
    </section>
    <aside class="panel opts">
      <h2>Options</h2>
      <label>Model
        <select bind:value={model}>{#each models as m}<option value={m}>{m}</option>{/each}</select>
      </label>
      <label class="row"><input type="checkbox" bind:checked={thinking} /> Thinking</label>
      <label>Temperature <input type="number" min="0" max="2" step="0.1" bind:value={temperature} /></label>
      <label>Max tokens <input type="number" min="1" step="1" bind:value={maxTokens} /></label>
      <label>System prompt <textarea rows="3" bind:value={system}></textarea></label>
      <label>API key <span class="muted">(only when the console's proxy keys are on)</span>
        <input type="password" autocomplete="off" value={apiKey} oninput={(e) => saveKey(e.currentTarget.value)} />
      </label>
    </aside>
  </div>
</div>

<style>
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 14px; align-items: start; }
  @media (max-width: 860px) { .layout { grid-template-columns: minmax(0, 1fr); } }
  .chat { display: grid; gap: 10px; }
  .msgs { display: grid; gap: 10px; max-height: 62vh; overflow: auto; }
  .msg { padding: 8px 10px; border-radius: var(--radius); background: var(--surface-2); }
  .msg.user { background: var(--accent-fill); }
  .who { font-size: var(--fs-xs); color: var(--muted); margin-bottom: 2px; font-weight: 600; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; }
  .text.r { color: var(--ink-2); font-size: var(--fs-sm); }
  .think { margin-bottom: 6px; font-size: var(--fs-sm); }
  .think summary { cursor: pointer; color: var(--ink-2); }
  .stats { font-size: var(--fs-xs); color: var(--muted); margin-top: 4px; font-variant-numeric: tabular-nums; }
  .caret { color: var(--accent); }
  .compose { display: grid; gap: 8px; }
  .compose textarea { width: 100%; }
  .acts { display: flex; gap: 8px; }
  .opts { display: grid; gap: 10px; font-size: var(--fs-sm); }
  .opts label { display: grid; gap: 4px; }
  .opts label.row { display: flex; align-items: center; gap: 6px; }
  .errline { color: var(--critical); }
</style>
