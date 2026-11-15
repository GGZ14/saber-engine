<script>
  // The console shell: the page nav, the global time range + auto-refresh, the theme, the
  // admin-token gate. Pages live in ./pages and reload on `ui.tick` (each refresh) and range.
  import { onMount, onDestroy } from 'svelte';
  import { setToken, getToken, api, AuthError, RateLimited } from './lib/api.js';
  import { ui, RANGES, REFRESH, setRange, setRefresh, applyTheme, refreshStatus, refreshWho, startRefresh } from './lib/store.svelte.js';
  import Overview from './pages/Overview.svelte';
  import Throughput from './pages/Throughput.svelte';
  import Spec from './pages/Spec.svelte';
  import Memory from './pages/Memory.svelte';
  import Hardware from './pages/Hardware.svelte';
  import Requests from './pages/Requests.svelte';
  import Usage from './pages/Usage.svelte';
  import Logs from './pages/Logs.svelte';
  import Playground from './pages/Playground.svelte';
  import Settings from './pages/Settings.svelte';
  import Setup from './pages/Setup.svelte';
  import Security from './pages/Security.svelte';
  import { W, loadSetup } from './lib/setup.svelte.js';

  const PAGES = [
    { id: 'overview', label: 'Overview', c: Overview },
    { id: 'throughput', label: 'Throughput', c: Throughput },
    { id: 'spec', label: 'Speculative', c: Spec },
    { id: 'memory', label: 'Memory', c: Memory },
    { id: 'hardware', label: 'Hardware', c: Hardware },
    { id: 'requests', label: 'Requests', c: Requests },
    { id: 'usage', label: 'Usage', c: Usage },
    { id: 'playground', label: 'Playground', c: Playground },
    { id: 'logs', label: 'Logs', c: Logs },
    { id: 'settings', label: 'Settings', c: Settings },
    { id: 'security', label: 'Security', c: Security },
    { id: 'setup', label: 'Setup', c: Setup, hidden: true },
  ];
  const hashPage = () => {
    const h = location.hash.replace(/^#\/?/, '').split('?')[0];
    return PAGES.find((p) => p.id === h)?.id ?? 'overview';
  };
  let page = $state(hashPage());
  const Current = $derived(PAGES.find((p) => p.id === page)?.c ?? Overview);
  let tokenInput = $state('');
  let timers = [];

  function onHash() { page = hashPage(); }
  let loginError = $state('');
  let loggingIn = $state(false);
  async function login() {
    const t = tokenInput.trim();
    if (!t) return;
    loggingIn = true;
    loginError = '';
    setToken(t);
    try {
      const a = await api('/api/auth');
      tokenInput = '';
      ui.needToken = false;
      ui.lockedFor = 0;
      ui.who = a.who;
      ui.tokenRequired = a.required;
      refreshStatus();
      const s = await loadSetup();
      if (s?.needed && !location.hash.replace(/^#\/?/, '')) location.hash = '#setup';
      ui.tick++;
    } catch (e) {
      setToken('');
      if (e instanceof RateLimited) { ui.lockedFor = e.retryAfter; loginError = e.message; }
      else if (e instanceof AuthError) loginError = 'That token is not this console\'s admin token.';
      else loginError = e.message;
    } finally {
      loggingIn = false;
    }
  }
  function logout() {
    setToken('');
    ui.who = null;
    ui.needToken = true;
  }

  onMount(async () => {
    // first run: the wizard shows while config.toml is absent or the setup is not finished
    await refreshWho();
    const s = ui.needToken ? null : await loadSetup();
    if (s?.needed && !location.hash.replace(/^#\/?/, '')) location.hash = '#setup';
    refreshStatus();
    startRefresh();
    timers.push(setInterval(refreshStatus, 3000));
    addEventListener('hashchange', onHash);
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const onMq = () => { if (ui.mode === 'system') ui.themeKey++; };
    mq.addEventListener('change', onMq);
    timers.push(() => mq.removeEventListener('change', onMq));
  });
  onDestroy(() => {
    removeEventListener('hashchange', onHash);
    for (const t of timers) typeof t === 'function' ? t() : clearInterval(t);
  });

  const dot = $derived(
    !ui.status ? 'neutral'
      : ui.status.state === 'ready' ? 'good'
      : ['loading', 'draining', 'stopping'].includes(ui.status.state) ? 'warning'
      : ['crashed', 'fatal-86'].includes(ui.status.state) ? 'critical' : 'neutral',
  );
</script>

<header class="top">
  <div class="row1">
    <a class="brand" href="#overview">
      <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true"><path d="M3 13 13 3" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" /></svg>
      <span class="name">Saber Console</span>
      <span class="status-pill st-{dot}" title="Engine: {ui.status?.state ?? 'unknown'}"><i></i><span class="state">{ui.status?.state ?? '…'}</span></span>
    </a>
    <div class="controls">
      <div class="seg" role="radiogroup" aria-label="Time range">
        {#each RANGES as r}
          <button role="radio" aria-checked={ui.range === r.id} class:on={ui.range === r.id} onclick={() => setRange(r.id)}>{r.id}</button>
        {/each}
      </div>
      <label class="refresh">
        <span class="muted">Refresh</span>
        <select value={ui.refresh} onchange={(e) => setRefresh(e.currentTarget.value)} aria-label="Auto-refresh">
          {#each REFRESH as r}<option value={r.id}>{r.id}</option>{/each}
        </select>
      </label>
      <button class="btn small" onclick={() => ui.tick++} title="Refresh now" aria-label="Refresh now">
        <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
      </button>
      {#if getToken() && !ui.needToken}
        <button class="btn small" onclick={logout} title="Forget the admin token in this tab">Sign out</button>
      {/if}
      <div class="seg" role="radiogroup" aria-label="Theme">
        {#each ['system', 'light', 'dark'] as m}
          <button role="radio" aria-checked={ui.mode === m} class:on={ui.mode === m} onclick={() => applyTheme(m)}>{m[0].toUpperCase() + m.slice(1)}</button>
        {/each}
      </div>
    </div>
  </div>
  <nav aria-label="Pages">
    {#each PAGES.filter((p) => !p.hidden || page === p.id || W.setup?.needed) as p}
      <a href="#{p.id}" class:setup={p.id === 'setup'} class:on={page === p.id} aria-current={page === p.id ? 'page' : undefined}>{p.label}</a>
    {/each}
  </nav>
</header>

<main>
  {#if ui.needToken}
    <section class="panel login" aria-labelledby="login-h">
      <h1 id="login-h">Sign in</h1>
      <p>This console needs its admin token. It is kept in this browser tab only (closing the tab forgets it).</p>
      <form onsubmit={(e) => { e.preventDefault(); login(); }}>
        <label for="token">Admin token</label>
        <div class="row">
          <input id="token" type="password" autocomplete="current-password" bind:value={tokenInput} placeholder="sa-admin-…" spellcheck="false" />
          <button class="btn primary" type="submit" disabled={loggingIn || !tokenInput.trim()}>{loggingIn ? 'Checking…' : 'Sign in'}</button>
        </div>
      </form>
      {#if loginError}<p class="err" role="alert">{loginError}</p>{/if}
      {#if ui.lockedFor}<p class="muted">Too many failed attempts from this address: wait {ui.lockedFor} s.</p>{/if}
      <p class="muted small">Lost it? On the server run <code>saber ui --gen-admin-token</code> (it prints a new token once and keeps only its hash), then restart the console.</p>
    </section>
  {:else}
    {#key page}
      <Current />
    {/key}
  {/if}
</main>

<style>
  .top {
    position: sticky;
    top: env(safe-area-inset-top, 0px);
    z-index: 5;
    background: var(--surface);
    border-bottom: 1px solid var(--ring);
    padding-top: 8px;
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
  .row1 { display: flex; align-items: center; justify-content: space-between; gap: 8px 12px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 10px; color: var(--ink); text-decoration: none; }
  .name { font-weight: 650; letter-spacing: 0.01em; }
  .state { color: var(--ink-2); font-weight: 500; }
  .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .refresh { display: inline-flex; align-items: center; gap: 6px; font-size: var(--fs-sm); }
  nav {
    display: flex;
    gap: 2px;
    overflow-x: auto;
    margin-top: 6px;
    scrollbar-width: none;
  }
  nav a {
    color: var(--ink-2);
    text-decoration: none;
    font-size: var(--fs-sm);
    padding: 6px 10px 8px;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
  }
  nav a.on { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
  nav a.setup { color: var(--accent); font-weight: 600; }
  main {
    max-width: 1280px;
    margin: 0 auto;
    padding-block: 16px 32px;
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
    display: grid;
    gap: 14px;
  }
  .login { max-width: 460px; margin: 48px auto; width: 100%; display: grid; gap: 10px; }
  .login h1 { margin: 0; font-size: var(--fs-lg, 1.25rem); }
  .login p { margin: 0; }
  .login form { display: grid; gap: 6px; }
  .login .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .login input { flex: 1 1 220px; }
  .login .err { color: var(--critical, #c0392b); }
  .small { font-size: var(--fs-xs); }
</style>
