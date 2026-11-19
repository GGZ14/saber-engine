<script>
  // The setup wizard: System check -> Model -> Profile -> Network and security -> Review and
  // launch. Every step is saved on the server (config.toml [setup]), so a reload resumes here.
  // Back is always open; Next passes the step's gate (the server enforces the same gates).
  import { onMount, onDestroy } from 'svelte';
  import { api, stream, bytes, int, setToken, login } from '../lib/api.js';
  import { ui, guarded, refreshStatus } from '../lib/store.svelte.js';
  import { W, loadSetup, loadSystem, loadCatalog, followDownloads, setupCall, copyText, gb } from '../lib/setup.svelte.js';
  import DownloadRow from '../lib/DownloadRow.svelte';
  import KvTip from '../lib/KvTip.svelte';

  const step = $derived(W.setup?.state?.step ?? 'system');
  const steps = $derived((W.setup?.steps ?? []).filter((s) => s.id !== 'done'));
  const idx = $derived(steps.findIndex((s) => s.id === step));
  const model = $derived(W.catalog?.models?.find((m) => m.id === W.setup?.state?.model) ?? null);

  let unfollow = null;
  onMount(async () => {
    await loadSetup();
    loadSystem();
    loadCatalog();
    unfollow = followDownloads();
  });
  onDestroy(() => { unfollow?.(); stopLogs(); clearInterval(applyTimer); });

  async function go(target) {
    const r = await setupCall('/api/setup/goto', { step: target });
    if (r.ok) {
      document.getElementById('wiz-title')?.focus();
      if (target === 'profile') loadEstimate();
      if (target === 'review') loadReview();
    }
  }
  const next = () => go(steps[idx + 1]?.id ?? 'review');
  const back = () => go(steps[idx - 1]?.id ?? 'system');
  async function skip() {
    const r = await setupCall('/api/setup/skip');
    if (r.ok) location.hash = '#settings';
  }

  // ---- system ----
  const ICON = { pass: 'Pass', warn: 'Warning', fail: 'Problem', info: 'Info' };
  const sysCounts = $derived.by(() => {
    const c = { pass: 0, warn: 0, fail: 0, info: 0 };
    for (const k of W.system?.checks ?? []) c[k.status]++;
    return c;
  });

  // ---- model ----
  async function pick(m) {
    await setupCall('/api/setup/model', { id: m.id });
    loadSystem();
    loadCatalog();
  }
  // one card or two (the 27B): the TP=1 form downloads its own single-card packs
  async function pickTp(tp, device) {
    const r = await setupCall('/api/setup/tp', device == null ? { tp } : { tp, device });
    if (r.ok) { loadCatalog(); loadSystem(); }
  }
  const TP_TEXT = {
    2: 'Each card holds half of the model: the full 262,144-token context, 8 sequences and the fastest decode.',
    1: 'The whole model on one GPU, and the other card stays free. One conversation at a time, up to ~128k tokens of context; other requests queue. Choose 2 or 4 sequences in Settings if several people or agents use it at once (they share the KV: ~171k / ~127k tokens at a 64k context).',
  };
  const cardsText = (n) => `This machine has ${n} supported card${n === 1 ? '' : 's'}.`;
  const VERDICT = { fits: 'Fits this machine', tight: 'Fits, with a caveat', 'no-fit': 'Does not fit this machine' };

  // ---- profile ----
  let profiles = $state([]);
  let estimate = $state(null);
  let estErr = $state('');
  async function loadProfiles() {
    try {
      const p = await guarded(() => api('/api/profiles'));
      const preset = model?.preset ?? 'flashnext';
      const sec = preset === 'flashnext' ? 'engine' : 'serve';
      // the 27B on one card: the same five profiles, sized for one card's KV pool
      profiles = (model?.tp === 1 && sec === 'serve' ? p?.builtin_tp1?.serve : p?.builtin?.[sec]) ?? [];
    } catch (e) { W.error = e.message; }
  }
  async function loadEstimate() {
    try {
      const cfg = (await guarded(() => api('/api/config')))?.config;
      if (!cfg) return;
      const c = await guarded(() => api('/api/config/validate', { method: 'POST', body: { config: cfg } }));
      estimate = c?.estimate ?? null;
      estErr = c?.report?.errors?.map((e) => e.message).join('; ') ?? '';
    } catch (e) { estErr = e.message; }
  }
  async function pickProfile(name) {
    const r = await setupCall('/api/setup/profile', { name });
    if (r.ok) loadEstimate();
  }
  $effect(() => { if (step === 'profile' && model) { loadProfiles(); loadEstimate(); } });

  // ---- network ----
  let net = $state(null);
  let exposure = $state('local');
  let apiKeyOn = $state(false);
  let keyLabel = $state('default');
  let keyVal = $state('');
  let adminTok = $state('');
  let copied = $state('');
  $effect(() => {
    if (step === 'network' && W.setup && !net) {
      const n = W.setup.network;
      net = { engine_port: n.engine_port, console_port: n.console_port || 8090 };
      exposure = n.console_host && n.console_host !== '127.0.0.1' && n.console_host !== 'localhost' ? 'network' : 'local';
      apiKeyOn = n.keys.length > 0;
      // A stored key is a hash; its text was shown once, when it was made
      if (n.keys.length) { keyLabel = n.keys[0].label; keyVal = ''; }
    }
  });
  async function gen(kind) {
    try {
      const r = await guarded(() => api('/api/setup/keygen', { method: 'POST', body: { kind } }));
      if (kind === 'admin') adminTok = r.key; else keyVal = r.key;
    } catch (e) { W.error = e.message; }
  }
  async function copy(id, text) {
    if (await copyText(text)) { copied = id; setTimeout(() => { if (copied === id) copied = ''; }, 1600); }
  }
  async function saveNetwork() {
    const body = {
      engine_port: Number(net.engine_port),
      console_port: Number(net.console_port),
      console_host: exposure === 'network' ? '0.0.0.0' : '',
      keys: [],
    };
    const stored = W.setup?.network?.keys ?? [];
    if (apiKeyOn) {
      const label = keyLabel.trim() || 'default';
      // a new key replaces the first; a stored one is kept by its label (sent without its text)
      if (keyVal) body.keys.push({ label, key: keyVal });
      else if (stored.length) body.keys.push({ label: stored[0].label });
      // the keys made on the Security page stay
      for (const k of stored.slice(1)) if (k.label !== label) body.keys.push({ label: k.label });
    }
    if (exposure === 'network' && adminTok) body.admin_token = adminTok;
    if (exposure === 'local') body.admin_token = '';
    const r = await setupCall('/api/setup/network', body);
    if (!r.ok) return false;
    // this browser keeps the token and the key, so the console and the playground keep working
    if (body.admin_token) { setToken(body.admin_token); login(body.admin_token).catch(() => {}); }
    if (keyVal) { try { localStorage.setItem('saber-playground-key', keyVal); } catch (e) {} }
    return true;
  }
  async function networkNext() { if (await saveNetwork()) next(); }
  const needsToken = $derived(exposure === 'network' && !adminTok && !W.setup?.network?.admin_token_set);

  // ---- review and launch ----
  let cmd = $state(null);
  let job = $state(null);
  let logs = $state([]);
  let stopLogs = () => {};
  let applyTimer = null;
  async function loadReview() {
    try { cmd = await guarded(() => api('/api/command')); } catch (e) { cmd = { error: e.message }; }
    try { job = (await guarded(() => api('/api/apply')))?.job ?? null; } catch (e) {}
  }
  $effect(() => { if (step === 'review' && !cmd) loadReview(); });
  async function launch() {
    logs = [];
    stopLogs();
    stopLogs = stream('/api/logs/stream', 'log', (l) => { logs = [...logs.slice(-400), l]; });
    try {
      job = await guarded(() => api('/api/setup/launch', { method: 'POST', body: {} }));
      W.error = '';
    } catch (e) { W.error = e.message; return; }
    clearInterval(applyTimer);
    applyTimer = setInterval(async () => {
      try {
        const a = await api('/api/apply');
        job = a.job;
        ui.status = a.status;
        if (job && job.state !== 'running') { clearInterval(applyTimer); loadSetup(); refreshStatus(); }
      } catch (e) {}
    }, 800);
  }
  const ready = $derived(ui.status?.state === 'ready');
  const base = $derived(`${location.origin}/v1`);
  const key = $derived(keyVal || (W.setup?.network?.keys?.length ? playgroundKey() : ''));
  function playgroundKey() {
    try { return localStorage.getItem('saber-playground-key') || '<your API key>'; } catch (e) { return '<your API key>'; }
  }
  const servedModel = $derived(ui.status?.model ?? 'default');
  const curl = $derived(
    `curl ${base}/chat/completions \\\n  -H "Content-Type: application/json" \\\n${key ? `  -H "Authorization: Bearer ${key}" \\\n` : ''}  -d '{"model": "${servedModel}", "messages": [{"role": "user", "content": "Hello!"}]}'`,
  );
  const py = $derived(
    `from openai import OpenAI\n\nclient = OpenAI(base_url="${base}", api_key="${key || 'none'}")\nreply = client.chat.completions.create(\n    model="${servedModel}",\n    messages=[{"role": "user", "content": "Hello!"}],\n)\nprint(reply.choices[0].message.content)`,
  );
  async function finish(withoutStart) {
    const r = await setupCall('/api/setup/finish', { without_start: withoutStart });
    if (r.ok) location.hash = withoutStart ? '#overview' : '#playground';
  }
  let logBox = $state(null);
  $effect(() => { logs.length; if (logBox) logBox.scrollTop = logBox.scrollHeight; });
</script>

<div class="page wizard">
  <header class="whead">
    <div>
      <h1 class="title" id="wiz-title" tabindex="-1">Set up Saber</h1>
      <p class="muted lead">A few steps from a fresh machine to a running model. Every step is saved: you can close this page and come back.</p>
    </div>
    <button class="link" onclick={skip} disabled={W.busy}>I'll configure manually</button>
  </header>

  <ol class="stepper" aria-label="Setup steps">
    {#each steps as s, i}
      <li class:cur={s.current} class:done={i < idx}>
        <button disabled={!s.reachable || W.busy} onclick={() => go(s.id)} aria-current={s.current ? 'step' : undefined}>
          <span class="num">{i < idx ? '✓' : i + 1}</span><span class="lbl">{s.label}</span>
        </button>
      </li>
    {/each}
  </ol>

  {#if idx >= 0}<p class="steptxt">Step {idx + 1} of {steps.length}: {steps[idx].label}</p>{/if}

  {#if W.error}<p class="banner b-critical" role="alert">{W.error}</p>{/if}

  {#if !W.setup}
    <section class="panel"><p class="empty">Loading the setup…</p></section>
  {:else if step === 'system'}
    <section class="panel">
      <div class="c-head">
        <div>
          <h2>System check</h2>
          <div class="legend"><span class="muted">Read-only: nothing here touches the GPUs.</span></div>
        </div>
        <button class="btn small" onclick={loadSystem}>Check again</button>
      </div>
      {#if !W.system}
        <p class="empty">Checking the GPUs, drivers, container runtime and disks…</p>
      {:else}
        <p class="summary">
          {#if sysCounts.fail}<strong class="t-fail">{sysCounts.fail} problem{sysCounts.fail > 1 ? 's' : ''}</strong> to fix before a model can run{sysCounts.warn ? `, ${sysCounts.warn} warning${sysCounts.warn > 1 ? 's' : ''}` : ''}. You can still continue and pick the mock engine to try the console.
          {:else if sysCounts.warn}<strong class="t-warn">Ready, with {sysCounts.warn} warning{sysCounts.warn > 1 ? 's' : ''}.</strong>
          {:else}<strong class="t-pass">Everything checks out.</strong>{/if}
        </p>
        <ul class="checks">
          {#each W.system.checks as c (c.id)}
            <li class="check s-{c.status}">
              <span class="badge-s" aria-label={ICON[c.status]}>
                {#if c.status === 'pass'}<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8.5 6.5 12 13 4.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" /></svg>
                {:else if c.status === 'fail'}<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" /></svg>
                {:else if c.status === 'warn'}<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M8 3v6M8 12.5v.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" /></svg>
                {:else}<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M8 7v5M8 4v.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" /></svg>{/if}
              </span>
              <div class="cbody">
                <div class="ctitle">{c.title} <span class="cstat">{ICON[c.status]}</span></div>
                <div class="cdetail">{c.detail}</div>
                {#if c.fix}<div class="fix"><strong>How to fix:</strong> {c.fix}</div>{/if}
              </div>
            </li>
          {/each}
        </ul>
        {#if W.system.gpus.length}
          <details class="tbl">
            <summary>GPU details</summary>
            <div class="scroll">
              <table>
                <thead><tr><th class="l">GPU</th><th class="l">Arch</th><th>VRAM</th><th class="l">Slot link</th><th class="l">Card link</th><th class="l">PCI</th><th class="l">Peers</th></tr></thead>
                <tbody>
                  {#each W.system.gpus as g}
                    <tr><td class="l">{g.name || '–'}{g.integrated ? ' (integrated)' : ''}</td><td class="l">{g.gfx}</td><td>{g.vram_bytes ? bytes(g.vram_bytes, 0) : '–'}</td>
                      <td class="l">{g.slot_max.width ? `x${g.slot_max.width} @ ${g.slot_max.speed_gts} GT/s` : '–'}</td>
                      <td class="l">{g.link_max.width ? `x${g.link_max.width} @ ${g.link_max.speed_gts} GT/s` : '–'}</td>
                      <td class="l">{g.bdf || '–'}</td><td class="l">{g.p2p_peers.join(', ') || 'none'}</td></tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </details>
        {/if}
      {/if}
    </section>
  {:else if step === 'model'}
    <section class="panel">
      <div class="c-head">
        <div>
          <h2>Choose a model</h2>
          <div class="legend"><span class="muted">{W.catalog ? `Downloads go to ${W.catalog.roots.packs} · ${gb(W.catalog.free_bytes)} free` : ''}</span></div>
        </div>
      </div>
      {#if !W.catalog}
        <p class="empty">Loading the model catalog…</p>
      {:else if !W.catalog.models.length}
        <p class="empty">The model catalog is empty ({W.catalog.source}).</p>
      {:else}
        <div class="models" role="radiogroup" aria-label="Models">
          {#each W.catalog.models as m (m.id)}
            {@const sel = W.setup.state.model === m.id}
            <article class="mcard" class:sel>
              <label class="mpick">
                <input type="radio" name="model" checked={sel} onchange={() => pick(m)} disabled={W.busy} />
                <span class="mname">{m.name}</span>
                <span class="kind">{m.kind}</span>
              </label>
              <p class="mdesc">{m.description}</p>
              <div class="mfacts">
                <span>{m.components.length ? m.note || gb(m.remaining_bytes) : 'No download'}</span>
                <span class="fit f-{m.fit.verdict}">{VERDICT[m.fit.verdict]}</span>
                {#if m.installed && m.components.length}<span class="fit f-fits">Installed</span>{/if}
              </div>
              {#if m.fit.reasons.length}
                <ul class="reasons">{#each m.fit.reasons as r}<li>{r}</li>{/each}</ul>
              {/if}
              {#if sel && m.tp_choices.length > 1 && m.kind !== 'mock'}
                <fieldset class="tpc">
                  <legend>How many cards</legend>
                  <div class="tpopts">
                    {#each m.tp_forms as f (f.tp)}
                      <label class="tpopt" class:sel={m.tp === f.tp} class:off={!f.possible}>
                        <input type="radio" name="tp-{m.id}" checked={m.tp === f.tp} disabled={!f.possible || W.busy} onchange={() => pickTp(f.tp)} />
                        <span>
                          <span class="tph"><strong>{f.tp === 2 ? 'Two cards (TP=2)' : 'One card (TP=1)'}</strong>{#if f.tp === m.recommended_tp}<span class="rec">recommended here</span>{/if}</span>
                          <span class="tpdesc">{TP_TEXT[f.tp]}{f.note ? ` Download: ${f.note}.` : ''}</span>
                          {#if !f.possible}<span class="tpdesc why">{cardsText(W.catalog.gpus)} {f.tp === 2 ? 'Two cards are needed.' : ''}</span>{/if}
                          {#if f.possible && f.installed}<span class="fit f-fits">Installed</span>{/if}
                        </span>
                      </label>
                    {/each}
                  </div>
                  {#if m.tp === 1 && W.catalog.gpus > 1}
                    <label class="card">Card
                      <select value={W.catalog.device ?? 0} disabled={W.busy} onchange={(e) => pickTp(1, Number(e.currentTarget.value))}>
                        {#each Array.from({ length: W.catalog.gpus }, (_, i) => i) as i}<option value={i}>Card {i}</option>{/each}
                      </select>
                    </label>
                  {/if}
                </fieldset>
              {:else if sel && m.kind !== 'mock'}
                <p class="note tpn">Runs on two cards (TP=2) only: no single-card pack is hosted for this model.{W.catalog.gpus < 2 ? ` ${cardsText(W.catalog.gpus)}` : ''}</p>
              {/if}
              {#if sel && m.components.length}
                <div class="comps">
                  {#each m.components as c (c.id)}<DownloadRow model={m} comp={c} />{/each}
                </div>
              {/if}
            </article>
          {/each}
        </div>
      {/if}
    </section>
  {:else if step === 'profile'}
    <section class="panel">
      <div class="c-head">
        <div>
          <h2>Pick a profile</h2>
          <div class="legend"><span class="muted">Profiles tune the engine for a goal. You can change any setting later in Settings.</span></div>
        </div>
      </div>
      {#if model && model.kind !== 'mock'}
        <p class="note tpline">Runs on <strong>{model.tp === 1 ? 'one card (TP=1)' : 'two cards (TP=2)'}</strong>{model.tp_choices?.length > 1 ? ' · change it in the Model step' : ''}.</p>
      {/if}
      {#if !profiles.length}
        <p class="empty">Loading the profiles…</p>
      {:else}
        <div class="profiles" role="radiogroup" aria-label="Profiles">
          {#each profiles as p (p.name)}
            {@const sel = W.setup.state.profile === p.name}
            <label class="pcard" class:sel>
              <input type="radio" name="profile" checked={sel} onchange={() => pickProfile(p.name)} disabled={W.busy} />
              <span class="pname">{p.name}{p.name === 'Balanced' ? ' (recommended)' : ''}</span>
              <span class="pdesc">{p.description}</span>
            </label>
          {/each}
        </div>
      {/if}
      {#if W.setup.state.profile}
        <div class="est">
          <h3>Memory estimate</h3>
          {#if !estimate}
            <p class="muted">Computing…</p>
          {:else}
            {#if estimate.label}<p class="note">{estimate.label}</p>{/if}
            <div class="estrow">
              <div><div class="val">{estimate.seqs_at_ctx ?? '–'}<span class="unit"> / {estimate.requested_seqs}</span></div><div class="sub">requests at the full {int(estimate.max_ctx)}-token context</div></div>
              <div><div class="val">{estimate.kv_capacity_tokens != null ? int(estimate.kv_capacity_tokens) : '–'}</div><div class="sub">tokens of KV cache</div></div>
              {#if estimate.vram_bytes}<div><div class="val">{bytes(estimate.vram_bytes, 0)}</div><div class="sub">{estimate.tp === 1 ? 'VRAM on the card' : 'VRAM per card'}</div></div>{/if}
            </div>
            {#if !estimate.fits}<p class="t-fail" role="alert">The estimate says this does not fit: pick another profile or a smaller context in Settings.</p>{/if}
            {#each estimate.warnings ?? [] as w}<p class="note">{w}</p>{/each}
            <KvTip {estimate} />
            <p class="note">An estimate from the pack and the model's config; the engine checks the real budget when it starts.</p>
          {/if}
          {#if estErr}<p class="t-fail">{estErr}</p>{/if}
        </div>
      {/if}
    </section>
  {:else if step === 'network'}
    <section class="panel">
      <div class="c-head"><div><h2>Network and security</h2><div class="legend"><span class="muted">Who can reach the console and the API.</span></div></div></div>
      {#if net}
        <div class="form">
          <fieldset>
            <legend>Who can open this console</legend>
            <label class="opt"><input type="radio" bind:group={exposure} value="local" /> <span><strong>Only this computer</strong> (recommended). The console listens on 127.0.0.1.</span></label>
            <label class="opt"><input type="radio" bind:group={exposure} value="network" /> <span><strong>Other computers on my network</strong>. Listens on every address and requires an admin token.</span></label>
          </fieldset>
          {#if exposure === 'network'}
            <div class="secret">
              <label for="admintok">Admin token</label>
              <div class="row">
                <input id="admintok" type="text" readonly value={adminTok || (W.setup.network.admin_token_set ? '(a token is already set)' : '')} placeholder="Generate a token" spellcheck="false" />
                <button class="btn small" onclick={() => gen('admin')}>Generate</button>
                <button class="btn small" disabled={!adminTok} onclick={() => copy('admin', adminTok)}>{copied === 'admin' ? 'Copied' : 'Copy'}</button>
              </div>
              <p class="note">Anyone opening the console from another computer types this token once. Keep it somewhere safe. The new address takes effect the next time the console starts.</p>
            </div>
          {/if}
          <div class="ports">
            <label>Console port <input type="number" min="1024" max="65535" bind:value={net.console_port} /></label>
            <label>Engine port <input type="number" min="1024" max="65535" bind:value={net.engine_port} /></label>
          </div>
          <p class="note">Apps connect through the console at <code>{location.origin}/v1</code>; the engine's own port stays internal. A new console port or address takes effect the next time the console starts.</p>
          <fieldset>
            <legend>API key</legend>
            <label class="opt"><input type="checkbox" bind:checked={apiKeyOn} /> <span>Require an API key for the OpenAI-compatible API (recommended when other computers or apps connect)</span></label>
            {#if apiKeyOn}
              <div class="secret">
                <div class="row">
                  <input aria-label="Key label" class="lbl" type="text" bind:value={keyLabel} placeholder="label" />
                  <input aria-label="API key" type="text" readonly value={keyVal} placeholder={W.setup?.network?.keys?.length ? 'stored (shown once when made); generate to replace' : 'Generate a key'} spellcheck="false" />
                  <button class="btn small" onclick={() => gen('api')}>Generate</button>
                  <button class="btn small" disabled={!keyVal} onclick={() => copy('key', keyVal)}>{copied === 'key' ? 'Copied' : 'Copy'}</button>
                </div>
              </div>
            {/if}
          </fieldset>
        </div>
      {/if}
    </section>
  {:else if step === 'review'}
    <section class="panel">
      <div class="c-head"><div><h2>Review and launch</h2></div></div>
      <dl class="kv review">
        <dt>Model</dt><dd>{model?.name ?? W.setup.state.model}</dd>
        {#if model && model.kind !== 'mock'}<dt>Cards</dt><dd>{model.tp === 1 ? `one card (TP=1), card ${W.catalog?.device ?? 0}` : 'two cards (TP=2)'}</dd>{/if}
        <dt>Profile</dt><dd>{W.setup.state.profile}</dd>
        <dt>Console</dt><dd>{W.setup.network.console_host && W.setup.network.console_host !== '127.0.0.1' ? `every address, port ${W.setup.network.console_port || 8090} (admin token set)` : `this computer only, port ${W.setup.network.console_port || 8090}`}{(W.setup.network.console_port || 8090) !== W.setup.network.running_console_port || W.setup.network.console_host ? ` · from the next console start (now ${location.host})` : ''}</dd>
        <dt>API key</dt><dd>{W.setup.network.keys.length ? 'required' : 'not required'}</dd>
        <dt>Config file</dt><dd><code>{W.setup.config_path}</code></dd>
      </dl>
      <h3 class="sp">The exact command</h3>
      {#if cmd?.error}<p class="t-fail">{cmd.error}</p>
      {:else if cmd}<pre class="tail cmd">{cmd.engine_line}</pre>{:else}<p class="muted">…</p>{/if}

      {#if ready && job?.state !== 'running'}
        <div class="readybox" role="status">
          <h2>Saber is running</h2>
          <p>The model is loaded and answering. Try it in the <a href="#playground">Playground</a>, or call it from your own code:</p>
          <div class="snip">
            <div class="snh"><h3>curl</h3><button class="btn small" onclick={() => copy('curl', curl)}>{copied === 'curl' ? 'Copied' : 'Copy'}</button></div>
            <pre class="tail">{curl}</pre>
          </div>
          <div class="snip">
            <div class="snh"><h3>Python (OpenAI SDK)</h3><button class="btn small" onclick={() => copy('py', py)}>{copied === 'py' ? 'Copied' : 'Copy'}</button></div>
            <pre class="tail">{py}</pre>
          </div>
          <div class="actions">
            <button class="btn primary" onclick={() => finish(false)}>Finish and open the Playground</button>
          </div>
        </div>
      {:else}
        {#if job}
          <ol class="jobsteps" aria-label="Launch progress">
            {#each job.steps as s}
              <li class="js-{s.state}"><span>{s.label}</span>{#if s.message}<span class="muted"> · {s.message}</span>{/if}</li>
            {/each}
          </ol>
          {#if job.state === 'failed' || job.state === 'rolled-back'}
            <p class="t-fail" role="alert">The engine did not start: {job.message ?? 'see the log below'}. Check the log, change the settings (Back), and press Start again.</p>
          {/if}
        {/if}
        <div class="actions">
          <button class="btn primary" disabled={job?.state === 'running'} onclick={launch}>{job?.state === 'running' ? 'Starting…' : job ? 'Start again' : 'Start the engine'}</button>
          <button class="link" onclick={() => finish(true)}>Finish without starting</button>
        </div>
        {#if logs.length || job?.state === 'running'}
          <h3 class="sp">Boot log</h3>
          <pre class="tail log" bind:this={logBox} aria-live="polite">{#each logs as l}<span class:s-err={l.stream === 'stderr'} class:s-console={l.stream === 'console'}>{l.text}
</span>{/each}</pre>
        {/if}
      {/if}
    </section>
  {/if}

  {#if W.setup && step !== 'review'}
    <footer class="wfoot">
      <button class="btn" onclick={back} disabled={idx <= 0 || W.busy}>Back</button>
      <span class="muted gate">{step === 'network' && needsToken ? 'Generate an admin token to let other computers in.' : W.setup.blocker ?? ''}</span>
      {#if step === 'network'}
        <button class="btn primary" onclick={networkNext} disabled={W.busy || needsToken}>Next</button>
      {:else}
        <button class="btn primary" onclick={next} disabled={W.busy || !!W.setup.blocker}>Next</button>
      {/if}
    </footer>
  {:else if W.setup}
    <footer class="wfoot"><button class="btn" onclick={back} disabled={W.busy}>Back</button></footer>
  {/if}
</div>

<style>
  .wizard { max-width: 920px; margin: 0 auto; width: 100%; }
  .whead { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px 16px; flex-wrap: wrap; }
  .lead { margin: 4px 0 0; font-size: var(--fs-sm); }
  .title:focus { outline: none; }
  .link { border: 0; background: none; color: var(--accent); cursor: pointer; font: inherit; font-size: var(--fs-sm); padding: 4px 0; text-decoration: underline; text-underline-offset: 2px; }
  .link:disabled { opacity: 0.5; cursor: default; }
  .stepper { list-style: none; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; padding: 0; margin: 0; }
  .stepper button { width: 100%; display: flex; align-items: center; gap: 8px; border: 1px solid var(--ring); background: var(--surface); color: var(--ink-2); border-radius: var(--radius); padding: 8px 10px; cursor: pointer; font-size: var(--fs-sm); text-align: left; min-width: 0; }
  .stepper button:disabled { cursor: default; opacity: 0.6; }
  .stepper .num { flex: none; width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; background: var(--surface-2); font-size: var(--fs-xs); font-weight: 650; }
  .stepper .lbl { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .stepper li.cur button { border-color: var(--accent); color: var(--ink); font-weight: 600; }
  .stepper li.cur .num { background: var(--accent); color: #fff; }
  .stepper li.done .num { background: var(--accent-fill); color: var(--accent); }
  .steptxt { display: none; margin: -6px 0 0; font-size: var(--fs-sm); font-weight: 600; }
  @media (max-width: 760px) {
    .stepper .lbl { display: none; }
    .stepper button { justify-content: center; padding: 6px; }
    .steptxt { display: block; }
  }
  .banner { margin: 0; padding: 8px 10px; border-radius: 5px; font-size: var(--fs-sm); }
  .b-critical { background: color-mix(in srgb, var(--critical) 14%, transparent); }
  .summary { margin: 4px 0 10px; font-size: var(--fs-sm); }
  .t-pass { color: var(--good-text); }
  .t-warn { color: color-mix(in srgb, var(--warning) 60%, var(--ink)); }
  .t-fail { color: var(--critical); font-size: var(--fs-sm); }
  .checks { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
  .check { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid var(--grid); }
  .check:first-child { border-top: 0; }
  .badge-s { flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; color: var(--on-status); margin-top: 1px; }
  .s-pass .badge-s { background: var(--good); }
  .s-warn .badge-s { background: var(--warning); color: #3a2a00; }
  .s-fail .badge-s { background: var(--critical); }
  .s-info .badge-s { background: var(--deemph); }
  .cbody { min-width: 0; display: grid; gap: 2px; }
  .ctitle { font-weight: 600; font-size: var(--fs-sm); }
  .cstat { font-weight: 500; color: var(--muted); font-size: var(--fs-xs); margin-left: 4px; }
  .cdetail { font-size: var(--fs-sm); color: var(--ink-2); overflow-wrap: anywhere; }
  .fix { font-size: var(--fs-sm); background: var(--surface-2); border-radius: 4px; padding: 6px 8px; margin-top: 4px; overflow-wrap: anywhere; }
  .models { display: grid; gap: 10px; }
  .mcard { border: 1px solid var(--ring); border-radius: var(--radius); padding: 12px 14px; display: grid; gap: 6px; }
  .mcard.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .mpick { display: flex; align-items: center; gap: 8px; cursor: pointer; flex-wrap: wrap; }
  .mname { font-weight: 650; }
  .kind { font-size: var(--fs-xs); background: var(--surface-2); color: var(--ink-2); padding: 1px 7px; border-radius: 9px; }
  .mdesc { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
  .mfacts { display: flex; gap: 6px 14px; flex-wrap: wrap; font-size: var(--fs-sm); align-items: center; }
  .fit { font-size: var(--fs-xs); font-weight: 600; padding: 1px 8px; border-radius: 9px; }
  .f-fits { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good-text); }
  .f-tight { background: color-mix(in srgb, var(--warning) 22%, transparent); color: color-mix(in srgb, var(--warning) 45%, var(--ink)); }
  .f-no-fit { background: color-mix(in srgb, var(--critical) 14%, transparent); color: var(--critical); }
  .reasons { margin: 0; padding-left: 18px; font-size: var(--fs-sm); color: var(--ink-2); }
  .comps { border-top: 1px solid var(--grid); margin-top: 4px; }
  .tpc { border: 0; padding: 0; margin: 4px 0 0; display: grid; gap: 6px; min-width: 0; }
  .tpc legend { font-weight: 600; font-size: var(--fs-sm); padding: 0; margin-bottom: 4px; }
  .tpopts { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; }
  .tpopt { display: flex; gap: 8px; align-items: flex-start; border: 1px solid var(--ring); border-radius: var(--radius); padding: 8px 10px; cursor: pointer; font-size: var(--fs-sm); min-width: 0; }
  .tpopt.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .tpopt.off { cursor: default; opacity: 0.7; }
  .tpopt input { margin-top: 3px; flex: none; }
  .tpopt > span { display: grid; gap: 3px; min-width: 0; }
  .tpopt > span > .fit { justify-self: start; }
  .tph { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 0; }
  .tpdesc { font-size: var(--fs-xs); color: var(--ink-2); }
  .tpdesc.why { color: var(--critical); }
  .rec { margin-left: 6px; font-size: var(--fs-xs); font-weight: 600; padding: 1px 7px; border-radius: 9px; background: var(--accent-fill); color: var(--accent); }
  .card { display: grid; gap: 4px; font-size: var(--fs-sm); max-width: 200px; }
  .tpn { margin: 0; }
  .tpline { margin: 0 0 10px; }
  .profiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
  .pcard { border: 1px solid var(--ring); border-radius: var(--radius); padding: 10px 12px; display: grid; grid-template-columns: auto 1fr; gap: 4px 8px; cursor: pointer; align-content: start; }
  .pcard.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .pcard input { margin-top: 3px; }
  .pname { font-weight: 600; font-size: var(--fs-sm); }
  .pdesc { grid-column: 2; font-size: var(--fs-xs); color: var(--ink-2); }
  .est { margin-top: 14px; border-top: 1px solid var(--grid); padding-top: 12px; display: grid; gap: 6px; }
  .estrow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  @media (max-width: 520px) { .estrow { grid-template-columns: minmax(0, 1fr); } }
  .form { display: grid; gap: 14px; }
  fieldset { border: 0; padding: 0; margin: 0; display: grid; gap: 6px; }
  legend { font-weight: 600; font-size: var(--fs-sm); margin-bottom: 4px; }
  .opt { display: flex; gap: 8px; align-items: flex-start; font-size: var(--fs-sm); cursor: pointer; }
  .opt input { margin-top: 3px; }
  .secret { display: grid; gap: 4px; }
  .secret > label { font-size: var(--fs-sm); font-weight: 600; }
  .row { display: flex; gap: 6px; flex-wrap: wrap; }
  .row input[readonly] { flex: 1 1 260px; font-family: var(--mono); }
  .row input.lbl { flex: 0 1 120px; }
  .ports { display: flex; gap: 14px; flex-wrap: wrap; }
  .ports label { display: grid; gap: 4px; font-size: var(--fs-sm); font-weight: 600; }
  .ports input { width: 140px; }
  .review { margin-bottom: 10px; }
  .sp { margin: 12px 0 6px; }
  .cmd { white-space: pre-wrap; overflow-wrap: anywhere; min-height: 0; }
  .log { max-height: 320px; overflow-y: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
  .actions { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
  .jobsteps { list-style: none; display: flex; flex-wrap: wrap; gap: 6px; padding: 0; margin: 12px 0 0; font-size: var(--fs-xs); }
  .jobsteps li { padding: 3px 9px; border-radius: 9px; background: var(--surface-2); color: var(--ink-2); }
  .jobsteps .js-running { background: var(--accent-fill); color: var(--accent); font-weight: 600; }
  .jobsteps .js-done { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good-text); }
  .jobsteps .js-failed { background: color-mix(in srgb, var(--critical) 14%, transparent); color: var(--critical); }
  .readybox { margin-top: 14px; border: 1px solid color-mix(in srgb, var(--good) 40%, transparent); background: color-mix(in srgb, var(--good) 6%, transparent); border-radius: var(--radius); padding: 12px 14px; display: grid; gap: 10px; }
  .readybox p { margin: 0; font-size: var(--fs-sm); }
  .snip { display: grid; gap: 4px; min-width: 0; }
  .snh { display: flex; justify-content: space-between; align-items: center; }
  .snip pre { min-height: 0; white-space: pre; }
  .wfoot { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 2px 0 10px; }
  .gate { font-size: var(--fs-sm); text-align: center; flex: 1; }
</style>
