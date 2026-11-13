// The Settings plane's shared state: the flag schema, the saved config, the DRAFT the user is
// editing, and the server's check of that draft (/api/config/validate: per-field errors and
// warnings, the memory estimate, the effective command line with each flag's source, the diff
// against the running config, whether applying restarts the engine). The draft is checked on
// every change, debounced.
import { api, getToken, AuthError } from './api.js';
import { guarded, save } from './store.svelte.js';

export const TIERS = ['basic', 'advanced', 'diagnostic'];

function loadTier() {
  try { return localStorage.getItem('saber-settings-tier') || 'basic'; } catch (e) { return 'basic'; }
}

export const S = $state({
  schema: null,
  saved: null,
  path: '',
  draft: null,
  check: null,
  checking: false,
  error: '',
  tier: loadTier(),
  job: null,
  applyOpen: false,
  seq: 0,
});

export function setTier(t) {
  S.tier = t;
  save('saber-settings-tier', t);
}

const clone = (x) => JSON.parse(JSON.stringify(x));

/** The schema section of a preset (`flashnext` boots `saber engine`, the others `saber serve`). */
export function sectionOf(preset) {
  return preset === 'flashnext' ? 'engine' : 'serve';
}

export async function loadSettings() {
  try {
    const [schema, cfg] = await Promise.all([guarded(() => api('/api/schema')), guarded(() => api('/api/config'))]);
    if (!schema || !cfg) return;
    S.schema = schema;
    S.saved = cfg.config;
    S.path = cfg.path;
    if (!S.draft) S.draft = clone(cfg.config);
    // the last apply (a rollback or a failure shows as a banner)
    try { S.job = (await api('/api/apply')).job; } catch (e) {}
    S.error = '';
    scheduleCheck(0);
  } catch (e) {
    if (!(e instanceof AuthError)) S.error = e.message;
  }
}

/** Throw away the draft: back to the saved config. */
export function discard() {
  S.draft = clone(S.saved);
  scheduleCheck(0);
}

let timer = null;
export function scheduleCheck(delay = 300) {
  clearTimeout(timer);
  timer = setTimeout(runCheck, delay);
}

async function runCheck() {
  if (!S.draft) return;
  const seq = ++S.seq;
  S.checking = true;
  try {
    const c = await api('/api/config/validate', { method: 'POST', body: { config: S.draft } });
    if (seq === S.seq) S.check = c;
  } catch (e) {
    if (seq === S.seq) S.check = { report: { ok: false, errors: [{ field: 'config', level: 'error', message: e.message }], warnings: [] } };
  } finally {
    if (seq === S.seq) S.checking = false;
  }
}

/** Set (or with `undefined`, reset) one flag override in the draft. */
export function setFlag(name, value) {
  const flags = { ...(S.draft.engine.flags || {}) };
  if (value === undefined) delete flags[name];
  else flags[name] = value;
  S.draft.engine.flags = flags;
  scheduleCheck();
}

/** A config field by dotted path (`engine.port`). */
export function setField(path, value) {
  const parts = path.split('.');
  let o = S.draft;
  for (const p of parts.slice(0, -1)) o = o[p];
  o[parts[parts.length - 1]] = value;
  scheduleCheck();
}

export function replaceDraft(cfg) {
  S.draft = clone(cfg);
  scheduleCheck(0);
}

/** Issues for one field (errors first). */
export function issuesFor(field) {
  const r = S.check?.report;
  if (!r) return [];
  return [...r.errors.filter((i) => i.field === field), ...r.warnings.filter((i) => i.field === field)];
}

/** The effective flag row (value + source) from the last check. */
export function effectiveOf(name) {
  return S.check?.effective?.find((e) => e.name === name) ?? null;
}

export async function startApply(note = '') {
  const r = await api('/api/config/apply', { method: 'POST', body: { config: S.draft, note } });
  S.job = r;
  return r;
}

export async function pollJob() {
  const r = await api('/api/apply');
  S.job = r.job;
  if (r.job && r.job.state !== 'running') {
    // the server's config may be the applied one or the rolled-back one: reload it
    const cfg = await api('/api/config');
    S.saved = cfg.config;
    S.draft = clone(cfg.config);
    scheduleCheck(0);
  }
  return r;
}

/** POST a raw text body (profile import). */
export async function postText(path, text) {
  const headers = { 'X-Saber-Console': '1', 'Content-Type': 'text/plain' };
  const tok = getToken();
  if (tok) headers.Authorization = `Bearer ${tok}`;
  const r = await fetch(path, { method: 'POST', headers, body: text });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}

/** A flag value as the text control shows it. */
export function valueText(v) {
  if (v === undefined || v === null) return '';
  if (Array.isArray(v)) return v.join(', ');
  return String(v);
}

/** Parse a control's text into the override's JSON type for `flag` (`undefined` = reset). */
export function parseValue(flag, text) {
  const t = String(text).trim();
  if (t === '') return undefined;
  if (flag.multiple && !flag.delimiter) {
    const parts = t.split(',').map((x) => x.trim()).filter(Boolean);
    return parts.length > 1 ? parts : parts[0];
  }
  if (flag.multiple) return t.split(',').map((x) => x.trim()).filter(Boolean).join(flag.delimiter);
  if (flag.kind === 'int' || flag.kind === 'uint') {
    return /^-?\d+$/.test(t) ? Number(t) : t;
  }
  if (flag.kind === 'float') {
    const n = Number(t);
    return Number.isFinite(n) && /^-?[\d.eE+-]+$/.test(t) ? n : t;
  }
  return t;
}
