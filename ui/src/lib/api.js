// The console's /api client. A token (needed when the console was started with an admin token)
// is kept in this TAB's sessionStorage (gone when the tab closes; never in localStorage) and
// sent as a Bearer header, or as ?token= for an EventSource, which cannot set headers. Signing
// in also makes the SERVER set a signed HttpOnly cookie, so a new tab or a restart of the
// browser stays signed in for 30 days without the token in page storage.

const KEY = 'saber-admin-token';

// a token an older console build left in localStorage is moved to this tab and forgotten
try {
  const old = localStorage.getItem(KEY);
  if (old) {
    if (!sessionStorage.getItem(KEY)) sessionStorage.setItem(KEY, old);
    localStorage.removeItem(KEY);
  }
} catch (e) {}

export function getToken() {
  try { return sessionStorage.getItem(KEY) || ''; } catch (e) { return ''; }
}
export function setToken(t) {
  try { t ? sessionStorage.setItem(KEY, t) : sessionStorage.removeItem(KEY); } catch (e) {}
}

/** Sign in: the server checks the token and remembers this browser (a signed 30-day cookie). */
export async function login(token) {
  const r = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Saber-Console': '1' },
    body: JSON.stringify({ token }),
  });
  const no = await refusal(r);
  if (no) throw no;
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  setToken(token); // the tab's fallback when cookies are blocked
}

/** Sign out: the server clears the cookie; this tab forgets the token. */
export async function logoutServer() {
  setToken('');
  try { await fetch('/api/auth/logout', { method: 'POST', headers: { 'X-Saber-Console': '1' } }); } catch (e) {}
}

/** A sign-in link (`/#token=...`, printed by the console at start): sign in with it, then
 *  remove the token from the address bar. The fragment never leaves the browser. */
export async function consumeSignInLink() {
  const m = /^#token=([^&]+)/.exec(location.hash);
  if (!m) return;
  const token = decodeURIComponent(m[1]);
  history.replaceState(null, '', location.pathname + location.search);
  try { await login(token); } catch (e) {}
}

export class AuthError extends Error {}
/** 429: this address failed auth too often; `retryAfter` seconds. */
export class RateLimited extends Error {
  constructor(msg, retryAfter) { super(msg); this.retryAfter = retryAfter; }
}

async function refusal(r) {
  if (r.status === 401) return new AuthError('admin token required');
  if (r.status === 429) {
    const d = await r.json().catch(() => ({}));
    return new RateLimited(d.error || 'too many failed attempts', Number(r.headers.get('Retry-After')) || 60);
  }
  return null;
}

export async function api(path, { method = 'GET', body } = {}) {
  const headers = { Accept: 'application/json' };
  const tok = getToken();
  if (tok) headers.Authorization = `Bearer ${tok}`;
  if (method !== 'GET') headers['X-Saber-Console'] = '1';
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const r = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const no = await refusal(r);
  if (no) throw no;
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}

export function stream(path, event, onData) {
  const tok = getToken();
  const url = tok ? `${path}?token=${encodeURIComponent(tok)}` : path;
  const es = new EventSource(url);
  es.addEventListener(event, (e) => {
    try { onData(JSON.parse(e.data)); } catch (err) {}
  });
  return () => es.close();
}

// formatting
export function compact(v, digits = 1) {
  if (v == null || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(digits) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(digits) + 'K';
  if (a >= 100) return Math.round(v).toLocaleString();
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(a === 0 ? 0 : 2);
}

export function duration(ms) {
  if (ms == null || ms < 0) return '–';
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

// ----

/** A GET that returns text (downloads that need the token header). */
export async function apiText(path) {
  const headers = {};
  const tok = getToken();
  if (tok) headers.Authorization = `Bearer ${tok}`;
  const r = await fetch(path, { headers });
  const no = await refusal(r);
  if (no) throw no;
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.text();
}

/** Save `text` as a file in the browser. */
export function saveText(name, text, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function bytes(v, digits = 1) {
  if (v == null || !Number.isFinite(v)) return '–';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  let i = 0;
  let x = Math.abs(v);
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
  return `${(Math.sign(v) * x).toFixed(i === 0 ? 0 : digits)} ${u[i]}`;
}

export function ms(v) {
  if (v == null || !Number.isFinite(v)) return '–';
  if (v >= 60000) return `${(v / 60000).toFixed(1)} min`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v >= 100) return `${Math.round(v)} ms`;
  return `${v.toFixed(1)} ms`;
}

export function pct(v, digits = 1) {
  return v == null || !Number.isFinite(v) ? '–' : `${(v * 100).toFixed(digits)}%`;
}

export function int(v) {
  return v == null || !Number.isFinite(v) ? '–' : Math.round(v).toLocaleString();
}

export function when(tsMs) {
  if (tsMs == null) return '–';
  const d = new Date(tsMs);
  const today = new Date();
  const same = d.toDateString() === today.toDateString();
  return same
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/** `name{a="x",b="y"}` -> { name, labels: {a: 'x', b: 'y'} }. */
export function parseKey(k) {
  const i = k.indexOf('{');
  if (i < 0) return { name: k, labels: {} };
  const labels = {};
  const re = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g;
  let m;
  while ((m = re.exec(k.slice(i)))) labels[m[1]] = m[2];
  return { name: k.slice(0, i), labels };
}

/**
 * Group a series window's columns of family `name` by one label (summing the others, e.g.
 * over ranks): -> { [labelValue]: values[] }.
 */
export function groupBy(win, name, label) {
  const out = {};
  for (const [k, vals] of Object.entries(win?.series ?? {})) {
    const p = parseKey(k);
    if (p.name !== name) continue;
    const g = label ? p.labels[label] ?? '' : '';
    const acc = out[g] ?? (out[g] = vals.map(() => null));
    vals.forEach((v, i) => { if (v != null) acc[i] = (acc[i] ?? 0) + v; });
  }
  return out;
}

/** Sum of every non-null value (a rate column's window total = sum x interval). */
export function total(vals) {
  let s = 0;
  let any = false;
  for (const v of vals ?? []) if (v != null) { s += v; any = true; }
  return any ? s : null;
}

export function last(vals) {
  for (let i = (vals?.length ?? 0) - 1; i >= 0; i--) if (vals[i] != null) return vals[i];
  return null;
}

export function meanOf(vals) {
  const v = (vals ?? []).filter((x) => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** Element-wise a / b (null where either is missing or b is 0). */
export function ratioCols(a, b, scale = 1) {
  return (a ?? []).map((x, i) => (x == null || b?.[i] == null || b[i] === 0 ? null : (x / b[i]) * scale));
}

/** The fixed categorical order (never cycled: a ninth series folds into "Other"). */
export const SERIES = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7', '--series-8'];
export const seriesColor = (i) => (i === 0 ? '--accent' : SERIES[i] ?? '--deemph');
