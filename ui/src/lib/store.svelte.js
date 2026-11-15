// Console-wide state: the global time range, auto-refresh, theme, the engine status and the
// admin-token gate. Pages read `ui.tick` in an $effect to reload on each refresh.
import { api, AuthError, RateLimited, getToken } from './api.js';

export const RANGES = [
  { id: '15m', s: 900 },
  { id: '1h', s: 3600 },
  { id: '6h', s: 21600 },
  { id: '24h', s: 86400 },
  { id: '7d', s: 7 * 86400 },
  { id: '30d', s: 30 * 86400 },
];
export const REFRESH = [
  { id: 'off', s: 0 },
  { id: '5s', s: 5 },
  { id: '15s', s: 15 },
  { id: '1m', s: 60 },
];
/** The in-memory metrics ring holds 24 h; live pages clamp to it. */
export const RING_S = 86400;

function load(k, d) {
  try { return localStorage.getItem(k) || d; } catch (e) { return d; }
}
export function save(k, v) {
  try { localStorage.setItem(k, v); } catch (e) {}
}

export const ui = $state({
  range: load('saber-range', '15m'),
  refresh: load('saber-refresh', '5s'),
  mode: load('saber-theme', 'system'),
  themeKey: 0,
  tick: 0,
  needToken: false,
  /** Seconds left of a failed-auth lockout (429), else 0. */
  lockedFor: 0,
  /** Who this session is (`admin` / `local`) and whether a token is required. */
  who: null,
  tokenRequired: false,
  status: null,
});

export function rangeS() {
  return RANGES.find((r) => r.id === ui.range)?.s ?? 900;
}
export function liveS() {
  return Math.min(rangeS(), RING_S);
}

/** Run an /api call; a 401 raises the token gate instead of throwing. */
export async function guarded(fn) {
  try {
    const v = await fn();
    ui.needToken = false;
    return v;
  } catch (e) {
    if (e instanceof AuthError) ui.needToken = true;
    else if (e instanceof RateLimited) { ui.needToken = true; ui.lockedFor = e.retryAfter; }
    else throw e;
  }
}

/** Who this session is (`/api/auth`); a 401 raises the login screen. */
export async function refreshWho() {
  try {
    const a = await guarded(() => api('/api/auth'));
    if (a) { ui.who = a.who; ui.tokenRequired = a.required; }
  } catch (e) {}
}

export async function refreshStatus() {
  // signed out: nothing to ask until a token is entered
  if (ui.needToken && !getToken()) return;
  try { ui.status = await guarded(() => api('/api/status')); } catch (e) {}
}

let timer = null;
export function startRefresh() {
  clearInterval(timer);
  const s = REFRESH.find((r) => r.id === ui.refresh)?.s ?? 0;
  if (s) timer = setInterval(() => { ui.tick++; }, s * 1000);
}
export function setRefresh(id) {
  ui.refresh = id;
  save('saber-refresh', id);
  startRefresh();
}
export function setRange(id) {
  ui.range = id;
  save('saber-range', id);
  ui.tick++;
}
export function applyTheme(m) {
  ui.mode = m;
  save('saber-theme', m);
  if (m === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = m;
  ui.themeKey++;
}

/** The metrics ring columns (/api/metrics) over the live range. */
export async function fetchCols(seconds, maxPoints = 1500) {
  return guarded(() => api(`/api/metrics?seconds=${seconds}&max_points=${maxPoints}`));
}
/** Per-family series (/api/series). */
export async function fetchSeries(names, seconds, maxPoints = 600) {
  return guarded(() => api(`/api/series?names=${encodeURIComponent(names.join(','))}&seconds=${seconds}&max_points=${maxPoints}`));
}
