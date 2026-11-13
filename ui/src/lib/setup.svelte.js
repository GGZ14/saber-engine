// The setup wizard's and the model manager's shared state: the wizard state (/api/setup), the
// system check (/api/system), the catalog with fit + install state (/api/catalog) and the live
// download list (/api/downloads/stream, SSE).
import { api, stream } from './api.js';
import { guarded } from './store.svelte.js';

export const W = $state({
  setup: null,
  system: null,
  catalog: null,
  downloads: [],
  error: '',
  busy: false,
  loadedSystem: false,
});

export async function loadSetup() {
  try {
    const s = await guarded(() => api('/api/setup'));
    if (s) W.setup = s;
    return s;
  } catch (e) {
    W.error = e.message;
  }
}

export async function loadSystem() {
  try {
    const s = await guarded(() => api('/api/system'));
    if (s) W.system = s;
    W.loadedSystem = true;
  } catch (e) {
    W.error = e.message;
  }
}

export async function loadCatalog() {
  try {
    const c = await guarded(() => api('/api/catalog'));
    if (c) W.catalog = c;
  } catch (e) {
    W.error = e.message;
  }
}

let unsub = null;
let users = 0;
let lastDone = new Set();
/** Follow the download list (refcounted: the wizard and the model manager share one stream). */
export function followDownloads() {
  users++;
  if (!unsub) {
    unsub = stream('/api/downloads/stream', 'downloads', (jobs) => {
      W.downloads = jobs;
      // a finished download changes what is installed
      const done = new Set(jobs.filter((j) => j.state === 'done').map((j) => j.id));
      if ([...done].some((id) => !lastDone.has(id))) {
        loadCatalog();
        // the wizard's gates read what is installed
        if (W.setup) loadSetup();
      }
      lastDone = done;
    });
  }
  return () => {
    users--;
    if (users <= 0 && unsub) {
      unsub();
      unsub = null;
      users = 0;
    }
  };
}

/** A wizard call that answers the setup view; errors land in W.error and are returned. */
export async function setupCall(path, body = {}) {
  W.busy = true;
  try {
    const s = await guarded(() => api(path, { method: 'POST', body }));
    if (s) W.setup = s;
    W.error = '';
    return { ok: true, s };
  } catch (e) {
    W.error = e.message;
    return { ok: false, error: e.message };
  } finally {
    W.busy = false;
  }
}

export async function startDownload(model, component) {
  try {
    await guarded(() => api('/api/downloads', { method: 'POST', body: component ? { model, component } : { model } }));
    W.error = '';
  } catch (e) {
    W.error = e.message;
  }
}

export async function dlAction(id, action) {
  try {
    await guarded(() => api(`/api/downloads/${id}/${action}`, { method: 'POST', body: {} }));
    W.error = '';
  } catch (e) {
    W.error = e.message;
  }
}

/** The newest download per (model, component). */
export function jobFor(model, component) {
  for (let i = W.downloads.length - 1; i >= 0; i--) {
    const j = W.downloads[i];
    if (j.model === model && j.component === component) return j;
  }
  return null;
}

export const ACTIVE = ['queued', 'listing', 'downloading', 'verifying', 'installing', 'paused'];

/** Copy text; falls back to a hidden textarea where the clipboard API is unavailable (plain http
 *  on a LAN address is not a secure context). */
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    const t = document.createElement('textarea');
    t.value = text;
    t.setAttribute('readonly', '');
    t.style.position = 'fixed';
    t.style.opacity = '0';
    document.body.appendChild(t);
    t.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (err) {}
    t.remove();
    return ok;
  }
}

export function eta(s) {
  if (s == null || !Number.isFinite(s)) return '';
  if (s < 60) return `${Math.max(1, Math.round(s))} s left`;
  if (s < 3600) return `${Math.round(s / 60)} min left`;
  return `${(s / 3600).toFixed(1)} h left`;
}

export function rate(bps) {
  if (bps == null || !Number.isFinite(bps)) return '';
  if (bps >= 1e9) return `${(bps / 1e9).toFixed(2)} GB/s`;
  if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`;
  if (bps >= 1e3) return `${(bps / 1e3).toFixed(0)} kB/s`;
  return `${Math.round(bps)} B/s`;
}

export function gb(b) {
  if (b == null) return '–';
  if (b >= 1e9) return `${(b / 1e9).toFixed(b >= 1e11 ? 0 : 1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} kB`;
  return `${b} B`;
}
