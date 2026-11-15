// Drag-reorder of dashboard panels. A Svelte action on a page container: every direct
// child with the `panel` class (in the page, and in each `.grid2` row inside it) gets a grip
// button; drag a grip onto another panel, or focus it and press the arrow keys, to move the
// panel. The order is CSS `order` only (Svelte's DOM is never moved), kept per page in this
// browser's localStorage; rows, titles and tiles keep their places, panels trade the slots
// panels hold. `resetLayout(key)` forgets a page's order.

const PREFIX = 'saber-layout:';

function loadOrder(key) {
  try { return JSON.parse(localStorage.getItem(PREFIX + key) || '[]'); } catch (e) { return []; }
}
function saveOrder(key, ids) {
  try { localStorage.setItem(PREFIX + key, JSON.stringify(ids)); } catch (e) {}
}

export function resetLayout(key) {
  try {
    for (const k of Object.keys(localStorage)) if (k === PREFIX + key || k.startsWith(PREFIX + key + '/')) localStorage.removeItem(k);
  } catch (e) {}
}

function slug(s) {
  return (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function idOf(el, i) {
  if (el.dataset.panel) return el.dataset.panel;
  const h = el.querySelector('h2, h3');
  return slug(h?.textContent) || slug(el.getAttribute('aria-label')) || `p${i}`;
}

/** One container's sorting (the page, or a row). */
function container(node, key) {
  let dragging = null;
  let order = loadOrder(key);
  const children = () => [...node.children].filter((c) => c.nodeType === 1);
  const movable = () => children().filter((c) => c.classList.contains('panel'));

  function apply() {
    const all = children();
    const mv = movable();
    const ids = mv.map((el, i) => idOf(el, i));
    // the slots panels hold, in DOM order; the saved order picks which panel takes each
    const slots = all.map((el, i) => (el.classList.contains('panel') ? i : -1)).filter((i) => i >= 0);
    const known = order.filter((id) => ids.includes(id));
    const rest = ids.filter((id) => !known.includes(id));
    const seq = [...known, ...rest];
    all.forEach((el, i) => { if (!el.classList.contains('panel')) el.style.order = String(i * 10); });
    seq.forEach((id, k) => {
      const el = mv[ids.indexOf(id)];
      if (el) el.style.order = String(slots[k] * 10);
    });
    mv.forEach((el, i) => grip(el, ids[i]));
    return seq;
  }

  function move(id, to) {
    const seq = apply();
    const from = seq.indexOf(id);
    if (from < 0) return;
    const t = Math.max(0, Math.min(seq.length - 1, to));
    seq.splice(from, 1);
    seq.splice(t, 0, id);
    order = seq;
    saveOrder(key, order);
    apply();
  }

  function grip(el, id) {
    el.dataset.panelId = id;
    if (el.querySelector(':scope > .grip')) return;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'grip';
    b.title = 'Drag to move this panel (or focus and use the arrow keys)';
    b.setAttribute('aria-label', 'Move panel');
    b.innerHTML = '<svg viewBox="0 0 10 16" width="8" height="13" aria-hidden="true"><g fill="currentColor"><circle cx="2" cy="2" r="1.4"/><circle cx="8" cy="2" r="1.4"/><circle cx="2" cy="8" r="1.4"/><circle cx="8" cy="8" r="1.4"/><circle cx="2" cy="14" r="1.4"/><circle cx="8" cy="14" r="1.4"/></g></svg>';
    b.addEventListener('mousedown', () => { el.draggable = true; });
    b.addEventListener('touchstart', () => { el.draggable = true; }, { passive: true });
    b.addEventListener('keydown', (e) => {
      const seq = apply();
      const at = seq.indexOf(el.dataset.panelId);
      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { e.preventDefault(); move(el.dataset.panelId, at - 1); b.focus(); }
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { e.preventDefault(); move(el.dataset.panelId, at + 1); b.focus(); }
    });
    el.addEventListener('dragstart', (e) => {
      if (!el.draggable) return;
      dragging = el.dataset.panelId;
      el.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      try { e.dataTransfer.setData('text/plain', dragging); } catch (err) {}
      e.stopPropagation();
    });
    el.addEventListener('dragend', () => {
      el.draggable = false;
      el.classList.remove('dragging');
      dragging = null;
      for (const c of movable()) c.classList.remove('drop-target');
    });
    el.addEventListener('dragover', (e) => {
      if (!dragging || dragging === el.dataset.panelId) return;
      e.preventDefault();
      el.classList.add('drop-target');
    });
    el.addEventListener('dragleave', () => el.classList.remove('drop-target'));
    el.addEventListener('drop', (e) => {
      if (!dragging) return;
      e.preventDefault();
      e.stopPropagation();
      el.classList.remove('drop-target');
      const seq = apply();
      move(dragging, seq.indexOf(el.dataset.panelId));
    });
    el.prepend(b);
  }

  node.classList.add('sortable');
  apply();
  // panels that appear later ({#if} blocks, data arriving) join the order
  const mo = new MutationObserver(() => apply());
  mo.observe(node, { childList: true });
  return () => mo.disconnect();
}

/** `use:sortable={'throughput'}` on a page's root. */
export function sortable(node, key) {
  const stops = [container(node, key)];
  const rows = new WeakSet();
  const scan = () => {
    node.querySelectorAll(':scope > .grid2').forEach((row, i) => {
      if (rows.has(row)) return;
      rows.add(row);
      stops.push(container(row, `${key}/row${i}`));
    });
  };
  scan();
  // rows inside {#if} blocks appear later
  const mo = new MutationObserver(scan);
  mo.observe(node, { childList: true });
  return { destroy() { mo.disconnect(); for (const s of stops) s(); } };
}
