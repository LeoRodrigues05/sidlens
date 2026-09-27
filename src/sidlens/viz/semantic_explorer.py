"""Render a portable, offline explorer for aligned item/SID catalogs."""

from __future__ import annotations

import json


def render_explorer(data: dict) -> str:
    """Return standalone HTML; catalog strings are never interpreted as markup.

    ``items`` and each variant's ``codes`` must use the same row order. The
    returned document makes no network requests and can be opened with file://.
    """
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Escape HTML parsing delimiters, including user-provided </script>, even
    # though the data script uses application/json rather than executable JS.
    for character, replacement in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        payload = payload.replace(character, replacement)
    return _HTML.replace("__SIDLENS_PAYLOAD__", payload, 1)


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIDLens · Semantic ID explorer</title>
<style>
:root{color-scheme:light;--ink:#172a35;--muted:#526674;--line:#d9e3e8;--paper:#fff;--wash:#f3f7f8;--accent:#006f72;--light:#e7f4f2;--amber:#805712}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{background:#14333d;color:white;padding:30px max(24px,calc((100vw - 1420px)/2));border-bottom:5px solid #3eb0a3}header p{color:#c8dee4;margin:8px 0 0}h1{font-size:28px;line-height:1.2;margin:5px 0 10px;letter-spacing:-.7px}h2{font-size:18px;margin:0 0 12px}h3{font-size:14px;margin:0 0 8px}.eyebrow{font-size:12px;letter-spacing:1.8px;text-transform:uppercase;font-weight:700}
main{max-width:1468px;margin:auto;padding:24px}section,.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px}.notice{border-left:4px solid #d3a349;color:var(--amber);background:#fffaf0;font-size:13px;padding:12px 16px;margin:0 0 18px}.muted{color:var(--muted)}.small{font-size:12px}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.controls{display:grid;grid-template-columns:minmax(220px,1fr) minmax(220px,2fr);gap:16px}label{display:block;font-size:12px;font-weight:700;color:var(--muted);margin-bottom:5px}select,input,button{font:inherit}select,input{border:1px solid #bacbd3;background:white;border-radius:7px;padding:9px 11px;color:var(--ink);max-width:100%;min-width:0}input{width:100%}button{cursor:pointer;border:1px solid #bacbd3;border-radius:7px;background:white;color:var(--ink);padding:8px 12px}button:hover{border-color:var(--accent);background:var(--light)}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #55b7b1;outline-offset:2px}button.primary{background:var(--accent);color:white;border-color:var(--accent)}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.between{justify-content:space-between}.grow{flex:1}.stack{margin-top:16px}.crumbs{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.crumbs button{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px}.crumbs .current{background:var(--light);color:var(--accent);border-color:#94c9c3}.chip{display:inline-flex;gap:7px;align-items:center;border-radius:18px;padding:4px 10px;background:var(--light);color:var(--accent);font-size:12px}.chip button{padding:0 2px;border:0;background:transparent}.examples{display:flex;gap:8px;flex-wrap:wrap}.examples button{font-size:12px;text-align:left}.summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.metric{font-size:30px;line-height:1.15;font-weight:700;letter-spacing:-.8px}.metric-label{font-size:12px;color:var(--muted);margin-top:5px}.insight-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:22px}.rank{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 12px;margin:9px 0;font-size:13px}.rank-bar{height:4px;border-radius:5px;background:var(--light);grid-column:1/-1}.rank-bar span{display:block;background:#54a69a;height:100%;border-radius:5px}.word-list{display:flex;gap:7px;flex-wrap:wrap}.word-list span{padding:4px 8px;background:var(--wash);border-radius:6px;font-size:12px}.branches{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}.branch{display:flex;flex-direction:column;align-items:flex-start;gap:5px;text-align:left;padding:12px}.branch .top{display:flex;width:100%;justify-content:space-between;gap:8px}.branch .category{font-size:12px;color:var(--muted)}.branch .example{font-size:11px;color:var(--muted);max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;background:var(--wash);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}td,th{border-bottom:1px solid var(--line);padding:11px 10px;vertical-align:top}td.sid{white-space:nowrap}td.title{min-width:230px;max-width:460px}td.category{min-width:170px}.table-code{border:0;background:var(--light);color:var(--accent);font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;padding:5px 7px;white-space:nowrap}.pagination{margin-top:16px}.empty{padding:25px;color:var(--muted);text-align:center}.hidden{display:none!important}details summary{cursor:pointer;font-size:14px;font-weight:600}details p{font-size:13px;color:var(--muted)}footer{max-width:1420px;margin:0 auto 28px;padding:0 24px;color:var(--muted);font-size:12px}.error{padding:24px;color:#9a2a1f}.ellipsis{overflow-wrap:anywhere}
@media(max-width:760px){main{padding:12px}section,.card{padding:15px}header{padding:22px 18px}.controls,.insight-grid{grid-template-columns:1fr}.metric{font-size:23px}.summary-grid{gap:10px}.branches{grid-template-columns:1fr 1fr}.branch{min-width:0}.branch .top{flex-wrap:wrap}h1{font-size:24px}}
</style>
</head>
<body>
<header><div class="eyebrow">SIDLens / Catalog inspection</div><h1>What does each semantic ID mean?</h1><p id="subtitle">Explore items that share code positions, prefixes, and complete IDs.</p></header>
<main id="app">
<div class="notice"><strong>Descriptive associations only.</strong> Shared categories or title words show what co-occurs in this catalog; they do not establish what a code means or what a trained recommender uses. A complete SID can map to multiple items. Every matching item is retained below.</div>
<section aria-label="Catalog filters">
<div class="controls"><div><label for="variant">Codebook / tokenizer</label><select id="variant"></select></div><div><label for="search">Search within selection · title, ASIN, item ID, or brand</label><input id="search" type="search" placeholder="Search items…" autocomplete="off"></div></div>
<div class="stack"><label>Selected prefix</label><nav id="breadcrumbs" class="crumbs" aria-label="SID prefix"></nav></div>
<div class="row stack"><div><label for="next-digit">Append next digit</label><select id="next-digit"></select></div><button id="reset" style="align-self:flex-end">Reset filters</button></div>
<details class="stack"><summary>Filter any code position</summary><p>Useful for comparing MQ positions or inspecting a digit independently of earlier digits. Position numbers start at 1. These conditions are combined with the selected prefix.</p><div class="row"><div><label for="position">Position</label><select id="position"></select></div><div><label for="position-value">Code value</label><select id="position-value"></select></div><button id="add-position" style="align-self:flex-end">Add condition</button></div><div id="conditions" class="row stack"></div></details>
<div id="example-wrap" class="stack hidden"><label>Suggested selections</label><div id="examples" class="examples"></div></div>
</section>
<section aria-label="Selection summary" aria-live="polite">
<div class="summary-grid"><div><div id="item-count" class="metric">—</div><div class="metric-label">matching items</div></div><div><div id="sid-count" class="metric">—</div><div class="metric-label">distinct complete SIDs</div></div><div><div id="collision-count" class="metric">—</div><div class="metric-label">items sharing a complete SID in this selection</div></div></div>
<p id="selection-note" class="small muted"></p><div class="insight-grid"><div><h3 id="category-heading">Shared category labels</h3><div id="categories"></div></div><div><h3>Common title words</h3><div id="words" class="word-list"></div><p class="small muted">Counts are numbers of item titles containing each word; common function words and numeric tokens are excluded.</p></div></div>
</section>
<section><div class="row between"><h2 id="branch-heading">Next digit groups</h2><span id="branch-note" class="small muted"></span></div><div id="branches" class="branches"></div><button id="more-branches" class="stack hidden">Show all groups</button></section>
<section><div class="row between"><h2>Items and complete IDs</h2><span class="row"><span id="download-status" class="small muted" role="status"></span><button id="download" class="primary">Download matching items · CSV</button></span></div><p id="table-note" class="small muted">Select an ID to inspect every item with that complete SID.</p><div class="scroll"><table><thead><tr><th>Complete SID</th><th>Item</th><th>ASIN / item ID</th><th>Brand</th><th>Category path</th></tr></thead><tbody id="items"></tbody></table></div><div class="row between pagination"><span id="page-status" class="small muted"></span><div class="row"><label for="page-size" style="margin:0">Rows</label><select id="page-size"><option>25</option><option selected>100</option><option>250</option></select><button id="previous">Previous</button><button id="next">Next</button></div></div></section>
<section><details><summary>Whole-codebook prefix statistics</summary><p>These statistics are computed over the whole catalog and do not change with the filters above. Pair agreement measures whether two distinct items sharing a prefix have the same category label. The null baseline comes from shuffled labels with the prefix groups held fixed. Singleton groups contain no item pairs. A higher excess is an association with the supplied category labels, not proof of model behavior.</p><div class="scroll"><table><thead><tr><th>Prefix depth</th><th>Groups</th><th>Items in singleton groups</th><th>Labelled pairs</th><th>Category agreement</th><th>Null agreement</th><th>Excess</th></tr></thead><tbody id="statistics"></tbody></table></div></details></section>
</main><footer id="footer">Self-contained catalog explorer. All filtering and CSV export happen in your browser.</footer>
<script id="catalog-data" type="application/json">__SIDLENS_PAYLOAD__</script>
<script>
'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  let data;
  try { data = JSON.parse($('catalog-data').textContent); }
  catch (error) { $('app').textContent = 'Could not read the embedded catalog: ' + error.message; return; }
  const items = Array.isArray(data.items) ? data.items : [];
  const variants = data.variants || {};
  const names = Object.keys(variants);
  if (!names.length) { $('app').textContent = 'No codebook variants are available in this catalog.'; return; }
  const fmt = new Intl.NumberFormat('en-US');
  const state = { variant: names[0], prefix: [], conditions: {}, query: '', page: 0, pageSize: 100, allBranches: false };
  let matched = [];
  const searchable = items.map((item) => [item.title, item.asin, item.item_id, item.brand].filter((x) => x != null).join(' ').toLocaleLowerCase());
  const stopwords = new Set(('a an and are as at be by can do for from has have in into is it its of on or our the their these this to up with you your new set pack pcs inch inches cm mm oz ml size piece pieces each all not no').split(' '));
  const text = (value) => value == null ? '' : String(value);
  const sidText = (code) => code.map((value) => '(' + value + ')').join('');
  const category = (item) => text(item.cat_l3 || item.cat_l2 || item.cat_l1 || 'Unlabelled');
  const categoryPath = (item) => [item.cat_l1,item.cat_l2,item.cat_l3].filter(Boolean).map(text).join(' › ') || 'Unlabelled';
  const codes = () => variants[state.variant].codes || [];
  const depth = () => {
    const first = codes().find((code) => Array.isArray(code) && code.length);
    return first ? first.length : 0;
  };
  function element(tag, value, className) {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = text(value);
    if (className) node.className = className;
    return node;
  }
  function option(value, label) { const node = element('option', label); node.value = value; return node; }
  function button(label, action, className) { const node = element('button', label, className); node.type = 'button'; node.addEventListener('click', action); return node; }
  function ratio(value, signed = false) {
    return Number.isFinite(value) ? (signed && value > 0 ? '+' : '') + (value * 100).toFixed(1) + '%' : '—';
  }
  function numeric(value) { return Number.isFinite(value) ? fmt.format(value) : '—'; }
  function accepts(index, excludedPosition = null) {
    const code = codes()[index];
    if (!Array.isArray(code) || !code.length) return false;
    if (!state.prefix.every((value, position) => code[position] === value)) return false;
    for (const [position, value] of Object.entries(state.conditions)) {
      if (Number(position) !== excludedPosition && code[Number(position)] !== value) return false;
    }
    return !state.query || searchable[index].includes(state.query.toLocaleLowerCase());
  }
  function updateHash() {
    const params = new URLSearchParams();
    params.set('variant', state.variant);
    if (state.prefix.length) params.set('prefix', state.prefix.join(','));
    if (Object.keys(state.conditions).length) params.set('positions', Object.entries(state.conditions).map(([key,value]) => key + ':' + value).join(','));
    if (state.query) params.set('q', state.query);
    try { history.replaceState(null, '', '#' + params.toString()); } catch (_) { /* Filtering also works in restricted viewers. */ }
  }
  function readHash() {
    const params = new URLSearchParams(location.hash.slice(1));
    if (Object.hasOwn(variants, params.get('variant'))) state.variant = params.get('variant');
    const values = params.get('prefix');
    const prefix = values ? values.split(',').map(Number) : [];
    state.prefix = prefix.every(Number.isInteger) ? prefix.slice(0, depth()) : [];
    state.conditions = {};
    for (const entry of (params.get('positions') || '').split(',')) {
      const parts = entry.split(':');
      if (parts.length !== 2) continue;
      const [position, value] = parts.map(Number);
      if (Number.isInteger(position) && position >= 0 && position < depth() && Number.isInteger(value)) state.conditions[position] = value;
    }
    state.query = params.get('q') || '';
    state.page = 0;
    state.allBranches = false;
  }
  function resetSelection() { state.prefix = []; state.conditions = {}; state.query = ''; state.page = 0; state.allBranches = false; }
  function selectPrefix(prefix) { state.prefix = prefix.slice(0, depth()); state.page = 0; state.allBranches = false; render(); }
  function selectComplete(code) {
    state.conditions = {};
    state.query = '';
    selectPrefix(code);
    $('breadcrumbs').scrollIntoView({behavior:'smooth', block:'center'});
  }
  function renderNavigation() {
    $('variant').value = state.variant;
    $('search').value = state.query;
    const crumbs = $('breadcrumbs'); crumbs.replaceChildren();
    crumbs.append(button('All items', () => selectPrefix([]), state.prefix.length ? '' : 'current'));
    state.prefix.forEach((value,index) => {
      crumbs.append(element('span', '›', 'muted'));
      crumbs.append(button('(' + value + ')', () => selectPrefix(state.prefix.slice(0,index+1)), index === state.prefix.length-1 ? 'current' : ''));
    });
    const next = $('next-digit'); next.replaceChildren();
    const nextCounts = new Map();
    if (state.prefix.length < depth()) {
      for (const index of matched) {
        const value = codes()[index][state.prefix.length];
        nextCounts.set(value, (nextCounts.get(value) || 0) + 1);
      }
      next.append(option('', 'Choose digit ' + (state.prefix.length + 1) + '…'));
      for (const [value,count] of [...nextCounts].sort((a,b) => a[0]-b[0])) next.append(option(value, '(' + value + ') · ' + fmt.format(count) + ' items'));
    } else { next.append(option('', 'Complete SID selected')); }
    next.disabled = state.prefix.length >= depth() || !nextCounts.size;
    const position = $('position'); const previous = position.value;
    position.replaceChildren();
    for (let index = 0; index < depth(); index++) position.append(option(index, 'Digit ' + (index+1)));
    if ([...position.options].some((entry) => entry.value === previous)) position.value = previous;
    renderPositionValues();
    const conditions = $('conditions'); conditions.replaceChildren();
    for (const [position,value] of Object.entries(state.conditions)) {
      const chip = element('span', 'Digit ' + (Number(position)+1) + ' = (' + value + ')', 'chip');
      const remove = button('×', () => { delete state.conditions[position]; state.page = 0; render(); });
      remove.setAttribute('aria-label', 'Remove condition on digit ' + (Number(position)+1));
      chip.append(remove); conditions.append(chip);
    }
  }
  function renderPositionValues() {
    const position = Number($('position').value);
    const values = new Map();
    for (let index = 0; index < items.length; index++) if (accepts(index, position)) {
      const value = codes()[index][position];
      if (value !== undefined) values.set(value, (values.get(value)||0)+1);
    }
    const select = $('position-value'); select.replaceChildren();
    select.append(option('', 'Choose code…'));
    for (const [value,count] of [...values].sort((a,b) => a[0]-b[0])) select.append(option(value, '(' + value + ') · ' + fmt.format(count) + ' items'));
    if (Object.hasOwn(state.conditions, position)) select.value = state.conditions[position];
    $('add-position').disabled = !values.size;
  }
  function renderSummary() {
    const sids = new Map(), categories = new Map(), words = new Map();
    for (const index of matched) {
      const item = items[index]; const key = codes()[index].join(',');
      sids.set(key, (sids.get(key)||0)+1);
      const label = category(item); categories.set(label, (categories.get(label)||0)+1);
      const tokens = new Set(text(item.title).toLocaleLowerCase().match(/[\p{L}][\p{L}\p{N}'-]*/gu) || []);
      for (const token of tokens) if (token.length >= 3 && !stopwords.has(token)) words.set(token,(words.get(token)||0)+1);
    }
    let collisionItems = 0, collisionGroups = 0;
    for (const count of sids.values()) if (count > 1) { collisionItems += count; collisionGroups++; }
    $('item-count').textContent = fmt.format(matched.length);
    $('sid-count').textContent = fmt.format(sids.size);
    $('collision-count').textContent = fmt.format(collisionItems);
    const invalid = items.length - codes().filter((code) => Array.isArray(code) && code.length).length;
    $('selection-note').textContent = 'Showing ' + fmt.format(matched.length) + ' of ' + fmt.format(items.length) + ' catalog items. ' +
      (collisionGroups ? fmt.format(collisionGroups) + ' complete SIDs currently have more than one matching item. ' : '') +
      (invalid > 0 ? fmt.format(invalid) + ' items have no SID for this variant. ' : '') +
      'Category labels use the deepest available level (L3, then L2, then L1). Search and position conditions affect all selection counts.';
    const categoryBox = $('categories'); categoryBox.replaceChildren();
    const sorted = [...categories].sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0]));
    for (const [label,count] of sorted.slice(0,8)) {
      const rank = element('div', undefined, 'rank');
      rank.append(element('span', label, 'ellipsis'),element('span',fmt.format(count) + ' · ' + ratio(count/matched.length),'muted'));
      const bar = element('div',undefined,'rank-bar'); const fill = element('span'); fill.style.width = 100*count/matched.length + '%'; bar.append(fill); rank.append(bar); categoryBox.append(rank);
    }
    if (!sorted.length) categoryBox.append(element('p','No items match these filters.','muted small'));
    else if (sorted.length > 8) categoryBox.append(element('p','Showing the 8 most frequent of ' + fmt.format(sorted.length) + ' labels.','muted small'));
    const wordBox = $('words'); wordBox.replaceChildren();
    for (const [word,count] of [...words].sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0])).slice(0,24)) wordBox.append(element('span',word + ' · ' + fmt.format(count)));
    if (!words.size) wordBox.append(element('span','No title words available.'));
  }
  function renderBranches() {
    const branchBox = $('branches'); branchBox.replaceChildren();
    const position = state.prefix.length;
    const more = $('more-branches'); more.classList.add('hidden');
    if (position >= depth()) {
      $('branch-heading').textContent = 'Complete SID'; $('branch-note').textContent = '';
      branchBox.append(element('p', 'No further digits. The table below lists every matching item for this complete SID.', 'muted small')); return;
    }
    const groups = new Map();
    for (const index of matched) {
      const value = codes()[index][position];
      if (!groups.has(value)) groups.set(value,{count:0, categories:new Map(), example:items[index].title});
      const group = groups.get(value); group.count++;
      const label = category(items[index]); group.categories.set(label,(group.categories.get(label)||0)+1);
    }
    $('branch-heading').textContent = 'Digit ' + (position+1) + ' · next groups';
    $('branch-note').textContent = fmt.format(groups.size) + ' observed values · ordered by code';
    const ordered = [...groups].sort((a,b) => a[0]-b[0]);
    for (const [value,group] of ordered.slice(0,state.allBranches ? ordered.length : 24)) {
      const entry = button('', () => selectPrefix([...state.prefix,value]), 'branch');
      const top = element('span',undefined,'top'); top.append(element('strong','(' + value + ')','mono'),element('span',fmt.format(group.count) + ' items','small')); entry.append(top);
      const [label,count] = [...group.categories].sort((a,b) => b[1]-a[1])[0];
      entry.append(element('span',label + ' · ' + ratio(count/group.count),'category ellipsis'));
      entry.append(element('span',group.example || 'Untitled item','example'));
      branchBox.append(entry);
    }
    if (!ordered.length) branchBox.append(element('p','No next-digit groups match these filters.','muted small'));
    if (ordered.length > 24 && !state.allBranches) { more.classList.remove('hidden'); more.textContent = 'Show all ' + fmt.format(ordered.length) + ' groups'; }
  }
  function renderTable() {
    const table = $('items'); table.replaceChildren();
    const pages = Math.max(1,Math.ceil(matched.length/state.pageSize)); state.page = Math.min(state.page,pages-1);
    const start = state.page*state.pageSize;
    for (const index of matched.slice(start,start+state.pageSize)) {
      const item = items[index], row = element('tr');
      const sid = element('td',undefined,'sid'); const sidButton = button(sidText(codes()[index]), () => selectComplete(codes()[index]), 'table-code');
      sidButton.title = 'Inspect all catalog items with this complete SID'; sid.append(sidButton); row.append(sid);
      row.append(element('td',item.title || 'Untitled item','title ellipsis'));
      const identity = element('td',undefined,'mono small'); identity.append(element('div',item.asin || '—'),element('div',item.item_id == null ? '' : 'ID ' + item.item_id,'muted')); row.append(identity);
      row.append(element('td',item.brand || '—'),element('td',categoryPath(item),'category'));
      table.append(row);
    }
    if (!matched.length) { const row = element('tr'), cell = element('td','No matching items. Clear a filter or return to All items.','empty'); cell.colSpan=5; row.append(cell); table.append(row); }
    $('page-status').textContent = (matched.length ? fmt.format(start+1) + '–' + fmt.format(Math.min(start+state.pageSize,matched.length)) : '0') + ' of ' + fmt.format(matched.length) + ' items · page ' + (state.page+1) + ' / ' + pages;
    $('previous').disabled = state.page === 0; $('next').disabled = state.page >= pages-1;
    $('download').disabled = !matched.length;
  }
  function renderStatistics() {
    const table = $('statistics'); table.replaceChildren();
    for (const stats of variants[state.variant].summary || []) {
      const row = element('tr');
      for (const value of [numeric(stats.depth),numeric(stats.n_groups),ratio(stats.singleton_item_share),numeric(stats.n_labelled_pairs),ratio(stats.pair_category_agreement),ratio(stats.null_pair_mean),ratio(stats.pair_excess,true)]) row.append(element('td',value));
      table.append(row);
    }
    if (!table.childElementCount) { const row = element('tr'), cell = element('td','No whole-codebook statistics supplied.','muted'); cell.colSpan = 7; row.append(cell); table.append(row); }
  }
  function renderExamples() {
    const examples = $('examples'); examples.replaceChildren();
    for (const example of data.examples || []) {
      if (example.variant !== state.variant || !Array.isArray(example.prefix)) continue;
      examples.append(button((example.label || sidText(example.prefix)), () => { state.conditions={}; state.query=''; selectPrefix(example.prefix); }));
    }
    $('example-wrap').classList.toggle('hidden',!examples.childElementCount);
  }
  function render() {
    matched = [];
    for (let index = 0; index < items.length; index++) if (accepts(index)) matched.push(index);
    renderNavigation(); renderSummary(); renderBranches(); renderTable(); renderStatistics(); renderExamples(); updateHash();
  }
  function csvCell(value) {
    // Quoting preserves embedded commas/newlines/quotes. Prefix formula-like
    // text for safe spreadsheet opening; numeric SID digits remain numeric.
    let valueText = text(value);
    if (/^[\s\u0000-\u001f]*[=+@-]/.test(valueText)) valueText = "'" + valueText;
    return '"' + valueText.replace(/"/g,'""') + '"';
  }
  // Hosted on claude.ai the page runs in a sandboxed frame where an <a download>
  // click silently does nothing; saves must go through the viewer's downloads
  // capability. Everywhere else (file://, any static host) window.claude is
  // absent or use() resolves null, and the plain blob link is the right path.
  const saver = window.claude && typeof window.claude.use === 'function'
    ? window.claude.use('downloads').catch(() => null) : Promise.resolve(null);
  function linkDownload(blob, filename) {
    const url = URL.createObjectURL(blob), link = element('a');
    link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  async function downloadCsv() {
    const headers = ['variant','sid',...Array.from({length:depth()},(_,index) => 'code' + (index+1)),'asin','item_id','title','brand','cat_l1','cat_l2','cat_l3'];
    const lines = [headers.map(csvCell).join(',')];
    for (const index of matched) {
      const item = items[index], code = codes()[index];
      lines.push([state.variant,sidText(code),...code,item.asin,item.item_id,item.title,item.brand,item.cat_l1,item.cat_l2,item.cat_l3].map(csvCell).join(','));
    }
    const blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'],{type:'text/csv;charset=utf-8'});
    const filename = 'sidlens_' + state.variant.replace(/[^a-zA-Z0-9_-]/g,'_') + '_' + (state.prefix.length ? state.prefix.join('-') : 'all') + '.csv';
    const status = $('download-status'); status.textContent = '';
    const downloads = await saver;
    if (!downloads) { linkDownload(blob, filename); return; }
    try { await downloads.save({filename, data: blob}); }
    catch (error) {
      // A declined prompt is the viewer's choice, not a failure worth a message.
      const code = error && error.code;
      if (code === 'declined') return;
      status.textContent = code === 'rate_limited' ? 'A save prompt is already open. Try again in a moment.'
        : 'Download is unavailable in this view. Open the file directly to export.';
    }
  }
  names.forEach((name) => $('variant').append(option(name,name)));
  $('subtitle').textContent = (data.category ? text(data.category) + ' · ' : '') + fmt.format(items.length) + ' catalog items · ' + names.length + ' codebook variants. Follow a prefix to inspect its items.';
  if (data.generated_at) $('footer').append(document.createTextNode(' Generated ' + text(data.generated_at) + '.'));
  $('variant').addEventListener('change',() => { state.variant=$('variant').value; resetSelection(); render(); });
  $('next-digit').addEventListener('change',() => { if ($('next-digit').value !== '') selectPrefix([...state.prefix,Number($('next-digit').value)]); });
  $('position').addEventListener('change',renderPositionValues);
  $('add-position').addEventListener('click',() => { if ($('position-value').value !== '') { state.conditions[Number($('position').value)]=Number($('position-value').value); state.page=0; state.allBranches=false; render(); } });
  $('reset').addEventListener('click',() => { resetSelection(); render(); });
  $('more-branches').addEventListener('click',() => { state.allBranches=true; renderBranches(); });
  $('search').addEventListener('input',() => { state.query=$('search').value; state.page=0; state.allBranches=false; render(); });
  $('page-size').addEventListener('change',() => { state.pageSize=Number($('page-size').value); state.page=0; renderTable(); });
  $('previous').addEventListener('click',() => { state.page--; renderTable(); });
  $('next').addEventListener('click',() => { state.page++; renderTable(); });
  $('download').addEventListener('click',downloadCsv);
  window.addEventListener('hashchange',() => { readHash(); render(); });
  readHash(); render();
})();
</script>
</body>
</html>
"""
