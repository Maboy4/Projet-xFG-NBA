/**
 * court.js — Terrain NBA + Shot Chart + Heatmap SVG.
 * Coordonnées dataset : LOC_X ∈ [-250, 250], LOC_Y ∈ [-50, 420]
 * Mapping SVG        : x = LOC_X + 250,  y = LOC_Y + 50   (viewBox 0 0 500 470)
 */

// LOC_X en pieds [-25, +25], axe X inversé dans ce dataset
// LOC_Y en pieds depuis le panier [0, ~42]
export function toSVG(locX, locY) {
  return {
    x: -locX * 10 + 250,
    y:  locY * 10 + 41.75,
  };
}

// ── Dessin du terrain ─────────────────────────────────────────────────────────
export function drawCourt(svgEl) {
  const ns  = 'http://www.w3.org/2000/svg';
  const bx  = 250, by = 41.75;  // panier

  function el(tag, attrs, parent) {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    (parent ?? svgEl).appendChild(e);
    return e;
  }

  const line  = 'rgba(255,255,255,0.25)';
  const acct  = '#e8863a';
  const base  = { stroke: line, fill: 'none', 'stroke-width': 1.5 };

  // ── Filtre blur pour la heatmap (déclaré ici, utilisé plus tard) ────────────
  const defs = el('defs', {});
  const flt  = el('filter', { id: 'heat-blur', x: '-40%', y: '-40%', width: '180%', height: '180%' }, defs);
  el('feGaussianBlur', { stdDeviation: '14', in: 'SourceGraphic' }, flt);

  // Fond
  el('rect', { x: 0, y: 0, width: 500, height: 470, fill: '#0f1117' });

  // Couche heatmap (vide au chargement)
  el('g', { id: 'heat-layer', filter: 'url(#heat-blur)' });

  // ── Lignes terrain ──────────────────────────────────────────────────────────
  el('line', { x1: 0,   y1: 0,   x2: 500, y2: 0,   ...base });  // ligne de fond
  el('line', { x1: 0,   y1: 0,   x2: 0,   y2: 420, ...base });  // côté gauche
  el('line', { x1: 500, y1: 0,   x2: 500, y2: 420, ...base });  // côté droit
  el('line', { x1: 30,  y1: 0,   x2: 30,  y2: 92,  ...base });  // corner 3 gauche
  el('line', { x1: 470, y1: 0,   x2: 470, y2: 92,  ...base });  // corner 3 droit
  el('path', { d: `M 30 92 A 237.5 237.5 0 0 1 470 92`,         ...base });  // arc 3 pts
  el('rect', { x: 170, y: 0, width: 160, height: 190,           ...base });  // raquette
  el('path', { d: `M 210 ${by} A 40 40 0 0 1 290 ${by}`,        ...base });  // zone restrictive
  el('line', { x1: 170, y1: 190, x2: 330, y2: 190,              ...base });  // ligne LF
  el('path', { d: `M 170 190 A 80 80 0 0 1 330 190`,            ...base });  // demi-cercle LF (plein)
  el('path', { d: `M 170 190 A 80 80 0 0 0 330 190`, stroke: line, fill: 'none',
               'stroke-width': 1.5, 'stroke-dasharray': '6 4' });             // demi-cercle LF (pointillé)
  el('circle', { cx: bx, cy: by, r: 7.5, stroke: acct, fill: 'none', 'stroke-width': 1.5 });
  el('line',   { x1: 220, y1: 30, x2: 280, y2: 30, stroke: acct, fill: 'none', 'stroke-width': 1.5 });
  el('circle', { cx: 250, cy: 420, r: 60,           ...base });  // cercle médiane
  el('line',   { x1: 0,   y1: 420, x2: 500, y2: 420, ...base }); // ligne médiane

  // Couche shot dots (vide au chargement, dessinée PAR-DESSUS les lignes)
  el('g', { id: 'shot-layer' });
}

// ── Heatmap : cercles floutés pour la densité de tirs ─────────────────────────
export function renderHeatmap(svgEl, shots) {
  const layer = svgEl.querySelector('#heat-layer');
  if (!layer) return;
  layer.innerHTML = '';

  const ns = 'http://www.w3.org/2000/svg';
  for (const shot of shots) {
    const { x: cx, y: cy } = toSVG(shot.LOC_X, shot.LOC_Y);
    if (cx < -60 || cx > 560 || cy < -60 || cy > 530) continue;

    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', cx);
    c.setAttribute('cy', cy);
    c.setAttribute('r',  '22');
    c.setAttribute('fill', '#f97316');
    c.setAttribute('fill-opacity', '0.07');
    layer.appendChild(c);
  }
}

// ── Shot chart : points individuels ──────────────────────────────────────────
export function renderShots(svgEl, shots) {
  const layer = svgEl.querySelector('#shot-layer');
  if (!layer) return;
  layer.innerHTML = '';

  const ns = 'http://www.w3.org/2000/svg';
  for (const shot of shots) {
    const { x: cx, y: cy } = toSVG(shot.LOC_X, shot.LOC_Y);
    if (cx < 0 || cx > 500 || cy < 0 || cy > 470) continue;

    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', cx);
    c.setAttribute('cy', cy);
    c.setAttribute('r',  shot.SHOT_MADE_FLAG ? '2.8' : '2.4');
    c.setAttribute('fill',         shot.SHOT_MADE_FLAG ? '#4ade80' : '#f87171');
    c.setAttribute('fill-opacity', shot.SHOT_MADE_FLAG ? '0.8'    : '0.5');
    layer.appendChild(c);
  }
}

export function clearChart(svgEl) {
  const hl = svgEl.querySelector('#heat-layer');
  const sl = svgEl.querySelector('#shot-layer');
  if (hl) hl.innerHTML = '';
  if (sl) sl.innerHTML = '';
}
