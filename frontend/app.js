import { drawCourt, renderHeatmap, renderShots, clearChart } from './court.js';

const API = 'http://127.0.0.1:8080';

// ── Cache ─────────────────────────────────────────────────────────────────────
let allPlayers = [];
let sortAsc    = false;

// ── DOM ───────────────────────────────────────────────────────────────────────
const svgEl        = document.getElementById('court');
const searchInput  = document.getElementById('player-search');
const acList       = document.getElementById('autocomplete-list');
const statsBox     = document.getElementById('player-stats');
const statName     = document.getElementById('stat-name');
const statFGM      = document.getElementById('stat-fgm');
const statFGA      = document.getElementById('stat-fga');
const statFG       = document.getElementById('stat-fg');
const statXFG      = document.getElementById('stat-xfg');
const statPOE      = document.getElementById('stat-poe');
const heroDiv      = document.getElementById('player-hero');
const heroName     = document.getElementById('hero-name');
const heroPOE      = document.getElementById('hero-poe');
const heroFG       = document.getElementById('hero-fg');
const heroXFG      = document.getElementById('hero-xfg');
const heroFGA      = document.getElementById('hero-fga');
const courtHint    = document.getElementById('court-hint');
const lbList       = document.getElementById('leaderboard-list');
const sortDescBtn  = document.getElementById('sort-desc');
const sortAscBtn   = document.getElementById('sort-asc');

// ── Init ──────────────────────────────────────────────────────────────────────
drawCourt(svgEl);

async function init() {
  const res  = await fetch(`${API}/leaderboard?limit=500`);
  allPlayers = await res.json();
  renderLeaderboard();
}

// ── Leaderboard ───────────────────────────────────────────────────────────────
function renderLeaderboard() {
  const players = sortAsc ? [...allPlayers].reverse() : allPlayers.slice(0, 20);
  lbList.innerHTML = '';

  players.forEach((p, i) => {
    const sign  = p.DIFF >= 0 ? '+' : '';
    const cls   = p.DIFF >= 0 ? 'lb-poe-pos' : 'lb-poe-neg';
    const row   = document.createElement('div');
    row.className = 'lb-row';
    row.innerHTML = `
      <span class="lb-rank">${i + 1}</span>
      <span class="lb-name">${p.PLAYER_NAME}</span>
      <span class="lb-poe ${cls}">${sign}${p.DIFF}%</span>
    `;
    row.addEventListener('click', () => selectPlayer(p));
    lbList.appendChild(row);
  });
}

sortDescBtn.addEventListener('click', () => {
  sortAsc = false;
  sortDescBtn.classList.add('active');
  sortAscBtn.classList.remove('active');
  renderLeaderboard();
});
sortAscBtn.addEventListener('click', () => {
  sortAsc = true;
  sortAscBtn.classList.add('active');
  sortDescBtn.classList.remove('active');
  renderLeaderboard();
});

// ── Sélection joueur ──────────────────────────────────────────────────────────
async function selectPlayer(player) {
  // Carte stats sidebar
  statName.textContent = player.PLAYER_NAME;
  statFGM.textContent  = player.FGM;
  statFGA.textContent  = player.FGA;
  statFG.textContent   = `${player.FG_PCT}%`;
  statXFG.textContent  = `${player.xFG_PCT}%`;
  const sign = player.DIFF >= 0 ? '+' : '';
  statPOE.textContent = `${sign}${player.DIFF}%`;
  statPOE.className   = 'stat-value poe-value ' + (player.DIFF >= 0 ? 'poe-positive' : 'poe-negative');
  statsBox.classList.remove('hidden');

  // Hero header au-dessus du terrain
  heroName.textContent  = player.PLAYER_NAME;
  heroPOE.textContent   = `POE ${sign}${player.DIFF}%`;
  heroPOE.className     = 'pill ' + (player.DIFF >= 0 ? '' : 'pill-neg');
  heroFG.textContent    = `FG% ${player.FG_PCT}%`;
  heroXFG.textContent   = `xFG% ${player.xFG_PCT}%`;
  heroFGA.textContent   = `${player.FGA} tirs`;
  heroDiv.classList.remove('hidden');
  courtHint.classList.add('hidden');

  // Surbrillance leaderboard
  lbList.querySelectorAll('.lb-row').forEach(r => r.classList.remove('selected'));
  [...lbList.querySelectorAll('.lb-name')]
    .find(n => n.textContent === player.PLAYER_NAME)
    ?.closest('.lb-row')?.classList.add('selected');

  // Shot chart + heatmap
  try {
    const res  = await fetch(`${API}/player/${encodeURIComponent(player.PLAYER_NAME)}/shots`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderHeatmap(svgEl, data.shots);
    renderShots(svgEl, data.shots);
  } catch (err) {
    console.warn('Shot chart indisponible :', err.message);
    clearChart(svgEl);
  }
}

// ── Autocomplétion ────────────────────────────────────────────────────────────
let acIndex = -1;

searchInput.addEventListener('input', () => {
  const q = searchInput.value.trim().toLowerCase();
  acIndex = -1;

  if (q.length < 1) { acList.classList.add('hidden'); return; }

  const matches = allPlayers.filter(p => p.PLAYER_NAME.toLowerCase().includes(q)).slice(0, 8);
  if (!matches.length) { acList.classList.add('hidden'); return; }

  acList.innerHTML = '';
  for (const p of matches) {
    const li   = document.createElement('li');
    const sign = p.DIFF >= 0 ? '+' : '';
    li.textContent = `${p.PLAYER_NAME} (${sign}${p.DIFF}%)`;
    li.addEventListener('mousedown', e => {
      e.preventDefault();
      searchInput.value = p.PLAYER_NAME;
      acList.classList.add('hidden');
      selectPlayer(p);
    });
    acList.appendChild(li);
  }
  acList.classList.remove('hidden');
});

searchInput.addEventListener('keydown', e => {
  const items = [...acList.querySelectorAll('li')];
  if (!items.length) return;
  if (e.key === 'ArrowDown')  { e.preventDefault(); acIndex = Math.min(acIndex + 1, items.length - 1); }
  else if (e.key === 'ArrowUp')   { e.preventDefault(); acIndex = Math.max(acIndex - 1, -1); }
  else if (e.key === 'Enter')     { e.preventDefault(); if (acIndex >= 0) items[acIndex].dispatchEvent(new MouseEvent('mousedown')); return; }
  else if (e.key === 'Escape')    { acList.classList.add('hidden'); return; }
  items.forEach((li, i) => li.classList.toggle('active', i === acIndex));
  if (acIndex >= 0) items[acIndex].scrollIntoView({ block: 'nearest' });
});

document.addEventListener('click', e => {
  if (!e.target.closest('#search-container')) acList.classList.add('hidden');
});

init();
