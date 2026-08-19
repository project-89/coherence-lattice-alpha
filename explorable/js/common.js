/* Shared utilities for the Coherence Lattice explorable. */

// --- Theme system ------------------------------------------------------
//
// `palette` is a live getter-object: every property read re-evaluates the
// current theme from the DOM (the CSS variables under :root[data-theme]).
// Canvas code calls `palette.bg`, `palette.text`, etc. on every draw; when
// the theme flips, the next frame paints with the new colours automatically.
//
// Exports:
//   palette          — live colour object (drop-in for hex strings)
//   getTheme()       — returns 'light' | 'dark'
//   setTheme(theme)  — persist + apply
//   installThemeToggle() — injects a floating button (auto-runs on import)

const _cssVars = {
  bg:              '--canvas-bg',
  bgSoft:          '--canvas-bg-soft',
  text:            '--canvas-text',
  textSoft:        '--canvas-text-soft',
  textFaint:       '--canvas-text-faint',
  grid:            '--canvas-grid',
  axis:            '--canvas-axis',
  axisBold:        '--canvas-axis-bold',
  blue:            '--c-blue',
  green:           '--c-green',
  orange:          '--c-orange',
  burgundy:        '--c-burgundy',
  purple:          '--c-purple',
  accent:          '--accent',
  alive:           '--alive',
  vortex:          '--vortex',
};

// Derived composite colours. Not mapped to CSS vars individually — computed
// from the current theme so they track the toggle.
function _derived(key) {
  if (key === 'bondDefault') {
    // Mid-grey line that reads on both cream and dark backgrounds.
    return getTheme() === 'dark'
      ? 'rgba(180, 175, 160, 0.55)'
      : 'rgba(60, 60, 60, 0.5)';
  }
  if (key === 'bondFaint') {
    return getTheme() === 'dark'
      ? 'rgba(120, 115, 100, 0.4)'
      : 'rgba(30, 30, 30, 0.45)';
  }
  return null;
}

// Light-theme fallback values (used only before DOM is available, e.g. at
// module-init time on a headless test).
const _fallback = {
  bg: '#f5f0e4', bgSoft: '#fdfaf3',
  text: '#2a2a2a', textSoft: '#555', textFaint: '#888',
  grid: '#e0dbcb', axis: '#c8c0ad', axisBold: '#c0b9a0',
  blue: '#2a5f8f', green: '#2d7d4f', orange: '#d97236',
  burgundy: '#7d2d4f', purple: '#a855f7',
  accent: '#2a5f8f', alive: '#d97236', vortex: '#7d2d4f',
};

function _cssVar(name, fallback) {
  if (typeof document === 'undefined') return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (v && v.trim()) || fallback;
}

export const palette = new Proxy({}, {
  get(_target, key) {
    const cssName = _cssVars[key];
    if (cssName) return _cssVar(cssName, _fallback[key] || '#888');
    const derived = _derived(key);
    if (derived !== null) return derived;
    return _fallback[key] || '#888';
  },
  has(_target, key) {
    return key in _cssVars || key in _fallback || _derived(key) !== null;
  },
});

export function getTheme() {
  if (typeof document === 'undefined') return 'dark';
  const explicit = document.documentElement.getAttribute('data-theme');
  if (explicit) return explicit;
  return 'dark'; // dark is the default when nothing is set
}

export function setTheme(theme) {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('coh-theme', theme); } catch (_) {}
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

function _loadPersistedTheme() {
  if (typeof document === 'undefined') return;
  try {
    const saved = localStorage.getItem('coh-theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.setAttribute('data-theme', saved);
    }
  } catch (_) {}
}

export function installThemeToggle() {
  if (typeof document === 'undefined') return;
  _loadPersistedTheme();
  if (document.querySelector('.theme-toggle')) return;  // already installed

  const btn = document.createElement('button');
  btn.className = 'theme-toggle';
  btn.type = 'button';
  btn.setAttribute('aria-label', 'Toggle dark mode');
  btn.title = 'Toggle dark mode';
  function refreshIcon() {
    btn.textContent = getTheme() === 'dark' ? '\u263C' : '\u263E'; // ☼ / ☾
  }
  refreshIcon();
  btn.addEventListener('click', () => {
    setTheme(getTheme() === 'dark' ? 'light' : 'dark');
    refreshIcon();
  });
  // Install on DOM-ready so <body> exists.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => document.body.appendChild(btn));
  } else {
    document.body.appendChild(btn);
  }
}

// Auto-install so every page that imports anything from common.js gets the toggle.
installThemeToggle();


// --- Modified Bessel functions I_0 and I_1 (series + asymptotic) ---

export function bessel_I0(x) {
  const ax = Math.abs(x);
  if (ax < 3.75) {
    const t = (x / 3.75) ** 2;
    return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 +
           t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
  }
  const t = 3.75 / ax;
  const poly = 0.39894228 + t * (0.01328592 + t * (0.00225319 +
               t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 +
               t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
  return (Math.exp(ax) / Math.sqrt(ax)) * poly;
}

export function bessel_I1(x) {
  const ax = Math.abs(x);
  if (ax < 3.75) {
    const t = (x / 3.75) ** 2;
    const poly = 0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 +
                 t * (0.02658733 + t * (0.00301532 + t * 0.00032411)))));
    return x * poly;
  }
  const t = 3.75 / ax;
  let poly = 0.39894228 + t * (-0.03988024 + t * (-0.00362018 +
             t * (0.00163801 + t * (-0.01031555 + t * (0.02282967 +
             t * (-0.02895312 + t * (0.01787654 + t * (-0.00420059))))))));
  poly = (Math.exp(ax) / Math.sqrt(ax)) * poly;
  return x < 0 ? -poly : poly;
}

// R_0(K) = I_1(K) / I_0(K) — the von Mises order parameter
export function R0(K) {
  if (K < 1e-10) return K / 2;
  return bessel_I1(K) / bessel_I0(K);
}

// --- Core constants ---

export const K_BKT = 2 / Math.PI;
export const K_BULK = 16 / (Math.PI * Math.PI);

// --- Alpha formula (self-consistent, 20 iterations) ---

export function alpha_BKT(z = 4, K = K_BKT) {
  const V = R0(K) ** z;
  const base = Math.PI / z;
  let a = V;
  for (let i = 0; i < 20; i++) {
    const n = 1 / Math.sqrt(Math.E) + a / (2 * Math.PI);
    a = V * Math.pow(base, n);
  }
  return a;
}

// --- Tunable inline values (draggable numbers) ---

export function makeTunable(el, { min, max, step, value, format, onChange }) {
  let v = value ?? parseFloat(el.textContent);
  const fmt = format ?? ((x) => x.toFixed(2));
  el.textContent = fmt(v);
  el.classList.add('tune');

  let dragging = false;
  let startX = 0;
  let startV = 0;

  const range = max - min;
  const pixelsPerUnit = 200 / range;

  function start(e) {
    dragging = true;
    el.classList.add('active');
    startX = (e.touches ? e.touches[0].clientX : e.clientX);
    startV = v;
    e.preventDefault();
  }
  function move(e) {
    if (!dragging) return;
    const x = (e.touches ? e.touches[0].clientX : e.clientX);
    const dx = x - startX;
    let nv = startV + dx / pixelsPerUnit;
    nv = Math.round(nv / step) * step;
    nv = Math.max(min, Math.min(max, nv));
    if (nv !== v) {
      v = nv;
      el.textContent = fmt(v);
      if (onChange) onChange(v);
    }
    e.preventDefault();
  }
  function end() {
    dragging = false;
    el.classList.remove('active');
  }

  el.addEventListener('mousedown', start);
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', end);
  el.addEventListener('touchstart', start, { passive: false });
  window.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('touchend', end);

  return {
    get value() { return v; },
    set value(nv) {
      v = Math.max(min, Math.min(max, nv));
      el.textContent = fmt(v);
      if (onChange) onChange(v);
    },
  };
}

// --- Slider control (boxed, labeled) ---

export function makeSlider(container, { label, min, max, step, value, format, onChange }) {
  const fmt = format ?? ((x) => x.toFixed(2));
  const ctrl = document.createElement('div');
  ctrl.className = 'control';
  ctrl.innerHTML = `
    <label>${label} <span class="val">${fmt(value)}</span></label>
    <input type="range" min="${min}" max="${max}" step="${step}" value="${value}">
  `;
  container.appendChild(ctrl);
  const input = ctrl.querySelector('input');
  const val = ctrl.querySelector('.val');
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    val.textContent = fmt(v);
    if (onChange) onChange(v);
  });
  return {
    get value() { return parseFloat(input.value); },
    set value(v) {
      input.value = v;
      val.textContent = fmt(v);
    },
  };
}

// --- Canvas helpers ---

export function setupHiDPICanvas(canvas, w, h) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = w * ratio;
  canvas.height = h * ratio;
  // Responsive CSS size: never exceed logical width, shrink on smaller screens
  canvas.style.maxWidth = w + 'px';
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  canvas.style.aspectRatio = `${w} / ${h}`;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  return ctx;
}

// --- Coherence metrics for a 2D grid ---
//
// Given phases and grid size, compute I_phase_hat (alignment in [0,1]),
// ρ (structural richness, spatial std-dev of local alignment, normalized),
// and C (coherence capital, their product).

export function coherenceMetrics(thetas, L) {
  // Global alignment: mean neighbor cos, in [-1, 1]
  let s = 0, nb = 0;
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      const i = y * L + x;
      if (x + 1 < L) { s += Math.cos(thetas[y * L + (x + 1)] - thetas[i]); nb++; }
      if (y + 1 < L) { s += Math.cos(thetas[(y + 1) * L + x] - thetas[i]); nb++; }
    }
  }
  const Iphase = nb ? s / nb : 0;
  const Iphat = (1 + Iphase) / 2;

  // ρ — structural-richness proxy: circular diversity of the phase field,
  // 1 − |mean resultant|. Both factors of C must vanish at a dead end:
  // in noise Iphase ≈ 0 (no alignment), in trivial unison ρ = 0 (nothing
  // left to be in sync about). The product C peaks only at structured
  // synchrony — distinct locked regions at distinct phases — which is the
  // theory's definition of capital. (The old patch-variance ρ vanished in
  // noise too, so C mislabeled chaos and sagged to ~0 after full sync.)
  let cs = 0, sn = 0;
  const n = L * L;
  for (let i = 0; i < n; i++) { cs += Math.cos(thetas[i]); sn += Math.sin(thetas[i]); }
  const rho = 1 - Math.sqrt(cs * cs + sn * sn) / n;

  const Ipos = Math.max(0, Iphase);
  const C = Ipos * rho;
  return { Iphase, Iphat, Ipos, rho, C };
}

// --- Compact coherence-capital strip renderer ---
//
// Draws a small panel below a grid showing I_phase, ρ, C bars plus a
// live sparkline of C(t). Call on every draw() pass with the latest
// metrics and a rolling history array.

export function drawMetricsStrip(ctx, x, y, w, h, metrics, history) {
  const { Ipos, rho, C } = metrics;

  // Panel background
  ctx.fillStyle = '#fdfaf3';
  ctx.strokeStyle = '#c8c0ad';
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);

  // Three bars on left
  const barsW = 140;
  const barsX = x + 10;
  const barsY = y + 10;
  const barH = 10;
  const gap = 18;
  const entries = [
    ['I_phase', Ipos, 1, '#2a5f8f'],
    ['ρ',       rho,  1, '#7d2d4f'],
    ['C',       C,    0.5, '#d97236'],
  ];
  entries.forEach(([label, val, vmax, color], i) => {
    const by = barsY + i * gap;
    ctx.fillStyle = '#888';
    ctx.font = 'italic 11px serif';
    ctx.textAlign = 'right';
    ctx.fillText(label, barsX + 32, by + 8);
    ctx.fillStyle = '#fdfaf3';
    ctx.strokeStyle = '#d8d4c8';
    ctx.lineWidth = 0.5;
    ctx.fillRect(barsX + 38, by, barsW - 86, barH);
    ctx.strokeRect(barsX + 38, by, barsW - 86, barH);
    ctx.fillStyle = color;
    ctx.fillRect(barsX + 38, by, (val / vmax) * (barsW - 86), barH);
    ctx.fillStyle = color;
    ctx.font = '11px "SF Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(val.toFixed(3), barsX + barsW - 42, by + 9);
  });

  // Sparkline of C(t) on right
  const spX = x + barsW + 20;
  const spY = y + 10;
  const spW = w - (barsW + 40);
  const spH = h - 20;

  ctx.strokeStyle = '#e0dbcb';
  ctx.strokeRect(spX, spY, spW, spH);

  // Scale: C up to ~0.5
  const Cmax = 0.5;
  const zeroY = spY + spH;
  const topY = spY;

  ctx.strokeStyle = '#d97236';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < history.length; i++) {
    if (history[i] == null) continue;
    const sx = spX + (i / history.length) * spW;
    const sy = zeroY - Math.min(1, history[i] / Cmax) * (zeroY - topY);
    if (!started) { ctx.moveTo(sx, sy); started = true; }
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();

  ctx.fillStyle = '#888';
  ctx.font = 'italic 10px serif';
  ctx.textAlign = 'left';
  ctx.fillText('C(t)', spX + 4, spY + 12);
}

// --- Visibility-gated animation loop ---
//
// Runs `fn` on every animation frame, but pauses when the canvas is scrolled
// offscreen (via IntersectionObserver). This eliminates the scroll jank from
// many simultaneous simulations.
//
// Design notes:
//   - Starts with visible = true so the first frame always renders.
//   - Runs fn() once synchronously before the RAF loop, so figures never
//     appear blank even if the observer fires late or the browser has a
//     hiccup initializing the intersection state.
//   - IntersectionObserver then takes over; offscreen figures pause.
//
// Returns { stop() } to cancel explicitly.

export function runWhenVisible(canvas, fn) {
  let visible = true;
  let rafId = null;

  if (typeof IntersectionObserver !== 'undefined') {
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) visible = e.isIntersecting;
    }, { threshold: 0, rootMargin: '120px' });
    obs.observe(canvas);
  }

  // First draw happens immediately so the figure is never blank.
  try { fn(); } catch (err) { console.error('[runWhenVisible] initial draw failed:', err); }

  function loop() {
    if (visible) {
      try { fn(); } catch (err) { console.error('[runWhenVisible] frame failed:', err); }
    }
    rafId = requestAnimationFrame(loop);
  }
  loop();

  return {
    stop() { if (rafId) cancelAnimationFrame(rafId); },
  };
}

// --- Color helpers (HSL lerp for smooth transitions) ---

export function lerpColor(a, b, t) {
  const pa = a.match(/\d+/g).map(Number);
  const pb = b.match(/\d+/g).map(Number);
  return `rgb(${Math.round(pa[0] + (pb[0] - pa[0]) * t)},` +
         `${Math.round(pa[1] + (pb[1] - pa[1]) * t)},` +
         `${Math.round(pa[2] + (pb[2] - pa[2]) * t)})`;
}

// Map K from [0, K_bulk] to a dead→alive color
export function KColor(K, Kmax = K_BULK) {
  const t = Math.min(1, Math.max(0, K / Kmax));
  // Dead gray → alive orange
  return lerpColor('rgb(168, 163, 154)', 'rgb(217, 114, 54)', t);
}

// --- Interactive equation wiring ---
//
// HTML pattern:
//   <div class="equation-block" id="eq-foo">
//     <div class="eq-display">
//       <span class="eq-symbol state" data-symbol="theta">θ</span> = ...
//     </div>
//     <p class="eq-prompt">Click any colored symbol to see what it means.</p>
//     <div class="eq-detail"></div>
//     <div class="eq-howitworks">
//       <span class="label">In words</span> Prose explanation.
//     </div>
//   </div>
//
// Call: wireEquation('eq-foo', { theta: {name, pronounce, description}, ... })

export function wireEquation(blockId, symbolData) {
  const container = document.getElementById(blockId);
  if (!container) return;
  const detail = container.querySelector('.eq-detail');
  const prompt = container.querySelector('.eq-prompt');
  const symbols = container.querySelectorAll('.eq-symbol');

  symbols.forEach((s) => {
    s.addEventListener('click', () => {
      const key = s.dataset.symbol;
      const info = symbolData[key];
      if (!info) return;

      const wasActive = s.classList.contains('active');
      symbols.forEach((x) => x.classList.remove('active'));

      if (wasActive) {
        detail.classList.remove('visible');
        detail.innerHTML = '';
        if (prompt) prompt.style.display = '';
        return;
      }

      s.classList.add('active');
      detail.innerHTML = `
        <div class="name">${info.name}</div>
        ${info.pronounce ? `<div class="pronounce">pronounced <em>${info.pronounce}</em></div>` : ''}
        <div class="description">${info.description}</div>
      `;
      detail.classList.add('visible');
      if (prompt) prompt.style.display = 'none';
    });
  });
}

// --- WebAudio sonification ---
//
// The subject of these essays is coupled oscillators, so sonification can
// be literal: a voice per oscillator, pitch from its instantaneous
// frequency, and what you hear (beats, a detuned cluster pulling into one
// pitch) is the real signal mix. Where a figure uses a mapping instead
// (e.g. loudness = agreement), its caption must say so.
//
// One shared AudioContext (created on first user gesture — browsers block
// audio before that), one low master gain; each figure gets its own
// toggle + sub-gain so sound never stacks across figures.

let _actx = null;
let _masterGain = null;

export function audioContext() {
  if (!_actx) {
    _actx = new (window.AudioContext || window.webkitAudioContext)();
    _masterGain = _actx.createGain();
    _masterGain.gain.value = 0.14;
    _masterGain.connect(_actx.destination);
  }
  if (_actx.state === 'suspended') _actx.resume();
  return _actx;
}

// A per-figure sound toggle. Appends a button to `container`, off by
// default. Returns { on, gain, onChange(fn) }; `gain` exists after the
// first enable.
export function makeSoundToggle(container, { label = 'sound' } = {}) {
  const btn = document.createElement('button');
  btn.className = 'sound-toggle';
  btn.textContent = '\u{1F507} ' + label;
  btn.setAttribute('aria-pressed', 'false');
  const state = { on: false, gain: null, _subs: [] };
  btn.addEventListener('click', () => {
    state.on = !state.on;
    btn.textContent = (state.on ? '\u{1F50A} ' : '\u{1F507} ') + label;
    btn.setAttribute('aria-pressed', String(state.on));
    const ctx = audioContext();
    if (!state.gain) {
      state.gain = ctx.createGain();
      state.gain.gain.value = 0;
      state.gain.connect(_masterGain);
    }
    state.gain.gain.setTargetAtTime(state.on ? 1 : 0, ctx.currentTime, 0.06);
    state._subs.forEach(f => f(state.on));
  });
  container.appendChild(btn);
  state.onChange = f => state._subs.push(f);
  return state;
}

// n sine voices under a sound toggle. Built lazily on first enable (the
// AudioContext needs a user gesture to exist). `.set(i, freq, amp)` is
// safe to call every animation frame, sound on or off; amp is per-voice
// in [0,1] and is divided by n so a full chorus sums to unit level.
export function makeVoices(sound, n, { freq = 220 } = {}) {
  let voices = null;
  function build() {
    const ctx = audioContext();
    voices = Array.from({ length: n }, () => {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      g.gain.value = 0;
      osc.connect(g);
      g.connect(sound.gain);
      osc.start();
      return { osc, g };
    });
  }
  sound.onChange(on => { if (on && !voices) build(); });
  return {
    set(i, f, a) {
      if (!voices || !sound.on) return;
      const t = _actx.currentTime;
      if (f > 0) voices[i].osc.frequency.setTargetAtTime(f, t, 0.04);
      voices[i].g.gain.setTargetAtTime(Math.max(0, a) / n, t, 0.04);
    },
  };
}
