/* Shared utilities for the Coherence Lattice explorable. */

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
  // Global alignment
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

  // Local alignment per 3×3 patch, then spatial std-dev → ρ
  const ps = 3;
  const locals = [];
  for (let py = 0; py < L; py += ps) {
    for (let px = 0; px < L; px += ps) {
      let ls = 0, lc = 0;
      for (let y = py; y < Math.min(py + ps, L); y++) {
        for (let x = px; x < Math.min(px + ps, L); x++) {
          const i = y * L + x;
          if (x + 1 < Math.min(px + ps, L)) {
            ls += Math.cos(thetas[y * L + (x + 1)] - thetas[i]);
            lc++;
          }
          if (y + 1 < Math.min(py + ps, L)) {
            ls += Math.cos(thetas[(y + 1) * L + x] - thetas[i]);
            lc++;
          }
        }
      }
      if (lc > 0) locals.push((1 + ls / lc) / 2);
    }
  }
  const mean = locals.reduce((a, b) => a + b, 0) / locals.length;
  const variance = locals.reduce((acc, x) => acc + (x - mean) ** 2, 0) / locals.length;
  const rho = Math.min(1, Math.sqrt(variance) / 0.35);

  const C = Iphat * rho;
  return { Iphat, rho, C };
}

// --- Compact coherence-capital strip renderer ---
//
// Draws a small panel below a grid showing I_phase, ρ, C bars plus a
// live sparkline of C(t). Call on every draw() pass with the latest
// metrics and a rolling history array.

export function drawMetricsStrip(ctx, x, y, w, h, metrics, history) {
  const { Iphat, rho, C } = metrics;

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
    ['I_phase', Iphat, 1, '#2a5f8f'],
    ['ρ',       rho,   1, '#7d2d4f'],
    ['C',       C,     0.5, '#d97236'],
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
