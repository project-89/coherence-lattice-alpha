// Interaction QA: drive buttons/sliders on each page, sample readouts,
// screenshot the affected figure. Steps per page defined below.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const BASE = 'http://localhost:8137';
const OUT = path.join(__dirname, 'qa_interact');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// step kinds:
//  {click: 'Button text'} — click button whose textContent includes it
//  {slider: cssSelector, value: v} — set range input + dispatch input
//  {check: 'label', wait: ms} — wait, then record all .readout texts + canvas fig shot
//  {shot: 'label', sel: css} — screenshot element containing sel (or its figure)
const SUITES = [
  {
    page: 'foundations/sections/02-capital.html',
    steps: [
      { check: 'default-panels', wait: 4000 },
      { slider: '#k-slider, input[type=range]', value: 5, index: 0 },
      { check: 'K-high', wait: 5000 },
      { slider: 'input[type=range]', value: 0.05, index: 0 },
      { check: 'K-low', wait: 5000 },
    ],
  },
  {
    page: 'foundations/sections/03-clr.html',
    steps: [
      { click: 'Warm start' },
      { check: 'warm-start', wait: 6000 },
    ],
  },
  {
    page: 'foundations/sections/04-coherence-theorem.html',
    steps: [
      { check: 't0', wait: 1000 },
      { check: 't8s', wait: 8000 },
      { check: 't20s', wait: 12000 },
    ],
  },
  {
    page: 'foundations/sections/05-memory-basins.html',
    steps: [
      { click: 'Scramble' },
      { check: 'post-scramble', wait: 5000 },
      { click: 'Partial cue' },
      { check: 'post-partial-cue', wait: 5000 },
    ],
  },
  {
    page: 'alpha/sections/06-plm-npd.html',
    steps: [{ check: 'strip-render', wait: 5000 }],
  },
  {
    page: 'paradigm/sections/08-shape.html',
    steps: [
      { click: 'Apply another perspective' },
      { click: 'Apply another perspective' },
      { check: 'after-2-perspectives', wait: 3000 },
      { click: 'Deep' },
      { check: 'deep', wait: 3000 },
    ],
  },
  {
    page: 'paradigm/sections/09-strange-loop.html',
    steps: [
      { click: 'Sphere' },
      { check: 'sphere', wait: 3000 },
      { click: 'Torus' },
      { check: 'torus', wait: 3000 },
    ],
  },
  {
    page: 'paradigm/sections/12-witness.html',
    steps: [
      { click: 'Perturb the self-model' },
      { check: 'post-perturb', wait: 4000 },
    ],
  },
  {
    page: 'paradigm/sections/16-brains-language.html',
    steps: [
      { click: 'Align (CRT)' },
      { check: 'aligned', wait: 5000 },
      { click: 'Compare: tangent' },
      { check: 'toggled', wait: 3000 },
    ],
  },
  {
    page: 'transformers/sections/03-manifold.html',
    steps: [
      { slider: 'input[type=range]', value: 1, index: 0 },
      { check: 'sync-1', wait: 4000 },
    ],
  },
];

(async () => {
  const only = process.argv[2];
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--window-size=1280,1000'],
  });
  const report = [];
  for (const suite of SUITES) {
    if (only && !suite.page.includes(only)) continue;
    const slug = suite.page.replace(/\.html$/, '').replace(/\//g, '_');
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 950 });
    const logs = [];
    page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
    await page.goto(`${BASE}/${suite.page}`, { waitUntil: 'networkidle0', timeout: 30000 });
    await sleep(800);
    // scroll through page to boot all observers
    const height = await page.evaluate(() => document.body.scrollHeight);
    for (let y = 0; y < height; y += 500) { await page.evaluate((yy) => window.scrollTo(0, yy), y); await sleep(120); }

    const results = [];
    let shot = 0;
    for (const st of suite.steps) {
      if (st.click) {
        const ok = await page.evaluate((txt) => {
          const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.trim().includes(txt));
          if (!btn) return false;
          btn.scrollIntoView({ block: 'center' });
          btn.click();
          return true;
        }, st.click);
        results.push({ click: st.click, ok });
        await sleep(400);
      } else if (st.slider) {
        const ok = await page.evaluate((sel, v, idx) => {
          const els = [...document.querySelectorAll(sel)].filter((e) => e.type === 'range');
          const el = els[idx || 0];
          if (!el) return false;
          el.scrollIntoView({ block: 'center' });
          el.value = v;
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
          return true;
        }, st.slider, st.value, st.index);
        results.push({ slider: st.slider, value: st.value, ok });
        await sleep(400);
      } else if (st.check) {
        await sleep(st.wait || 2000);
        const readouts = await page.evaluate(() =>
          [...document.querySelectorAll('.readout, [id*="readout"], [id*="counter"]')].map((r) => r.textContent.trim().replace(/\s+/g, ' '))
        );
        // screenshot every canvas figure container
        const nCan = await page.evaluate(() => {
          document.querySelectorAll('canvas').forEach((c, i) => {
            const cont = c.closest('figure, .figure') || c.parentElement;
            cont.setAttribute('data-qa2', String(i));
          });
          return document.querySelectorAll('canvas').length;
        });
        for (let i = 0; i < nCan; i++) {
          const el = await page.$(`[data-qa2="${i}"]`);
          if (!el) continue;
          await page.evaluate((s) => document.querySelector(s).scrollIntoView({ block: 'center' }), `[data-qa2="${i}"]`);
          await sleep(600);
          try { await el.screenshot({ path: path.join(OUT, `${slug}__${String(shot).padStart(2, '0')}-${st.check}-c${i}.png`) }); } catch (e) {}
        }
        if (nCan === 0) await page.screenshot({ path: path.join(OUT, `${slug}__${String(shot).padStart(2, '0')}-${st.check}.png`), fullPage: true });
        shot++;
        results.push({ check: st.check, readouts });
      }
    }
    report.push({ page: suite.page, results, logs });
    console.log(`${suite.page}: done, ${logs.length} log lines`);
    await page.close();
  }
  fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close();
})();
