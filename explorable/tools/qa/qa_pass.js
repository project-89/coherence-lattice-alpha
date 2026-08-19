// Headless visual-QA harness for The Living Lattice explorable.
// For each page: load, collect console errors/warnings + pageerrors,
// slow-scroll to trigger IntersectionObservers, then screenshot each
// canvas figure block (after letting its animation run) plus key
// non-canvas interactives. Emits qa/<slug>/fig-N.png + qa/report.json.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const BASE = 'http://localhost:8137';
const OUT = path.join(__dirname, 'qa');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const PAGES = [
  'foundations/sections/01-oscillators.html',
  'foundations/sections/02-capital.html',
  'foundations/sections/03-clr.html',
  'foundations/sections/04-coherence-theorem.html',
  'foundations/sections/05-memory-basins.html',
  'paradigm/sections/05-the-wall.html',
  'paradigm/sections/06-the-edge.html',
  'paradigm/sections/07-flux.html',
  'paradigm/sections/08-shape.html',
  'paradigm/sections/09-strange-loop.html',
  'paradigm/sections/12-witness.html',
  'paradigm/sections/13-scale-ladder.html',
  'paradigm/sections/14-one-constant.html',
  'paradigm/sections/15-rotation.html',
  'paradigm/sections/16-brains-language.html',
  'paradigm/sections/17-coda.html',
  'transformers/sections/01-coupled-oscillators.html',
  'transformers/sections/02-dead-heads.html',
  'transformers/sections/03-manifold.html',
  'transformers/sections/04-lossless.html',
  'transformers/sections/05-fiber-bundle.html',
  'primers/sections/00-phase-and-sound.html',
  'primers/sections/01-rotations.html',
  'primers/sections/06-von-mises.html',
  'primers/sections/07-rotation-groups.html',
  'primers/sound-workshop/sections/01-harmonograph.html',
  'primers/sound-workshop/sections/02-chladni.html',
  'primers/sound-workshop/sections/03-why-the-pattern.html',
  'primers/sound-workshop/sections/04-chladni-lab.html',
  'primers/topology-manifolds/sections/01-topology.html',
  'primers/topology-manifolds/sections/02-manifolds.html',
  'primers/topology-manifolds/sections/03-counting-loops.html',
  'primers/inside-a-transformer/sections/01-tokens.html',
  'primers/inside-a-transformer/sections/02-attention.html',
  'primers/inside-a-transformer/sections/03-residual-stream.html',
  'primers/inside-a-transformer/sections/04-sphere.html',
  'primers/gauge-holonomy/sections/01-parallel-transport.html',
  'primers/gauge-holonomy/sections/02-gauge.html',
  'primers/gauge-holonomy/sections/03-holonomy.html',
  'primers/graphs-spectra/sections/01-modes.html',
  'primers/graphs-spectra/sections/02-laplacian.html',
  'primers/graphs-spectra/sections/03-fiedler.html',
  'primers/lohe-hypersphere/sections/01-sphere-dynamics.html',
  'primers/lohe-hypersphere/sections/02-effective-coupling.html',
  'primers/lohe-hypersphere/sections/03-tokens-as-paths.html',
  'primers/living-k/sections/01-quenched.html',
  'primers/living-k/sections/02-the-rule.html',
  'primers/living-k/sections/03-repair.html',
  'primers/bkt-transition/sections/01-vortices.html',
  'primers/bkt-transition/sections/02-the-argument.html',
  'primers/bkt-transition/sections/03-the-wall.html',
  'primers/coarse-graining/sections/01-squinting.html',
  'primers/coarse-graining/sections/02-flows.html',
  'primers/coarse-graining/sections/03-universality.html',
];

const slug = (p) => p.replace(/\.html$/, '').replace(/\//g, '_');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const colorScheme = process.argv[2] === 'dark' ? 'dark' : 'light';
  const only = process.argv[3]; // optional substring filter
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--window-size=1280,1000', '--force-device-scale-factor=1'],
  });
  const report = [];
  for (const rel of PAGES) {
    if (only && !rel.includes(only)) continue;
    const s = slug(rel) + (colorScheme === 'dark' ? '_dark' : '');
    const dir = path.join(OUT, s);
    fs.mkdirSync(dir, { recursive: true });
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 950 });
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: colorScheme }]);
    const logs = [];
    page.on('console', (m) => {
      const t = m.type();
      if (t === 'error' || t === 'warning') logs.push(`[${t}] ${m.text()}`);
    });
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
    page.on('requestfailed', (r) => logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`));
    try {
      await page.goto(`${BASE}/${rel}`, { waitUntil: 'networkidle0', timeout: 30000 });
    } catch (e) {
      logs.push(`[goto-fail] ${e.message}`);
    }
    await sleep(1200);
    // slow scroll to bottom to trigger every IntersectionObserver, then back up
    const height = await page.evaluate(() => document.body.scrollHeight);
    for (let y = 0; y < height; y += 500) {
      await page.evaluate((yy) => window.scrollTo(0, yy), y);
      await sleep(180);
    }
    await page.evaluate(() => window.scrollTo(0, 0));
    await sleep(400);

    // collect figure targets: each canvas's figure-ish container, deduped, plus
    // known non-canvas interactives.
    const targets = await page.evaluate(() => {
      const seen = new Set();
      const boxes = [];
      const pick = (el, label) => {
        if (!el || seen.has(el)) return;
        seen.add(el);
        el.setAttribute('data-qa-idx', String(boxes.length));
        boxes.push({ label });
      };
      document.querySelectorAll('canvas').forEach((c, i) => {
        const cont =
          c.closest('figure, .figure, .fig, .demo, .interactive, .sim, .panel-wrap, .viz') ||
          c.parentElement;
        pick(cont, `canvas-${i}`);
      });
      document
        .querySelectorAll('.ladder-wrap, .emergent, .cobweb-wrap, .criteria-grid, .table-wrap')
        .forEach((el, i) => pick(el, `div-${i}`));
      return boxes.map((b) => b.label);
    });

    let shot = 0;
    for (let i = 0; i < targets.length; i++) {
      const sel = `[data-qa-idx="${i}"]`;
      const el = await page.$(sel);
      if (!el) continue;
      await page.evaluate(
        (s) => document.querySelector(s).scrollIntoView({ block: 'center' }),
        sel
      );
      await sleep(2500); // let the sim run / settle
      try {
        await el.screenshot({ path: path.join(dir, `fig-${String(shot).padStart(2, '0')}-${targets[i]}.png`) });
        shot++;
      } catch (e) {
        logs.push(`[shot-fail ${targets[i]}] ${e.message}`);
      }
    }
    if (shot === 0) {
      await page.screenshot({ path: path.join(dir, 'fullpage.png'), fullPage: true });
    }
    report.push({ page: rel, scheme: colorScheme, figures: shot, logs });
    console.log(`${rel} [${colorScheme}]: ${shot} figs, ${logs.length} log lines`);
    await page.close();
  }
  fs.writeFileSync(path.join(OUT, `report_${colorScheme}.json`), JSON.stringify(report, null, 2));
  await browser.close();
})();
