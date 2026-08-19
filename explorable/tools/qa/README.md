# Headless visual-QA harness for The Living Lattice

Two scripts that QA the explorable without the Chrome extension, using
`puppeteer-core` driving the system Chrome. Written 2026-08-13 during the
full-site QA pass; they found the ρ-metric bug, the §16 rotation-invariance
bug, and the trivial-unison winding trap (see
`../../../general_theory_intelligence_consciousness/HANDOFF_2026-08-13_explorable_qa_session.md`).

Setup (any scratch dir):

```bash
npm init -y && npm install puppeteer-core
# serve the site first:
cd <explorable root> && python3 -m http.server 8137
```

- `qa_pass.js [light|dark] [page-substring]` — loads every page, collects
  console errors/warnings/pageerrors, slow-scrolls to trigger every
  IntersectionObserver (figures boot lazily via `runWhenVisible`), then
  screenshots each canvas figure block after letting its sim run ~2.5 s.
  Output: `qa/<slug>/fig-*.png` + `qa/report_<scheme>.json`.
- `qa_interact.js [page-substring]` — drives the per-page interaction
  suites defined at the top of the file (button clicks by text, range
  sliders by index), samples all `.readout` elements after each step, and
  screenshots each figure. Output: `qa_interact/*.png` + `report.json`
  (note: each run overwrites `report.json` — copy it aside between
  filtered runs).

Both assume the local server on :8137 and macOS Chrome at
`/Applications/Google Chrome.app/...`. Screenshots are then eyeballed
(the QA lesson stands: `node --check` and "serves 200" are NOT enough —
and neither is "0 console errors": every one of the three real bugs this
harness caught rendered cleanly and threw nothing).
