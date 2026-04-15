# Handoff — Coherence Lattice Alpha Project

**Last context save**: 2026-04-15
**State at handoff**: Paper is preprint-ready and pushed to GitHub. Explorable explanation is 6/15 sections complete.

This is the document a fresh session should read first, in combination with the files listed in **§ Must-read files on pickup** below.

---

## § What this project is

The paper derives the fine structure constant **α = 1/137.036** from first principles — a single dynamical principle (the **Coherence Learning Rule**, CLR) on the diamond lattice, with zero free parameters. The core equation of the broader paradigm is

```
  I ≥ 0    where  I(t) := dC/dt = d/dt (I_phase · ρ)
```

— the intelligence flux is non-negative under the CLR (the **Coherence Theorem**). α falls out as the operating point the CLR selects when topology (vortices) is present.

The paper is at **papers/publication/coherence_lattice_alpha/paper.tex**, and has a standalone copy at **papers/standalone/coherence_lattice_alpha/** that's published to GitHub. An **explorable explanation** (Bret-Victor-style interactive essay) is in the `explorable/` subdirectory of each.

---

## § Where things live

### Two parallel copies — keep in sync

```
papers/publication/coherence_lattice_alpha/   ← SOURCE OF TRUTH. Edit here.
papers/standalone/coherence_lattice_alpha/    ← Git repo. Pushed to GitHub. Sync from publication.
```

The standalone copy is the git repo synced to GitHub. The publication copy is the "upstream" source of truth — edits happen there, then sync to standalone before committing.

**GitHub remote**: `git@github.com:project-89/coherence-lattice-alpha.git`
**Latest branch**: `main`

### File layout (in each copy)

```
├── paper.tex                 # main paper, 52-page LaTeX
├── paper.pdf                 # compiled
├── paper.bbl                 # bibliography (committed)
├── references.bib            # BibTeX source
├── AGENTS.md                 # agent onboarding for the paper/physics
├── HANDOFF.md                # THIS FILE
├── README.md                 # repo landing page (paradigm-first framing)
├── LICENSE                   # CC BY-NC 4.0 (paper) + AGPL-3.0 (code)
├── Makefile                  # `make` compiles the paper
├── scripts/                  # 27 Python verification scripts
├── figures/                  # 6 paper figures (PNG)
├── data/                     # JSON precomputed results
└── explorable/               # interactive essay (15 sections planned)
    ├── OUTLINE.md            # section-by-section plan & status
    ├── index.html            # landing page with TOC
    ├── css/style.css         # shared styling
    ├── js/common.js          # shared utilities
    └── sections/             # 01–15, six done so far
```

---

## § Must-read files on pickup

When starting a new session, read these in order:

1. **`AGENTS.md`** — paradigm, derivation chain, proven identities, first principles, all scripts with runtime, conventions, and key numbers. This is the physics spine. ✋ **This file has been manually edited outside of our sessions — treat it as authoritative.**

2. **`HANDOFF.md`** (this file) — project state, what's done, what's next, gotchas.

3. **`explorable/OUTLINE.md`** — 15-section plan for the explorable. Shows status (✅ done / TODO) and has detailed plans for remaining sections.

4. **`paper.tex`** (at least skim the abstract + Section 1 + the boxed `I ≥ 0` theorem in §1.2 + the Open Derivation paragraph in §5-9) — for context on what's in the paper.

5. **`explorable/js/common.js`** — the shared JS module. Essential patterns: `setupHiDPICanvas`, `wireEquation`, `runWhenVisible`, `coherenceMetrics`, `drawMetricsStrip`, `R0` (Bessel ratio). Read this before writing any new section.

6. **`explorable/sections/06-plm-npd.html`** — the most recent and feature-complete section. Use as template for future sections. Shows the guided-tour pattern, clickable equation blocks, drive-pattern × local-coherence coloring, bonds toggle.

7. **`explorable/sections/01-prelude.html`** — simplest section, shows the basic canvas + multi-figure structure.

8. **`references.bib`** — bibliography. Notable entry: `Sharpe2026Coherence` is the unpublished companion paper on the broader Coherence Theorem (`I ≥ 0`). Labeled "In preparation" — don't imply it exists publicly.

### Context-wide reference docs

- **`/Users/parzival/workspace/oneirocom/project89/wayfaring/CLAUDE.md`** — wayfaring/Project 89 codebase overview (high-level context)
- **`/Users/parzival/workspace/oneirocom/project89/wayfaring/06_intelligence/coherence_lattice/CLAUDE.md`** — coherence_lattice project entry point (points to AGENT.md, research registry protocol)
- **`papers/core/main.tex`** — the companion Coherence Theorem paper (unpublished, provides the broader substrate-agnostic framework from which the alpha derivation is one application)

---

## § What's been completed

### Paper (preprint-ready)

- **Framing refactor**: Main boxed result is `1/α_BKT = 137.032` at 29 ppm with zero free parameters (rigorous). LCE correction to 137.035999 at 1.5 ppb presented as plausibility argument. Convergence pattern + candidate enumeration given as evidence; the specific R₀² embedding weight is explicitly flagged as an open derivation.
- **`I ≥ 0` elevation**: Theorem 1.1 (Coherence Ascent) now defines `I(t) := dC/dt` and states `I(t) ≥ 0` boxed. Remark 1 cites the forthcoming companion paper while noting the derivation is self-contained.
- **Casimir mass spectrum removed**: Section 6.2 replaced with brief "lepton generations via hexagonal DOFs are covered in a follow-up paper" note.
- **Acknowledgments**: GPD (Get Physics Done) framework credited.
- **Licenses**: CC BY-NC 4.0 for paper/figures, AGPL-3.0 for scripts.
- **GitHub repo**: live at https://github.com/project-89/coherence-lattice-alpha. All changes committed and pushed.

### Explorable explanation (6/15 sections done)

See `explorable/OUTLINE.md` for the full plan. Completed:

| # | Title | Key interactives |
|---|-------|------------------|
| 01 | Prelude | One osc + waveform, two uncoupled, two coupled (K slider + frequency gap), single bond with CLR (K_eq ≈ 2.1) |
| 02 | Oscillators on a graph | Unit-circle math (draggable phases), Kuramoto equation (clickable symbols), firefly ring, firefly grid |
| 03 | Coherence capital | Three-regime comparison, main interactive with grid + C-contour phase portrait |
| 04 | The CLR | Derivation, clickable CLR equation, potential V(K), live bond with bistability, death-threshold bifurcation diagram, **Coherence Theorem climax with `I ≥ 0` boxed** |
| 05 | How a binary field emerges | Three bonds → small ring → full grid with K-histogram + I(t) + C(t) + phases/bonds/both toggle |
| 06 | PLMs and memory | PLM detection (connected components), guided 5-step memory tour, patterns playground with drive × local-coherence coloring, coherence-capital tracking, SVG NPD hierarchy |

Section 6's memory tour is the pedagogical highlight — it makes the "K-field = memory" claim visceral by imprinting a pattern, scrambling phases, and watching phases re-lock into the same pattern from the retained K-field.

---

## § What's next (sections 07–15)

From `explorable/OUTLINE.md` — detailed specs are in that file. Summary:

- **07 — Spontaneous vortices**: topology emerges from CLR dynamics. Vortex cores, winding numbers, Fiedler protection.
- **08 — Why the diamond lattice**: the 5-filter selection theorem. SC/BCC/FCC/HCP/diamond comparison, d-dimensional generalization.
- **09 — The BKT wall**: vortex marginality theorem visualized — the CLR wants K high, topology forces K ≤ K_BKT, equilibrium is at the boundary.
- **10 — Living vs static**: the 143 → 137 transition. Trajectory integration vs fixed-point evaluation. The PLM Freezing Lemma in action.
- **11 — The α formula, piece by piece**: `α = R₀(2/π)⁴ × (π/4)^(1/√e + α/2π)`. Interactive clickable formula with sensitivity sliders.
- **12 — Why three dimensions**: the d-dial showing only d=3 gives physical α.
- **13 — Closing the gap with linked clusters**: LCE convergence 28,800 → 6.7 → 1.5 ppb. Candidate enumeration for the R₀² open problem.
- **14 — From α to g**: QED series, 11.4 matching digits.
- **15 — Coda**: full chain visualized, paradigm recap, future companions, "I ≥ 0 — the universe climbing."

---

## § Gotchas / lessons learned

These bit us during construction — memorize them.

### HTML/JS

- **Bare `>` and `<` in prose break module script loading.** Always use `&gt;` and `&lt;` in prose text like "cos(Δθ) &gt; 4/r" or "K &gt; 0" — otherwise the HTML parser confuses itself, the `<script type="module">` tag isn't reached, and you see silent non-loading with no console errors. Took ~two hours to diagnose the first time.
- **Use `runWhenVisible(canvas, fn)` for every animation loop** — it does the initial sync draw + IntersectionObserver-gated RAF. `visible = true` default + initial draw inside a try/catch. Robust against observer timing.
- **Every IIFE in a section shares one `<script type="module">`.** If one throws during top-level setup (e.g., references a missing DOM id), all later IIFEs never execute. Always add clean error handling.
- **Consistent breadcrumb pattern** — use `<nav class="breadcrumb"><a>← contents</a><span class="sep">·</span><span class="section-num">section NN / 15</span></nav>`.

### Pedagogy

- **The "click" of learning is visible in the K-field (bonds), not the cell colors alone.** Coloring cells by phase or by drive is ambiguous. The cleanest approach is **drive-signature × local-coherence** — cells fade to neutral cream when unlocked, bloom into pattern colors when locked.
- **Guided tours beat free controls** when the goal is "make a specific point land." Section 6's memory tour is a 5-step Next/Prev walk that beats free knobs.
- **Browser extensions (especially MetaMask's SES lockdown) can break ES module loading.** If a user reports nothing rendering, first ask them to test in incognito.
- **Always explain what each axis is** in plots (Section 4 potential V(K) needed a prose block explicitly naming X = K, Y = cost).

### Physics

- **K is dynamical.** Every computation using fixed K is a quenched approximation — always label it. This is in AGENTS.md and `.gpd/AGENT_BOOTSTRAP.md`.
- **`r = 5.9` is determined self-consistently** by `⟨K⟩_bulk = 16/π²` on the chiral vortex phase profile. In standalone demos where we expose it as a slider, say so in prose.
- **Bistability** — at moderate frequency gaps, whether a bond lives or dies depends on initial K (cold vs warm start). This IS the physics; it models memory.
- **Single-bond CLR overshoots K_bulk**. A single bond in isolation goes to K_eq ≈ 2.1, not 16/π² ≈ 1.62. The lower value emerges only when vortex topology is present and pushes back. This is the key insight for Chapter 9.

### Open derivation (critical — don't paper over)

The LCE embedding weight **R₀²/(z(z−1))** is physically motivated but not rigorously derived. The paper flags this in §5-9 ("Open derivation" paragraph) and in the Open Questions list. It's defended with three pieces of evidence:
1. Progressive convergence across LCE orders (28,800 → 6.7 → 1.5 ppb with oscillation around CODATA)
2. Magnitude consistent with geometric suppression R₀²/(z(z−1)) ≈ 0.008 per layer
3. Candidate enumeration in `scripts/two_vertex_lce.py` rules out a dozen plausible alternatives

Do not quietly strengthen the claim. Until someone writes a rigorous LCE derivation producing R₀² as the unique shared-bond coherence factor, it stays flagged.

---

## § Practical how-tos

### Compile the paper

```
cd papers/publication/coherence_lattice_alpha
make    # or: pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

### Start the explorable dev server

```
cd papers/standalone/coherence_lattice_alpha/explorable
python3 -m http.server 8089 --bind 0.0.0.0
```

Then desktop: http://localhost:8089/  · LAN: http://192.168.1.81:8089/

### Sync changes from publication → standalone

```
SRC=papers/publication/coherence_lattice_alpha
DST=papers/standalone/coherence_lattice_alpha
cp "$SRC/paper.tex" "$DST/"
cp "$SRC/paper.pdf" "$DST/"
cp "$SRC/paper.bbl" "$DST/"
cp "$SRC/references.bib" "$DST/"
cp -r "$SRC/explorable/" "$DST/"
```

(Sync only the files actually edited — don't nuke directory contents unnecessarily.)

### Commit and push

```
cd papers/standalone/coherence_lattice_alpha
git add <files>
git commit -m "..."
git push origin main
```

### Verify scripts still work

```
cd papers/standalone/coherence_lattice_alpha
python3 scripts/alpha_137_verification.py      # → 1/α = 137.032051
python3 scripts/g_factor_from_lattice.py       # → g = 2.002319371
python3 scripts/living_vs_static_alpha.py      # → static 143, living 137
```

(Python environment: pyenv 3.11 under `.venv/` in the `coherence_lattice/` root — or any venv with numpy+scipy+matplotlib.)

---

## § Outstanding / unresolved

- **Frequency-learning demo removed from Section 6.** The corner/perimeter-driver + uniform interior physics didn't cleanly demonstrate the user's token-coupling use case. Revisit if a meaningful temporal-drive demo is needed — probably requires spatial drive-pattern + temporal-frequency together.
- **Inline SVG in section 06 NPD diagram** uses no xmlns attribute; works in Chrome/Safari/Firefox but verify on Edge before final publication.
- **Mobile scroll performance** is now acceptable thanks to IntersectionObserver, but section 6 (four figures) may still be heavy on older phones.

---

## § Quick-start template for a new section

```
explorable/sections/XX-slug.html  (new file)

Structure:
1. <!DOCTYPE html> boilerplate, link to ../css/style.css
2. <nav class="breadcrumb"> with section NN / 15
3. <h1>Title<span class="subtitle">subtitle</span></h1>
4. <p class="lead">Opening paragraph.</p>
5. Prose + <figure id="fig-X"> blocks with <canvas> + <div class="controls">
6. Interactive equation blocks where applicable:
   <div class="equation-block" id="eq-X">
     <div class="eq-display"> ...clickable <span class="eq-symbol"> spans... </div>
     <p class="eq-prompt">Click any symbol...</p>
     <div class="eq-detail"></div>
     <div class="eq-howitworks"><span class="label">In words</span>...</div>
   </div>
7. <nav class="nav-links"> prev/next at bottom
8. <script type="module">
     import { setupHiDPICanvas, wireEquation, R0, runWhenVisible, coherenceMetrics, drawMetricsStrip } from '../js/common.js';
     wireEquation('eq-X', { symbolA: {name, pronounce, description}, ... });
     (() => { /* IIFE for Figure X */ ... runWhenVisible(canvas, () => { step(); draw(); }); })();
   </script>
```

Key: **USE `&gt;` AND `&lt;` IN PROSE INSTEAD OF `>` AND `<`** if they appear outside tags.
