# Explorable Explanation — Section Outline

Companion interactive essay for *The Coherence Learning Rule on the Diamond Lattice*.
Target audience: curious physicists, programmers, and interested general readers.
Style: Bret Victor (explorable explanations). Vanilla JS + Canvas, no framework.

## Design principles

- Each section does one thing well
- Every equation has a clickable-symbol interactive block (via `wireEquation()`)
- Prose references the interactive; the interactive changes the prose
- Mobile-responsive canvases via `setupHiDPICanvas()` in common.js
- Consistent palette: blue (state), green (parameters), orange (structural), burgundy (operators)
- Max width 720px body; figures break out wider when needed

## Overall arc

A three-act structure:

- **Act I — The Paradigm** (Chapters 1–4): build up from one oscillator to the intelligence flux `I ≥ 0`
- **Act II — Emergence** (Chapters 5–7): run the rule on a network, watch structure crystallize
- **Act III — Physics** (Chapters 8–14): specialize to the diamond lattice and read out α, with coda at 15

---

## Status

| # | Title | Status |
|---|-------|--------|
| 01 | Prelude | ✅ DONE |
| 02 | Oscillators on a graph | ✅ DONE |
| 03 | Coherence capital | ✅ DONE |
| 04 | The Coherence Learning Rule | ✅ DONE |
| 05 | How a binary field emerges | ✅ DONE |
| 06 | Phase-locked modes: geometry and memory | ✅ DONE |
| 07 | Spontaneous vortices | ✅ DONE |
| 08 | Why the diamond lattice | ✅ DONE |
| 09 | The electron (identity & topology) | ✅ DONE |
| 10 | How the lattice makes it a fermion | ✅ DONE |
| 11 | The BKT wall | ✅ DONE |
| 12 | Living versus static | ✅ DONE |
| 13 | The α formula, piece by piece | ✅ DONE |
| 14 | Why three dimensions | ✅ DONE |
| 15 | Closing the gap with linked clusters | TODO |
| 16 | From α to g | TODO |
| 17 | Coda: what just happened | TODO |

---

## Completed sections (brief)

### 01 — Prelude

One oscillator → two uncoupled → coupled (Kuramoto) → living K (CLR).
Introduces phase, frequency, sine waveform, coupling.
Four interactive figures. Ends with teaser: K settles at ~2.1 on a single bond; the BKT value 16/π² is lower — that difference is where α lives.

### 02 — Oscillators on a graph

Unit circle math (drag two phase dots, see cos / sin / Δθ).
Kuramoto equation with clickable-symbol block + "in words" explanation.
Firefly analogy (Southeast Asian fireflies).
I_phase definition with clickable-symbol block.
Firefly ring (24 oscillators) + 14×14 grid, both with live I_phase bar.
Ends: "Total sync is a seizure. We need something that rewards both sync AND complexity."

### 03 — Coherence capital

Bach fugue / "not a seizure" framing.
Introduces ρ (structural richness) — simplified as spatial variance of local I_phase.
Coherence capital C = I_phase · ρ with clickable-symbol block.
Three-regime comparison grid: chaos / coordinated complexity / trivial unison.
Main interactive: grid + bars (I_phase, ρ, C) + phase portrait with C-contours.
K slider 0–1 (sweet spot around 0.2).
Ends: setup for CLR as the rule that climbs C.

### 04 — The Coherence Learning Rule

Derivation sketch (product rule on dC/dt).
Full CLR equation: `K̇_ij = η[R₀(K_ij)·cos(Δθ_ij) − 2K_ij/r] + η·I_phase·S_ij`
Interactive block with 7 clickable symbols.
Shannon (Hebbian) and Fiedler channels.
Potential landscape V(K) = K²/r − cos(Δθ)·ln I₀(K).
Reading guide for the V(K) plot (X = K, Y = cost, marble = current K).
Live bond interactive: oscillators + potential + K(t).
Bistability aside (cold start vs warm start).
Death threshold bifurcation diagram (K* vs cos(Δθ), r slider).
Binary K-field aside.
**Climax: the Coherence Theorem.**
  - `I(t) := dC/dt` — intelligence flux
  - `I ≥ 0` boxed large
  - "By construction, not by accident"
"Arrow of coherence" — negative entropy, intelligence as self-organization, "The universe, climbing."
Aside teases Sections 5–15.

### 05 — How a binary field emerges

**Completed.** Three progressive figures (two bonds sharing a node → small ring → full 14×14 grid). Two interactive equation blocks (aggregate I(t) and fixed-point K*). Full grid has bonds toggle (phases / bonds / both), K-histogram, I(t) and C(t) traces. Sets up the idea of K-field as the lattice's learned connectivity.

### 06 — Phase-locked modes: geometry and memory

**Completed.** Four figures: PLM detection with connected-component coloring (16×16 grid, PLM-size bar chart, K-threshold slider), guided 5-step memory tour (drive on → imprinted → drive off → scramble phases → pattern reforms), patterns playground (6 preset drives, drive/learning toggles, drive-pattern × local-coherence coloring), coherence capital tracking on playgrounds. Two interactive equation blocks (PLM definition, PLM Freezing Lemma). SVG NPD hierarchy diagram. Bonds-toggle on every figure.

**Note:** A frequency-entrainment playground was prototyped but removed — the corner/perimeter-driver physics was unclear and didn't demonstrate frequency learning cleanly on a uniform 2D lattice. Revisit if a cleaner temporal-drive demo is needed later.

### 07 — Spontaneous vortices

**Completed.** Four figures: (1) two-circles winding interactive — twelve phase dots on a loop, phase-circle S¹ pointer with lap counter, "let them oscillate" toggle demonstrating topology invariance under common rotation. (2) 16×16 atan2-seeded vortex with draggable/resizable loop that computes live winding on any rectangular perimeter; vortex/antivortex and pair presets; smooth-perturbation demo of topological protection. (3) 20×20 spontaneous nucleation from random phases under Kuramoto + Shannon-CLR, guided narrative from random soup → local alignment → vortex nucleation → core death. (4) 32×32 side-by-side Shannon vs Shannon+Fiedler on atan2 seed, with **real Lanczos-based Fiedler eigensolver** (`explorable/js/fiedler.js` + 17 passing tests in `fiedler.test.mjs`); cells darken where bonds die (K-health overlay); red X marks dead bonds; orange dashes mark Fiedler-maintained weak bonds; ± plaquette markers show topology preserved; 2D annihilation at ~step 40k explicitly flagged as the §08 segue (vortex lines require d ≥ 3).

**Physics covered:** π₁(S¹) = ℤ topology; winding integrality under smooth deformation; budget-conserving Fiedler sensitivity (v_{2,i} − v_{2,j})²; Proposition 2.3 (Generalized Survival) made visible — Fiedler keeps would-be-dead core bonds above threshold; 2D point-defect annihilation as the empirical reason diamond must be 3D.

---

### 08 — Why the diamond lattice

**Core idea:** Diamond is not chosen — it is *selected*. Five filters uniquely identify the 3D diamond lattice among all possible lattices.

**Figure 1: Selection checklist + 3D view of all candidates.**
Dropdown with seven candidate lattices (SC/BCC/FCC/HCP/diamond/2D honeycomb/2D square). A five-column filter table (z ≥ 4, bipartite, O_h, 2 sites/cell, d ≥ 3) updates with ✓/✗ marks; diamond is the unique row passing every column. Above the table, a rotatable 3D canvas shows the currently selected lattice with proper bipartite colouring (A = burgundy, B = blue, or monochromatic grey when non-bipartite). Drag rotates, shift+drag pans, wheel zooms.

**Figure 2: Deep-dive into diamond.** Same 3D renderer but on diamond only. Mode buttons: Full lattice / A-sublattice only (reveals FCC) / B-sublattice only / One tetrahedron (isolates central atom + 4 NN at 109.47°) / (111) honeycomb layer 1, 2, 3 (each isolates one of the three buckled honeycomb sheets that stack to form the 3D crystal; camera auto-snaps to [111]). View shortcuts: Isometric / Along [111]. Auto-rotate toggle.

**Figure 3: 3D vortex line on diamond.** Seeded atan2 phase θ(x,y,z) = atan2(y, x) (independent of z) on a 216-atom diamond lattice (nCells = 3). Atom fill colour = phase via HSL hue; outline colour = sublattice (burgundy A, blue B). Three view presets: oblique (default), View down [001] (pinwheel face-on), View from side (vortex line vertical, stacked rainbow disks). "Let them oscillate" toggle advances all phases by a common ω·dt per frame — hue cycles, pinwheel is preserved; winding invariance made visible. Reuses `lattice3d.js` machinery and the same HSL hue mapping as §7.

**Prose shipped:**
- Five filters spelled out in numbered list
- Per-lattice explanations clickable from either dropdown or column headers
- Mentions the A_d family generalisation
- Segue prose into §9 (electron) already in place
- 2D→3D motivation carried over from §7's annihilation

---

### 09 — The electron

**Core idea:** We have a vortex (§7) and a diamond lattice (§8). Put them together and we have an electron. This section pulls together charge, the Dirac equation, spin-1/2 setup, and names what we've built.

**Interactives:**
- **Dirac cone emergence:** Bloch Hamiltonian at nodal k-points on the diamond Brillouin zone. Hover a k-point cursor, see the two bands cross linearly (not parabolically). That linear crossing is the Dirac spectrum.
- **Electron assembly diagram (SVG):** nodes for Vortex, Diamond, Bipartite, Frame (greyed/deferred). Arrows to Charge, Dirac equation, Spin-½, Mass. Clickable with per-node explanations.
- **Dirac equation interactive:** clickable-symbol `iℏ γ^μ ∂_μ ψ = m ψ`. Clicking γ^μ reveals the Clifford anticommutator `{γ^μ, γ^ν} = 2g^μν` and says: this algebra forces spin-½.

**Prose:**
- Topology made a charge; the bipartite lattice made a spectrum; the combination is a fermion.
- The chiral downfold (paper §4.2): vortex bound state localises on one sublattice.
- Clifford algebra as the geometry of non-commutativity that encodes spin.
- What's still deferred (frame sector, companion paper) and what's still to come (α, §10; g-factor, §15).

---

### 11 — The BKT wall

**Completed.** Three figures: (1) Vortex free energy F(R) with live K-slider from 0.30 to 1.00 and three frozen reference curves (K=0.50 unbound, K=2/π marginal, K=0.80 confined), live burgundy curve tracks the slider, regime strip above the plot snaps between confined/marginal/unbound. (2) Vortex-antivortex pair on a 2D phase lattice with draggable cores, K slider, force-arrow visualisation (attractive/repulsive/zero coloured by regime), Play button that releases the pair so it annihilates (K > K_BKT), flees (K < K_BKT), or drifts randomly (K = K_BKT). (3) Hero "wall" figure: 1D K-axis showing the CLR potential V(K) with marble rolling upward under Shannon-channel gradient descent, BKT wall at K_BKT, K_bulk = 16/π² marked, K_eq ≈ 2.11 marked. Play button sends marble up; Remove-vortex toggle drops the wall and lets the marble overshoot to K_eq. Two equation blocks (F(R), K_bulk derivation) plus a short base = π/z recap. Segue paragraph into §12 teases the living-vs-static exponent distinction.

---

### 12 — Living versus static

**Completed.** Three figures: (1) Side-by-side α calculator — left panel shows shaded area under exp(−σ²l) from l=0 to l=1 (static integration, n = 0.787, 1/α = 143.13); right panel shows a single green bar at l=1 (living endpoint, n = 1/√e = 0.607, 1/α = 137.032). Live σ² slider with "physical σ²=1/2" reset button. Both 1/α values computed via self-consistent BKT iteration, CODATA reference printed below. (2) PLM trajectory figure: four bonds released from different K₀ (0.1, 0.6, 1.2, 2.4) all converging exponentially to K* = 16/π² (upper plot); lower plot shows the Debye–Waller factor exp(−σ²K(t)) along each trajectory, all collapsing onto the single endpoint exp(−σ²K*). Play/pause controls, 40-unit runtime. (3) Window-width sweep: slider w ∈ [0, 1] highlights the averaged window under the decay curve (green line at w=0, collapsing to the full area at w=1); 1/α readout updates continuously from 137.03 (green) to 143.13 (purple), with a gradient bar and marker showing position. "Living" and "Static" snap buttons. Two equation blocks (PLM convergence rate λ = −η V″(K*); window-averaged n(w) integral). Closing restates the BKT α formula and teases §13's factor-by-factor walkthrough.

---

### 13 — The α formula, piece by piece

**Completed.** Hero clickable equation at 28px display scale — every coloured symbol opens a detail panel beneath the equation with chapter source, pronunciation, current numeric value at the BKT point, and 2–4 sentence physics description. Seven clickable chips (α, R₀, 2/π, the exponent 4, π/4 base, 1/√e, α/2π Schwinger correction) cross-linking all twelve earlier chapters. Iteration animation: orange dot on a 1/α number line between 100 and 175, CODATA 137.036 marked in green. Starts at 1/α₀ = 118.32 (bare UV), snaps step-by-step to 137.038 → 137.032 → converged. ppm distance to CODATA printed live. Iteration history list on the left shows the n_eff and 1/α at each step. Sensitivity figure: three ±5% sliders (R₀, π/4 base, 1/√e), each with a coloured bar beneath showing the full ±5% swing in 1/α space — R₀ = 55-unit range, π/4 = 8-unit, 1/√e = 2-unit. CODATA dashed marker on ruler. "Zero free parameters, twelve chapters deep" aside inventories every factor's provenance. Closing teases §14's dimensional dial.

---

### 14 — Why three dimensions

**Completed.** One big d-dial figure with d ∈ [2, 5] slider (step 0.01) and four snap-buttons. Log-scaled 1/α ruler from 10 to 3000 with coloured regime bands (purple = too strong <100, green = physical 100–200, orange = too weak >200). CODATA 137.036 dashed marker. Integer d positions (2, 3, 4, 5) drawn as faded dots on the ruler. Current value highlighted in the regime's colour. Four lattice-sketch cards below — one per integer d — showing coordination number z=d+1 as a radial bonds diagram with central atom. Cards click through to the slider; the active card highlights. "Why d=2 and d=4 fail" two-card section explaining atom collapse at strong α and barely-bound atoms at weak α. Final aside distinguishes anthropic from forced dimensionality and gives the three-constraint argument (CLR attractor + topologically stable vortex + physical α) that intersects only at d=3. Closing previews §15's LCE correction.

---

### 15 — Closing the gap with linked clusters

**Core idea:** BKT gives 29 ppm accuracy. A linked-cluster expansion over diamond subgraphs adds vacuum polarization running and closes to 1.5 ppb. Includes the honest open-problem flag.

**Interactives:**
- **Convergence table:** progressive LCE orders with residuals (28,800 → 6.7 → 1.5 ppb).
- **Subgraph visualizer:** show the specific subgraphs — star, plaquette, dumbbell — that contribute at each order. Click a subgraph, see its contribution.
- **The R₀² question:** interactive showing the candidate embedding weights and their δc values. The specific form R₀²/(z(z−1)) is the only survivor.

**Prose:**
- BKT is 29 ppm already
- QED vacuum polarization runs α from lattice to Q=0
- The LCE computes the crossover scale
- Layer by layer: 28,800 → 6.7 → 1.5 ppb
- **Honest flag:** the R₀² form is physically motivated but not rigorously derived. Convergence pattern + tight constraint are evidence, but formal derivation is open.
- Net result: 1/α = 137.035999

---

### 16 — From α to g

**Core idea:** α → standard QED series → g-factor to 11.4 digits. Not an independent prediction; a consistency check showing the lattice α, fed through textbook QED, reproduces g.

**Interactives:**
- **QED series:** clickable Taylor series in α/π. Each term's contribution shown. Total a_e = Σ C_n (α/π)^n.
- **Comparison:** digit-by-digit comparison of lattice g vs measured g. 11 matching digits, then divergence.
- **Sensitivity:** wiggle α, watch g response. Shows α is the bottleneck.

**Prose:**
- QED perturbation series for a_e = (g−2)/2
- Schwinger: a_e(1) = α/(2π)
- Higher orders: C_2, C_3, C_4, C_5 all known from standard QED
- Plug in lattice α → get g = 2.002319304355
- 11.4 matching digits with experiment
- The chain: lattice → α → QED → g. No part of QED is modified.

---

### 17 — Coda: what just happened

**Core idea:** Step back. Summarize the whole climb. Reframe the paradigm. Hint at what's next (companion papers, open questions).

**Interactives:**
- **The full derivation chain visualized:** interactive table from axioms to α with every step clickable, each linking back to the chapter that derived it.
- **Timeline:** hypotheses → derivations → numerical verifications → publication. Shows the process.

**Prose:**
- Recap: 17 steps, zero free parameters
- The paradigm in three sentences
- What's coming: lepton generations (hexagonal DOFs), gauge sector (SO(3) frame), gravity (phase gradients)
- Open problems: R₀² derivation, large-L convergence, frame sector dynamics, VCO bench test
- Closing: "We have been looking at a mathematical structure. Whether it is the structure of reality, an effective description of a deeper one, or a coincidence of extraordinary precision is not for this paper to decide. What this paper shows is that the inequality `I ≥ 0`, written on the diamond lattice, reproduces the fine structure constant to six decimal places. That is what just happened."

---

## File locations

```
explorable/
├── index.html                 # Landing + TOC
├── OUTLINE.md                 # this file
├── css/style.css              # shared styling
├── js/common.js               # setupHiDPICanvas, R0, wireEquation, etc.
├── js/fiedler.js              # Lanczos Fiedler eigensolver (§7 Fig 4)
├── js/lattice3d.js            # canvas 3D renderer (§8 Fig 2+3)
└── sections/
    ├── 01-prelude.html        ✅
    ├── 02-oscillators.html    ✅
    ├── 03-coherence.html      ✅
    ├── 04-clr.html            ✅
    ├── 05-binary-field.html   ✅
    ├── 06-plm-npd.html        ✅
    ├── 07-vortices.html       ✅
    ├── 08-diamond.html        ✅
    ├── 09-electron.html       ✅
    ├── 10-fermion.html        ✅
    ├── 11-bkt-wall.html       ✅
    ├── 12-living-vs-static.html ✅
    ├── 13-alpha-formula.html  ✅
    ├── 14-dimension.html      ✅
    ├── 15-lce.html            TODO
    ├── 16-g-factor.html       TODO
    └── 17-coda.html           TODO
```

## Development notes

- Local preview: `python3 -m http.server 8089 --bind 0.0.0.0` from explorable/
- Source of truth: `papers/publication/coherence_lattice_alpha/explorable/`
- Standalone (git repo): `papers/standalone/coherence_lattice_alpha/explorable/`
- Always sync before committing

## Key recurring visual/math elements

- **Palette:** cream bg (#fdfaf3), graphite text, deep blue (#2a5f8f) for state, green (#2d7d4f) for parameters, warm orange (#d97236) for alive/structural, burgundy (#7d2d4f) for dead/operators
- **Canvas sizing:** 680 wide default, heights vary by content. Responsive via `aspect-ratio` in CSS.
- **Equations:** all significant equations get `wireEquation()` treatment with per-symbol clickable explanations, pronunciations, descriptions.
- **In words:** every equation block has a plain-English "In words" paragraph that tells the story of what the equation does.
