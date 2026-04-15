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
| 05 | How a binary field emerges | TODO |
| 06 | Phase-locked modes: geometry and memory | ✅ DONE |
| 07 | Spontaneous vortices | TODO |
| 08 | Why the diamond lattice | TODO |
| 09 | The BKT wall | TODO |
| 10 | Living versus static | TODO |
| 11 | The α formula, piece by piece | TODO |
| 12 | Why three dimensions | TODO |
| 13 | Closing the gap with linked clusters | TODO |
| 14 | From α to g | TODO |
| 15 | Coda: what just happened | TODO |

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

---

## TODO sections — planned content

### 07 — Spontaneous vortices

**Core idea:** Topology emerges from dynamics. Start with random phases, run the CLR, and phase vortices nucleate spontaneously. They are topologically protected: cannot be removed by smooth deformation.

**Interactives:**
- **Spontaneous nucleation:** 2D grid running CLR from random initial conditions. Vortex cores highlighted as they form. Watch winding numbers appear. Live readout of ±1 charges.
- **Vortex anatomy:** zoom in on one vortex. Color-coded phase wheel around core. Show the dead bonds at the core (Fiedler-protected). Show I_phase_local dropping inside the core, recovering outside.
- **Winding number demonstration:** trace a loop on the grid. Show the integer winding number = 2π n. Interactive: draw a loop with mouse, get the winding.

**Prose:**
- Topological defects: what they are, why they matter
- Vortex = phase winds by 2π around a loop
- Quantized: winding must be an integer
- Show spontaneous formation — NOT hand-placed
- The Fiedler channel: protects core bonds. Without Fiedler, core collapses.
- Charge quantization = topological quantization
- Setup for 3D (diamond): vortices are *lines* in 3D, more stable than in 2D

---

### 08 — Why the diamond lattice

**Core idea:** Diamond is not chosen — it is *selected*. Five filters uniquely identify the 3D diamond lattice among all possible lattices.

**Interactives:**
- **Lattice chooser:** dropdown/buttons for simple cubic, BCC, FCC, HCP, diamond, honeycomb, square. For each, show which filters pass/fail. Live result: "Diamond is the unique pass."
- **3D diamond viewer:** rotatable canvas view of the diamond lattice. z = 4 visible. Bipartite structure colored. Tetrahedral bond geometry.
- **Dirac emergence:** at nodal points in the structure factor, show how the Dirac cone appears. Interactive: move k around the Brillouin zone, see the cone form.

**Prose:**
- Five filters:
  1. `z ≥ d+1 = 4` (Bravais rank bound)
  2. Bipartite structure → chiral Bloch Hamiltonian
  3. O_h octahedral symmetry → nodal line protection
  4. 2 sites per unit cell → structure factor zeros
  5. `d ≥ 3` → vortex line persistence
- Enumeration table: SC fails (1 site), BCC fails (not bipartite), FCC fails (not bipartite), HCP fails (no O_h), diamond passes
- d dimensional generalization: A_d root lattices
- Diamond is forced

---

### 09 — The BKT wall

**Core idea:** The CLR wants K high. The topology of a vortex requires K ≤ K_BKT = 2/π. The equilibrium sits exactly at the wall. That's where α lives.

**Interactives:**
- **Vortex free energy curve:** F(R) = (πK − 2)·ln(R/a). Three regimes visualized: confined (K > K_BKT, F → +∞), marginal (K = K_BKT, F = 0), unbound (K < K_BKT, F → −∞).
- **K dial + single vortex:** user slides K, shows whether vortex is stable/confined/unbinding. Live readout.
- **The wall interactive:** CLR urge arrow pushing K up; BKT wall at K_BKT; equilibrium exactly at the boundary. Show K_bulk = 16/π² as the derived consequence.

**Prose:**
- Vortex marginality theorem (Thm 5.5 in paper)
- Three regimes of F(R)
- The constraint boundary as the equilibrium
- K_bulk = 16/π² = z · K_BKT² derivation
- The topological constraint is what breaks the unconstrained Shannon equilibrium K_eq ≈ 2.1 down to K_bulk

---

### 10 — Living versus static

**Core idea:** The entire difference between 143 and 137 comes from *how* the exponent is evaluated. Static lattice integrates over the RG trajectory. Living lattice (with PLM Lemma) evaluates at the fixed point. This chapter is a pure side-by-side comparison.

**Interactives:**
- **Two-panel comparison:** left = static integration (area under curve), right = living fixed-point (endpoint value). Identical inputs (σ² = 1/2, z = 4, etc.). Different outputs (143 vs 137).
- **Exponent sweep:** slide between "fully integrate" and "evaluate at endpoint," show how 1/α changes. The correct answer is *endpoint*.
- **PLM Lemma illustration:** shows a bond K trajectory converging exponentially to K*. The endpoint is what determines the exponent, not the trajectory.

**Prose:**
- Static calculation: `n_static = ∫_0^1 exp(-σ²l) dl ≈ 0.787 → 1/α = 143`
- Living calculation: `n_living = exp(-σ²) = 1/√e ≈ 0.607 → 1/α = 137`
- PLM Lemma proves K freezes at attractor
- Evaluate AT the attractor, not along the trajectory
- This is the single most important distinction between this framework and standard lattice field theory.

---

### 11 — The α formula, piece by piece

**Core idea:** Build up the complete α formula factor by factor, with each piece grounded in earlier chapters.

**Formula:** `α = R₀(2/π)⁴ × (π/4)^(1/√e + α/2π)`

**Interactives:**
- **Clickable-symbol equation** with full explanation of each piece. Symbols:
  - R₀(2/π) — von Mises (Ch 2)
  - ^4 — star graph vertex, z = 4 (Ch 8)
  - (π/4) — variance ratio base (derived)
  - 1/√e — DW intensity (Ch 10)
  - α/2π — Schwinger self-consistency
- **Self-consistent iteration:** watch α converge over 3 steps from the initial V = R₀⁴ guess.
- **Sensitivity slider:** wiggle each factor ±5%, see how 1/α responds. Shows which pieces matter most.

**Prose:**
- Walk through the formula factor by factor
- Each factor has a chapter's worth of meaning behind it
- Self-consistency — why α depends on itself (Schwinger QED)
- Result: 1/α = 137.032 (29 ppm from CODATA)

---

### 12 — Why three dimensions

**Core idea:** The α formula has a "d dial." Only d = 3 produces a physical α. This is the most surprising demonstration in the essay.

**Interactives:**
- **Dimension dial:** slide d from 2 to 5, watch 1/α(d) update.
  - d=2: ~36 (too strong; atoms collapse)
  - d=3: 137.032 ✓
  - d=4: ~390 (too weak; EM barely binds)
  - d=5: ~1290
- **Two constraints meeting:** show the ratio K_BKT/K_cross = π/(d+1) must be in [0,1] (d ≥ 2) AND R₀^z factor must give physical α.
- **d = 3 as anthropic or forced?:** frame it as the unique dimension where α is perturbative yet strong enough.

**Prose:**
- The dimension dial and what it reveals
- Why d=2 fails (too strong; also point vortices annihilate)
- Why d≥4 fails (too weak EM)
- Only d=3 is viable. Is this "why we live in 3D"?

---

### 13 — Closing the gap with linked clusters

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

### 14 — From α to g

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

### 15 — Coda: what just happened

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
└── sections/
    ├── 01-prelude.html        ✅
    ├── 02-oscillators.html    ✅
    ├── 03-coherence.html      ✅
    ├── 04-clr.html            ✅
    ├── 05-binary-field.html   ✅
    ├── 06-plm-npd.html        ✅
    ├── 07-vortices.html       TODO
    ├── 08-diamond.html        TODO
    ├── 09-bkt-wall.html       TODO
    ├── 10-living-vs-static.html TODO
    ├── 11-alpha-formula.html  TODO
    ├── 12-dimension.html      TODO
    ├── 13-lce.html            TODO
    ├── 14-g-factor.html       TODO
    └── 15-coda.html           TODO
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
