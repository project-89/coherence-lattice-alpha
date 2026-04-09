# The Coherence Learning Rule on the Diamond Lattice

**From Coupled Oscillators to the Fine Structure Constant**

## The Living Lattice

Standard lattice field theory treats couplings as fixed parameters tuned by hand to match experiment. The coherence lattice inverts this: couplings are dynamical variables governed by their own equation of motion — the **Coherence Learning Rule** (CLR).

The CLR is a single principle: every bond coupling evolves to maximize *coherence capital* — the product of phase alignment and structural richness. This creates a tension: alignment wants all phases equal (trivial), while structural richness wants complex topology (incoherent). The CLR finds the compromise.

The result is a **living lattice** — a self-organizing network of coupled oscillators where:

- **Couplings self-select.** Bond strengths converge to a binary field: alive bonds at the BKT critical coupling, dead bonds at zero. No parameter tuning.
- **Topology emerges.** Phase vortices nucleate spontaneously from the co-evolution of phases and couplings. These are not inserted by hand — they are produced by the dynamics.
- **The electron is a vortex.** A topological defect on the diamond lattice carries quantized charge, spin-1/2, chirality, Dirac dispersion, and an anomalous magnetic moment — all emergent from lattice mathematics. No fields are added; no particles are postulated.
- **α is a byproduct.** The fine structure constant emerges at the topologically constrained optimum of the CLR. It measures where the self-optimizing vacuum screens a magnetic perturbation — a property of the living lattice, not of the vortex.

## Why the Diamond Lattice

Diamond is not a choice but a consequence. Five physics filters — Bravais rank bound, bipartite structure, octahedral symmetry, 2-site unit cell, and vortex line persistence in d ≥ 3 — uniquely select the diamond lattice among all 3D crystal structures. It is the minimal lattice supporting Dirac fermions with topological protection.

## The Derivation

The CLR drives couplings upward. Vortex topology prevents them from exceeding the BKT critical point. The equilibrium is at the constraint boundary: K_bulk = 16/π².

From this single operating point, the entire electromagnetic coupling follows:

```
α = R₀(2/π)⁴ × (π/4)^(1/√e + α/2π)
```

A linked-cluster expansion over diamond lattice subgraphs adds vacuum polarization:

```
1/α = 137.035999    (1.5 ppb from CODATA)
  g = 2.002319304355 (11.4 matching digits)
```

Zero free parameters. The only inputs are π, e, the modified Bessel functions I₀ and I₁, and the coordination number z = 4.

### The Key Distinction: Living vs Static

A static lattice at the same BKT critical coupling gives **1/α = 143** — wrong by 4.4%. The difference is *how* the power-law exponent is evaluated. Standard lattice field theory integrates along the RG trajectory. The CLR's Phase-Locked Mode Lemma proves couplings freeze at a fixed point, so the Debye-Waller factor evaluates *at the attractor* rather than averaging over the path. This single distinction — endpoint evaluation vs path integration — accounts for the entire gap between 143 and 137.

## Repository Structure

```
├── paper.tex           # Main paper (LaTeX, revtex4-2)
├── paper.pdf           # Compiled PDF
├── paper.bbl           # Bibliography
├── references.bib      # BibTeX source
├── figures/            # All figures (PNG)
├── scripts/            # Verification scripts (Python)
├── data/               # Precomputed results (JSON)
└── LICENSE             # CC BY-NC 4.0 (paper) + AGPL-3.0 (code)
```

## Reproducing Results

```bash
pip install numpy scipy matplotlib

# Core α formula (< 1 second)
python scripts/alpha_137_verification.py

# g-factor via QED series (< 1 second)
python scripts/g_factor_from_lattice.py

# Living vs static comparison (< 1 second)
python scripts/living_vs_static_alpha.py

# Green's function G_diff = 1/z (< 10 seconds)
python scripts/diamond_greens_function.py

# Two-vertex linked-cluster expansion (< 5 seconds)
python scripts/two_vertex_lce.py
```

## Key Results

| Quantity | This Work | Measurement | Precision |
|----------|-----------|-------------|-----------|
| 1/α | 137.035999 | 137.035999206 | 1.5 ppb |
| g | 2.002319304355 | 2.002319304361 | 11.4 digits |
| Static lattice (no CLR) | 143.134 | — | Wrong by 4.4% |

## License

- **Paper and figures**: CC BY-NC 4.0
- **Code**: AGPL-3.0

See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{sharpe2026coherence,
  author = {Sharpe, Michael},
  title = {The Coherence Learning Rule on the Diamond Lattice: From Coupled Oscillators to the Fine Structure Constant},
  year = {2026},
  note = {Preprint}
}
```
