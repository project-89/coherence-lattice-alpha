# The Coherence Learning Rule on the Diamond Lattice

**From Coupled Oscillators to the Fine Structure Constant**

This repository contains the paper, figures, and verification scripts for deriving the fine structure constant α = 1/137.036 from first principles.

## Abstract

We derive the fine structure constant α = 1/137.036 from a single dynamical principle—the Coherence Learning Rule (CLR)—operating on coupled oscillators arranged on the diamond lattice (z = 4). The derivation chains four proven lattice identities through BKT critical-point physics to yield:

```
1/α = 137.035999  (1.5 ppb from CODATA)
g = 2.002319304355  (11.4 matching digits)
```

with zero free parameters. The only inputs are π, e, the modified Bessel functions I₀ and I₁, and the coordination number z = 4.

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
# Install dependencies
pip install numpy scipy matplotlib

# Verify α formula
python scripts/alpha_137_verification.py

# Compute g-factor from lattice α
python scripts/g_factor_from_lattice.py

# Compare living vs static lattice
python scripts/living_vs_static_alpha.py

# Verify Green's function G_diff = 1/z
python scripts/diamond_greens_function.py
```

## Key Results

| Quantity | This Work | Measurement | Gap |
|----------|-----------|-------------|-----|
| 1/α | 137.035999 | 137.035999206 | 1.5 ppb |
| g | 2.002319304355 | 2.002319304361 | 11.4 digits |

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
