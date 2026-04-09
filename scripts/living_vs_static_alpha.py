#!/usr/bin/env python3
"""
Living lattice vs static lattice: why the CLR determines α.

This script demonstrates the central claim of the paper: the fine structure
constant α = 1/137 requires a LIVING lattice (dynamical K via the CLR).
A static lattice at the same BKT critical point gives 1/α = 143 — wrong
by 4.4%.

The distinction:

  STATIC (standard lattice field theory):
    K is a fixed parameter. The power-law correction integrates the
    Debye-Waller coherence transfer h(l) = exp(-σ² l) continuously
    over the RG trajectory:
      n_static = ∫₀¹ exp(-σ² l) dl = (1 - exp(-σ²))/σ² = 0.787
      → 1/α = 143.1

  LIVING (CLR dynamics):
    K is dynamical. The PLM Lemma proves K converges to a frozen fixed
    point. The power-law correction evaluates the DW factor AT the
    attractor, not integrated along the trajectory:
      n_living = exp(-σ²) = 1/√e = 0.607
      → 1/α = 137.0

The living lattice gives 137. The dead lattice gives 143. The CLR is
what makes the lattice alive.

References:
  - PLM K-field freezing lemma: paper §5.5, Lemma 5.1
  - Coherence transfer function: dn/dl = -σ² n  →  n(l) = exp(-σ² l)
  - σ² = 1/(πK_BKT) = 1/2: paper §5.8, Appendix A.6

Usage:
  python living_vs_static_alpha.py
"""

import numpy as np
from scipy.special import i0, i1


# === Constants ===
K_BKT = 2 / np.pi        # BKT critical coupling (universal)
z = 4                     # diamond lattice coordination number
eta_BKT = 0.25            # anomalous dimension at BKT: 1/(2πK)
sigma2 = 1 / (np.pi * K_BKT)  # = 1/2 (normalized NN variance, proven)
base = np.pi / z          # lattice-to-RG variance ratio (proven)
CODATA = 137.035999206    # Morel et al. 2020


def R0(K):
    """Bessel ratio I₁(K)/I₀(K) — the XY model order parameter."""
    return i1(K) / i0(K)


def alpha_formula(n_eff):
    """Compute α = R₀^z × base^n_eff with self-consistent Schwinger."""
    V = R0(K_BKT) ** z
    # Self-consistent iteration: n_total = n_eff + α/(2π)
    alpha = V * base ** n_eff  # initial guess (no Schwinger)
    for _ in range(10):
        n_total = n_eff + alpha / (2 * np.pi)
        alpha = V * base ** n_total
    return alpha


# === Main comparison ===
print("=" * 65)
print("  LIVING LATTICE vs STATIC LATTICE: THE α DERIVATION")
print("=" * 65)

# Common ingredients
V = R0(K_BKT) ** z
print(f"\nCommon to both models:")
print(f"  K_BKT        = 2/π       = {K_BKT:.8f}")
print(f"  R₀(K_BKT)    = I₁/I₀    = {R0(K_BKT):.8f}")
print(f"  V = R₀^z     (vertex)   = {V:.10f}")
print(f"  base = π/z   (variance) = {base:.10f}")
print(f"  σ² = 1/(πK)  (proven)   = {sigma2:.8f}")
print(f"  η = 1/(2πK)  (BKT)      = {eta_BKT:.8f}")


# --- Static lattice ---
print("\n" + "-" * 65)
print("  STATIC LATTICE (standard lattice field theory)")
print("-" * 65)
print("  K is a fixed parameter. RG running integrates DW continuously.")
print("  Coherence transfer: h(l) = exp(-σ² l)")
print("  Effective exponent: n = ∫₀¹ h(l) dl = (1 - exp(-σ²))/σ²")

n_static = (1 - np.exp(-sigma2)) / sigma2
alpha_static = alpha_formula(n_static)
inv_alpha_static = 1 / alpha_static

print(f"\n  n_static = (1 - e^(-1/2)) / (1/2) = {n_static:.8f}")
print(f"  α_static = {alpha_static:.10f}")
print(f"  1/α_static = {inv_alpha_static:.4f}")
print(f"  Error vs CODATA: {(inv_alpha_static - CODATA) / CODATA * 1e6:+.0f} ppm")


# --- Living lattice ---
print("\n" + "-" * 65)
print("  LIVING LATTICE (CLR dynamics, PLM Lemma)")
print("-" * 65)
print("  K is dynamical. PLM Lemma: K frozen at attractor.")
print("  Coherence at fixed point: n = exp(-σ²) = DW factor")
print("  Power-law correction evaluates AT the attractor.")

n_living = np.exp(-sigma2)
alpha_living = alpha_formula(n_living)
inv_alpha_living = 1 / alpha_living

print(f"\n  n_living = exp(-1/2) = 1/√e = {n_living:.8f}")
print(f"  α_living = {alpha_living:.10f}")
print(f"  1/α_living = {inv_alpha_living:.4f}")
print(f"  Error vs CODATA: {(inv_alpha_living - CODATA) / CODATA * 1e6:+.0f} ppm")
print(f"  (residual closed by LCE → 1.5 ppb, see two_vertex_lce.py)")


# --- Comparison ---
print("\n" + "=" * 65)
print("  COMPARISON")
print("=" * 65)
print(f"  CODATA 2020:  1/α = {CODATA}")
print(f"  Living (CLR): 1/α = {inv_alpha_living:.6f}  "
      f"({abs(inv_alpha_living - CODATA) / CODATA * 1e6:+.0f} ppm)")
print(f"  Static:       1/α = {inv_alpha_static:.6f}  "
      f"({abs(inv_alpha_static - CODATA) / CODATA * 1e6:+.0f} ppm)")
print(f"\n  The living lattice is "
      f"{abs(inv_alpha_static - CODATA) / abs(inv_alpha_living - CODATA):.0f}× "
      f"closer to experiment.")
print(f"  Static error: {abs(inv_alpha_static - CODATA) / CODATA * 100:.1f}%")
print(f"  Living error: {abs(inv_alpha_living - CODATA) / CODATA * 100:.3f}% "
      f"(before LCE)")


# --- Why the difference ---
print("\n" + "=" * 65)
print("  WHY THE DIFFERENCE")
print("=" * 65)
print(f"""
  The RG coherence equation:  dn/dl = -σ² n
  Solution:                   n(l) = exp(-σ² l)

  At l = 0 (UV):  n = 1     (fully coherent)
  At l = 1 (IR):  n = 1/√e  (DW decoherence)

  Static model: INTEGRATES over the full trajectory [0, 1].
    Each scale contributes proportionally to its coherence.
    Average coherence = (1 - 1/√e) / (1/2) = {n_static:.4f}

  Living model: EVALUATES at the fixed point l = 1.
    CLR gradient descent freezes K at the attractor.
    Fixed-point coherence = exp(-σ²) = {n_living:.4f}

  Endpoint < average (because coherence decays monotonically).
  Less running → larger 1/α → correct answer.
""")


# --- Robustness check ---
print("=" * 65)
print("  ROBUSTNESS: ALTERNATIVE EXPONENTS")
print("=" * 65)
print(f"  {'Exponent':25s} {'Value':>8s}   {'1/α':>10s}   {'Error':>10s}")
print(f"  {'-'*25} {'-'*8}   {'-'*10}   {'-'*10}")

alternatives = [
    ("exp(-σ²) [living]", np.exp(-sigma2)),
    ("(1-exp(-σ²))/σ² [static]", n_static),
    ("σ²", sigma2),
    ("η = 1/4", eta_BKT),
    ("2η = 1/2", 2 * eta_BKT),
    ("1 - 2η", 1 - 2 * eta_BKT),
    ("1 (full step)", 1.0),
    ("0 (no running)", 0.0),
]

for label, n_val in alternatives:
    a = alpha_formula(n_val)
    inv_a = 1 / a
    err = (inv_a - CODATA) / CODATA * 100
    marker = " ◄" if label.startswith("exp") else ""
    print(f"  {label:25s} {n_val:8.6f}   {inv_a:10.4f}   {err:+9.3f}%{marker}")

print(f"\n  Only exp(-σ²) = 1/√e gives the correct 1/α.")
print(f"  This is the unique fixed-point evaluation from the PLM Lemma.")


# --- Chain of logic ---
print("\n" + "=" * 65)
print("  CHAIN OF LOGIC")
print("=" * 65)
print("""
  1. CLR dynamics: K̇ ∝ [R₀(K)·cos(Δθ) - 2K/r]      [Axiom 1]
  2. PLM attractor: cos(Δθ_ij) = constant               [Attractor Thm]
  3. K frozen at PLM: K → K*                             [PLM Lemma]
  4. BKT selected: K* = 2/π                              [Vortex margin.]
  5. σ² = 1/(πK*) = 1/2                                  [Proven, 3 ways]
  6. Fixed-point evaluation: n = exp(-σ²) = 1/√e         [THIS SCRIPT]
  7. Formula: α = R₀⁴ × (π/4)^(1/√e + α/2π) = 1/137.032
  8. + LCE vacuum polarization → 1/α = 137.035999 (1.5 ppb)

  Step 6 closes the assembly gap: the CLR's gradient descent to a
  fixed point means the power-law factor is a POINT EVALUATION,
  not a PATH INTEGRAL. Remove the CLR → 1/α = 143 (wrong by 4.4%).
""")
