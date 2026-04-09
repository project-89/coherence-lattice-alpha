#!/usr/bin/env python3
"""
two_vertex_lce.py
=================
Two-vertex linked-cluster correction to the crossover coefficient c.

The single-vertex binomial gives:
  c_1vtx = [1 - (1-V)^(z-1)] / V = 2.9747
  l_1vtx = (1-V)^(z-1) = 0.97486
  1/α = 137.035998  (+0.005 ppm)

CODATA requires c_exact = 2.9859, gap δc = +0.0112.

This script computes the two-vertex (dumbbell) correction from inter-vertex
correlations through shared bonds and plaquettes on the diamond lattice.

The key physics: two adjacent vertices A,B share a bond. Their vertex
amplitudes V_A = R₀^z and V_B = R₀^z are correlated because the shared
bond contributes one factor of R₀ to both. The correlation
Cov(V_A, V_B) = R₀^{2(z-1)} × [⟨cos²θ⟩ - R₀²] modifies the effective
crossover by making adjacent vertices more likely to be simultaneously
coherent, increasing the total "blocking fraction" and thus increasing c.

LT-ID: EXP-2VTX-LC
Status: EXPERIMENTAL
"""

import numpy as np
from scipy.special import i0, i1, iv
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# Constants
# =====================================================================

K_BKT = 2.0 / np.pi
z = 4
R0 = i1(K_BKT) / i0(K_BKT)
V_star = R0**z
Q_lat = 2.0 / np.sqrt(3)
INV_ALPHA_CODATA = 137.035999
g_measured = 2.00231930436256

# Bessel functions at K_BKT
I0_K = i0(K_BKT)
I1_K = i1(K_BKT)
I2_K = iv(2, K_BKT)
I3_K = iv(3, K_BKT)
I4_K = iv(4, K_BKT)

# R_n = I_n(K)/I_0(K)  — nth order coherence
R1 = I1_K / I0_K   # = R0
R2 = I2_K / I0_K
R3 = I3_K / I0_K
R4 = I4_K / I0_K

# BKT self-consistent alpha
alpha_bkt = V_star
for _ in range(30):
    n_exp = 1.0 / np.sqrt(np.e) + alpha_bkt / (2 * np.pi)
    alpha_bkt = V_star * (np.pi / 4)**n_exp
inv_alpha_BKT = 1.0 / alpha_bkt

# VP integral
def vp_F(t):
    f = lambda u: 6 * u * (1 - u) * np.log(1 + t * u * (1 - u))
    return quad(f, 0, 1)[0]

def inv_alpha_from_l(l_val):
    Q = Q_lat * np.exp(-l_val)
    return inv_alpha_BKT + vp_F(Q**2) / (3 * np.pi)

# Exact c from CODATA
def _gap(l_val):
    return inv_alpha_from_l(l_val) - INV_ALPHA_CODATA
l_exact = brentq(_gap, 0.9, 1.0)
c_exact = (1 - l_exact) / V_star

# Single-vertex binomial
c_binom = (1 - (1 - V_star)**(z-1)) / V_star
l_binom = (1 - V_star)**(z-1)

# QED g-factor
C_QED = [0.5, -0.328478965579193, 1.181241456587, -1.9113, 7.795]

def compute_g(alpha_val):
    x = alpha_val / np.pi
    a_e = sum(C_QED[i] * x**(i+1) for i in range(5))
    return 2 * (1 + a_e)

def count_digits(val, ref):
    if val == ref: return 99.0
    ratio = abs(val - ref) / abs(ref)
    return -np.log10(ratio) if ratio > 0 else 99.0

# Sensitivity: d(1/α)/dl
dl = 1e-8
d_inv_alpha_dl = (inv_alpha_from_l(l_binom + dl) - inv_alpha_from_l(l_binom - dl)) / (2 * dl)

print("=" * 72)
print("  TWO-VERTEX LINKED-CLUSTER CORRECTION")
print("=" * 72)

print(f"\n  ── Base Constants ──")
print(f"    R₀ = I₁(K)/I₀(K) = {R0:.10f}")
print(f"    R₂ = I₂(K)/I₀(K) = {R2:.10f}")
print(f"    R₃ = I₃(K)/I₀(K) = {R3:.10f}")
print(f"    R₄ = I₄(K)/I₀(K) = {R4:.10f}")
print(f"    V = R₀^{z} = {V_star:.10e}")
print(f"    1/α(BKT) = {inv_alpha_BKT:.10f}")
print(f"    c_binomial = {c_binom:.10f}")
print(f"    c_exact    = {c_exact:.10f}")
print(f"    δc needed  = {c_exact - c_binom:+.10f}")
print(f"    d(1/α)/dl  = {d_inv_alpha_dl:.6f}")

# =====================================================================
# Part 1: Bond Coherence Statistics
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 1: BOND COHERENCE AT K_BKT = {K_BKT:.8f}")
print(f"{'='*72}")

# ⟨cos(nθ)⟩ = I_n(K)/I_0(K) = R_n  for von Mises distribution
# For a single bond (i,j), θ = θ_i - θ_j:
#   ⟨cosθ⟩ = R₀
#   ⟨cos²θ⟩ = 1/2 + R₂/2  (trig identity: cos²θ = (1+cos2θ)/2)
#   ⟨cos³θ⟩ = (3R₁ + R₃)/4
#   ⟨cos⁴θ⟩ = (3 + 4R₂ + R₄)/8
cos2_mean = 0.5 + R2 / 2
cos1_mean = R0
sigma2_bond = cos2_mean - R0**2  # Var(cos θ)

print(f"\n  Single-bond coherence statistics:")
print(f"    ⟨cosθ⟩ = R₀ = {R0:.10f}")
print(f"    ⟨cos²θ⟩ = 1/2 + R₂/2 = {cos2_mean:.10f}")
print(f"    Var(cosθ) = σ²_bond = {sigma2_bond:.10f}")
print(f"    σ_bond = {np.sqrt(sigma2_bond):.10f}")
print(f"    R₀² = {R0**2:.10f}")
print(f"    σ²/R₀² = {sigma2_bond / R0**2:.6f}  (relative fluctuation)")

# =====================================================================
# Part 2: Inter-Vertex Correlation
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 2: INTER-VERTEX CORRELATION")
print(f"{'='*72}")

# V_A = ∏_{j=1}^z cos(θ_A - θ_{n_j})  for A's z neighbors {n_1,...,n_z}
# V_B = ∏_{k=1}^z cos(θ_B - θ_{m_k})  for B's z neighbors {m_1,...,m_z}
# A and B share bond (A,B). So one of A's neighbors is B and vice versa.
#
# In mean-field (independent bonds):
#   ⟨V_A⟩ = R₀^z = V
#   ⟨V_B⟩ = R₀^z = V
#   ⟨V_A V_B⟩_indep = V² (if A,B bonds were independent)
#
# But the shared bond (A,B) with phase θ_{AB} = θ_A - θ_B contributes:
#   - cos(θ_{AB}) to V_A
#   - cos(θ_{AB}) to V_B  (same factor!)
#
# So ⟨V_A V_B⟩ = ⟨cos²(θ_{AB})⟩ × R₀^{2(z-1)}  (in mean-field for other bonds)
#              = (1/2 + R₂/2) × R₀^{2(z-1)}
#
# While: ⟨V_A⟩⟨V_B⟩ = R₀^{2z}
#
# Covariance:
#   Cov(V_A, V_B) = R₀^{2(z-1)} × [cos2_mean - R₀²]
#                 = R₀^{2(z-1)} × σ²_bond

cov_VA_VB = R0**(2*(z-1)) * sigma2_bond
corr_coeff = cov_VA_VB / V_star**2  # normalized correlation

# Also compute: the "excess" second moment
# ⟨V_A V_B⟩ / (⟨V_A⟩⟨V_B⟩) = cos2_mean / R₀²
ratio_2nd_moment = cos2_mean / R0**2

print(f"\n  Vertex amplitude correlations (adjacent A,B sharing one bond):")
print(f"    ⟨V_A V_B⟩/⟨V_A⟩⟨V_B⟩ = ⟨cos²θ⟩/R₀² = {ratio_2nd_moment:.6f}")
print(f"    Cov(V_A, V_B) = R₀^{2*(z-1)} × σ² = {cov_VA_VB:.6e}")
print(f"    Correlation coefficient ρ = Cov/(V²) = {corr_coeff:.6f}")
print(f"    This means adjacent vertices are {corr_coeff:.1%} more likely")
print(f"    to be simultaneously coherent than independent vertices.")

# =====================================================================
# Part 3: Diamond Lattice Green's Function (momentum space)
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 3: DIAMOND LATTICE GREEN'S FUNCTION")
print(f"{'='*72}")

# Diamond lattice: FCC Bravais + 2-atom basis
# Primitive vectors (FCC):
#   a1 = (0, 1/2, 1/2), a2 = (1/2, 0, 1/2), a3 = (1/2, 1/2, 0)
# Basis: A at (0,0,0), B at (1/4,1/4,1/4)
# NN vectors (A→B):
#   δ1 = (1,1,1)/4, δ2 = (1,-1,-1)/4, δ3 = (-1,1,-1)/4, δ4 = (-1,-1,1)/4
#
# Structure factor: f(q) = Σ_j exp(iq·δ_j)
# |f(q)|² = z + 2Σ_{j<k} cos(q·(δ_j - δ_k))
#
# Lattice Laplacian (2x2): M(q) = [[z, -f(q)], [-f*(q), z]]
# Eigenvalues: z ± |f(q)|
#
# Green's function:
#   G_AA(r=0) - G_AA(r) = (1/Ω) ∫_BZ [1 - cos(q·r)] × z / [(z² - |f|²) K] d³q
#   G_AA(r=0) - G_AB(δ) = (1/Ω) ∫_BZ [1 - Re(f(q)e^{-iq·δ₁})/z] × z / [(z² - |f|²) K] d³q

# NN vectors (A→B), in units of a (cubic lattice constant)
deltas = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) / 4.0

def structure_factor_sq(qx, qy, qz):
    """Compute |f(q)|² for the diamond lattice."""
    f_real = 0.0
    f_imag = 0.0
    for d in deltas:
        phase = qx * d[0] + qy * d[1] + qz * d[2]
        f_real += np.cos(phase)
        f_imag += np.sin(phase)
    return f_real**2 + f_imag**2

def structure_factor(qx, qy, qz):
    """Compute f(q) (complex) for diamond lattice."""
    f = 0.0 + 0.0j
    for d in deltas:
        phase = qx * d[0] + qy * d[1] + qz * d[2]
        f += np.exp(1j * phase)
    return f

# Compute G(0,0) - G(0,nn) by BZ integration
# Use reciprocal lattice of FCC:
#   b1 = 2π(-1,1,1), b2 = 2π(1,-1,1), b3 = 2π(1,1,-1)
# BZ volume = (2π)³ × 4 (for FCC with a=1)

# Numerical integration over the BZ
# Use a grid in reduced coordinates (t1,t2,t3) ∈ [0,1]³
# q = t1*b1 + t2*b2 + t3*b3

b1 = 2 * np.pi * np.array([-1, 1, 1])
b2 = 2 * np.pi * np.array([1, -1, 1])
b3 = 2 * np.pi * np.array([1, 1, -1])

# BZ volume
BZ_vol = abs(np.dot(b1, np.cross(b2, b3)))
print(f"    BZ volume = {BZ_vol:.4f} = (2π)³ × {BZ_vol / (2*np.pi)**3:.4f}")

N_grid = 64  # grid points per dimension
dt = 1.0 / N_grid

# Second-neighbor vectors (A→A, via one B vertex)
# From A at origin, go to B at δ_j, then from B go to another A at δ_j - δ_k (for k≠j)
# NNN vector: δ_j - δ_k + δ_j = 2δ_j - δ_k  ... no, that's wrong.
# Actually from A(0) → B(δ_j) → A(δ_j - δ_k + some FCC vector)
# Wait: B at position δ_j has neighbors at δ_j + δ_k' where δ_k' are the
# REVERSE NN vectors: from B to A, the vectors are -δ_k.
# So: A(0) → B(δ_j) → A(δ_j - δ_k) for k = 1,...,z
# For k = j: A(δ_j - δ_j) = A(0) (back to start)
# For k ≠ j: A(δ_j - δ_k) = second neighbor

nnn_vectors = []
for j in range(z):
    for k in range(z):
        if k == j:
            continue
        nnn = deltas[j] - deltas[k]
        nnn_vectors.append(nnn)

# Unique NNN vectors
nnn_unique = []
nnn_mults = []
for v in nnn_vectors:
    found = False
    for i, u in enumerate(nnn_unique):
        if np.allclose(v, u) or np.allclose(v, -u):
            nnn_mults[i] += 1
            found = True
            break
    if not found:
        nnn_unique.append(v)
        nnn_mults.append(1)

print(f"\n  Second-neighbor (A→A) vectors on diamond:")
print(f"    Total NNN paths: {len(nnn_vectors)} = z(z-1) = {z*(z-1)}")
print(f"    Unique NNN distances: {len(nnn_unique)}")
for i, (v, m) in enumerate(zip(nnn_unique, nnn_mults)):
    dist = np.linalg.norm(v)
    print(f"      NNN_{i}: ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}), |r| = {dist:.4f}, multiplicity = {m}")

# ── BZ Integration ──
print(f"\n  Computing lattice Green's function (BZ integration, N={N_grid})...")

# We need:
# (a) G_AA(0) - G_AB(nn) = ∫ [z - Re(f·e^{-iq·δ₁})] / [(z²-|f|²)K] dq / BZ_vol
#     This should = 1/z (exact, as proven)
#
# (b) G_AA(0) - G_AA(nnn) = ∫ [1 - cos(q·r_nnn)] × z / [(z²-|f|²)K] dq / BZ_vol × 2
#     Wait — for same-sublattice:
#     The G on the same sublattice uses the diagonal element.
#     G_AA(q) = z / [(z² - |f(q)|²) K]  (2×2 inverse, diagonal)
#     G_AA(0) - G_AA(r) = (1/BZ_vol) ∫ [1 - cos(q·r)] × z / [(z²-|f|²)K] dq

# The integrand diverges at q=0 (Goldstone mode): z² - |f(0)|² = z² - z² = 0
# But the [1 - cos(q·r)] factor vanishes too, so the integral converges.
# Need to handle this carefully.

# For numerical stability, compute the differences directly.

sum_nn = 0.0  # G_AA(0) - G_AB(nn)
sum_nnn = np.zeros(len(nnn_unique))  # G_AA(0) - G_AA(nnn_i)

delta_1 = deltas[0]  # reference NN vector

for i1 in range(N_grid):
    t1 = (i1 + 0.5) * dt
    for i2 in range(N_grid):
        t2 = (i2 + 0.5) * dt
        for i3 in range(N_grid):
            t3 = (i3 + 0.5) * dt
            q = t1 * b1 + t2 * b2 + t3 * b3

            f = structure_factor(q[0], q[1], q[2])
            f_sq = abs(f)**2

            denom = z**2 - f_sq
            if denom < 1e-12:
                continue  # skip Goldstone mode

            # (a) NN: G_AA(0) - G_AB(nn)
            # From the 2x2 Green's function:
            # G_AB(q) = f(q) / [(z²-|f|²)K]
            # G_AA(0) - G_AB(nn) = (1/BZ_vol) ∫ [G_AA(q) - G_AB(q)e^{-iq·δ₁}] dq
            # = (1/BZ_vol) ∫ [z - Re(f(q)e^{-iq·δ₁})] / [(z²-|f|²)K] dq
            phase_nn = np.dot(q, delta_1)
            numerator_nn = z - np.real(f * np.exp(-1j * phase_nn))
            sum_nn += numerator_nn / (denom * K_BKT) * dt**3

            # (b) NNN: G_AA(0) - G_AA(nnn_i)
            for idx, r_nnn in enumerate(nnn_unique):
                phase_nnn = np.dot(q, r_nnn)
                numerator_nnn = (1 - np.cos(phase_nnn)) * z
                sum_nnn[idx] += numerator_nnn / (denom * K_BKT) * dt**3

# The integrals are already normalized by dt³ (which approximates 1/BZ_vol × dV)
# But we need to account for BZ_vol:
# ∫ ... dq/BZ_vol = Σ ... × (BZ_vol/N³)/BZ_vol = Σ ... × dt³
# So the sums are already correctly normalized.

# Actually wait: q = t1*b1 + t2*b2 + t3*b3, and the Jacobian is |det[b1,b2,b3]| = BZ_vol
# So ∫ f(q) d³q/(2π)³ = ∫₀¹ f(q(t)) BZ_vol dt1 dt2 dt3 / (2π)³
# Hmm, we want ∫ f(q) d³q / BZ_vol
# = ∫₀¹ f(q(t)) × (BZ_vol × dt³) / BZ_vol = ∫₀¹ f(q(t)) dt³
# So the midpoint sum Σ f(q_i) × dt³ is correct.

G_nn = sum_nn  # = G_AA(0) - G_AB(nn)

print(f"\n  Lattice Green's function results:")
print(f"    G_AA(0) - G_AB(nn)  = {G_nn:.10f}")
print(f"    Expected (1/z = 1/{z}) = {1.0/z:.10f}")
print(f"    Ratio to exact       = {G_nn * z:.10f}")

for idx, (r_nnn, m) in enumerate(zip(nnn_unique, nnn_mults)):
    dist = np.linalg.norm(r_nnn)
    print(f"    G_AA(0) - G_AA(nnn_{idx}) = {sum_nnn[idx]:.10f}  (|r|={dist:.3f}, mult={m})")

# Phase difference variances
sigma2_nn = 2 * G_nn  # ⟨(Δθ_nn)²⟩ = 2(G(0)-G(nn))
print(f"\n  Phase difference variances:")
print(f"    ⟨(Δθ)²⟩_nn   = 2 × G_nn = {sigma2_nn:.10f}")
print(f"    Expected (1/(πK)) = {1.0/(np.pi * K_BKT):.10f}")
print(f"    (lattice σ² = 1/2 from BKT derivation)")

for idx, r_nnn in enumerate(nnn_unique):
    sigma2_nnn = 2 * sum_nnn[idx]
    R_nnn_gauss = np.exp(-sigma2_nnn / 2)
    R_nnn_indep = R0**2  # independent bonds
    dist = np.linalg.norm(r_nnn)
    print(f"    ⟨(Δθ)²⟩_nnn_{idx} = {sigma2_nnn:.10f}  → R_nnn^Gauss = {R_nnn_gauss:.8f} (indep: {R_nnn_indep:.8f})")

# =====================================================================
# Part 4: Models for Two-Vertex Correction to c
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 4: MODELS FOR TWO-VERTEX δc")
print(f"{'='*72}")

delta_c_needed = c_exact - c_binom
print(f"\n  Target: δc = {delta_c_needed:+.10f}")
print(f"  This shifts l by δl = -{delta_c_needed} × V = {-delta_c_needed * V_star:.6e}")
print(f"  Which shifts 1/α by δ(1/α) = {-delta_c_needed * V_star * d_inv_alpha_dl:.6e}")

def report_model(name, delta_c):
    """Report the effect of a δc correction."""
    c_new = c_binom + delta_c
    l_new = 1 - c_new * V_star
    inv_a = inv_alpha_from_l(l_new)
    ppm = (INV_ALPHA_CODATA - inv_a) / INV_ALPHA_CODATA * 1e6
    g_val = compute_g(1.0 / inv_a)
    g_dig = count_digits(g_val, g_measured)
    match = "✓" if abs(ppm) < abs((INV_ALPHA_CODATA - inv_alpha_from_l(l_binom)) / INV_ALPHA_CODATA * 1e6) else "✗"
    print(f"    {name:<55s}  δc={delta_c:+.8f}  1/α={inv_a:.7f}  ppm={ppm:+.5f}  g_dig={g_dig:.1f}  {match}")
    return c_new, l_new, inv_a, ppm, g_dig

print(f"\n  {'Model':<55s}  {'δc':>10s}  {'1/α':>14s}  {'ppm':>10s}  {'g_dig':>6s}")
print(f"  {'-'*100}")

# ── Model A: Shared-bond correlation ──
# Cov(V_A, V_B) = R₀^{2(z-1)} × σ²_bond
# Per vertex: z/2 neighbor pairs (each bond counted once)
# δc_A = z/2 × R₀^{2(z-1)} × σ²_bond / V²
# (normalized by V² since c = Σ/V)
delta_c_A = (z / 2) * cov_VA_VB / V_star
report_model("A: z/2 × Cov(V_A,V_B)/V", delta_c_A)

# ── Model B: Excess second moment ──
# The shared bond gives ⟨cos²θ⟩ instead of R₀² = ⟨cosθ⟩²
# The "excess" per bond pair: cos2_mean / R₀² - 1 = σ²/R₀²
# This multiplies the vertex pair amplitude V²
# Per vertex: z/2 bonds, each contributing
delta_c_B = (z / 2) * (ratio_2nd_moment - 1) * V_star
report_model("B: z/2 × (⟨cos²θ⟩/R₀² - 1) × V", delta_c_B)

# ── Model C: Plaquette-mediated inter-vertex ──
# Hexagonal plaquettes on diamond pass through 6 bonds and 6 vertices.
# A plaquette connecting bond (A,n_i) at vertex A to bond (B,m_j) at vertex B
# through the shared bond creates an inter-vertex correlation.
# Number of plaquettes per bond: 6 (from our lattice computation)
# Each plaquette has 4 "external" bonds with amplitude R₀⁴
# The inter-vertex correction per bond: 6 × R₀⁴
# Per vertex: z/2 bonds
n_plaq_per_bond = 6
delta_c_C = (z / 2) * n_plaq_per_bond * R0**4
report_model("C: z/2 × n_plaq_per_bond × R₀⁴", delta_c_C)

# ── Model D: NNN Green's function correction ──
# The lattice Green's function at NNN distance differs from 2×G_nn.
# This modifies the effective R at NNN distance.
# Use the Gaussian estimate.
G_nnn_0 = sum_nnn[0] if len(sum_nnn) > 0 else 0
sigma2_nnn_0 = 2 * G_nnn_0
R_nnn_gauss = np.exp(-sigma2_nnn_0 / 2)
R_nnn_indep = R0**2
delta_R_nnn = R_nnn_gauss - R_nnn_indep
# The excess NNN coherence modifies the two-vertex cluster amplitude
# Per bond: the excess NNN coherence times (z-1) exits at each vertex
delta_c_D = (z / 2) * (z - 1) * delta_R_nnn / R0**2 * V_star
report_model("D: z/2 × (z-1) × δR_nnn/R₀² × V", delta_c_D)

# ── Model E: Geometric series in shared-bond scattering ──
# A walker scattered at A has probability V/z of going to B through shared bond.
# At B, probability V of being scattered, probability 1/z of going back to A.
# Geometric series: each round trip has amplitude (V/z)².
# The effective vertex blocking enhancement:
# δV_eff = V² × (z-1)/z² × 1/(1 - V²/z²) - V²(z-1)/z²
# ≈ V² × (z-1)/z² × V²/z² (leading correction)
delta_V_E = V_star**2 * (z-1) / z**2 * V_star**2 / z**2
delta_c_E = delta_V_E / V_star * (z / 2)
report_model("E: Markov backscatter (V⁴ order)", delta_c_E)

# ── Model F: Direct from lattice Green's function ──
# The correction to l from the lattice propagator
# The NNN phase variance σ²_nnn vs the independent-bond estimate 2σ²_nn
# gives the lattice correction
sigma2_nn_val = sigma2_nn
for idx in range(len(nnn_unique)):
    s2_nnn = 2 * sum_nnn[idx]
    delta_sigma2 = s2_nnn - 2 * sigma2_nn_val  # deviation from independent bonds
    delta_R = np.exp(-s2_nnn/2) - np.exp(-2*sigma2_nn_val/2)
    m = nnn_mults[idx]
    # Each NNN pair contributes δR to the inter-vertex correlation
    # Correction to c from NNN pairs: (m/2) × δR × (z-1) / V
    delta_c_F_i = (m / 2) * delta_R * (z - 1) / R0
    name = f"F{idx}: NNN_{idx} lattice G correction (mult={m})"
    report_model(name, delta_c_F_i)

# ── Model G: Combined shared-bond + plaquette ──
# The shared bond gives a direct correlation (Model A)
# PLUS the plaquette loops give additional correlations (Model C)
# These are independent contributions at leading order
delta_c_G = delta_c_A + delta_c_C
report_model("G: A + C (bond + plaquette)", delta_c_G)

# ── Model H: Effective medium (self-consistent) ──
# V_eff = V × [1 + z × (Cov/V²)] self-consistently
# This enhances V, which changes c
V_eff_H = V_star * (1 + z * cov_VA_VB / V_star**2)
c_eff_H = (1 - (1 - V_eff_H)**(z-1)) / V_eff_H
delta_c_H = c_eff_H - c_binom
report_model("H: V_eff self-consistent", delta_c_H)

# ── Model I: σ² scaling (variance ratio) ──
# The key quantity is the ratio ⟨cos²θ⟩/R₀².
# For the crossover, V = R₀^z, and the inter-vertex correlation
# replaces one R₀² factor with ⟨cos²θ⟩ in the two-vertex amplitude.
# Per bond: one shared R₀² → ⟨cos²θ⟩, excess = σ²
# The two-vertex correction to the effective V:
# δV = (z/2) × V/R₀² × σ² = (z/2) × R₀^{z-2} × σ²
delta_V_I = (z / 2) * R0**(z-2) * sigma2_bond
delta_c_I = -(z-1) * delta_V_I  # dc/dV = -(z-1) at leading order + correction from (1-V)^(z-2)
# More precisely: l = (1-V_eff)^(z-1), δl = -(z-1)(1-V)^(z-2) δV
# δl = -(z-1)(1-V)^(z-2) × (z/2) × R₀^(z-2) × σ²
delta_l_I = -(z-1) * (1-V_star)**(z-2) * (z/2) * R0**(z-2) * sigma2_bond
# Convert to δc: δl = -δc × V, so δc = -δl/V
delta_c_I_from_l = -delta_l_I / V_star
report_model("I: V_eff = V + z/2 R₀^(z-2) σ² → l", delta_c_I_from_l)

# ── Model J: Exact from CODATA (reference) ──
report_model("EXACT (from CODATA)", delta_c_needed)

# =====================================================================
# Part 5: Sensitivity Analysis — What δ is needed?
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 5: WHAT INTER-VERTEX COUPLING STRENGTH CLOSES THE GAP?")
print(f"{'='*72}")

# The needed δc = 0.0112 corresponds to:
# δl = -δc × V = -0.0112 × 0.00845 = -9.5 × 10⁻⁵
delta_l_needed = -delta_c_needed * V_star
print(f"\n  δl needed = {delta_l_needed:.6e}")
print(f"  δ(1/α) needed = {delta_l_needed * d_inv_alpha_dl:.6e}")

# If δl = -(z-1)(1-V)^(z-2) × δV_eff, what δV_eff is needed?
delta_V_needed = -delta_l_needed / ((z-1) * (1-V_star)**(z-2))
print(f"\n  δV_eff needed = {delta_V_needed:.6e}")
print(f"  δV/V = {delta_V_needed/V_star:.6f}")

# What inter-vertex coupling ε (in V_eff = V × (1+ε)) is needed?
epsilon_needed = delta_V_needed / V_star
print(f"  ε = δV/V = {epsilon_needed:.6f}")
print(f"  This means V_eff/V = {1+epsilon_needed:.6f}")

# How does this compare to the bond variance?
print(f"\n  Comparison with bond statistics:")
print(f"    σ²_bond = {sigma2_bond:.6f}")
print(f"    σ²/R₀² = {sigma2_bond/R0**2:.6f}")
print(f"    z × σ²/R₀² = {z * sigma2_bond/R0**2:.6f}")
print(f"    ε_needed = {epsilon_needed:.6f}")
print(f"    Ratio ε/(z σ²/R₀²) = {epsilon_needed / (z * sigma2_bond/R0**2):.4f}")

# =====================================================================
# Part 6: Transfer Matrix for Dumbbell
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 6: EXACT DUMBBELL TRANSFER MATRIX")
print(f"{'='*72}")

# Model the dumbbell as a Markov chain.
# A walker enters vertex A from one external bond.
# At each vertex: with probability V, "scattered" (coherent vertex) →
#   re-emitted uniformly to one of z bonds (including back).
# With probability (1-V), "transparent" → continues to a random exit
#   among the (z-1) other bonds (excluding entry).
#
# States: (vertex, entry_bond_type)
#   - A_ext: at vertex A, entered from external bond of A
#   - A_int: at vertex A, entered from internal bond (from B)
#   - B_ext: at vertex B, entered from external bond of B
#   - B_int: at vertex B, entered from internal bond (from A)
#   - EXIT: exited through external bond (absorbing)
#
# For the crossover, we track whether the walker exits "cleanly"
# (without being scattered) or not.
#
# Actually, for the LCE, what matters is the probability that the
# walker traverses the cluster without encountering a coherent vertex.
# This is the "transparency" of the cluster.

# For a single vertex (star), the transparency for the walker to
# pass from entry to a SPECIFIC exit is:
#   T_1exit = (1-V)/(z-1)  (probability of no scattering × choosing this exit)
# Total transparency (to ANY exit):
#   T_star = (1-V)  (probability of no scattering)

# For the dumbbell, the walker enters A from external bond a₁.
# Clean paths (no scattering at either vertex):
#   A(a₁) → exit through a₂, a₃: prob (1-V_A) × 2/(z-1) = (1-V) × 2/3 each
#   A(a₁) → B(internal) → exit through b₁,b₂,b₃:
#     prob (1-V_A) × 1/(z-1) × (1-V_B) × (z-1)/(z-1)
#     = (1-V) × 1/3 × (1-V)
#     = (1-V)² / 3
# Total clean exit probability:
#   T_clean = (1-V) × 2/3 + (1-V)² / 3
#           = (1-V)/3 × [2 + (1-V)]
#           = (1-V)/3 × (3 - V)
#           = (1-V)(1 - V/3)

T_clean_A = (1 - V_star) * (1 - V_star/3)
T_star_A = (1 - V_star)  # single vertex transparency

# But we also need to account for scattered walkers that return.
# A walker scattered at A can go to B (prob 1/z), and if it then
# exits B cleanly, it doesn't count as "clean" for the transparency.
# OR: a walker that passes A cleanly, enters B, gets scattered at B,
# and is sent back to A (prob V/z), where it might pass A cleanly again.
#
# The key insight: "transparency" of the dumbbell means the walker
# passes through without being scattered at ANY vertex it encounters.
# If it enters A, passes to B, gets scattered at B — that's a failure
# even if it eventually exits through B's external bonds.
#
# So the full transparency includes:
#   T_dumbbell = P(no scatter at A) × P(no scatter at B | passed through (A,B) bond)
#              = (1-V) × [(z-2)/(z-1) + 1/(z-1) × (1-V)]
# Wait, let me redo this.
#
# Walker enters A from external bond a₁.
# At A, not scattered (prob 1-V). Then goes to one of A's (z-1) exits:
#   - With prob (z-2)/(z-1): goes to external bond a₂ or a₃ → EXIT (clean)
#   - With prob 1/(z-1): goes to internal bond → enters B
#     At B, not scattered (prob 1-V) → exits through one of B's (z-1) exits:
#       - All z-1 are external bonds of B (since B's only internal bond leads to A,
#         but the walker entered from A, so it exits to z-1 other bonds of B)
#       Wait: B has z=4 bonds: {(B,A), b₁, b₂, b₃}. Walker entered from (B,A).
#       So z-1 = 3 exits from B, all external. All clean exits.
#       Prob: 1 (given no scatter at B)
# So: T_dumbbell = (1-V) × [(z-2)/(z-1) + 1/(z-1) × (1-V)]
#               = (1-V) × [(z-2 + 1-V)/(z-1)]
#               = (1-V) × [(z-1-V)/(z-1)]
#               = (1-V) × [1 - V/(z-1)]

# For z=4:
T_dumbbell = (1 - V_star) * (1 - V_star / (z - 1))

print(f"\n  Dumbbell transparency (clean exit probability):")
print(f"    T_star (single vertex)  = (1-V) = {1-V_star:.10f}")
print(f"    T_dumbbell (two vertex) = (1-V)(1-V/(z-1)) = {T_dumbbell:.10f}")
print(f"    T_dumbbell / T_star     = {T_dumbbell / (1-V_star):.10f}")
print(f"    = 1 - V/(z-1) = {1 - V_star/(z-1):.10f}")

# The effective crossover from the dumbbell picture:
# Each of the (z-1) exit channels from A either goes to:
#   - An external bond (z-2 of them): transparency (1-V)
#   - The internal bond to B (1 of them): transparency (1-V)(1-V/(z-1))...
#
# Wait, I need to reconsider. The crossover l = (1-V)^(z-1) means
# each exit channel independently has transparency (1-V), and we need
# ALL (z-1) to be transparent. For the modified picture:
#
# Of the (z-1) exits at vertex A:
#   (z-2) go to external bonds → transparency (1-V) each
#   1 goes to vertex B → transparency (1-V) × (1-V) × ... (depends on B's structure)
#
# The exit toward B has transparency:
#   T_to_B = (1-V_A)(entry from A side) × T_through_B
# But T_through_B depends on what we're computing.
#
# Actually, let me think about this differently.
#
# In the single-vertex picture:
#   l = ∏_{exit_i at v} (1-V) = (1-V)^(z-1)
#
# In the two-vertex picture, the exit channel from A toward B gets modified.
# For that channel, the walker must pass through A (prob 1-V_A) AND through B
# (prob depends on B's structure, and includes possible back-scattering from B).
#
# The effective transparency of the exit channel A→B:
# T_{A→B} = (1-V_A) × [prob walker doesn't return from B to A and trigger scatter]
#
# A walker that passes A toward B: at B, it's scattered with prob V_B.
# If scattered, sent back to A with prob 1/z (through the internal bond).
# At A, scattered with prob V_A, sent to B with prob 1/z. Etc.
#
# The probability of the walker returning to A (ever) from B:
# P_return = V_B/z × [1 + V_A/z × V_B/z + (V_A/z × V_B/z)² + ...]
#          = V_B/z × 1/(1 - V_A V_B/z²)
#          = V_B/(z - V_A V_B/z)
#
# If the walker returns to A, it then faces A's scattering again (prob V_A).
# A returned walker that scatters at A with prob V_A exits through a random
# bond, with prob (z-1)/z of exiting through an external bond (lost to BKT)
# and prob 1/z of going back to B (another round trip).
#
# This is getting complicated. Let me set up the exact Markov chain.

# States for the Markov chain:
# 0: At vertex A, not yet scattered (entering from internal or external)
# 1: At vertex B, not yet scattered
# 2: Clean exit through A's external bond (absorbing) — "VP"
# 3: Clean exit through B's external bond (absorbing) — "VP"
# 4: Scattered exit (absorbing) — "BKT"
#
# Actually, for the transparency calculation, we just need:
# P(walker exits cleanly through any external bond without EVER being scattered)
# vs P(walker is scattered at some vertex).

# Let me define it as a linear system.
# Let p_A = probability of clean exit, given walker is at A (from internal bond)
# Let p_B = probability of clean exit, given walker is at B (from internal bond)
#
# At vertex A (entered from internal bond, i.e., from B):
#   - Not scattered (prob 1-V): goes to one of (z-1) exits:
#     (z-2) external → clean exit with prob (z-2)/(z-1)
#     1 internal (back to B) → continue with prob p_B from there × 1/(z-1)
#   - Scattered (prob V): NOT a clean exit → prob 0 contribution to transparency
# BUT: if scattered, the walker still exists. It's re-emitted:
#   - To external bond: prob (z-1)/z → scattered exit (not clean)
#   - To internal bond (to B): prob 1/z → now at B but "scattered"
#     At B, if not scattered: exits cleanly (but the walker WAS already scattered at A)
#     So this doesn't count as "clean" for transparency purposes.
#
# So for TRANSPARENCY (probability of ZERO scattering events):
# p_A = (1-V) × [(z-2)/(z-1) + 1/(z-1) × p_B]
# p_B = (1-V) × [(z-2)/(z-1) + 1/(z-1) × p_A]
#
# By symmetry: p_A = p_B = p
# p = (1-V) × [(z-2)/(z-1) + p/(z-1)]
# p × [1 - (1-V)/(z-1)] = (1-V)(z-2)/(z-1)
# p × [(z-1-(1-V))/(z-1)] = (1-V)(z-2)/(z-1)
# p × (z-2+V) = (1-V)(z-2)
# p = (1-V)(z-2) / (z-2+V)

p_internal = (1 - V_star) * (z - 2) / (z - 2 + V_star)

# For a walker entering A from an EXTERNAL bond:
# p_ext = (1-V) × [(z-2)/(z-1) + 1/(z-1) × p_internal]
p_external_A = (1 - V_star) * ((z-2)/(z-1) + p_internal/(z-1))

# For comparison, single vertex: p = 1-V (just pass through)
# The effective transparency per exit channel:
# In single vertex: each exit contributes factor (1-V)
# In two-vertex: the exit toward B contributes (1-V) × p_internal instead of (1-V)²

# The l in the full lattice:
# For each vertex A:
#   - (z-2) exit channels toward second-shell vertices → transparency (1-V) each
#     [but these vertices also have their own corrections... this becomes infinite series]
#   - 1 exit channel toward nearest neighbor B → transparency (1-V) × p_B/(1-V) = p_B
#     where p_B includes back-scattering from B

# Actually: for the LINKED-CLUSTER correction, we need:
# l_dumbbell = transparency of dumbbell cluster
# l_two_stars = (1-V)^(z-1) × (1-V)^(z-1)  [independent stars]
# δl_2vtx = l_dumbbell - l_two_stars ???
# No — the LCE weight is computed on the cluster subgraph.

# Let me reconsider. The linked cluster expansion for l works as follows:
# On the infinite lattice, l = transparency of the infinite lattice.
# l = exp[ Σ_G W(G) × (lattice embedding number per vertex) ]
# where W(G) is the linked cluster weight for connected graph G.
#
# For the star graph (single vertex + z bonds):
# W(star) = ln(1-V)  [each vertex contributes ln(1-V) independently]
# Per exit channel: ln(1-V)
# For z-1 exits per vertex: (z-1)ln(1-V) = ln(1-V)^(z-1)
#
# Wait, that gives l = exp((z-1)ln(1-V)) = (1-V)^(z-1). Correct!
#
# For the dumbbell (two vertices + shared bond):
# W(dumbbell) = ln(T_dumbbell_2vtx) - 2 × W(star)
# where T_dumbbell_2vtx is the transparency of the two-vertex cluster
# computed ON the cluster subgraph (not the full lattice).
#
# On the dumbbell cluster: vertex A has z=4 bonds (3 external + 1 to B),
# vertex B has z=4 bonds (3 external + 1 to A).
# The transparency of this cluster:
# T_cluster = probability that walker enters from one side and exits cleanly
#
# But what does "transparency of the cluster" mean exactly?
# It's the contribution to ln(l) from the cluster.

# I think the correct approach is:
# l = (1-V)^(z-1) × (1 + δ_2vtx)
# where δ_2vtx comes from the dumbbell correction.
#
# From the Markov chain: the exit toward B has effective transparency
# p_internal instead of (1-V). So:
#
# l_corrected = ∏_{exits of A} T_exit
# where T_exit = (1-V) for external exits, and T_exit = p_internal for internal (to B)
#
# But EVERY exit of A goes to some neighbor, and every neighbor is a dumbbell.
# So l_corrected = ∏_{all exits} T_exit = p_internal^(z-1)
# Wait no — only ONE exit per vertex goes to each specific neighbor.
# Each vertex has z-1 exits, each going to a different neighbor.
# Each such exit leads to a dumbbell. So:
# l_corrected = p_internal^(z-1)
#
# Hmm, but that double-counts. Each dumbbell is shared between two vertices.
# Per vertex, z/2 dumbbells. Each dumbbell modifies one exit.
# But each vertex has z-1 exits...
#
# I think the correct formula is:
# ln(l_corrected) = (z-1) × ln(1-V) + (z-1) × [ln(p_internal) - ln(1-V)]
# Wait, that gives l = p_internal^(z-1), ignoring the dumbbell sharing.
#
# The LCE correction per vertex from dumbbells:
# (z exits per vertex, z/2 unique dumbbells per vertex)
# Each dumbbell affects ONE exit of A and ONE exit of B.
# So per vertex, (z-1) exits are modified by (z-1) dumbbells (one per exit).
# But each dumbbell is shared, so the correction per vertex is (z-1)/2.
# No wait — each vertex has z-1 exits, each exit goes to a distinct neighbor,
# each forms a dumbbell. There are z-1 dumbbells per vertex,
# but each is shared with the neighbor vertex. So per vertex, (z-1)/2 dumbbell corrections.
#
# Hmm, this is the standard LCE subtlety. Let me use the standard formula:
# l = exp(Σ_cluster L_cluster × n_cluster_per_vertex)
# where L_cluster is the linked cluster weight and n is the embedding number.
#
# For single bond (star exit):
# L_bond = ln(1-V)
# n_bond = z-1 (exits per vertex)
# ln(l) = (z-1) ln(1-V)
#
# For dumbbell (two vertices sharing one bond):
# L_dumbbell = correction to ln(transparency) from the two-vertex cluster
#            = ln(p_internal) - ln(1-V)  (excess over single-bond)
# n_dumbbell = z/2 per vertex (unique dumbbells per vertex)?
# Or n_dumbbell = (z-1) per vertex (one per exit)?
#
# Wait: each vertex has z-1 exits. Each exit goes to a neighbor.
# The dumbbell correction modifies the transparency of that exit channel.
# So each vertex has z-1 dumbbell corrections, one per exit.
# But each dumbbell is shared between two vertices, so:
# n_dumbbell_per_vertex = (z-1) × (1/2) = (z-1)/2
# (the 1/2 avoids double counting)
#
# No, that's not right either. In the LCE, the correction from a dumbbell
# is computed ONCE for the dumbbell and then distributed to both vertices.
# The standard formula: each dumbbell contributes its weight to the
# extensive quantity, and we divide by N_vertices.
#
# For the extensive quantity ln L_total = N_vertices × ln l:
# ln L_total = N_vertices × (z-1) × ln(1-V)  [single vertex]
#            + N_bonds × Δ_dumbbell  [dumbbell correction]
# where N_bonds = N_vertices × z/2.
#
# So: ln l = (z-1) ln(1-V) + (z/2) × Δ_dumbbell
# where Δ_dumbbell is the linked-cluster weight per bond (dumbbell).
#
# The dumbbell weight Δ_dumbbell:
# On the dumbbell cluster (2 vertices, 1 shared bond + 6 external):
# The transparency of the dumbbell as a whole = ?
#
# Hmm, I think I need to define this more carefully.
# Let me compute it as follows:
#
# The total "blocking rate" per vertex is:
# Γ = (1 - l) / V = c (the crossover coefficient)
# In the single-vertex picture: Γ = [1 - (1-V)^(z-1)] / V = c_binom
# The dumbbell correction to Γ is δc.
#
# From the Markov chain, the exit toward B has effective transparency
# p_internal = (1-V)(z-2)/(z-2+V) instead of (1-V).
# The "excess blocking" from this one exit is:
# δblocking = (1-V) - p_internal = (1-V) - (1-V)(z-2)/(z-2+V)
#           = (1-V) × [1 - (z-2)/(z-2+V)]
#           = (1-V) × V/(z-2+V)

delta_blocking_per_exit = (1 - V_star) * V_star / (z - 2 + V_star)

# Each vertex has (z-1) exits, but only z of them go to a dumbbell
# (all z bonds form dumbbells). Wait, (z-1) exits per vertex, each
# going to a distinct neighbor. So (z-1) dumbbells per vertex.
# But the correction per dumbbell at this vertex is delta_blocking_per_exit.
# The correction per bond (shared between two vertices) is delta_blocking_per_exit
# (each vertex gets the correction for its own exit toward the shared bond).
# No double-counting issue because the correction at A is about A's exit toward B.

# Total correction to blocking rate at vertex A:
# δ(1-l) = (z-1) × δblocking_per_exit × (perturbative correction)
# But this overcounts: the dumbbell effect modifies (z-1) exits,
# but we need to account for the fact that only ONE of the (z-1) exits
# goes through any specific dumbbell.

# Actually, each vertex has (z-1) exit channels. In the single-vertex
# picture, each exit has transparency (1-V). In the two-vertex picture,
# each exit's transparency is modified by the backscattering from the
# neighbor. So:
#
# l_corrected = ∏_{exit i=1}^{z-1} T_i
# where T_i = p_internal for exit toward neighbor i
# (each neighbor is the other end of a dumbbell)
#
# l_corrected = p_internal^(z-1)

l_dumbbell_full = p_internal**(z-1)
c_dumbbell_full = (1 - l_dumbbell_full) / V_star

print(f"\n  Markov chain results:")
print(f"    p_internal = (1-V)(z-2)/(z-2+V) = {p_internal:.10f}")
print(f"    (1-V) = {1-V_star:.10f}")
print(f"    p_internal/(1-V) = {p_internal/(1-V_star):.10f}")
print(f"    = (z-2)/(z-2+V) = {(z-2)/(z-2+V_star):.10f}")
print(f"\n    l_binom = (1-V)^(z-1) = {l_binom:.15f}")
print(f"    l_dumbbell = p^(z-1) = {l_dumbbell_full:.15f}")
print(f"    l_exact (CODATA) = {l_exact:.15f}")
print(f"    δl = l_dumbbell - l_binom = {l_dumbbell_full - l_binom:.6e}")
print(f"    δl_needed = l_exact - l_binom = {l_exact - l_binom:.6e}")
print(f"\n    c_binom    = {c_binom:.10f}")
print(f"    c_dumbbell = {c_dumbbell_full:.10f}")
print(f"    c_exact    = {c_exact:.10f}")
print(f"    δc_dumbbell = {c_dumbbell_full - c_binom:+.10f}")
print(f"    δc_needed   = {c_exact - c_binom:+.10f}")

inv_a_dumbbell = inv_alpha_from_l(l_dumbbell_full)
ppm_dumbbell = (INV_ALPHA_CODATA - inv_a_dumbbell) / INV_ALPHA_CODATA * 1e6
g_dumbbell = compute_g(1.0 / inv_a_dumbbell)
gdig_dumbbell = count_digits(g_dumbbell, g_measured)

print(f"\n    1/α(dumbbell) = {inv_a_dumbbell:.10f}")
print(f"    ppm = {ppm_dumbbell:+.6f}")
print(f"    g digits = {gdig_dumbbell:.2f}")

# =====================================================================
# Part 7: Combined result — single + two-vertex
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 7: COMBINED RESULT")
print(f"{'='*72}")

# The exact formula from the Markov chain:
# p = (1-V)(z-2)/(z-2+V)
# l = p^(z-1) = [(1-V)(z-2)/(z-2+V)]^(z-1)
#
# Let's also try a "half-correction" model where the dumbbell
# correction is weighted by 1/2 (since each bond is shared)

l_half = ((1-V_star) * ((1-V_star) + (z-2)/(z-2+V_star)) / 2)**(z-1)
# Actually this doesn't make sense. Let me try the proper weighting.
# Each exit channel at vertex A goes to ONE specific neighbor.
# The transparency of that exit is p_internal (from the Markov chain).
# There's no 1/2 factor because we're computing the transparency at A.

# What if we use l = (1-V)^(z-2) × p_internal instead?
# This says: (z-2) exits are "bare" (1-V) and 1 exit is "dressed" (p_internal)
# But that doesn't make sense either — ALL exits go to neighbors.

# The correct formula should be l = p_internal^(z-1) since every exit
# goes to a neighbor with backscattering.

# Let's also try different powers to see which matches CODATA:
print(f"\n  Testing different numbers of 'dressed' exits:")
print(f"  {'k (dressed exits)':<20s}  {'l':<18s}  {'1/α':<14s}  {'ppm':<10s}  {'c':<12s}")
print(f"  {'-'*80}")

for k_dressed in range(z):
    if k_dressed == 0:
        l_k = (1 - V_star)**(z-1)
    else:
        l_k = (1 - V_star)**(z-1-k_dressed) * p_internal**k_dressed
    c_k = (1 - l_k) / V_star
    inv_a_k = inv_alpha_from_l(l_k)
    ppm_k = (INV_ALPHA_CODATA - inv_a_k) / INV_ALPHA_CODATA * 1e6
    label = f"k={k_dressed}"
    if k_dressed == 0:
        label += " (binomial)"
    elif k_dressed == z-1:
        label += " (all dressed)"
    print(f"  {label:<20s}  {l_k:.15f}  {inv_a_k:.7f}  {ppm_k:+.5f}  {c_k:.8f}")

print(f"  {'CODATA':<20s}  {l_exact:.15f}  {INV_ALPHA_CODATA:.7f}  {0:+.5f}  {c_exact:.8f}")

# =====================================================================
# Part 8: Interpolation — what fraction of exits are "dressed"?
# =====================================================================

print(f"\n{'='*72}")
print(f"  PART 8: INTERPOLATION")
print(f"{'='*72}")

# Find k_eff (fractional) that matches CODATA
# l = (1-V)^(z-1-k) × p^k = (1-V)^(z-1) × (p/(1-V))^k
# ln l = (z-1)ln(1-V) + k × ln(p/(1-V))
# Solve for k:
k_eff = (np.log(l_exact) - (z-1)*np.log(1-V_star)) / np.log(p_internal/(1-V_star))
print(f"\n  k_eff (fractional dressed exits) = {k_eff:.6f}")
print(f"  Out of z-1 = {z-1} total exits")
print(f"  Fraction dressed: {k_eff/(z-1):.4f}")
print(f"  This means {k_eff:.2f} of the {z-1} exit channels at each vertex")
print(f"  experience the two-vertex backscattering correction.")

# Physical interpretation:
# k_eff ≈ 1 would mean one exit channel is fully dressed (the nearest neighbor)
# k_eff ≈ z-1 = 3 would mean all exits are fully dressed
# k_eff ≈ 0 would mean no inter-vertex effects (binomial limit)

# =====================================================================
# Part 9: Summary
# =====================================================================

print(f"\n{'='*72}")
print(f"  SUMMARY")
print(f"{'='*72}")

# Best result
print(f"""
  Two-vertex Markov chain (all exits dressed):
    p = (1-V)(z-2)/(z-2+V) = {p_internal:.10f}
    l = p^(z-1)            = {l_dumbbell_full:.15f}
    1/α                    = {inv_a_dumbbell:.10f}
    ppm                    = {ppm_dumbbell:+.6f}
    g digits               = {gdig_dumbbell:.2f}

  Comparison:
    Binomial (single vtx): 1/α = {inv_alpha_from_l(l_binom):.10f}  ({(INV_ALPHA_CODATA - inv_alpha_from_l(l_binom))/INV_ALPHA_CODATA*1e6:+.5f} ppm)
    Dumbbell (two vtx):    1/α = {inv_a_dumbbell:.10f}  ({ppm_dumbbell:+.5f} ppm)
    CODATA:                1/α = {INV_ALPHA_CODATA:.10f}

  The dumbbell correction moves from {(INV_ALPHA_CODATA - inv_alpha_from_l(l_binom))/INV_ALPHA_CODATA*1e6:+.5f} ppm to {ppm_dumbbell:+.5f} ppm.

  Effective dressed-exit fraction needed for CODATA: k_eff/{z-1} = {k_eff/(z-1):.4f}
""")

# =====================================================================
# Part 10: Plot
# =====================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Two-Vertex Linked-Cluster Correction for α', fontsize=14, fontweight='bold')

# (a) Convergence: 1/α at each level
ax = axes[0, 0]
levels = ['BKT', 'Binom\n(1-V)³', 'Dumbbell\np^(z-1)', 'CODATA']
vals = [inv_alpha_BKT, inv_alpha_from_l(l_binom), inv_a_dumbbell, INV_ALPHA_CODATA]
colors_a = ['#e74c3c', '#3498db', '#2ecc71', '#333333']
bars = ax.bar(range(4), vals, color=colors_a, edgecolor='k', linewidth=0.5)
ax.axhline(INV_ALPHA_CODATA, color='k', ls='--', lw=1)
ax.set_xticks(range(4))
ax.set_xticklabels(levels, fontsize=8)
ax.set_ylabel('1/α')
ax.set_ylim(137.031, 137.037)
ax.set_title('(a) 1/α convergence')

# (b) Residual ppm
ax = axes[0, 1]
ppms = [(INV_ALPHA_CODATA - v)/INV_ALPHA_CODATA*1e6 for v in vals[:3]]
abs_ppms = [abs(p) for p in ppms]
bar_colors = ['red' if p > 0 else 'blue' for p in ppms]
ax.bar(range(3), abs_ppms, color=bar_colors, edgecolor='k')
ax.set_yscale('log')
ax.set_xticks(range(3))
ax.set_xticklabels(['BKT', 'Binomial', 'Dumbbell'], fontsize=9)
ax.set_ylabel('|Residual| (ppm)')
ax.set_title('(b) |Residual| (ppm)')
for i, p in enumerate(ppms):
    sign = '+' if p > 0 else '−'
    ax.text(i, abs_ppms[i] * 1.3, f'{sign}{abs_ppms[i]:.3f}', ha='center', fontsize=8)

# (c) p vs (1-V) — the dumbbell transparency
ax = axes[0, 2]
V_range = np.logspace(-4, -0.5, 200)
p_range = (1 - V_range) * (z-2) / (z-2 + V_range)
oneV_range = 1 - V_range
ax.plot(V_range, oneV_range, 'b--', label='(1-V) [single vtx]', lw=1.5)
ax.plot(V_range, p_range, 'g-', label='p = (1-V)(z-2)/(z-2+V)\n[dumbbell]', lw=2)
ax.axvline(V_star, color='gray', ls=':', label=f'V = R₀⁴ = {V_star:.4f}')
ax.set_xlabel('V = R₀^z')
ax.set_ylabel('Exit transparency')
ax.set_xscale('log')
ax.legend(fontsize=7)
ax.set_title('(c) Exit transparency vs V')

# (d) k_eff scan
ax = axes[1, 0]
k_range = np.linspace(0, z-1, 100)
l_range = (1-V_star)**(z-1-k_range) * p_internal**k_range
c_range = (1 - l_range) / V_star
ax.plot(k_range, c_range, 'b-', lw=2)
ax.axhline(c_exact, color='r', ls='--', label=f'c_exact = {c_exact:.4f}')
ax.axhline(c_binom, color='gray', ls=':', label=f'c_binom = {c_binom:.4f}')
ax.axvline(k_eff, color='g', ls='--', label=f'k_eff = {k_eff:.2f}')
ax.set_xlabel('k (number of dressed exits)')
ax.set_ylabel('Crossover coefficient c')
ax.legend(fontsize=7)
ax.set_title('(d) c vs dressed exits')

# (e) Bond coherence distribution
ax = axes[1, 1]
theta = np.linspace(-np.pi, np.pi, 500)
p_theta = np.exp(K_BKT * np.cos(theta)) / (2 * np.pi * I0_K)
ax.plot(theta * 180 / np.pi, p_theta, 'b-', lw=1.5, label='P(Δθ) von Mises')
ax.axvline(0, color='gray', ls=':', lw=0.5)
ax.fill_between(theta * 180 / np.pi, 0, p_theta, alpha=0.1)
ax.set_xlabel('Bond phase difference Δθ (degrees)')
ax.set_ylabel('P(Δθ)')
ax.set_title(f'(e) Bond distribution at K_BKT={K_BKT:.3f}')
ax.text(0.95, 0.95, f'⟨cosθ⟩ = R₀ = {R0:.4f}\n⟨cos²θ⟩ = {cos2_mean:.4f}\nσ² = {sigma2_bond:.4f}',
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.legend(fontsize=8)

# (f) Summary text
ax = axes[1, 2]
ax.axis('off')
summary = (
    "TWO-VERTEX LCE RESULT\n"
    "━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Single vertex (binomial):\n"
    f"  l = (1-V)^(z-1)\n"
    f"  c = {c_binom:.7f}\n"
    f"  1/α = {inv_alpha_from_l(l_binom):.7f}\n"
    f"  ppm = {(INV_ALPHA_CODATA - inv_alpha_from_l(l_binom))/INV_ALPHA_CODATA*1e6:+.4f}\n\n"
    "Two vertex (dumbbell):\n"
    f"  p = (1-V)(z-2)/(z-2+V)\n"
    f"  l = p^(z-1)\n"
    f"  c = {c_dumbbell_full:.7f}\n"
    f"  1/α = {inv_a_dumbbell:.7f}\n"
    f"  ppm = {ppm_dumbbell:+.4f}\n\n"
    f"CODATA: c = {c_exact:.7f}\n"
    f"k_eff = {k_eff:.3f} of {z-1} exits"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'two_vertex_lce.png'), dpi=150, bbox_inches='tight')
print(f"\n  Figure saved: {OUTPUT_DIR}/two_vertex_lce.png")
print("\n  DONE.")
