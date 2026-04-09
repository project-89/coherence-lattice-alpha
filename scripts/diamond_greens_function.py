#!/usr/bin/env python3
"""
Lattice Green's function on the 3D diamond lattice.

The Debye-Waller route to n = 1/√e:
If the nearest-neighbor phase variance in the spin-wave approximation is
  ⟨(Δθ)²⟩_nn = 2η = 1/2
then exp(-⟨Δθ²⟩/2) = exp(-1/4) for R₀ (single site)
or exp(-⟨Δθ²⟩) = exp(-1/2) = 1/√e for the exponent n (two-site correlator).

In spin-wave theory:
  ⟨(Δθ)²⟩_nn = (1/K) × [G(0,0) - G(0,nn)]
where G is the lattice Green's function of the graph Laplacian.

For the d=3 diamond lattice (z=4, bipartite, FCC sublattices):
  G_diff = G(0,0) - G(nearest neighbor)

If G_diff = K_BKT/2 = 1/π, then ⟨Δθ²⟩ = 1/(πK) at general K,
and at K = K_BKT = 2/π: ⟨Δθ²⟩ = 1/(π·2/π) = 1/2. ✓

So the KEY QUESTION is: G_diff = 1/π on the diamond lattice?
Or more generally, what is G_diff for the diamond lattice?

The diamond lattice Green's function can be computed:
1. Analytically via Fourier transform (momentum space)
2. Numerically on finite lattices
3. By relation to the FCC Green's function (Watson integrals)
"""
import numpy as np
from scipy.integrate import dblquad, tplquad
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'canon', 'v4_2026-02-14'))
import importlib
mod = importlib.import_module("184b_lt183_vortex_ring_g2_convergence")

print("=" * 75)
print("LATTICE GREEN'S FUNCTION ON THE 3D DIAMOND LATTICE")
print("=" * 75)
print()

# === 1. ANALYTICAL: Diamond lattice in momentum space ===
print("=" * 75)
print("1. ANALYTICAL: MOMENTUM-SPACE GREEN'S FUNCTION")
print("=" * 75)
print()

# The diamond lattice has 2 atoms per unit cell (FCC sublattices A and B).
# The bond vectors from A to B are the 4 tetrahedral vectors:
#   δ₁ = a/4 (1,1,1)
#   δ₂ = a/4 (1,-1,-1)
#   δ₃ = a/4 (-1,1,-1)
#   δ₄ = a/4 (-1,-1,1)
# where a is the cubic lattice constant.
#
# The hopping matrix in k-space is:
#   H(k) = [ 0      f(k) ]
#           [ f*(k)  0    ]
# where f(k) = Σ_μ exp(ik·δ_μ)
#
# For the Laplacian: L = z·I - A, where A is the adjacency matrix.
# In k-space: L(k) = [ z    -f(k) ]
#                     [-f*(k)  z   ]
#
# Eigenvalues: λ± = z ± |f(k)|
# where |f(k)|² = |Σ exp(ik·δ_μ)|²

# For our diamond lattice (d=3, simplex bond vectors):
# The simplex vectors in our code are:
deltas = mod.make_simplex_deltas(3)
print("  Simplex bond vectors (d=3):")
for i, d in enumerate(deltas):
    print(f"    δ_{i} = ({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f})")
nn_dist = np.linalg.norm(deltas[0])
print(f"  |δ| = {nn_dist:.6f}")
print()

# The Green's function G(R) = (1/N) Σ_k G(k) exp(ikR)
# For the lattice Laplacian: G(k) = L(k)^{-1}
# G(0,0) = diagonal element: G_AA(0,0) or G_BB(0,0)
# G(0,nn) = off-diagonal: G_AB(0,δ₁)
#
# G_diff = G_AA(0,0) - G_AB(0,δ₁)
#
# In the 2×2 block structure:
# L(k)^{-1} = (1/(z²-|f|²)) × [ z     f(k) ]
#                                 [ f*(k)  z   ]
#
# So G_AA(k) = z/(z²-|f(k)|²)
#    G_AB(k) = f(k)/(z²-|f(k)|²)
#
# G_AA(0,0) = (1/V_BZ) ∫_BZ z/(z²-|f(k)|²) dk
# G_AB(0,δ₁) = (1/V_BZ) ∫_BZ f(k)·exp(-ik·δ₁)/(z²-|f(k)|²) dk
#
# Wait, need to be more careful. G_AB(R_A, R_B) where R_B = R_A + δ₁.
# Actually for the real-space Green's function:
# G_AB(R) = (1/V_BZ) ∫ f(k) exp(ikR) / (z² - |f(k)|²) dk
#
# For the nearest neighbor: we need G(site 0, site nn) where site 0 is
# on sublattice A and nn is on sublattice B at displacement δ₁.
#
# G(0, δ₁) = G_AB(0) = (1/V_BZ) ∫ f(k)/(z² - |f(k)|²) dk
# (No phase factor because the nn is in the same unit cell)

# Actually, let me think about this differently.
# For a bipartite lattice with z = 4 (d=3 diamond):
# The graph Laplacian has eigenvalues λ± = z ± |f(k)|
#
# The Green's function of the Laplacian (resolvent at ω=0) diverges
# because λ=0 is an eigenvalue (k=0, |f(0)|=z).
# We need the REGULARIZED Green's function:
# G_reg = (L + εI)^{-1} and take ε→0, or equivalently
# G_diff = G_AA(0,0) - G_AB(0,δ₁) which is finite.
#
# G_diff = (1/V_BZ) ∫ [z/(z²-|f|²) - f(k)/(z²-|f|²)] dk
#        = (1/V_BZ) ∫ (z-f(k))/(z²-|f(k)|²) dk
#        = (1/V_BZ) ∫ (z-f(k))/((z-|f|)(z+|f|)) dk
#
# Hmm, f(k) is complex in general, so z-f(k) ≠ z-|f(k)|.
# Let me reconsider.
#
# For a REAL lattice (with real hopping), f(k) = Σ cos(k·δ_μ) + i Σ sin(k·δ_μ)
# |f(k)|² = [Σ cos(k·δ_μ)]² + [Σ sin(k·δ_μ)]²
#
# G_AA - G_AB for the NEAREST NEIGHBOR (connected by the Laplacian):
# Actually, L_{AB} = -A_{AB}, where A is the adjacency.
# The Laplacian acts as: (LΨ)_A = z Ψ_A - Σ_μ Ψ_{B,δ_μ}
#
# For the Laplacian Green's function on the FULL graph:
# (G_diff)_nn ≡ G(i,i) - G(i,j) where j is a nearest neighbor
#
# This is related to the effective resistance: R_eff(i,j) = G(i,i) + G(j,j) - 2G(i,j)
# For a bipartite lattice with equal sublattices: G(i,i) = G(j,j)
# So R_eff = 2(G(i,i) - G(i,j)) = 2·G_diff

# Let me compute G_diff numerically on the finite diamond lattice.

print("=" * 75)
print("2. NUMERICAL: G_diff ON FINITE DIAMOND LATTICE")
print("=" * 75)
print()

for L in [4, 6, 8, 10, 12]:
    positions, N, ei, ej, _ = mod.make_d_diamond_adjacency(3, L)

    # Build adjacency matrix
    n_bonds = len(ei)
    data = np.ones(2 * n_bonds)
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Laplacian: L = D - A where D = z*I for regular graph
    z = 4  # coordination
    # Actually, boundary sites may have fewer neighbors
    degree = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix((degree, (range(N), range(N))), shape=(N, N))
    Lap = D - A

    # Regularized Green's function: (L + εI)^{-1}
    eps = 1e-6
    Lap_reg = Lap + eps * speye(N)

    # Solve for the Green's function column at site 0:
    # G(:, 0) = (L + εI)^{-1} e_0
    e0 = np.zeros(N)
    e0[0] = 1.0
    G_col = spsolve(Lap_reg, e0)

    # G(0,0) = G_col[0]
    G_00 = G_col[0]

    # Find nearest neighbors of site 0
    nn_sites = ej[ei == 0].tolist() + ei[ej == 0].tolist()
    nn_sites = list(set(nn_sites))

    # G(0, nn) averaged over nearest neighbors
    G_nn = np.mean([G_col[j] for j in nn_sites])
    G_diff = G_00 - G_nn

    # The spin-wave variance: ⟨(Δθ)²⟩ = G_diff / K (at coupling K)
    # At K = K_BKT = 2/π:
    K_BKT = 2.0 / np.pi
    dtheta_sq = G_diff / K_BKT

    # Does this equal 2η = 1/2?
    target_dtheta = 0.5

    n_nn = len(nn_sites)
    print(f"  L={L:2d}  N={N:5d}  nn={n_nn}  G(0,0)={G_00:.6f}  "
          f"G(0,nn)={G_nn:.6f}  G_diff={G_diff:.6f}  "
          f"⟨Δθ²⟩/K={G_diff:.6f}  ⟨Δθ²⟩@BKT={dtheta_sq:.6f}  "
          f"target={target_dtheta}")

print()

# === 3. Compare with known results ===
print("=" * 75)
print("3. COMPARISON WITH KNOWN LATTICE GREEN'S FUNCTIONS")
print("=" * 75)
print()

# For REFERENCE, known lattice Green's function values:
# Simple cubic (3D, z=6): G_diff = 1/(6·0.505) ≈ 0.330
# Actually, for simple cubic: G(0,0) - G(0,e_x) = 1/(2d) = 1/6 (in units where
# the hopping is 1 and the Laplacian is 2d - Σ_nn).
# Wait, this isn't right either. Let me use the standard result.
#
# The lattice Green's function for simple cubic:
# G(0) = (1/(2π)³) ∫∫∫ dk / (6 - 2cos k_x - 2cos k_y - 2cos k_z)
# This is Watson's integral W₃ = 0.505462... (per (2π)³)
# Actually G(0) itself diverges; the DIFFERENCE is finite.
#
# G(0) - G(e_x) = (1/(2π)³) ∫∫∫ (1-cos k_x) dk / (6-2Σcos kᵢ)
# For simple cubic this equals 1/6 exactly? No...
# Actually: for a random walk on Z³, the return probability is P(0)=G(0)·(2d).
# And the one-step difference is: G(0)-G(e₁) = 1/(2d) on Z^d for d≥3.
# Wait, that's not right either.

# Let me just look at the numerical results and see what G_diff converges to.

print("  Looking at L-dependence of G_diff:")
print()

# Compute for more L values to see convergence
G_diffs = []
for L in [4, 6, 8, 10, 12, 14, 16]:
    try:
        positions, N, ei, ej, _ = mod.make_d_diamond_adjacency(3, L)
        n_bonds = len(ei)
        data = np.ones(2 * n_bonds)
        rows = np.concatenate([ei, ej])
        cols = np.concatenate([ej, ei])
        A = csr_matrix((data, (rows, cols)), shape=(N, N))
        degree = np.array(A.sum(axis=1)).flatten()
        D = csr_matrix((degree, (range(N), range(N))), shape=(N, N))
        Lap = D - A
        eps = 1e-8
        Lap_reg = Lap + eps * speye(N)
        e0 = np.zeros(N)
        e0[0] = 1.0
        G_col = spsolve(Lap_reg, e0)
        nn_sites = list(set(ej[ei == 0].tolist() + ei[ej == 0].tolist()))
        G_diff = G_col[0] - np.mean([G_col[j] for j in nn_sites])
        G_diffs.append((L, N, G_diff))
        dtheta = G_diff / (2/np.pi)
        print(f"  L={L:2d}  N={N:5d}  G_diff={G_diff:.8f}  "
              f"⟨Δθ²⟩@BKT={dtheta:.8f}  vs 1/2={0.5}")
    except Exception as e:
        print(f"  L={L}: {e}")
        break

print()

# === 4. Analytical integral for the diamond lattice ===
print("=" * 75)
print("4. ANALYTICAL: MOMENTUM-SPACE INTEGRAL")
print("=" * 75)
print()

# For the diamond lattice (our simplex construction), the structure factor is:
# f(k) = Σ_{μ=0}^{3} exp(ik·δ_μ)
# |f(k)|² = 4 + 2 Σ_{μ<ν} cos(k·(δ_μ - δ_ν))
#
# The 6 bond-difference vectors δ_μ - δ_ν span the FCC lattice.
#
# G_diff = (1/V_BZ) ∫ (1 - Re[f(k)]·exp(-ik·δ₀)/|f(k)|) / (z² - |f(k)|²) × z dk
#
# Actually, let me derive this more carefully.
# On the bipartite diamond lattice:
# L = [z·I_A   -H_AB]
#     [-H_BA   z·I_B]
#
# where (H_AB)_{R,R'} = Σ_μ δ(R'-R-δ_μ) (hopping from B to A sublattice)
#
# In k-space: H_AB(k) = f(k) = Σ_μ exp(ik·δ_μ)
#
# L(k) = [z   -f(k)]
#         [-f*(k)  z]
#
# L(k)^{-1} = [z    f(k)] / (z² - |f|²)
#              [f*(k)  z ]
#
# G_AA(0,0) = (1/V_BZ) ∫ z/(z²-|f|²) dk  (sublattice A self-Green's fn)
# G_AB(0,δ₁) = (1/V_BZ) ∫ f(k)exp(-ik·δ₁)/(z²-|f|²) dk
#             = (1/V_BZ) ∫ [Σ_μ exp(ik·(δ_μ-δ₁))]/(z²-|f|²) dk
#
# Actually, for the nn Green's function, site 0 is on sublattice A,
# and its neighbor is on sublattice B at position δ₁.
# G(0_A, δ₁_B) = G_AB(δ₁) = (1/V_BZ) ∫ f*(k)·exp(ik·δ₁)/(z²-|f|²) dk
#
# Hmm, I'm getting confused with signs. Let me just compute numerically.
# The momentum-space integral should match the real-space result.

# Use the simplex vectors from the code:
d0, d1, d2, d3 = deltas[0], deltas[1], deltas[2], deltas[3]

def f_sq(kx, ky, kz):
    """Compute |f(k)|² for the diamond lattice."""
    f_real = sum(np.cos(kx*d[0] + ky*d[1] + kz*d[2]) for d in deltas)
    f_imag = sum(np.sin(kx*d[0] + ky*d[1] + kz*d[2]) for d in deltas)
    return f_real**2 + f_imag**2

# The BZ for our lattice depends on the reciprocal lattice.
# For a general lattice, (1/V_BZ)∫ → (1/(2π)³)∫ in Cartesian coordinates
# if the lattice has unit cell volume V_cell.
#
# Actually for our diamond lattice construction, the sites sit on
# an FCC-like lattice. The reciprocal lattice and BZ are complicated.
#
# Simpler approach: compute the integral over a large cube in k-space,
# exploiting the periodicity. The lattice is periodic with some period.
# For the simplex construction with edge length |δ|, the reciprocal
# lattice vectors are determined by δ_μ - δ_ν.

# Let me instead compute G_diff numerically from a dense k-mesh.
# The integrand is: [z - Re(f(k)·exp(-ik·δ₀)) × z/|f(k)|] / (z²-|f|²)
# Wait, I should be more careful.

# G_diff = G_AA(0) - G_AB(δ₀)
#        = (1/V_BZ) ∫ [z - f*(k)exp(ik·δ₀)] / (z²-|f|²) dk

# For our lattice, let's use a large k-grid and average.
# The key: |f(k=0)|² = (Σ 1)² = 16 = z², so the integrand diverges at k=0.
# We need to handle this carefully.

# Actually, z²-|f(0)|² = 16-16 = 0 → divergent! This is the standard
# zero mode of the Laplacian. For G_diff, the divergence cancels:
# Near k=0: f(k) ≈ z - k²σ²/2 + ...
# z - f*(k)exp(ik·δ₀) ≈ z - (z - k²σ²/2)·(1 + ik·δ₀ - ...)
# ≈ k²σ²/2 - iz·k·δ₀ + ...
# And z²-|f|² ≈ 2z·k²σ²/2 = z·k²σ²
# So the ratio → 1/(2z) + ... which is finite.

# Let me compute this numerically with a dense grid.
# Use the convention where the BZ is (-π,π)³ in some reciprocal coordinates.

# First, find the periodicity. The bond vectors are:
# δ_μ for μ=0..3. The lattice vectors of the diamond are:
# a₁ = δ₀ - δ₁, a₂ = δ₀ - δ₂, a₃ = δ₀ - δ₃
a1 = deltas[0] - deltas[1]
a2 = deltas[0] - deltas[2]
a3 = deltas[0] - deltas[3]
print(f"  FCC lattice vectors:")
print(f"    a₁ = {a1}")
print(f"    a₂ = {a2}")
print(f"    a₃ = {a3}")

# Reciprocal lattice vectors: b_i · a_j = 2π δ_ij
V_cell = np.abs(np.dot(a1, np.cross(a2, a3)))
b1 = 2 * np.pi * np.cross(a2, a3) / V_cell
b2 = 2 * np.pi * np.cross(a3, a1) / V_cell
b3 = 2 * np.pi * np.cross(a1, a2) / V_cell
print(f"  Unit cell volume: {V_cell:.6f}")
print(f"  Reciprocal vectors:")
print(f"    b₁ = {b1}")
print(f"    b₂ = {b2}")
print(f"    b₃ = {b3}")
print()

# Compute G_diff via numerical integration over the BZ
# k = s₁b₁ + s₂b₂ + s₃b₃ with s_i ∈ [0,1)
# (1/V_BZ) ∫ → ∫₀¹ ds₁ ds₂ ds₃ (BZ volume normalizes to 1 in these coords)

Nk = 100  # k-points per direction
s_vals = np.linspace(0.5/Nk, 1 - 0.5/Nk, Nk)  # avoid k=0 exactly

z = 4  # coordination

G_AA_sum = 0.0
G_AB_sum = 0.0  # for nearest neighbor δ₀
count = 0

for s1 in s_vals:
    for s2 in s_vals:
        for s3 in s_vals:
            k = s1 * b1 + s2 * b2 + s3 * b3

            # f(k) = Σ exp(ik·δ_μ)
            f_real = sum(np.cos(np.dot(k, d)) for d in deltas)
            f_imag = sum(np.sin(np.dot(k, d)) for d in deltas)
            f_sq_val = f_real**2 + f_imag**2

            denom = z**2 - f_sq_val
            if abs(denom) < 1e-12:
                continue

            G_AA_sum += z / denom
            # G_AB(δ₀) needs f*(k)exp(ik·δ₀):
            # f*(k) = f_real - i·f_imag
            # exp(ik·δ₀) = cos(k·δ₀) + i·sin(k·δ₀)
            phase = np.dot(k, deltas[0])
            exp_ikd = complex(np.cos(phase), np.sin(phase))
            f_star = complex(f_real, -f_imag)
            G_AB_val = (f_star * exp_ikd).real / denom
            G_AB_sum += G_AB_val

            count += 1

G_AA_k = G_AA_sum / count
G_AB_k = G_AB_sum / count
G_diff_k = G_AA_k - G_AB_k

K_BKT = 2.0 / np.pi
dtheta_sq = G_diff_k / K_BKT

print(f"  Momentum-space result (Nk={Nk}):")
print(f"    G_AA(0) = {G_AA_k:.8f}")
print(f"    G_AB(δ₀) = {G_AB_k:.8f}")
print(f"    G_diff = {G_diff_k:.8f}")
print(f"    ⟨(Δθ)²⟩ = G_diff/K = {dtheta_sq:.8f}")
print(f"    Target (2η = 1/2): 0.50000000")
print(f"    Gap: {(dtheta_sq - 0.5)/0.5:+.4%}")
print()

# === 5. The effective resistance interpretation ===
print("=" * 75)
print("5. EFFECTIVE RESISTANCE")
print("=" * 75)
print()

# The effective resistance between nn sites is R_eff = 2·G_diff
# For the diamond lattice: R_eff = 2·G_diff
R_eff = 2 * G_diff_k
print(f"  R_eff (nn) = 2·G_diff = {R_eff:.8f}")
print(f"  R_eff / z = {R_eff/z:.8f}")
print(f"  1/z = {1/z:.8f} = 0.25")
print()

# For a z-regular lattice in d dimensions, R_eff(nn) = 2/(z·(1-return_prob))
# For d≥3, the random walk is transient, and the return probability < 1.

# === 6. Putting it together ===
print("=" * 75)
print("6. DEBYE-WALLER INTERPRETATION")
print("=" * 75)
print()

print(f"  On the 3D diamond lattice (z=4):")
print(f"    G_diff (momentum space) = {G_diff_k:.8f}")
print(f"    ⟨(Δθ)²⟩ at K_BKT = G_diff/K_BKT = {dtheta_sq:.8f}")
print()

if abs(dtheta_sq - 0.5) / 0.5 < 0.05:
    print(f"  *** CLOSE TO 1/2! ***")
    print(f"  If ⟨Δθ²⟩ = 1/2 exactly:")
    print(f"    exp(-⟨Δθ²⟩/2) = exp(-1/4) = {np.exp(-0.25):.6f}")
    print(f"    (This would be the DW factor for the ORDER PARAMETER R₀)")
    print(f"    But R₀(2/π) = 0.303202, not {np.exp(-0.25):.6f}")
    print(f"")
    print(f"  The DW interpretation for the EXPONENT n:")
    print(f"    n = exp(-⟨Δθ²⟩) = exp(-1/2) = 1/√e ✓")
    print(f"    (Two-site correlator: squared DW factor)")
else:
    print(f"  ⟨Δθ²⟩ = {dtheta_sq:.6f}, NOT close to 1/2.")
    print(f"  The simple DW route does NOT explain 1/√e on this lattice.")
    print()
    print(f"  What ⟨Δθ²⟩ value would we need?")
    print(f"    For n = 1/√e: ⟨Δθ²⟩ = 1/2 = 0.5000")
    print(f"    For n = 2R₀:  ⟨Δθ²⟩ = -ln(2R₀) = {-np.log(2*0.303202):.6f}")
    print(f"    Actual: {dtheta_sq:.6f}")
    print(f"    Ratio actual/(1/2) = {dtheta_sq/0.5:.6f}")

print()
print("=" * 75)
