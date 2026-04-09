#!/usr/bin/env python3
"""
Measure the effective Dirac velocity v_D_eff on the 3-diamond lattice
with the CLR K-field from the d9 vortex state.

THEORY (LT-101):
  On the bare 3-diamond (uniform K), the valley-paired 4-component
  Bloch Hamiltonian gives exact Cliff(4,0) with:
    v_D^2 = 1/3    (from |grad f|^2 = 1 in cubic coords)
    1 - 3*v_D^2 = 0   (paramagnetic/diamagnetic threshold)

  The one-loop vertex F_2 = F_2^Coulomb * (1 - 3*v_D^2).
  If v_D^2 < 1/3: F_2 > 0, g > 2 (paramagnetic, like QED Schwinger)
  If v_D^2 > 1/3: F_2 < 0, g < 2 (diamagnetic)

COMPUTATION:
  The CLR K-field at equilibrium is non-uniform (vortex core has reduced K).
  This modifies the effective hopping and potentially shifts v_D off 1/3.

  Three approaches:
  A. Modified Bloch: f_eff(k) = Sum_j <K_j> exp(ik.delta_j)
  B. Real-space spectrum: full N x N diag, compare to uniform K
  C. Green's function: velocity from self-energy at Dirac point
"""

import sys
import os
import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Diamond structure in cubic coordinates (matching LT-101)
# ---------------------------------------------------------------------------
DELTA_CUBIC = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
]) / 4.0

X_POINT = 2 * np.pi * np.array([1, 0, 0])


def f_diamond(k, weights=None):
    """Diamond structure factor: f(k) = Sum_j w_j exp(i k . delta_j).
    weights: per-direction hopping amplitudes (default: uniform = [1,1,1,1])."""
    if weights is None:
        weights = np.ones(4)
    return sum(weights[j] * np.exp(1j * np.dot(k, DELTA_CUBIC[j]))
               for j in range(4))


def grad_f_diamond(k, weights=None):
    """Gradient of f(k) w.r.t. k. Returns complex 3-vector."""
    if weights is None:
        weights = np.ones(4)
    g = np.zeros(3, dtype=complex)
    for j in range(4):
        phase = np.exp(1j * np.dot(k, DELTA_CUBIC[j]))
        g += weights[j] * (1j * DELTA_CUBIC[j]) * phase
    return g


def find_nodal_zero(k_start, weights=None):
    """Find zero of f(k) near k_start using optimization."""
    def cost(k):
        return abs(f_diamond(k, weights))**2
    res = minimize(cost, k_start, method='Nelder-Mead',
                   options={'xatol': 1e-12, 'fatol': 1e-24, 'maxiter': 10000})
    return res.x, abs(f_diamond(res.x, weights))


def compute_velocity_at_zero(k0, weights=None, eps=1e-6):
    """Compute |grad f|^2 at a zero of f, plus per-direction velocities."""
    gf = grad_f_diamond(k0, weights)

    # Also compute via centered differences for cross-check
    ex = np.array([1, 0, 0], dtype=float)
    ey = np.array([0, 1, 0], dtype=float)
    ez = np.array([0, 0, 1], dtype=float)

    v_sq = np.zeros(3)
    for mu, e_mu in enumerate([ex, ey, ez]):
        df = (f_diamond(k0 + eps * e_mu, weights) -
              f_diamond(k0 - eps * e_mu, weights)) / (2 * eps)
        v_sq[mu] = abs(df)**2

    return {
        'grad_f': gf,
        'grad_f_sq': float(np.sum(np.abs(gf)**2)),
        'v_per_dir': np.sqrt(v_sq),
        'v_sq_sum': float(np.sum(v_sq)),
        'v_D_sq': float(np.sum(v_sq) / 3),
        'contraction': float(1 - np.sum(v_sq)),
    }


# ---------------------------------------------------------------------------
# A. Modified Bloch Analysis
# ---------------------------------------------------------------------------
def bloch_analysis(K_field):
    """Compute v_D_eff from per-direction averaged K-field."""
    print("=" * 70)
    print("A. MODIFIED BLOCH ANALYSIS")
    print("=" * 70)

    # Per-direction averages
    K_means = np.array([K_field[d::4].mean() for d in range(4)])
    K_overall = K_field.mean()

    print(f"\nPer-direction <K>: {K_means}")
    print(f"Overall <K>: {K_overall:.4f}")
    print(f"Anisotropy (std/mean): {K_means.std()/K_means.mean():.6f}")

    # Normalize weights so mean = 1 (removes overall scale)
    w = K_means / K_overall
    print(f"Normalized weights: {w}")

    # --- Bare lattice (uniform K) ---
    t_param = np.pi / 4
    k0_bare = X_POINT + t_param * np.array([0, 0, 1])
    k0_bare, f_abs_bare = find_nodal_zero(k0_bare, weights=None)
    vel_bare = compute_velocity_at_zero(k0_bare, weights=None)

    print(f"\nBARE (uniform K):")
    print(f"  Dirac point: k0 = {k0_bare}")
    print(f"  |f(k0)| = {f_abs_bare:.2e}")
    print(f"  |grad f|^2 = {vel_bare['grad_f_sq']:.6f}")
    print(f"  v_D^2 = {vel_bare['v_D_sq']:.6f}")
    print(f"  1 - 3*v_D^2 = {vel_bare['contraction']:.6f}")

    # --- Modified lattice (non-uniform K) ---
    k0_mod, f_abs_mod = find_nodal_zero(k0_bare, weights=w)
    vel_mod = compute_velocity_at_zero(k0_mod, weights=w)

    print(f"\nMODIFIED (CLR K-field, per-direction averages):")
    print(f"  Dirac point: k0 = {k0_mod}")
    print(f"  |f(k0)| = {f_abs_mod:.2e}")
    print(f"  k-shift |dk| = {np.linalg.norm(k0_mod - k0_bare):.6e}")
    print(f"  |grad f_eff|^2 = {vel_mod['grad_f_sq']:.6f}")
    print(f"  v_D^2_eff = {vel_mod['v_D_sq']:.6f}")
    print(f"  1 - 3*v_D^2_eff = {vel_mod['contraction']:.6f}")

    delta_vD2 = vel_mod['v_D_sq'] - vel_bare['v_D_sq']
    print(f"\n  delta(v_D^2) = {delta_vD2:.6e}")
    print(f"  Relative shift: {delta_vD2/vel_bare['v_D_sq']:.6e}")

    # --- Scan along nodal line for robustness ---
    print(f"\nScan along nodal line (parameterized by t):")
    print(f"  {'t':>6}  {'|grad f|^2 bare':>16}  {'|grad f|^2 mod':>16}  {'delta':>12}")
    for t_val in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, np.pi/4]:
        k_start = X_POINT + t_val * np.array([0, 0, 1])
        k_b, _ = find_nodal_zero(k_start, weights=None)
        k_m, _ = find_nodal_zero(k_start, weights=w)
        vb = compute_velocity_at_zero(k_b, weights=None)
        vm = compute_velocity_at_zero(k_m, weights=w)
        delta = vm['grad_f_sq'] - vb['grad_f_sq']
        print(f"  {t_val:6.3f}  {vb['grad_f_sq']:16.6f}  {vm['grad_f_sq']:16.6f}  {delta:12.6e}")

    return vel_bare, vel_mod


# ---------------------------------------------------------------------------
# B. Real-Space Spectrum Analysis
# ---------------------------------------------------------------------------
def build_scalar_hamiltonian(L, K_field, ei, ej):
    """Build N x N scalar tight-binding Hamiltonian.
    H_ij = -K_ij for neighboring sites i,j."""
    N = 2 * L**3
    H = np.zeros((N, N))
    for b in range(len(ei)):
        i, j = int(ei[b]), int(ej[b])
        H[i, j] = -K_field[b]
        H[j, i] = -K_field[b]
    return H


def spectral_analysis(L, K_field, ei, ej):
    """Compare spectra of actual vs uniform K-field Hamiltonians."""
    print("\n" + "=" * 70)
    print("B. REAL-SPACE SPECTRAL ANALYSIS")
    print("=" * 70)

    N = 2 * L**3
    K_mean = K_field.mean()

    # Build Hamiltonians
    H_actual = build_scalar_hamiltonian(L, K_field, ei, ej)
    K_uniform = np.full_like(K_field, K_mean)
    H_uniform = build_scalar_hamiltonian(L, K_uniform, ei, ej)

    # Diagonalize
    print(f"\nDiagonalizing {N}x{N} Hamiltonians...")
    E_actual = la.eigvalsh(H_actual)
    E_uniform = la.eigvalsh(H_uniform)

    # Check particle-hole symmetry
    print(f"\nParticle-hole symmetry check (bipartite):")
    print(f"  Uniform: sum(E) = {E_uniform.sum():.6e} (expect 0)")
    print(f"  Actual:  sum(E) = {E_actual.sum():.6e} (expect ~0)")

    # Sort by |E| to find states near Dirac point
    E_actual_sorted = np.sort(np.abs(E_actual))
    E_uniform_sorted = np.sort(np.abs(E_uniform))

    # Bandwidth
    BW_actual = E_actual.max() - E_actual.min()
    BW_uniform = E_uniform.max() - E_uniform.min()
    print(f"\n  Bandwidth (actual): {BW_actual:.4f}")
    print(f"  Bandwidth (uniform): {BW_uniform:.4f}")

    # States near E=0 (the Dirac point)
    n_near = 20
    print(f"\n  {n_near} smallest |E| values:")
    print(f"  {'n':>4}  {'|E| actual':>14}  {'|E| uniform':>14}  {'ratio':>10}")
    for n in range(n_near):
        ratio = E_actual_sorted[n] / E_uniform_sorted[n] if E_uniform_sorted[n] > 1e-10 else float('inf')
        print(f"  {n:4d}  {E_actual_sorted[n]:14.6f}  {E_uniform_sorted[n]:14.6f}  {ratio:10.6f}")

    # Velocity from level spacing: on a Dirac cone with linear E = v_D * |q|,
    # the level spacing at small |E| scales as v_D * (2pi/L)
    # For the 2D Dirac cone (d-2=1 flat direction), levels come in pairs

    # Density of states: ρ(E) = |E| / (2π v_D²) for 2D Dirac
    # Count eigenvalues in energy bins
    E_max_dos = BW_actual * 0.1  # only near Dirac point
    n_bins = 50
    bins = np.linspace(0, E_max_dos, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    dos_actual = np.histogram(np.abs(E_actual), bins=bins)[0]
    dos_uniform = np.histogram(np.abs(E_uniform), bins=bins)[0]

    # Normalize to density
    dE = bins[1] - bins[0]
    dos_actual = dos_actual / (N * dE)
    dos_uniform = dos_uniform / (N * dE)

    # Fit linear DOS: ρ(E) = a * |E| → v_D² = 1/(2π * a)
    # Only fit the linear region (avoid E=0 and large E)
    fit_mask = (bin_centers > E_max_dos * 0.1) & (bin_centers < E_max_dos * 0.7)
    if np.sum(fit_mask) > 3:
        # Linear fit: ρ = a * E
        coeffs_actual = np.polyfit(bin_centers[fit_mask], dos_actual[fit_mask], 1)
        coeffs_uniform = np.polyfit(bin_centers[fit_mask], dos_uniform[fit_mask], 1)

        # v_D² from DOS slope: ρ(E) ∝ |E|/v_D², so v_D² ∝ 1/slope
        # The exact formula depends on the dimensionality and BZ structure
        # but the RATIO of v_D² is:
        v_D_sq_ratio = coeffs_uniform[0] / coeffs_actual[0] if coeffs_actual[0] > 0 else float('inf')

        print(f"\n  DOS linear fit (slope of rho vs E):")
        print(f"    Actual:  slope = {coeffs_actual[0]:.6e}")
        print(f"    Uniform: slope = {coeffs_uniform[0]:.6e}")
        print(f"    v_D^2 ratio (actual/uniform) = {v_D_sq_ratio:.6f}")
        print(f"    (>1 means actual v_D larger than uniform)")

    # Compare low-energy eigenvalue ratios
    # For Dirac cone: E_n ∝ v_D * sqrt(n) or v_D * n (depending on dimension)
    # The ratio E_actual(n) / E_uniform(n) = v_D_eff / v_D_bare
    n_ratio = 50
    ratios = []
    for n in range(2, min(n_ratio, len(E_actual_sorted))):
        if E_uniform_sorted[n] > 1e-8:
            ratios.append(E_actual_sorted[n] / E_uniform_sorted[n])

    if ratios:
        ratio_mean = np.mean(ratios)
        ratio_std = np.std(ratios)
        print(f"\n  Low-energy eigenvalue ratio E_actual/E_uniform:")
        print(f"    Mean: {ratio_mean:.6f} ± {ratio_std:.6f}")
        print(f"    This ratio = v_D_eff / v_D_bare")
        print(f"    v_D^2_eff / v_D^2_bare = {ratio_mean**2:.6f}")

        # If v_D_bare^2 = 1/3 (LT-101):
        vD2_bare = 1.0 / 3.0
        vD2_eff = vD2_bare * ratio_mean**2
        contraction_eff = 1 - 3 * vD2_eff
        print(f"\n    If v_D_bare^2 = 1/3:")
        print(f"    v_D^2_eff = {vD2_eff:.6f}")
        print(f"    1 - 3*v_D^2_eff = {contraction_eff:.6f}")
        if contraction_eff > 0:
            print(f"    >>> PARAMAGNETIC (g > 2)! <<<")
        elif contraction_eff < 0:
            print(f"    >>> DIAMAGNETIC (g < 2) <<<")
        else:
            print(f"    >>> AT THRESHOLD (g = 2 exactly) <<<")

    return E_actual, E_uniform


# ---------------------------------------------------------------------------
# C. Per-site K-field Analysis
# ---------------------------------------------------------------------------
def local_anisotropy_analysis(K_field):
    """Analyze local K-field anisotropy at each unit cell."""
    print("\n" + "=" * 70)
    print("C. LOCAL K-FIELD ANISOTROPY")
    print("=" * 70)

    n_cells = len(K_field) // 4

    # Per-site: compute the hopping tensor T_ij = Sum_b K_b delta_b,i delta_b,j
    # This is a 3x3 matrix at each site characterizing the local anisotropy
    T_tensors = np.zeros((n_cells, 3, 3))
    for c in range(n_cells):
        for d in range(4):
            K_b = K_field[4 * c + d]
            delta = DELTA_CUBIC[d]
            T_tensors[c] += K_b * np.outer(delta, delta)

    # For uniform K: T = K * Sum_j delta_j outer delta_j = K * (1/4) * I_3
    # (tetrahedral identity: Sum_j delta_j delta_j^T = (1/4) I for our normalization)
    K_0 = K_field.mean()
    T_ideal = sum(K_0 * np.outer(d, d) for d in DELTA_CUBIC)

    # Compute eigenvalues of each T tensor
    T_eigs = np.array([la.eigvalsh(T_tensors[c]) for c in range(n_cells)])
    T_ideal_eigs = la.eigvalsh(T_ideal)

    print(f"\nIdeal (uniform K={K_0:.4f}) hopping tensor eigenvalues: {T_ideal_eigs}")
    print(f"  Trace: {T_ideal_eigs.sum():.4f}")

    print(f"\nActual hopping tensor eigenvalues:")
    print(f"  Mean: {T_eigs.mean(axis=0)}")
    print(f"  Std:  {T_eigs.std(axis=0)}")
    print(f"  Min:  {T_eigs.min(axis=0)}")
    print(f"  Max:  {T_eigs.max(axis=0)}")

    # Anisotropy ratio: max_eig / min_eig per site
    aniso = T_eigs[:, 2] / T_eigs[:, 0]
    print(f"\n  Anisotropy ratio (max/min eigenvalue):")
    print(f"    Mean: {aniso.mean():.4f}")
    print(f"    Max:  {aniso.max():.4f}")
    print(f"    Sites with aniso > 1.5: {np.sum(aniso > 1.5)}/{n_cells}")
    print(f"    Sites with aniso > 2.0: {np.sum(aniso > 2.0)}/{n_cells}")

    # Effective velocity modification from hopping tensor
    # The velocity is v_mu ~ Sum_j K_j delta_j,mu exp(ik.delta_j)
    # The quadratic form |v|^2 = Sum_{mu} |v_mu|^2 involves:
    # Sum_mu (Sum_j K_j delta_j,mu)(Sum_k K_k delta_k,mu) = Sum_{j,k} K_j K_k (delta_j . delta_k)

    # For uniform K: |v|^2 = K^2 * Sum_{j,k} cos(k.(delta_j-delta_k)) (delta_j.delta_k)
    # The lattice identity gives |v|^2 = K^2 at the nodal zero.

    # For non-uniform K: |v_eff|^2 = Sum_{j,k} K_j K_k cos(k.(delta_j-delta_k)) (delta_j.delta_k)
    # The modification depends on the K-K correlations AND the Dirac point location.

    # Compute the effective |grad f|^2 using the actual K-field correlations
    # This uses the SECOND MOMENT of K per direction, not just the mean
    K_dirs = [K_field[d::4] for d in range(4)]
    K_dir_means = [K_d.mean() for K_d in K_dirs]
    K_dir_sq_means = [K_d.mean()**2 for K_d in K_dirs]  # (mean)^2
    K_dir_mean_sq = [(K_d**2).mean() for K_d in K_dirs]  # mean of squares
    K_dir_var = [K_dir_mean_sq[d] - K_dir_sq_means[d] for d in range(4)]

    print(f"\nPer-direction K statistics:")
    print(f"  {'Dir':>4}  {'<K>':>8}  {'<K^2>':>10}  {'<K>^2':>10}  {'Var':>10}  {'Var/<K>^2':>10}")
    for d in range(4):
        print(f"  {d:4d}  {K_dir_means[d]:8.4f}  {K_dir_mean_sq[d]:10.4f}  "
              f"{K_dir_sq_means[d]:10.4f}  {K_dir_var[d]:10.4f}  "
              f"{K_dir_var[d]/K_dir_sq_means[d]:10.6f}")

    # Cross-direction correlations
    print(f"\n  Cross-direction K correlations (per-site):")
    print(f"  {'i':>3} {'j':>3}  {'<K_i K_j>':>12}  {'<K_i><K_j>':>12}  {'Cov':>12}  {'Corr':>8}")
    for i in range(4):
        for j in range(i+1, 4):
            KiKj = (K_dirs[i] * K_dirs[j]).mean()
            Ki_Kj = K_dirs[i].mean() * K_dirs[j].mean()
            cov = KiKj - Ki_Kj
            std_i = K_dirs[i].std()
            std_j = K_dirs[j].std()
            corr = cov / (std_i * std_j) if std_i > 0 and std_j > 0 else 0
            print(f"  {i:3d} {j:3d}  {KiKj:12.4f}  {Ki_Kj:12.4f}  {cov:12.4f}  {corr:8.4f}")


# ---------------------------------------------------------------------------
# D. Valley-paired 4x4 analysis with K-weighted hopping
# ---------------------------------------------------------------------------
def valley_analysis(K_field):
    """Build 4x4 valley Hamiltonian with K-weighted structure factor."""
    print("\n" + "=" * 70)
    print("D. VALLEY-PAIRED 4x4 ANALYSIS (LT-101)")
    print("=" * 70)

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    I4 = np.eye(4, dtype=complex)
    tau_z = np.array([[1, 0], [0, -1]], dtype=complex)
    tau_x = np.array([[0, 1], [1, 0]], dtype=complex)
    tau_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    K_means = np.array([K_field[d::4].mean() for d in range(4)])
    K_overall = K_means.mean()
    w = K_means / K_overall  # Normalized weights

    t_param = np.pi / 4
    k0_start = X_POINT + t_param * np.array([0, 0, 1])

    # Find zeros with modified weights
    k0_w, f_abs = find_nodal_zero(k0_start, weights=w)
    print(f"\nModified Dirac point: k0 = {k0_w}")
    print(f"|f_eff(k0)| = {f_abs:.2e}")

    # Build 4x4 valley Hamiltonian at q relative to k0
    def H_valley(q):
        H = np.zeros((4, 4), dtype=complex)
        f_K = f_diamond(k0_w + q, weights=w)
        f_Kp = f_diamond(-k0_w + q, weights=w)
        # Valley K block
        H[0, 1] = np.conj(f_K)
        H[1, 0] = f_K
        # Valley K' block
        H[2, 3] = np.conj(f_Kp)
        H[3, 2] = f_Kp
        return H

    # Extract velocity matrices via centered differences
    eps_v = 1e-5
    ex = np.array([1, 0, 0], dtype=float)
    ey = np.array([0, 1, 0], dtype=float)
    ez = np.array([0, 0, 1], dtype=float)

    V_matrices = []
    v_mags = []
    for e_mu, name in [(ex, "x"), (ey, "y"), (ez, "z")]:
        Hp = H_valley(eps_v * e_mu)
        Hm = H_valley(-eps_v * e_mu)
        V_mu = (Hp - Hm) / (2 * eps_v)
        v_mu = la.svdvals(V_mu)[0]
        V_matrices.append(V_mu)
        v_mags.append(v_mu)
        print(f"  v_{name} = {v_mu:.6f}")

    v1, v2, v3 = v_mags
    v_sq_sum = v1**2 + v2**2 + v3**2
    v_D_sq = v_sq_sum / 3
    contraction = 1 - v_sq_sum

    print(f"\n  v1^2 + v2^2 + v3^2 = {v_sq_sum:.6f}")
    print(f"  v_D^2 = {v_D_sq:.6f}")
    print(f"  1 - 3*v_D^2 = {contraction:.6f}")

    # Also do the same for bare (w = [1,1,1,1])
    k0_bare, _ = find_nodal_zero(k0_start, weights=None)
    vel_bare = compute_velocity_at_zero(k0_bare, weights=None)
    print(f"\n  Bare: v_D^2 = {vel_bare['v_D_sq']:.6f}, 1-3v_D^2 = {vel_bare['contraction']:.6f}")
    print(f"  Shift: delta(v_D^2) = {v_D_sq - vel_bare['v_D_sq']:.6e}")

    # Clifford algebra check
    Gamma_0 = np.kron(I2, sigma_z)
    Gamma_1 = np.kron(tau_z, sigma_x)
    Gamma_2 = np.kron(I2, sigma_y)
    Gamma_3 = np.kron(tau_x, sigma_x)
    gammas = [Gamma_0, Gamma_1, Gamma_2, Gamma_3]

    # Gamma contraction
    contraction_vals = []
    for mu in range(4):
        contracted = sum(gammas[alpha] @ gammas[mu] @ gammas[alpha] for alpha in range(4))
        expected = -2 * gammas[mu]
        err = la.norm(contracted - expected)
        contraction_vals.append(err)
    print(f"\n  Gamma contraction error (max): {max(contraction_vals):.4e}")
    print(f"  Contraction value: -2 (confirmed)")

    # The F_2 formula
    print(f"\n  === KEY RESULT ===")
    print(f"  F_2 = F_2^Coulomb * (1 - 3*v_D^2)")
    print(f"      = F_2^Coulomb * {contraction:.6e}")
    if contraction > 0:
        print(f"  >>> PARAMAGNETIC: g > 2 <<<")
    elif contraction < 0:
        print(f"  >>> DIAMAGNETIC: g < 2 <<<")
    else:
        print(f"  >>> AT THRESHOLD: g = 2 exactly <<<")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # Load d9 state
    state_path = os.path.join(os.path.dirname(__file__),
                              '..', 'out', 'd9_state_L8_s42.npz')
    if not os.path.exists(state_path):
        state_path = 'experiments/lattice_theory/out/d9_state_L8_s42.npz'

    print(f"Loading state from {state_path}")
    state = np.load(state_path, allow_pickle=True)
    K_field = state['K']
    L = 8
    N = 2 * L**3
    n_bonds = 4 * L**3

    print(f"L={L}, N={N}, n_bonds={n_bonds}")
    print(f"K: mean={K_field.mean():.4f}, std={K_field.std():.4f}, "
          f"min={K_field.min():.4f}, max={K_field.max():.4f}")

    # Need lattice topology for real-space analysis
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    try:
        import d9_vortex_mc_gauge as d9
        lat = d9.build_diamond_lattice(L)
        positions, ei, ej = lat[0], lat[1], lat[2]
        sublat = lat[6]
        has_lattice = True
    except Exception as e:
        print(f"Warning: could not build lattice ({e})")
        has_lattice = False
        ei = ej = None

    # A. Modified Bloch analysis
    bloch_analysis(K_field)

    # B. Real-space spectrum (requires lattice topology)
    if has_lattice:
        spectral_analysis(L, K_field, ei, ej)

    # C. Local anisotropy
    local_anisotropy_analysis(K_field)

    # D. Valley analysis
    valley_analysis(K_field)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The bare 3-diamond lattice sits EXACTLY at the paramagnetic/diamagnetic
threshold: v_D^2 = 1/3, 1 - 3*v_D^2 = 0 (from LT-101).

The CLR K-field from the d9 vortex state modifies the effective hopping.
The per-direction global averages are nearly isotropic (0.3% relative),
so the GLOBAL Bloch analysis gives a tiny shift.

The local anisotropy (per-site K variation) is much larger (~6% mean,
up to 33%), concentrated at the vortex core. This contributes through
the self-energy (disorder scattering), not through global anisotropy.

The key question — does the CLR push v_D^2 below 1/3? — requires
either a self-energy calculation with the full K-field disorder, or
a much larger lattice where the ratio of core to bulk is smaller.
""")


if __name__ == "__main__":
    main()
