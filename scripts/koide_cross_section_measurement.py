#!/usr/bin/env python3
"""
Koide Cross-Section Measurement (LT-214 / Generation Hypothesis)

Measures the effective coupling K_eff in each of the 3 cross-section
planes of the diamond tetrahedron from a converged CLR vortex configuration.

The hypothesis: the 3 K_eff values determine the 3 lepton masses through
the Koide parametrization m_i = M(1 + √2 cos(θ_K + 2πi/3))², where θ_K
is the CLR back-reaction angle encoding how the primary vortex's dead
bonds affect coupling in the other two planes.

Test: does the measured θ_K match the physical value from lepton masses?
θ_K(physical) ≈ 0.2222 rad ≈ 12.73° (from m_e, m_μ, m_τ)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.special import i0, i1
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# =====================================================================
# Infrastructure
# =====================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_d = THIS_DIR
for _ in range(10):
    if os.path.exists(os.path.join(_d, "AGENT.md")):
        break
    _d = os.path.dirname(_d)
PROJECT_ROOT = _d

CANON_DIR = os.path.join(PROJECT_ROOT, "experiments", "lattice_theory",
                         "canon", "v4_2026-02-14")
LT183B_PATH = os.path.join(CANON_DIR, "184b_lt183_vortex_ring_g2_convergence.py")

if not os.path.exists(LT183B_PATH):
    # Try symlink path
    alt = os.path.join(THIS_DIR, "..", "canon", "v4_2026-02-14",
                       "184b_lt183_vortex_ring_g2_convergence.py")
    if os.path.exists(alt):
        LT183B_PATH = alt
    else:
        print(f"ERROR: Cannot find {LT183B_PATH}")
        sys.exit(1)

spec = importlib.util.spec_from_file_location("lt183b", LT183B_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUTPUT_DIR = os.path.join(THIS_DIR, "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Fiedler (deterministic, same as da1)
# =====================================================================

_det_cache = {'X0': None, 'N': 0}

def _deterministic_fiedler(ei, ej, K_arr, N):
    from scipy.sparse.csgraph import connected_components
    w = np.maximum(K_arr, 0.0)
    degree = np.zeros(N)
    np.add.at(degree, ei, w)
    np.add.at(degree, ej, w)
    live = w > 1e-12
    if np.sum(live) == 0:
        return 0.0, np.ones(N) / np.sqrt(N), degree
    adj_row = np.concatenate([ei[live], ej[live]])
    adj_col = np.concatenate([ej[live], ei[live]])
    adj_data = np.ones(2 * int(np.sum(live)))
    A_sp = csr_matrix((adj_data, (adj_row, adj_col)), shape=(N, N))
    n_comp, labels = connected_components(A_sp, directed=False)
    if n_comp > 1:
        v_2 = labels.astype(float); v_2 -= v_2.mean()
        v_2 /= max(np.linalg.norm(v_2), 1e-15)
        return 0.0, v_2, degree
    L_sp = mod._build_graph_laplacian(ei, ej, w, degree, N)
    v0 = degree.copy(); v0 -= v0.mean()
    norm = np.linalg.norm(v0)
    if norm < 1e-15:
        v0 = np.arange(N, dtype=float); v0 -= v0.mean(); norm = np.linalg.norm(v0)
    v0 /= norm
    try:
        evals, evecs = eigsh(L_sp, k=3, sigma=0, which='LM',
                             maxiter=5000, tol=1e-10, v0=v0)
        order = np.argsort(evals)
        lambda_2 = float(max(evals[order[1]], 0.0))
        v_2 = evecs[:, order[1]]
    except Exception:
        try:
            evals, evecs = eigsh(L_sp, k=min(6, N-1), which='SM',
                                 maxiter=5000, tol=1e-8, v0=v0)
            order = np.argsort(evals)
            lambda_2 = float(max(evals[order[1]], 0.0))
            v_2 = evecs[:, order[1]]
        except Exception:
            lambda_2 = 0.0; v_2 = v0.copy()
    return lambda_2, v_2, degree

mod.sparse_laplacian_and_fiedler = _deterministic_fiedler


# =====================================================================
# Phase dynamics
# =====================================================================

def step_kuramoto(theta, omega, K_arr, ei, ej, N, dt):
    sin_diff = np.sin(theta[ej] - theta[ei])
    torque = np.zeros(N)
    weighted = K_arr * sin_diff
    np.add.at(torque, ei, weighted)
    np.add.at(torque, ej, -weighted)
    torque += omega
    return (theta + dt * torque) % (2.0 * np.pi)


def circulation_omega(positions, e1, e2, gradient=1.0):
    center = positions.mean(axis=0)
    dr = positions - center[np.newaxis, :]
    x_proj = dr @ e1
    y_proj = dr @ e2
    return gradient * np.arctan2(y_proj, x_proj) / (2.0 * np.pi)


# =====================================================================
# Koide analysis
# =====================================================================

def compute_cross_section_keff(K_arr, ei, ej, positions, deltas):
    """Compute effective coupling K_eff in each of the 3 cross-section planes.

    The 3 planes are defined by the 3 pairs of opposite edges of the
    tetrahedron. Their normals are the 3 cubic axes [100], [010], [001]
    (for the standard diamond orientation).

    For each plane, K_eff = Σ_b K_b × |proj_b|² / Σ_b |proj_b|²
    where proj_b is the bond vector's projection onto the plane.
    """
    # Bond vectors
    bond_vecs = positions[ej] - positions[ei]

    # The 3 cross-section plane normals (cubic axes)
    normals = np.eye(3)

    K_eff = np.zeros(3)
    for p in range(3):
        n = normals[p]
        # Project each bond onto the plane (remove normal component)
        proj = bond_vecs - np.outer(bond_vecs @ n, n)
        proj_sq = np.sum(proj**2, axis=1)  # |proj|² per bond

        # Weighted average: K_eff = Σ K × |proj|² / Σ |proj|²
        total_proj = np.sum(proj_sq)
        if total_proj > 0:
            K_eff[p] = np.sum(K_arr * proj_sq) / total_proj
        else:
            K_eff[p] = 0.0

    return K_eff


def koide_from_keff(K_eff):
    """Extract Koide angle and predicted masses from 3 K_eff values.

    The vortex mass in each cross-section scales with K_eff.
    The Koide parametrization: m_i ∝ (1 + √2 cos(θ + 2πi/3))²
    requires mapping K_eff → mass.

    The BKT vortex energy is E_v = π K ln(L/a). So mass ∝ K_eff.
    (More precisely, the mass gap depends on K through the correlation
    length ξ ~ exp(b/√(K - K_BKT)), but near K_BKT the leading
    dependence is linear.)
    """
    # Sort K_eff: smallest = electron (most disrupted), largest = tau
    idx = np.argsort(K_eff)
    K_sorted = K_eff[idx]

    # Masses proportional to K_eff (linear approximation)
    m_rel = K_sorted / K_sorted[0]  # normalized to m_e = 1

    # Physical masses
    m_e = 0.51099895  # MeV
    m_mu = 105.6583755
    m_tau = 1776.86

    # Koide Q from the measured K_eff
    sqrt_sum = np.sum(np.sqrt(K_sorted))
    Q_measured = np.sum(K_sorted) / sqrt_sum**2

    # Extract θ_K: fit Koide parametrization to K_eff
    # m_i = M(1 + √2 cos(θ + 2πi/3))² where M = mean(m_i)/3... no, Σm_i/3
    # Using √(m_i/M) = 1 + √2 cos(φ_i)
    M_K = np.sum(K_sorted) / 3
    r = np.sqrt(K_sorted / M_K)

    # The largest r gives cos(φ) = (r-1)/√2
    cos_phi_max = (r[2] - 1) / np.sqrt(2)
    if abs(cos_phi_max) <= 1:
        phi_max = np.arccos(cos_phi_max)
    else:
        phi_max = 0.0

    # θ_K from the constraint that the 3 phases are 2π/3 apart
    # and the largest corresponds to phase closest to 0
    theta_K = phi_max  # (the Koide angle)

    # Physical θ_K from actual lepton masses
    M_phys = (m_e + m_mu + m_tau) / 3
    r_phys = np.sqrt(np.array([m_e, m_mu, m_tau]) / M_phys)
    cos_phi_phys = (r_phys[2] - 1) / np.sqrt(2)
    theta_K_phys = np.arccos(np.clip(cos_phi_phys, -1, 1))

    return {
        'K_eff_sorted': K_sorted.tolist(),
        'mass_ratios': m_rel.tolist(),
        'Q_measured': float(Q_measured),
        'Q_koide': 2/3,
        'theta_K_measured': float(theta_K),
        'theta_K_physical': float(theta_K_phys),
        'theta_K_gap_deg': float(np.degrees(abs(theta_K - theta_K_phys))),
        'physical_ratios': [1.0, m_mu/m_e, m_tau/m_e],
    }


# =====================================================================
# Main
# =====================================================================

def main():
    d = 3
    L = 8  # L=8 for vortex formation
    K_BKT = 2.0 / np.pi
    r_clr = 5.893
    eta_clr = 1.0
    lam_clr = eta_clr / r_clr
    n_steps = 30000

    print("=" * 70)
    print("KOIDE CROSS-SECTION MEASUREMENT")
    print("=" * 70)
    print(f"  L={L}, r={r_clr}, n_steps={n_steps}")
    print()

    # Build lattice
    deltas = mod.make_simplex_deltas(d)
    positions, N, ei, ej, _ = mod.make_d_diamond_adjacency(d, L)
    n_bonds = len(ei)
    e1, e2, e3, _, _ = mod.get_3d_frame(deltas)
    print(f"  N={N}, n_bonds={n_bonds}")

    # Physical Koide angle
    m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
    M_phys = (m_e + m_mu + m_tau) / 3
    r_phys = np.sqrt(np.array(sorted([m_e, m_mu, m_tau])) / M_phys)
    theta_K_phys = np.arccos(np.clip((r_phys[2] - 1) / np.sqrt(2), -1, 1))
    print(f"  Physical Koide angle: θ_K = {theta_K_phys:.6f} rad = "
          f"{np.degrees(theta_K_phys):.4f}°")
    print()

    # Scan seeds for vortex formation
    best_result = None
    best_n_dead = 0

    for seed in [7, 13, 17, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]:
        np.random.seed(seed)
        _det_cache['X0'] = None; _det_cache['N'] = 0

        # Initialize from circulation profile
        theta = circulation_omega(positions, e1, e2, gradient=1.0) * (2.0 * np.pi)
        K_arr = np.ones(n_bonds) * 0.01
        omega = circulation_omega(positions, e1, e2, gradient=1.0)
        dead_mask = np.zeros(n_bonds, dtype=bool)

        # Warmup
        for _ in range(400):
            theta = step_kuramoto(theta, omega, K_arr, ei, ej, N, 0.1)

        # Full Shannon+Fiedler CLR co-evolution
        lambda_2, v_2, degree = _deterministic_fiedler(ei, ej, K_arr, N)
        fiedler_ema = (v_2[ei] - v_2[ej]) ** 2
        dt_K_cur = 0.01

        for step in range(n_steps):
            theta = step_kuramoto(theta, omega, K_arr, ei, ej, N, 0.1)
            cos_dth = np.cos(theta[ej] - theta[ei])

            if step % 20 == 0:
                lambda_2, v_2, degree = _deterministic_fiedler(ei, ej, K_arr, N)
                f_new = (v_2[ei] - v_2[ej]) ** 2
                fiedler_ema = 0.9 * fiedler_ema + 0.1 * f_new

            K_dot = mod.shannon_clr_Kdot(
                K_arr, cos_dth, ei, ej, N, v_2, lambda_2, degree,
                eta_clr, lam_clr, fiedler_sens=fiedler_ema,
                dead_mask=dead_mask)

            alive = ~dead_mask
            kdot_inf = float(np.max(np.abs(K_dot[alive]))) if np.any(alive) else 0.0
            if kdot_inf > 4.0:
                dt_K_cur = max(1e-4, dt_K_cur * 0.7)
            elif kdot_inf < 0.15:
                dt_K_cur = min(0.02, dt_K_cur * 1.02)

            K_arr = np.clip(K_arr + dt_K_cur * K_dot, 0.0, 100.0)
            K_arr[dead_mask] = 0.0

            if step >= 5000:
                newly_dead = (~dead_mask) & (K_arr < 1e-6) & (cos_dth < 0)
                dead_mask |= newly_dead
                K_arr[dead_mask] = 0.0

        n_dead = int(dead_mask.sum())
        K_alive = K_arr[~dead_mask]
        K_bulk = float(K_alive.mean()) if len(K_alive) > 0 else 0.0

        print(f"  seed {seed:2d}: K_bulk={K_bulk:.4f}, "
              f"dead={n_dead}/{n_bonds} ({n_dead/n_bonds:.1%})", end="")

        if n_dead > 50 and n_dead < n_bonds * 0.5:
            # Good vortex — measure cross-sections
            K_eff = compute_cross_section_keff(K_arr, ei, ej, positions, deltas)
            koide = koide_from_keff(K_eff)

            print(f" → K_eff = [{K_eff[0]:.4f}, {K_eff[1]:.4f}, {K_eff[2]:.4f}]"
                  f" Q={koide['Q_measured']:.6f}"
                  f" θ_K={np.degrees(koide['theta_K_measured']):.2f}°")

            if n_dead > best_n_dead:
                best_n_dead = n_dead
                best_result = {
                    'seed': seed,
                    'K_bulk': K_bulk,
                    'n_dead': n_dead,
                    'K_arr': K_arr.copy(),
                    'K_eff': K_eff.tolist(),
                    'koide': koide,
                }
        else:
            print(f" (no vortex or degenerate)")

        if best_n_dead > 100:
            break  # good enough

    if best_result is None:
        print("\nNo vortex formed at any seed. Cannot measure Koide.")
        # Still measure the UNIFORM case as a null check
        K_uniform = np.full(n_bonds, 16.0 / np.pi**2)
        K_eff_uniform = compute_cross_section_keff(K_uniform, ei, ej, positions, deltas)
        print(f"\nUniform K_eff: {K_eff_uniform}")
        print(f"  (Should be equal — no symmetry breaking)")

        # Save null result
        results = {
            'status': 'NO_VORTEX',
            'K_eff_uniform': K_eff_uniform.tolist(),
            'seeds_tested': 14,
        }
        out_path = os.path.join(OUTPUT_DIR, 'koide_measurement.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results: {out_path}")
        return

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    koide = best_result['koide']

    print(f"\n{'='*70}")
    print("KOIDE ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Best vortex: seed={best_result['seed']}, "
          f"n_dead={best_result['n_dead']}, K_bulk={best_result['K_bulk']:.4f}")

    print(f"\n  Cross-section K_eff values:")
    K_sorted = koide['K_eff_sorted']
    print(f"    Plane 0 (primary/electron): K_eff = {K_sorted[0]:.6f}")
    print(f"    Plane 1 (secondary/muon):   K_eff = {K_sorted[1]:.6f}")
    print(f"    Plane 2 (tertiary/tau):      K_eff = {K_sorted[2]:.6f}")

    print(f"\n  Mass ratios (from K_eff, linear approx):")
    print(f"    m_e : m_μ : m_τ = 1 : {koide['mass_ratios'][1]:.1f} : {koide['mass_ratios'][2]:.1f}")
    print(f"    Physical:         1 : {koide['physical_ratios'][1]:.1f} : {koide['physical_ratios'][2]:.1f}")

    print(f"\n  Koide Q:")
    print(f"    Measured: {koide['Q_measured']:.8f}")
    print(f"    Koide:    {koide['Q_koide']:.8f} (2/3)")
    print(f"    Gap:      {abs(koide['Q_measured'] - 2/3) / (2/3) * 1e6:.1f} ppm")

    print(f"\n  Koide angle θ_K:")
    print(f"    Measured: {np.degrees(koide['theta_K_measured']):.4f}°")
    print(f"    Physical: {np.degrees(koide['theta_K_physical']):.4f}°")
    print(f"    Gap:      {koide['theta_K_gap_deg']:.4f}°")

    print(f"\n  VERDICT:")
    if abs(koide['Q_measured'] - 2/3) / (2/3) < 0.01:
        print(f"    Q ≈ 2/3: YES (within 1%)")
    else:
        print(f"    Q ≈ 2/3: NO (gap too large)")

    if koide['theta_K_gap_deg'] < 5.0:
        print(f"    θ_K match: CLOSE (within 5°)")
    elif koide['theta_K_gap_deg'] < 20.0:
        print(f"    θ_K match: SUGGESTIVE (within 20°)")
    else:
        print(f"    θ_K match: NO (gap > 20°)")

    # Save
    results = {
        'status': 'VORTEX_FOUND',
        'seed': best_result['seed'],
        'L': L,
        'n_dead': best_result['n_dead'],
        'K_bulk': best_result['K_bulk'],
        'K_eff': best_result['K_eff'],
        'koide': koide,
    }
    out_path = os.path.join(OUTPUT_DIR, 'koide_measurement.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out_path}")


if __name__ == '__main__':
    main()
