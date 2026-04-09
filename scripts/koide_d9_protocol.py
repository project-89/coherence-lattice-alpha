#!/usr/bin/env python3
"""
Koide Cross-Section Measurement using D9 Vortex Protocol (LT-196 + LT-214)

Correct two-phase protocol:
  Phase 0: Kuramoto bootstrap — circulation ω-profile nucleates π₁ vortex
           (from LT-196, verified at L=6,8, vortex at step 90-120)
  Phase 1: Shannon+Fiedler CLR co-evolution sculpts K-field around the
           established vortex topology
  Phase 2: Measure K_eff in each of the 3 tetrahedral cross-section planes
           and extract the Koide angle θ_K

Key insight: the vortex must be nucleated BEFORE the CLR starts.
The CLR maintains and optimizes the topology, not creates it.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.special import i0, i1
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components

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

# Also try symlinked canon
if not os.path.exists(LT183B_PATH):
    alt = os.path.join(THIS_DIR, "..", "canon", "v4_2026-02-14",
                       "184b_lt183_vortex_ring_g2_convergence.py")
    if os.path.exists(alt):
        LT183B_PATH = alt

if not os.path.exists(LT183B_PATH):
    print(f"ERROR: Cannot find {LT183B_PATH}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("lt183b", LT183B_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUTPUT_DIR = os.path.join(THIS_DIR, "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Fiedler
# =====================================================================

def deterministic_fiedler(ei, ej, K_arr, N):
    w = np.maximum(K_arr, 0.0)
    degree = np.zeros(N)
    np.add.at(degree, ei, w)
    np.add.at(degree, ej, w)
    live = w > 1e-12
    if np.sum(live) == 0:
        return 0.0, np.ones(N) / np.sqrt(N), degree
    adj_row = np.concatenate([ei[live], ej[live]])
    adj_col = np.concatenate([ej[live], ei[live]])
    A_sp = csr_matrix((np.ones(2 * int(np.sum(live))), (adj_row, adj_col)), shape=(N, N))
    n_comp, labels = connected_components(A_sp, directed=False)
    if n_comp > 1:
        v2 = labels.astype(float); v2 -= v2.mean()
        v2 /= max(np.linalg.norm(v2), 1e-15)
        return 0.0, v2, degree
    L_sp = mod._build_graph_laplacian(ei, ej, w, degree, N)
    v0 = degree.copy(); v0 -= v0.mean()
    norm = np.linalg.norm(v0)
    if norm < 1e-15:
        v0 = np.arange(N, dtype=float); v0 -= v0.mean(); norm = np.linalg.norm(v0)
    v0 /= norm
    try:
        evals, evecs = eigsh(L_sp, k=3, sigma=0, which='LM', maxiter=5000, tol=1e-10, v0=v0)
        order = np.argsort(evals)
        return float(max(evals[order[1]], 0.0)), evecs[:, order[1]], degree
    except Exception:
        try:
            evals, evecs = eigsh(L_sp, k=min(6, N - 1), which='SM', maxiter=5000, tol=1e-8, v0=v0)
            order = np.argsort(evals)
            return float(max(evals[order[1]], 0.0)), evecs[:, order[1]], degree
        except Exception:
            return 0.0, v0, degree

mod.sparse_laplacian_and_fiedler = deterministic_fiedler


# =====================================================================
# Phase 0: Kuramoto Bootstrap (from D9/LT-196)
# =====================================================================

def kuramoto_bootstrap(positions, ei, ej, N, n_bonds, center, axis=2,
                       coupling=2.0, omega_gradient=1.0, n_steps=300,
                       dt=0.1):
    """Bootstrap π₁ vortex from circulation frequency profile.

    ω_i ∝ atan2(y, x) / (2π) drives differential rotation.
    Kuramoto coupling sin(θ_j - θ_i) synchronizes locally.
    Competition → stable vortex with winding ±1.

    Protocol from LT-196: nucleation at step 90-120 on L=6,8.
    """
    # Build neighbor table
    site_nbr_list = [[] for _ in range(N)]
    for b in range(n_bonds):
        site_nbr_list[ei[b]].append(ej[b])
        site_nbr_list[ej[b]].append(ei[b])
    z = max(len(nb) for nb in site_nbr_list)
    site_nbr = np.full((N, z), -1, dtype=np.int32)
    for i in range(N):
        for k, j in enumerate(site_nbr_list[i]):
            site_nbr[i, k] = j

    # Circulation frequency profile
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    omega = omega_gradient * np.arctan2(dy, dx) / (2 * np.pi)

    # Initialize phases: uniform or small random
    theta = np.zeros(N)

    winding_history = []
    step_nucleated = None

    for step in range(1, n_steps + 1):
        # Kuramoto torque
        torque = np.zeros(N)
        for k in range(z):
            nbr = site_nbr[:, k]
            valid = nbr >= 0
            torque[valid] += np.sin(theta[nbr[valid]] - theta[valid])
        torque *= coupling / z

        theta = (theta + dt * (omega + torque)) % (2 * np.pi)

        # Measure winding every 10 steps
        if step % 10 == 0:
            w = _measure_winding(theta, positions, center, axis)
            winding_history.append((step, w))
            if w != 0 and step_nucleated is None:
                step_nucleated = step
                print(f"    *** Vortex nucleated at step {step}! W={w} ***")

        if step % 50 == 0:
            cos_dphi = np.cos(theta[ej] - theta[ei])
            n_anti = (cos_dphi < 0).sum()
            w_now = winding_history[-1][1] if winding_history else 0
            print(f"    Kuramoto step {step:4d}: W={w_now}, "
                  f"<cos>={cos_dphi.mean():.4f}, anti={n_anti}")

    final_w = _measure_winding(theta, positions, center, axis)
    return theta, final_w, step_nucleated, omega


def _measure_winding(theta, positions, center, axis=2):
    """Measure π₁ winding number."""
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    rho = np.sqrt(dx**2 + dy**2)
    r_max = float(np.max(rho)) * 0.8
    r_min = 0.5
    mask = (rho > r_min) & (rho < r_max)
    if np.sum(mask) < 4:
        return 0
    angles = np.arctan2(dy[mask], dx[mask])
    order = np.argsort(angles)
    theta_sorted = theta[np.where(mask)[0][order]]
    dth = np.diff(theta_sorted)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    close = (theta_sorted[0] - theta_sorted[-1] + np.pi) % (2 * np.pi) - np.pi
    total = float(np.sum(dth) + close)
    return int(np.round(total / (2 * np.pi)))


# =====================================================================
# Phase 1: Shannon+Fiedler CLR Co-evolution
# =====================================================================

def clr_coevolve(theta, omega, K_arr, ei, ej, N, n_bonds, dead_mask,
                 eta, lam, n_steps=20000, dt_phase=0.1, dt_K_init=0.01,
                 prune_after=3000, report_interval=2000):
    """Co-evolve phases + K-field with full Shannon+Fiedler CLR."""

    lambda_2, v_2, degree = deterministic_fiedler(ei, ej, K_arr, N)
    fiedler_ema = (v_2[ei] - v_2[ej]) ** 2
    dt_K = dt_K_init

    for step in range(n_steps):
        # Phase dynamics
        sin_diff = np.sin(theta[ej] - theta[ei])
        torque = np.zeros(N)
        weighted = K_arr * sin_diff
        np.add.at(torque, ei, weighted)
        np.add.at(torque, ej, -weighted)
        torque += omega
        theta = (theta + dt_phase * torque) % (2 * np.pi)

        # CLR
        cos_dth = np.cos(theta[ej] - theta[ei])

        if step % 20 == 0:
            lambda_2, v_2, degree = deterministic_fiedler(ei, ej, K_arr, N)
            f_new = (v_2[ei] - v_2[ej]) ** 2
            fiedler_ema = 0.9 * fiedler_ema + 0.1 * f_new

        K_dot = mod.shannon_clr_Kdot(
            K_arr, cos_dth, ei, ej, N, v_2, lambda_2, degree,
            eta, lam, fiedler_sens=fiedler_ema, dead_mask=dead_mask)

        alive = ~dead_mask
        kdot_inf = float(np.max(np.abs(K_dot[alive]))) if np.any(alive) else 0.0
        if kdot_inf > 4.0:
            dt_K = max(1e-4, dt_K * 0.7)
        elif kdot_inf < 0.15:
            dt_K = min(0.02, dt_K * 1.02)

        K_arr = np.clip(K_arr + dt_K * K_dot, 0.0, 100.0)
        K_arr[dead_mask] = 0.0

        if step >= prune_after:
            newly_dead = (~dead_mask) & (K_arr < 1e-6) & (cos_dth < 0)
            dead_mask |= newly_dead
            K_arr[dead_mask] = 0.0

        if step % report_interval == 0:
            n_dead = int(dead_mask.sum())
            K_bulk = float(K_arr[K_arr > 0.1].mean()) if (K_arr > 0.1).any() else 0.0
            print(f"    CLR step {step:6d}: K_bulk={K_bulk:.4f}, "
                  f"dead={n_dead}/{n_bonds} ({n_dead/n_bonds:.1%}), "
                  f"dt_K={dt_K:.5f}")

    return K_arr, theta, dead_mask


# =====================================================================
# Phase 2: Koide Measurement
# =====================================================================

def measure_koide(K_arr, ei, ej, positions):
    """Measure K_eff in each cross-section plane and extract Koide angle."""
    bond_vecs = positions[ej] - positions[ei]
    normals = np.eye(3)  # cubic axes = cross-section plane normals

    K_eff = np.zeros(3)
    for p in range(3):
        n = normals[p]
        proj = bond_vecs - np.outer(bond_vecs @ n, n)
        proj_sq = np.sum(proj**2, axis=1)
        total = np.sum(proj_sq)
        if total > 0:
            K_eff[p] = np.sum(K_arr * proj_sq) / total

    # Sort: smallest = most disrupted (electron)
    idx = np.argsort(K_eff)
    K_sorted = K_eff[idx]

    # Koide Q
    sq = np.sqrt(np.maximum(K_sorted, 0))
    Q = np.sum(K_sorted) / max(np.sum(sq)**2, 1e-15)

    # Koide angle from the mass (K_eff) ratios
    M_K = np.sum(K_sorted) / 3
    if M_K > 0:
        r = np.sqrt(K_sorted / M_K)
        cos_phi = np.clip((r[2] - 1) / np.sqrt(2), -1, 1)
        theta_K = float(np.arccos(cos_phi))
    else:
        theta_K = 0.0

    # Physical reference
    m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
    M_phys = (m_e + m_mu + m_tau) / 3
    r_phys = np.sqrt(np.array(sorted([m_e, m_mu, m_tau])) / M_phys)
    theta_K_phys = float(np.arccos(np.clip((r_phys[2] - 1) / np.sqrt(2), -1, 1)))

    return {
        'K_eff': K_eff.tolist(),
        'K_eff_sorted': K_sorted.tolist(),
        'plane_order': idx.tolist(),
        'Q': float(Q),
        'theta_K': theta_K,
        'theta_K_deg': float(np.degrees(theta_K)),
        'theta_K_physical': theta_K_phys,
        'theta_K_physical_deg': float(np.degrees(theta_K_phys)),
        'mass_ratios': [1.0] + [K_sorted[i] / max(K_sorted[0], 1e-15) for i in [1, 2]],
        'physical_ratios': [1.0, m_mu / m_e, m_tau / m_e],
    }


# =====================================================================
# Main
# =====================================================================

def main():
    d = 3
    L = 8
    r_clr = 5.893
    eta = 1.0
    lam = eta / r_clr

    print("=" * 70)
    print("KOIDE CROSS-SECTION MEASUREMENT (D9 Protocol)")
    print("=" * 70)
    print(f"  L={L}, r={r_clr}")
    print(f"  Phase 0: Kuramoto bootstrap (LT-196)")
    print(f"  Phase 1: Shannon+Fiedler CLR (LT-214)")
    print(f"  Phase 2: Cross-section K_eff measurement")
    print()

    # Build lattice
    deltas = mod.make_simplex_deltas(d)
    positions, N, ei, ej, _ = mod.make_d_diamond_adjacency(d, L)
    n_bonds = len(ei)
    e1, e2, e3, _, _ = mod.get_3d_frame(deltas)
    center = positions.mean(axis=0)
    print(f"  N={N}, n_bonds={n_bonds}")

    # Try multiple axes for vortex nucleation
    results_all = []

    for axis in [0, 1, 2]:
        print(f"\n{'='*60}")
        print(f"  AXIS {axis} ({'xyz'[axis]})")
        print(f"{'='*60}")

        # Phase 0: Kuramoto bootstrap
        print(f"\n  Phase 0: Kuramoto bootstrap (axis={axis})")
        t0 = time.time()
        theta, winding, step_nuc, omega = kuramoto_bootstrap(
            positions, ei, ej, N, n_bonds, center,
            axis=axis, coupling=2.0, omega_gradient=1.0, n_steps=300)
        t_boot = time.time() - t0

        print(f"\n  Winding = {winding}, nucleated at step {step_nuc}")
        print(f"  Bootstrap time: {t_boot:.1f}s")

        if winding == 0:
            print(f"  NO VORTEX — skipping CLR for this axis")
            continue

        # Phase 1: CLR co-evolution
        print(f"\n  Phase 1: CLR co-evolution")
        K_arr = np.ones(n_bonds) * 0.01
        dead_mask = np.zeros(n_bonds, dtype=bool)

        t0 = time.time()
        K_arr, theta, dead_mask = clr_coevolve(
            theta, omega, K_arr, ei, ej, N, n_bonds, dead_mask,
            eta, lam, n_steps=20000, prune_after=3000,
            report_interval=4000)
        t_clr = time.time() - t0

        n_dead = int(dead_mask.sum())
        K_bulk = float(K_arr[K_arr > 0.1].mean()) if (K_arr > 0.1).any() else 0.0
        print(f"\n  K_bulk = {K_bulk:.4f}, dead = {n_dead}/{n_bonds} "
              f"({n_dead/n_bonds:.1%}), CLR time: {t_clr:.1f}s")

        # Phase 2: Koide measurement
        print(f"\n  Phase 2: Koide measurement")
        koide = measure_koide(K_arr, ei, ej, positions)

        print(f"\n  K_eff per plane: {[f'{k:.4f}' for k in koide['K_eff']]}")
        print(f"  Sorted:          {[f'{k:.4f}' for k in koide['K_eff_sorted']]}")
        print(f"  Plane order:     {koide['plane_order']} "
              f"(most disrupted → least)")
        print(f"\n  Koide Q = {koide['Q']:.8f}  (target: {2/3:.8f})")
        print(f"  θ_K = {koide['theta_K_deg']:.4f}°  "
              f"(physical: {koide['theta_K_physical_deg']:.4f}°)")
        print(f"  Mass ratios: 1 : {koide['mass_ratios'][1]:.1f} : "
              f"{koide['mass_ratios'][2]:.1f}")
        print(f"  Physical:    1 : {koide['physical_ratios'][1]:.1f} : "
              f"{koide['physical_ratios'][2]:.1f}")

        results_all.append({
            'axis': axis,
            'winding': winding,
            'step_nucleated': step_nuc,
            'K_bulk': K_bulk,
            'n_dead': n_dead,
            'koide': koide,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if not results_all:
        print("  No vortices formed on any axis.")
    else:
        print(f"\n  {'Axis':>5s} {'W':>3s} {'K_bulk':>8s} {'Dead%':>6s} "
              f"{'Q':>10s} {'θ_K':>8s} {'θ_phys':>8s} {'Gap':>8s}")
        print(f"  " + "-" * 60)
        for r in results_all:
            k = r['koide']
            gap = abs(k['theta_K_deg'] - k['theta_K_physical_deg'])
            print(f"  {r['axis']:5d} {r['winding']:3d} {r['K_bulk']:8.4f} "
                  f"{r['n_dead']/n_bonds:5.1%} "
                  f"{k['Q']:10.6f} {k['theta_K_deg']:7.2f}° "
                  f"{k['theta_K_physical_deg']:7.2f}° {gap:7.2f}°")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'koide_d9_measurement.json')
    with open(out_path, 'w') as f:
        json.dump({'L': L, 'r_clr': r_clr, 'results': results_all}, f, indent=2)
    print(f"\n  Results: {out_path}")


if __name__ == '__main__':
    main()
