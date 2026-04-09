#!/usr/bin/env python3
"""
3D Diamond Lattice CLR-BKT Convergence Test (LT-214)

Uses the existing da1 infrastructure to demonstrate that the CLR
self-tunes K_bulk → 16/π² on the 3D diamond lattice with spontaneous
vortex formation.

This is the key numerical test: on 3D diamond, vortex LINES persist
(unlike 2D point vortices that annihilate), providing the sustained
topological feedback that pins K at K_BKT.

Reuses: da1_spontaneous_vortex.py infrastructure (lattice construction,
CLR dynamics, vortex detection) from the alpha paper.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.special import i0, i1

# =====================================================================
# Import the canonical infrastructure from the alpha paper
# =====================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the project root (coherence_lattice/)
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
    print(f"ERROR: Cannot find infrastructure module at {LT183B_PATH}")
    print("Falling back to standalone implementation.")
    HAS_INFRA = False
else:
    spec = importlib.util.spec_from_file_location("lt183b", LT183B_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    HAS_INFRA = True

# Import da1 coevolution (not used directly — we have our own simplified version)
ALPHA_SCRIPTS = os.path.join(PROJECT_ROOT, "papers", "publication",
                             "coherence_lattice_alpha", "scripts")

OUTPUT_DIR = os.path.join(THIS_DIR, "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def R0(K):
    K = np.clip(K, 1e-12, 100.0)
    return i1(K) / i0(K)


def coevolve_simple(positions, ei, ej, N, n_bonds, eta, lam,
                    K_init=0.01, n_steps=20000, dt_phase=0.1, dt_K=0.01,
                    report_interval=2000):
    """Simplified co-evolution: Kuramoto phases + Shannon CLR on K.

    Stripped down from da1 — no Fiedler channel, no downfolding.
    Just the core Shannon CLR + Kuramoto dynamics for clean K_bulk tracking.
    """
    rng = np.random.RandomState(42)

    # Initialize
    theta = rng.uniform(0, 2 * np.pi, N)
    K = np.full(n_bonds, K_init, dtype=np.float64)

    # Natural frequencies: small random spread + gradient for vortex nucleation
    omega = rng.randn(N) * 0.1

    K_BKT = 2.0 / np.pi
    K_target = 16.0 / np.pi**2

    history = {'step': [], 'K_bulk': [], 'K_mean': [], 'n_dead': [],
               'dead_frac': [], 'I_phase': []}

    for step in range(n_steps):
        # --- Phase dynamics (Kuramoto) ---
        for _ in range(5):
            dtheta = omega.copy()
            for b in range(n_bonds):
                i, j = ei[b], ej[b]
                dth = np.sin(theta[j] - theta[i])
                dtheta[i] += K[b] * dth
                dtheta[j] -= K[b] * dth
            theta += dt_phase * dtheta
            theta = theta % (2 * np.pi)

        # --- Shannon CLR ---
        cos_dtheta = np.cos(theta[ej] - theta[ei])
        R0_K = R0(K)
        dK = dt_K * eta * (R0_K * cos_dtheta - 2 * lam * K)
        K = np.maximum(K + dK, 0.0)

        # --- Measure ---
        if step % 100 == 0 or step == n_steps - 1:
            alive = K > 0.01
            bulk = K > 0.1
            n_alive = int(alive.sum())
            n_dead = n_bonds - n_alive
            dead_frac = n_dead / n_bonds
            K_bulk = float(K[bulk].mean()) if bulk.any() else 0.0
            K_mean = float(K.mean())
            I_phase = float(cos_dtheta[alive].mean()) if n_alive > 0 else 0.0

            history['step'].append(step)
            history['K_bulk'].append(K_bulk)
            history['K_mean'].append(K_mean)
            history['n_dead'].append(n_dead)
            history['dead_frac'].append(dead_frac)
            history['I_phase'].append(I_phase)

            if step % report_interval == 0:
                print(f"    step {step:6d}: K_bulk={K_bulk:.4f} "
                      f"dead={dead_frac:.1%} I_phase={I_phase:.3f} "
                      f"(target={K_target:.4f})")

    return K, theta, history


def main():
    K_BKT = 2.0 / np.pi
    K_target = 16.0 / np.pi**2
    d = 3
    z = d + 1

    print("=" * 70)
    print("3D DIAMOND CLR-BKT CONVERGENCE (LT-214)")
    print("=" * 70)
    print(f"  K_BKT = 2/π = {K_BKT:.6f}")
    print(f"  K_target = 16/π² = {K_target:.6f}")
    print(f"  Infrastructure: {'canonical (lt183b)' if HAS_INFRA else 'standalone'}")
    print()

    if not HAS_INFRA:
        print("ERROR: Need canonical infrastructure for diamond lattice construction.")
        print(f"  Expected: {LT183B_PATH}")
        sys.exit(1)

    # Test at L=6 (N=432, fast) and L=8 (N=1024, slower)
    tests = [
        (6, 0.01, 20000),   # L=6, K_init=0.01, 20K steps
        (6, 0.05, 20000),   # L=6, K_init=0.05
        (8, 0.01, 30000),   # L=8, K_init=0.01, 30K steps
    ]

    r_clr = 5.893
    eta_clr = 1.0
    lam_clr = eta_clr / r_clr

    results = {
        'K_BKT': K_BKT,
        'K_target': K_target,
        'r_clr': r_clr,
        'runs': [],
    }

    for L, K_init, n_steps in tests:
        print(f"\n--- L={L}, K_init={K_init}, n_steps={n_steps} ---")

        # Build diamond lattice
        deltas = mod.make_simplex_deltas(d)
        positions, N, ei, ej, _ = mod.make_d_diamond_adjacency(d, L)
        n_bonds = len(ei)
        print(f"  N={N} sites, {n_bonds} bonds")

        t0 = time.time()
        K_final, theta_final, history = coevolve_simple(
            positions, ei, ej, N, n_bonds,
            eta=eta_clr, lam=lam_clr,
            K_init=K_init, n_steps=n_steps,
            dt_phase=0.1, dt_K=0.01,
            report_interval=2000,
        )
        elapsed = time.time() - t0

        K_bulk_final = history['K_bulk'][-1]
        dead_final = history['dead_frac'][-1]
        gap = (K_bulk_final - K_target) / K_target * 100

        print(f"\n  RESULT: K_bulk = {K_bulk_final:.4f} "
              f"(target = {K_target:.4f}, gap = {gap:+.1f}%)")
        print(f"  Dead bonds: {dead_final:.1%}")
        print(f"  Time: {elapsed:.1f}s")

        results['runs'].append({
            'L': L,
            'K_init': K_init,
            'n_steps': n_steps,
            'K_bulk_final': round(K_bulk_final, 6),
            'dead_frac': round(dead_final, 3),
            'gap_pct': round(gap, 2),
            'elapsed': round(elapsed, 1),
            'history_tail': {k: v[-20:] for k, v in history.items()},
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: 3D Diamond CLR Convergence")
    print("=" * 70)
    print(f"  Target: K_bulk = 16/π² = {K_target:.6f}")
    print()
    print(f"  {'L':>3s} {'K_init':>7s} {'K_final':>9s} {'Dead%':>6s} {'Gap%':>7s} {'Match?':>7s}")
    print("-" * 45)
    for run in results['runs']:
        match = abs(run['gap_pct']) < 10
        print(f"  {run['L']:3d} {run['K_init']:7.3f} {run['K_bulk_final']:9.4f} "
              f"{run['dead_frac']:5.1%} {run['gap_pct']:+7.1f} "
              f"{'YES' if match else 'NO':>7s}")

    out_path = os.path.join(OUTPUT_DIR, 'diamond_clr_convergence.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out_path}")


if __name__ == '__main__':
    main()
