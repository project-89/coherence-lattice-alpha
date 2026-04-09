#!/usr/bin/env python3
"""
CLR-BKT Convergence Test (LT-214)

Tests the central claim: the CLR self-tunes K_bulk to K_BKT on a 2D
square lattice with XY phases and vortex excitations.

Protocol:
1. Initialize L×L square lattice with random phases, uniform K = K_init
2. Co-evolve: Kuramoto phase dynamics + Shannon CLR on K
3. Measure K_bulk after convergence
4. Compare to K_BKT = 2/π ≈ 0.6366

Test at multiple K_init (above and below K_BKT) to verify universal
convergence to the BKT critical point.
"""
import numpy as np
from scipy.special import i0, i1
import time
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def R0(K):
    """Von Mises order parameter R₀(K) = I₁(K)/I₀(K)."""
    K = np.clip(K, 1e-12, 100.0)
    return i1(K) / i0(K)


def run_clr_xy(L, K_init, r, eta_K, eta_theta, n_steps, seed=42,
               noise_sigma=0.0):
    """Run coupled Kuramoto + CLR dynamics on L×L square lattice.

    The protocol follows the alpha paper (Section 4.1.1):
    - Random phases θ ∈ [0, 2π)
    - Small initial K (≪ 1) so phases develop topological winding
      BEFORE coupling grows large enough to order them
    - Thermal noise in phase dynamics to maintain vortex nucleation

    Returns time series of K_bulk (mean coupling over alive bonds).
    """
    rng = np.random.RandomState(seed)
    N = L * L

    # Initialize: random phases, SMALL uniform K
    theta = rng.uniform(0, 2 * np.pi, N)
    n_bonds = 2 * N
    K = np.full(n_bonds, K_init, dtype=np.float64)

    # Build neighbor lists
    def idx(x, y):
        return (y % L) * L + (x % L)

    # For each bond, store (site_a, site_b)
    bond_sites = np.zeros((n_bonds, 2), dtype=np.int32)
    # Horizontal bonds: bond b = i connects site i to site (i+1 mod L in row)
    for y in range(L):
        for x in range(L):
            b = y * L + x
            bond_sites[b, 0] = idx(x, y)
            bond_sites[b, 1] = idx(x + 1, y)
    # Vertical bonds: bond b = N + i connects site i to site i+L
    for y in range(L):
        for x in range(L):
            b = N + y * L + x
            bond_sites[b, 0] = idx(x, y)
            bond_sites[b, 1] = idx(x, y + 1)

    # Per-site neighbor bond lists
    site_bonds = [[] for _ in range(N)]
    for b in range(n_bonds):
        site_bonds[bond_sites[b, 0]].append(b)
        site_bonds[bond_sites[b, 1]].append(b)

    K_BKT = 2.0 / np.pi
    lam = 1.0 / r

    history = {
        'K_bulk': [],
        'K_mean': [],
        'I_phase': [],
        'n_alive': [],
        'n_vortex': [],
        'dead_frac': [],
        'step': [],
    }

    dt_theta = 0.1
    dt_K = eta_K

    for step in range(n_steps):
        # --- Phase dynamics (Kuramoto + thermal noise) ---
        for _ in range(5):  # 5 sub-steps per CLR step
            dtheta = np.zeros(N)
            for b in range(n_bonds):
                i, j = bond_sites[b]
                dth = np.sin(theta[j] - theta[i])
                dtheta[i] += K[b] * dth
                dtheta[j] -= K[b] * dth
            # Thermal noise maintains vortex nucleation at finite T
            if noise_sigma > 0:
                dtheta += noise_sigma * rng.randn(N)
            theta += dt_theta * dtheta
            theta = theta % (2 * np.pi)

        # --- CLR dynamics (Shannon channel) ---
        cos_dtheta = np.zeros(n_bonds)
        for b in range(n_bonds):
            i, j = bond_sites[b]
            cos_dtheta[b] = np.cos(theta[j] - theta[i])

        R0_K = R0(K)
        dK = dt_K * (R0_K * cos_dtheta - 2 * lam * K)
        K = np.maximum(K + dK, 0.0)

        # --- Measure ---
        if step % 100 == 0 or step == n_steps - 1:
            alive = K > 0.01
            n_alive = alive.sum()
            K_alive = K[alive]
            K_bulk = K_alive.mean() if n_alive > 0 else 0.0
            K_mean = K.mean()
            dead_frac = 1.0 - n_alive / n_bonds

            # Phase alignment (mean cos over alive bonds)
            I_phase = cos_dtheta[alive].mean() if n_alive > 0 else 0.0

            # Count vortices: winding around each plaquette
            # On square lattice, plaquettes are unit squares
            n_vortex = 0
            for y in range(L):
                for x in range(L):
                    # Plaquette corners: (x,y), (x+1,y), (x+1,y+1), (x,y+1)
                    i00 = idx(x, y)
                    i10 = idx(x + 1, y)
                    i11 = idx(x + 1, y + 1)
                    i01 = idx(x, y + 1)
                    # Sum phase differences around plaquette
                    dth1 = theta[i10] - theta[i00]
                    dth2 = theta[i11] - theta[i10]
                    dth3 = theta[i01] - theta[i11]
                    dth4 = theta[i00] - theta[i01]
                    # Wrap to [-π, π]
                    wind = 0
                    for dth in [dth1, dth2, dth3, dth4]:
                        dth = (dth + np.pi) % (2 * np.pi) - np.pi
                        wind += dth
                    # Winding number
                    n_w = round(wind / (2 * np.pi))
                    if n_w != 0:
                        n_vortex += 1

            history['K_bulk'].append(float(K_bulk))
            history['K_mean'].append(float(K_mean))
            history['I_phase'].append(float(I_phase))
            history['n_alive'].append(int(n_alive))
            history['n_vortex'].append(int(n_vortex))
            history['dead_frac'].append(float(dead_frac))
            history['step'].append(step)

            if step % 1000 == 0:
                pct_alive = n_alive / n_bonds * 100
                print(f"  step {step:6d}: K_bulk={K_bulk:.4f} "
                      f"K_mean={K_mean:.4f} dead={dead_frac:.1%} "
                      f"vortex={n_vortex} I_phase={I_phase:.3f} "
                      f"(K_BKT={K_BKT:.4f})")

    return history


def main():
    K_BKT = 2.0 / np.pi
    L = 16
    r = 6.0  # CLR signal-to-noise (must be > ~4.2 for BKT window)
    eta_K = 0.005  # slow CLR (adiabatic assumption)
    eta_theta = 0.1
    noise_sigma = 0.3  # thermal noise to maintain vortex nucleation
    n_steps = 20000

    # Two regimes:
    # A) Vortex sector: start with K_init << 1 so phases develop winding
    #    before coupling grows. This is the alpha paper's protocol.
    # B) Trivial sector: start with K_init > K_BKT (no vortices).
    #    Expect convergence to K_eq >> K_BKT (confirms Step i).
    #
    # The prediction: regime A → K_BKT, regime B → K_eq ≈ 2.17

    tests = [
        # (label, K_init, noise) — vortex sector (small K, vortices form)
        ("Vortex sector (K=0.01)", 0.01, noise_sigma),
        ("Vortex sector (K=0.05)", 0.05, noise_sigma),
        ("Vortex sector (K=0.1)", 0.10, noise_sigma),
        # With noise at higher K — can vortices nucleate thermally?
        ("Thermal nucleation (K=0.5)", 0.50, noise_sigma),
        ("Thermal nucleation (K=1.0)", 1.00, noise_sigma),
        # Trivial sector: no noise, high K — should converge to K_eq
        ("Trivial sector (K=1.0, no noise)", 1.00, 0.0),
    ]

    print("=" * 70)
    print("CLR-BKT CONVERGENCE TEST (LT-214)")
    print("=" * 70)
    print(f"  L = {L}, r = {r}, eta_K = {eta_K}")
    print(f"  n_steps = {n_steps}, noise_sigma = {noise_sigma}")
    print(f"  K_BKT = 2/π = {K_BKT:.6f}")
    print(f"  K_eq (unconstrained) ≈ 2.17")
    print()
    print("  Prediction:")
    print("    Vortex sector (small K_init + noise) → K_BKT = 0.637")
    print("    Trivial sector (large K_init, no noise) → K_eq ≈ 2.17")
    print()

    results = {
        'L': L,
        'r': r,
        'eta_K': eta_K,
        'n_steps': n_steps,
        'noise_sigma': noise_sigma,
        'K_BKT': K_BKT,
        'runs': [],
    }

    for label, K_init, noise in tests:
        print(f"\n--- {label} ---")
        t0 = time.time()
        history = run_clr_xy(L, K_init, r, eta_K, eta_theta, n_steps,
                             noise_sigma=noise)
        elapsed = time.time() - t0

        K_final = history['K_bulk'][-1]
        n_vortex_final = history['n_vortex'][-1]
        dead_final = history['dead_frac'][-1]
        gap_bkt = (K_final - K_BKT) / K_BKT * 100

        print(f"  Final: K_bulk={K_final:.4f} vortex={n_vortex_final} "
              f"dead={dead_final:.1%} (gap from K_BKT: {gap_bkt:+.1f}%)")
        print(f"  Time: {elapsed:.1f}s")

        results['runs'].append({
            'label': label,
            'K_init': K_init,
            'noise': noise,
            'K_final': round(K_final, 6),
            'n_vortex_final': n_vortex_final,
            'dead_frac_final': round(dead_final, 3),
            'gap_bkt_pct': round(gap_bkt, 2),
            'elapsed': round(elapsed, 1),
            'history': {k: v[-10:] for k, v in history.items()},
        })

    # Summary
    K_eq = 2.17  # unconstrained CLR equilibrium
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Label':<40s} {'K_final':>8s} {'Vortex':>7s} {'Dead':>6s} {'Target':>10s}")
    print("-" * 75)
    for run in results['runs']:
        has_vortex = run['n_vortex_final'] > 0
        if has_vortex:
            target = f"K_BKT={K_BKT:.3f}"
            match = abs(run['gap_bkt_pct']) < 20
        else:
            target = f"K_eq={K_eq:.2f}"
            match = abs(run['K_final'] - K_eq) / K_eq < 0.1
        status = "✓" if match else "✗"
        print(f"  {run['label']:<40s} {run['K_final']:8.4f} "
              f"{run['n_vortex_final']:>7d} {run['dead_frac_final']:>5.1%} "
              f"{target:>10s} {status}")

    out_path = os.path.join(OUTPUT_DIR, 'clr_bkt_convergence.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
