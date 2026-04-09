#!/usr/bin/env python3
"""
Plaquette Loop Corrections to α = 1/137
=========================================

The BKT alpha formula α = R₀^z × (π/4)^{1/√e + α/(2π)} gives 1/α = 137.032
vs measured 137.036 — a 29 ppm gap. The vertex RG flow result (EXP-VRG)
localized this gap to the **star graph approximation**: R₀^z = [I₁(K)/I₀(K)]^4
assumes the 4 bonds at a vertex are independent. On the full diamond lattice,
hexagonal plaquette loops create inter-bond correlations that modify the
vertex factor.

This script performs Monte Carlo measurement of the actual vertex factor V_MC
on the 3D diamond lattice XY model at K = K_BKT = 2/π, and determines whether
inter-bond correlations account for the 29 ppm gap.

Key observable:
  V_star = R₀(K)^4 = [I₁(K)/I₀(K)]^4   (star graph, bonds independent)
  V_MC = ⟨∏_{μ=1}^4 cos(θ₀ - θ_μ)⟩     (full lattice, bonds correlated)

LT-ID: EXP-PLAQ
Status: EXPERIMENTAL
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import i0, i1
from scipy.optimize import curve_fit, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try numba for ~100× speedup; fall back to pure numpy
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# Constants
# =====================================================================

K_BKT = 2.0 / np.pi       # BKT critical coupling
Z_DIAMOND = 4              # coordination number
BASE = np.pi / Z_DIAMOND   # = π/4
N_DW = np.exp(-0.5)        # = 1/√e (Debye-Waller intensity)

def R0_paper(K):
    """Order parameter in paper convention: I₁(K)/I₀(K)"""
    return i1(K) / i0(K)

R0_STAR = R0_paper(K_BKT)
V_STAR = R0_STAR ** Z_DIAMOND

# CODATA 2018 recommended value
ALPHA_CODATA = 1.0 / 137.035999206


# =====================================================================
# Part 1: 3D Diamond Lattice
# =====================================================================

def make_simplex_deltas(d):
    deltas = np.zeros((d + 1, d))
    for i in range(d + 1):
        for j in range(d):
            jj = j + 1
            s = np.sqrt(jj * (jj + 1))
            if i < jj:
                deltas[i, j] = 1.0 / s
            elif i == jj:
                deltas[i, j] = -jj / s
    return deltas


def build_diamond_lattice(L):
    """Build 3-diamond lattice with L^3 unit cells, 2 sites/cell, z=4.
    Returns: ei, ej, site_nbr, N, n_bonds
    """
    d = 3
    N_cells = L ** d
    N = 2 * N_cells
    deltas = make_simplex_deltas(d)
    A_mat = np.array([deltas[0] - deltas[i] for i in range(1, d + 1)])

    bonds = []

    def cell_idx(n_tuple):
        idx = 0
        for k in range(d):
            idx = idx * L + (n_tuple[k] % L)
        return idx

    for flat in range(N_cells):
        n = []
        tmp = flat
        for k in range(d - 1, -1, -1):
            n.append(tmp % L)
            tmp //= L
        n = list(reversed(n))
        iA = 2 * flat
        iB_same = 2 * flat + 1
        bonds.append((iA, iB_same))
        for dim in range(d):
            n_shifted = list(n)
            n_shifted[dim] = (n[dim] - 1) % L
            iB = 2 * cell_idx(tuple(n_shifted)) + 1
            bonds.append((iA, iB))

    ei = np.array([b[0] for b in bonds], dtype=np.int32)
    ej = np.array([b[1] for b in bonds], dtype=np.int32)
    n_bonds = len(ei)

    site_nbr_list = [[] for _ in range(N)]
    for b_idx in range(n_bonds):
        i, j = int(ei[b_idx]), int(ej[b_idx])
        site_nbr_list[i].append(j)
        site_nbr_list[j].append(i)

    site_nbr = np.array(site_nbr_list, dtype=np.int32)  # shape (N, 4)
    return ei, ej, site_nbr, N, n_bonds


def enumerate_hexagonal_plaquettes(ei, ej, site_nbr, N, n_bonds):
    """Find all 6-bond hexagonal chair rings on 3-diamond.
    Returns: plaq_sites array of shape (n_plaq, 6)
    """
    z = site_nbr.shape[1]
    bond_lookup = {}
    for b in range(n_bonds):
        i, j = int(ei[b]), int(ej[b])
        bond_lookup[(i, j)] = b
        bond_lookup[(j, i)] = b

    seen = set()
    plaq_sites_list = []

    for s0 in range(N):
        for k1 in range(z):
            s1 = int(site_nbr[s0, k1])
            for k2 in range(z):
                s2 = int(site_nbr[s1, k2])
                if s2 == s0:
                    continue
                for k3 in range(z):
                    s3 = int(site_nbr[s2, k3])
                    if s3 in (s1, s0):
                        continue
                    for k4 in range(z):
                        s4 = int(site_nbr[s3, k4])
                        if s4 in (s2, s1, s0):
                            continue
                        for k5 in range(z):
                            s5 = int(site_nbr[s4, k5])
                            if s5 in (s3, s2, s1):
                                continue
                            if (s5, s0) not in bond_lookup:
                                continue
                            sites = (s0, s1, s2, s3, s4, s5)
                            canon = _canonical_hex(sites)
                            if canon in seen:
                                continue
                            seen.add(canon)
                            plaq_sites_list.append(list(sites))

    n_plaq = len(plaq_sites_list)
    if n_plaq == 0:
        return np.zeros((0, 6), dtype=np.int32)
    return np.array(plaq_sites_list, dtype=np.int32)


def _canonical_hex(sites):
    n = len(sites)
    min_s = min(sites)
    starts = [i for i, s in enumerate(sites) if s == min_s]
    candidates = []
    for st in starts:
        candidates.append(tuple(sites[(st + i) % n] for i in range(n)))
        candidates.append(tuple(sites[(st - i) % n] for i in range(n)))
    return min(candidates)


# =====================================================================
# Part 2: Monte Carlo Engine
# =====================================================================

if HAS_NUMBA:
    @njit
    def _mc_sweep_numba(theta, K, site_nbr, N, delta_max, rng_state):
        """Single Metropolis sweep with numba acceleration."""
        n_accept = 0
        for _step in range(N):
            idx = np.random.randint(0, N)
            theta_old = theta[idx]
            theta_new = theta_old + (np.random.random() - 0.5) * 2 * delta_max

            # Energy change
            dE = 0.0
            for k in range(4):
                nbr = site_nbr[idx, k]
                dE += np.cos(theta_old - theta[nbr]) - np.cos(theta_new - theta[nbr])
            dE *= K  # note: H = -K Σ cos(...), so ΔH = K × (cos_old - cos_new)

            if dE <= 0.0 or np.random.random() < np.exp(-dE):
                theta[idx] = theta_new
                n_accept += 1
        return n_accept

    @njit
    def _overrelax_sweep_numba(theta, K, site_nbr, N):
        """Overrelaxation: reflect θ_i around local field direction."""
        for _step in range(N):
            idx = np.random.randint(0, N)
            hx, hy = 0.0, 0.0
            for k in range(4):
                nbr = site_nbr[idx, k]
                hx += np.cos(theta[nbr])
                hy += np.sin(theta[nbr])
            phi_local = np.arctan2(hy, hx)
            theta[idx] = 2.0 * phi_local - theta[idx]


def mc_sweep_numpy(theta, K, site_nbr, N, delta_max, rng):
    """Single Metropolis sweep — pure numpy fallback."""
    order = rng.permutation(N)
    n_accept = 0
    for idx in order:
        theta_old = theta[idx]
        theta_new = theta_old + rng.uniform(-delta_max, delta_max)
        nbrs = site_nbr[idx]
        cos_old = np.sum(np.cos(theta_old - theta[nbrs]))
        cos_new = np.sum(np.cos(theta_new - theta[nbrs]))
        dE = K * (cos_old - cos_new)
        if dE <= 0.0 or rng.random() < np.exp(-dE):
            theta[idx] = theta_new
            n_accept += 1
    return n_accept


def overrelax_sweep_numpy(theta, K, site_nbr, N, rng):
    """Overrelaxation sweep — pure numpy."""
    order = rng.permutation(N)
    for idx in order:
        nbrs = site_nbr[idx]
        hx = np.sum(np.cos(theta[nbrs]))
        hy = np.sum(np.sin(theta[nbrs]))
        phi_local = np.arctan2(hy, hx)
        theta[idx] = 2.0 * phi_local - theta[idx]


def run_mc(theta, K, site_nbr, N, n_sweeps, delta_max, rng):
    """Run n_sweeps of Metropolis + overrelaxation, adapting delta_max."""
    total_accept = 0
    total_proposed = 0
    for _sw in range(n_sweeps):
        if HAS_NUMBA:
            na = _mc_sweep_numba(theta, K, site_nbr, N, delta_max, 0)
            _overrelax_sweep_numba(theta, K, site_nbr, N)
        else:
            na = mc_sweep_numpy(theta, K, site_nbr, N, delta_max, rng)
            overrelax_sweep_numpy(theta, K, site_nbr, N, rng)
        total_accept += na
        total_proposed += N

        # Adapt delta_max every 100 sweeps
        if (_sw + 1) % 100 == 0 and total_proposed > 0:
            rate = total_accept / total_proposed
            if rate > 0.55:
                delta_max *= 1.05
            elif rate < 0.45:
                delta_max *= 0.95
            delta_max = np.clip(delta_max, 0.1, np.pi)
            total_accept = 0
            total_proposed = 0

    return delta_max


# =====================================================================
# Part 3: Observable Measurement
# =====================================================================

def measure_observables(theta, ei, ej, site_nbr, N, n_bonds):
    """Measure all five observables from a single configuration.

    Returns dict with:
      R0_MC:   ⟨cos(Δθ)⟩ per bond
      V_MC:    ⟨∏_μ cos(θ₀ - θ_μ)⟩ per vertex (4-bond vertex product)
      sigma2:  ⟨(Δθ)²⟩ per bond (wrapped to [-π,π])
      mag:     |Σ e^{iθ}|/N (order parameter)
    """
    dtheta = theta[ei] - theta[ej]
    # Wrap to [-π, π]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    R0_MC = np.mean(np.cos(dtheta))
    sigma2_MC = np.mean(dtheta ** 2)

    # Vertex product: for each site, product of cos(θ_i - θ_nbr) over 4 neighbors
    V_acc = 0.0
    for i in range(N):
        delta_i = theta[i] - theta[site_nbr[i]]
        V_acc += np.prod(np.cos(delta_i))
    V_MC = V_acc / N

    # Magnetization
    mx = np.mean(np.cos(theta))
    my = np.mean(np.sin(theta))
    mag = np.sqrt(mx**2 + my**2)

    return {
        'R0_MC': float(R0_MC),
        'V_MC': float(V_MC),
        'sigma2_MC': float(sigma2_MC),
        'mag': float(mag),
    }


def measure_plaquette(theta, plaq_sites):
    """Measure hexagonal plaquette product ⟨∏_{ring} cos(Δθ)⟩."""
    if len(plaq_sites) == 0:
        return 0.0
    n_plaq = len(plaq_sites)
    plaq_acc = 0.0
    for p in range(n_plaq):
        ring = plaq_sites[p]
        prod = 1.0
        for step in range(6):
            sa = ring[step]
            sb = ring[(step + 1) % 6]
            prod *= np.cos(theta[sa] - theta[sb])
        plaq_acc += prod
    return plaq_acc / n_plaq


# =====================================================================
# Part 4: Size Scan
# =====================================================================

def run_single(L, seed, n_therm, n_meas, meas_interval, K):
    """Run a single MC simulation at given L and seed."""
    rng = np.random.default_rng(seed)

    # Build lattice
    ei, ej, site_nbr, N, n_bonds = build_diamond_lattice(L)

    # Only enumerate plaquettes for small L (expensive for large L)
    if L <= 8:
        plaq_sites = enumerate_hexagonal_plaquettes(ei, ej, site_nbr, N, n_bonds)
        n_plaq = len(plaq_sites)
    else:
        plaq_sites = np.zeros((0, 6), dtype=np.int32)
        n_plaq = 0

    # Random initial configuration
    theta = rng.uniform(-np.pi, np.pi, N)

    # Thermalize
    delta_max = 1.0
    delta_max = run_mc(theta, K, site_nbr, N, n_therm, delta_max, rng)

    # Measure
    n_measurements = n_meas // meas_interval
    R0_samples = []
    V_samples = []
    sigma2_samples = []
    mag_samples = []
    plaq_samples = []

    for m in range(n_measurements):
        delta_max = run_mc(theta, K, site_nbr, N, meas_interval, delta_max, rng)
        obs = measure_observables(theta, ei, ej, site_nbr, N, n_bonds)
        R0_samples.append(obs['R0_MC'])
        V_samples.append(obs['V_MC'])
        sigma2_samples.append(obs['sigma2_MC'])
        mag_samples.append(obs['mag'])
        if n_plaq > 0:
            plaq_samples.append(measure_plaquette(theta, plaq_sites))

    result = {
        'L': L,
        'N': N,
        'n_bonds': n_bonds,
        'n_plaq': n_plaq,
        'seed': seed,
        'K': float(K),
        'n_therm': n_therm,
        'n_meas': n_meas,
        'meas_interval': meas_interval,
        'n_measurements': n_measurements,
        'R0_MC': float(np.mean(R0_samples)),
        'R0_MC_err': float(np.std(R0_samples) / np.sqrt(len(R0_samples))),
        'V_MC': float(np.mean(V_samples)),
        'V_MC_err': float(np.std(V_samples) / np.sqrt(len(V_samples))),
        'sigma2_MC': float(np.mean(sigma2_samples)),
        'sigma2_MC_err': float(np.std(sigma2_samples) / np.sqrt(len(sigma2_samples))),
        'mag': float(np.mean(mag_samples)),
        'mag_err': float(np.std(mag_samples) / np.sqrt(len(mag_samples))),
    }

    if len(plaq_samples) > 0:
        result['plaq_MC'] = float(np.mean(plaq_samples))
        result['plaq_MC_err'] = float(np.std(plaq_samples) / np.sqrt(len(plaq_samples)))
    else:
        result['plaq_MC'] = None
        result['plaq_MC_err'] = None

    return result


# =====================================================================
# Part 5: Finite-Size Extrapolation
# =====================================================================

def extrapolate(L_arr, val_arr, err_arr, label=""):
    """Fit f(L) = a + b/L^p and extrapolate to L→∞."""
    results = {}
    for p in [2, 3]:
        def model(L, a, b):
            return a + b / L**p

        try:
            popt, pcov = curve_fit(model, L_arr, val_arr, sigma=err_arr,
                                   absolute_sigma=True, p0=[val_arr[-1], 0.0])
            a_inf, b_coeff = popt
            a_inf_err = np.sqrt(pcov[0, 0])
            # Chi-squared
            residuals = val_arr - model(L_arr, *popt)
            chi2 = np.sum((residuals / err_arr)**2)
            dof = len(L_arr) - 2
            results[p] = {
                'a_inf': float(a_inf),
                'a_inf_err': float(a_inf_err),
                'b': float(b_coeff),
                'chi2_dof': float(chi2 / dof) if dof > 0 else float('inf'),
                'p': p,
            }
        except Exception as e:
            results[p] = {'error': str(e), 'p': p}

    # Choose better fit (lower chi2/dof)
    best = None
    for p, r in results.items():
        if 'error' not in r:
            if best is None or r['chi2_dof'] < results[best]['chi2_dof']:
                best = p

    return results, best


# =====================================================================
# Part 6: Corrected Alpha
# =====================================================================

def solve_alpha_self_consistent(V_vertex, n_dw=N_DW, base=BASE, tol=1e-18):
    """Solve α = V_vertex × base^{n_dw + α/(2π)} self-consistently."""
    alpha = 1.0 / 137.0
    for i in range(500):
        alpha_new = V_vertex * base ** (n_dw + alpha / (2 * np.pi))
        if abs(alpha_new - alpha) < tol:
            return alpha_new, i + 1
        alpha = alpha_new
    return alpha, 500


def solve_alpha_with_sigma2(V_vertex, sigma2_MC, base=BASE, tol=1e-18):
    """Solve α using MC-measured σ² for the DW exponent.

    Theory: n_DW = exp(-σ²) where σ² = ⟨(Δθ)²⟩ / (2·base)
    but the raw σ² from MC needs to be compared to the theory prediction σ² = 1/2.
    We use the ratio to correct the exponent.
    """
    # Theory: σ²_theory = 1/2 (proven). MC gives σ²_MC.
    # Correction: n_DW_corrected = exp(-sigma2_MC) since σ² ≡ ⟨Δθ²⟩_nn/(2·base)
    # But ⟨Δθ²⟩_nn is what we measure as sigma2_MC directly (it's the nn variance).
    # The DW exponent is n = exp(-⟨Δθ²⟩/base) = exp(-2η) where 2η = σ²
    # For the wrapped angle on the lattice, σ² = ⟨Δθ²⟩, and the proven relation
    # is 2η = σ²/base = ⟨Δθ²⟩/(π/4).
    # With σ²_MC as the measured ⟨Δθ²⟩:
    eta_MC = sigma2_MC / (2 * base)  # = σ²_MC / (π/2)
    n_dw_corrected = np.exp(-2 * eta_MC)

    alpha = 1.0 / 137.0
    for i in range(500):
        alpha_new = V_vertex * base ** (n_dw_corrected + alpha / (2 * np.pi))
        if abs(alpha_new - alpha) < tol:
            return alpha_new, n_dw_corrected, i + 1
        alpha = alpha_new
    return alpha, n_dw_corrected, 500


# =====================================================================
# Part 7: Analytical Cross-Check
# =====================================================================

def analytical_plaquette_correction(n_hex_per_bond, R0_val):
    """Linked-cluster estimate of the first correction to R₀ from hexagonal loops.

    The star graph assumes bonds are independent. The first correction comes
    from hexagonal (6-bond) loops that create inter-bond correlations.

    δR₀/R₀ ~ n_hex × R₀^4 (from the 4 extra bonds in the hex loop beyond
    the original bond pair).

    For the vertex factor V = R₀^4:
    δV/V ~ 4 × n_hex × R₀^4
    """
    delta_R0_frac = n_hex_per_bond * R0_val**4
    delta_V_frac = 4 * n_hex_per_bond * R0_val**4

    return {
        'n_hex_per_bond': n_hex_per_bond,
        'R0': float(R0_val),
        'delta_R0_frac': float(delta_R0_frac),
        'delta_V_frac': float(delta_V_frac),
        'delta_R0_abs': float(R0_val * delta_R0_frac),
        'delta_V_abs': float(R0_val**4 * delta_V_frac),
    }


# =====================================================================
# Part 8: Plotting
# =====================================================================

def make_plots(all_results, extrap_R0, extrap_V, extrap_sigma2, alpha_results):
    """Generate summary plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Plaquette Loop Corrections to α = 1/137", fontsize=14, fontweight='bold')

    # Collect per-L data
    L_vals = sorted(set(r['L'] for r in all_results))
    L_mean = {}
    for L in L_vals:
        runs = [r for r in all_results if r['L'] == L]
        L_mean[L] = {
            'R0': np.mean([r['R0_MC'] for r in runs]),
            'R0_err': np.sqrt(np.sum([r['R0_MC_err']**2 for r in runs])) / len(runs),
            'V': np.mean([r['V_MC'] for r in runs]),
            'V_err': np.sqrt(np.sum([r['V_MC_err']**2 for r in runs])) / len(runs),
            'sigma2': np.mean([r['sigma2_MC'] for r in runs]),
            'sigma2_err': np.sqrt(np.sum([r['sigma2_MC_err']**2 for r in runs])) / len(runs),
            'mag': np.mean([r['mag'] for r in runs]),
            'mag_err': np.sqrt(np.sum([r['mag_err']**2 for r in runs])) / len(runs),
        }

    Ls = np.array(L_vals)
    R0s = np.array([L_mean[L]['R0'] for L in L_vals])
    R0_errs = np.array([L_mean[L]['R0_err'] for L in L_vals])
    Vs = np.array([L_mean[L]['V'] for L in L_vals])
    V_errs = np.array([L_mean[L]['V_err'] for L in L_vals])

    # Panel (a): R₀ vs L
    ax = axes[0, 0]
    ax.errorbar(Ls, R0s, yerr=R0_errs, fmt='bo-', capsize=4, label=r'$R_0^{MC}(L)$')
    ax.axhline(R0_STAR, color='red', ls='--', label=f'Star graph: {R0_STAR:.5f}')
    if extrap_R0 and 'a_inf' in extrap_R0.get(2, {}):
        ax.axhline(extrap_R0[2]['a_inf'], color='green', ls=':',
                    label=f"$L \\to \\infty$: {extrap_R0[2]['a_inf']:.5f}")
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\langle \cos(\Delta\theta) \rangle$')
    ax.set_title(r'(a) Bond Coherence $R_0^{MC}$')
    ax.legend(fontsize=8)

    # Panel (b): V_MC vs L
    ax = axes[0, 1]
    ax.errorbar(Ls, Vs, yerr=V_errs, fmt='rs-', capsize=4, label=r'$V_{MC}(L)$')
    ax.axhline(V_STAR, color='blue', ls='--', label=f'Star graph: {V_STAR:.6f}')
    if extrap_V and 'a_inf' in extrap_V.get(2, {}):
        ax.axhline(extrap_V[2]['a_inf'], color='green', ls=':',
                    label=f"$L \\to \\infty$: {extrap_V[2]['a_inf']:.6f}")
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\langle \prod_\mu \cos(\theta_0 - \theta_\mu) \rangle$')
    ax.set_title(r'(b) Vertex Factor $V_{MC}$')
    ax.legend(fontsize=8)

    # Panel (c): Magnetization vs L
    ax = axes[0, 2]
    mags = np.array([L_mean[L]['mag'] for L in L_vals])
    mag_errs = np.array([L_mean[L]['mag_err'] for L in L_vals])
    ax.errorbar(Ls, mags, yerr=mag_errs, fmt='g^-', capsize=4, label=r'$|m|(L)$')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$|m| = |\sum e^{i\theta}|/N$')
    ax.set_title('(c) Order Parameter (Phase Diagnosis)')
    ax.legend(fontsize=8)

    # Panel (d): σ² vs L
    ax = axes[1, 0]
    s2s = np.array([L_mean[L]['sigma2'] for L in L_vals])
    s2_errs = np.array([L_mean[L]['sigma2_err'] for L in L_vals])
    ax.errorbar(Ls, s2s, yerr=s2_errs, fmt='m+-', capsize=4, label=r'$\sigma^2_{MC}(L)$')
    ax.axhline(0.5, color='red', ls='--', label=r'Theory: $\sigma^2 = 1/2$')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\langle (\Delta\theta)^2 \rangle$')
    ax.set_title(r'(d) Bond Variance $\sigma^2_{MC}$')
    ax.legend(fontsize=8)

    # Panel (e): 1/alpha comparison
    ax = axes[1, 1]
    labels = []
    vals = []
    colors = []

    labels.append('Star graph')
    vals.append(1.0 / alpha_results['alpha_star'])
    colors.append('royalblue')

    if 'alpha_corrected_V' in alpha_results:
        labels.append('MC vertex')
        vals.append(1.0 / alpha_results['alpha_corrected_V'])
        colors.append('forestgreen')

    if 'alpha_corrected_both' in alpha_results:
        labels.append(r'MC V+$\sigma^2$')
        vals.append(1.0 / alpha_results['alpha_corrected_both'])
        colors.append('darkorange')

    labels.append('CODATA')
    vals.append(1.0 / ALPHA_CODATA)
    colors.append('crimson')

    bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel(r'$1/\alpha$')
    ax.set_title(r'(e) Fine Structure Constant')
    ymin = min(vals) - 0.02
    ymax = max(vals) + 0.02
    ax.set_ylim(ymin, ymax)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel (f): Finite-size scaling fit
    ax = axes[1, 2]
    ax.errorbar(1.0/Ls**2, Vs, yerr=V_errs, fmt='rs', capsize=4, markersize=8,
                label=r'$V_{MC}(L)$')
    if extrap_V and 'a_inf' in extrap_V.get(2, {}):
        x_fit = np.linspace(0, 1.1 / min(Ls)**2, 100)
        a, b = extrap_V[2]['a_inf'], extrap_V[2]['b']
        ax.plot(x_fit, a + b * x_fit, 'g--',
                label=f'$a + b/L^2$, $a = {a:.6f}$')
    ax.axhline(V_STAR, color='blue', ls=':', alpha=0.7, label=f'Star: {V_STAR:.6f}')
    ax.set_xlabel(r'$1/L^2$')
    ax.set_ylabel(r'$V_{MC}$')
    ax.set_title(r'(f) Finite-Size Extrapolation of $V$')
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'plaquette_correction_alpha.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {outpath}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Plaquette corrections to alpha")
    parser.add_argument('--L_values', nargs='+', type=int, default=[4, 6, 8, 10, 12])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--n_therm', type=int, default=10000)
    parser.add_argument('--n_meas', type=int, default=50000)
    parser.add_argument('--meas_interval', type=int, default=5)
    parser.add_argument('--K', type=float, default=K_BKT)
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("  PLAQUETTE LOOP CORRECTIONS TO α = 1/137")
    print("=" * 70)
    print(f"\n  K = {args.K:.10f} (K_BKT = {K_BKT:.10f})")
    print(f"  L values: {args.L_values}")
    print(f"  Seeds per L: {args.n_seeds}")
    print(f"  Thermalization: {args.n_therm} sweeps")
    print(f"  Measurement: {args.n_meas} sweeps (interval {args.meas_interval})")
    print(f"  Numba available: {HAS_NUMBA}")
    print()

    # Star graph reference values
    R0_ref = R0_paper(args.K)
    V_ref = R0_ref ** Z_DIAMOND
    print(f"  Star graph reference at K = {args.K:.6f}:")
    print(f"    R₀ = I₁(K)/I₀(K) = {R0_ref:.10f}")
    print(f"    V  = R₀^4         = {V_ref:.10e}")
    print(f"    R₀^6              = {R0_ref**6:.10e}")
    print(f"    σ² (theory)       = 0.5")

    # --- Run MC for all (L, seed) ---
    all_results = []
    t_start = time.time()

    for L in args.L_values:
        for seed in range(args.n_seeds):
            t0 = time.time()
            print(f"\n  Running L={L}, seed={seed} ...", end="", flush=True)
            result = run_single(L, seed + 42, args.n_therm, args.n_meas,
                                args.meas_interval, args.K)
            dt = time.time() - t0
            print(f" done ({dt:.1f}s)  R₀={result['R0_MC']:.6f}  V={result['V_MC']:.6e}"
                  f"  σ²={result['sigma2_MC']:.4f}  |m|={result['mag']:.4f}")
            all_results.append(result)

    t_total = time.time() - t_start
    print(f"\n  Total MC time: {t_total:.1f}s")

    # Save raw results
    json_path = os.path.join(OUTPUT_DIR, 'plaquette_correction_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Raw results saved: {json_path}")

    # --- Per-L averages ---
    print("\n" + "=" * 70)
    print("  PER-SIZE AVERAGES")
    print("=" * 70)
    print(f"\n  {'L':>4s}  {'N':>6s}  {'R0_MC':>10s}  {'V_MC':>12s}  {'sigma2':>8s}  {'|m|':>8s}  {'plaq':>12s}")
    print("  " + "-" * 72)

    L_vals = sorted(set(r['L'] for r in all_results))
    per_L = {}
    for L in L_vals:
        runs = [r for r in all_results if r['L'] == L]
        per_L[L] = {
            'R0': np.mean([r['R0_MC'] for r in runs]),
            'R0_err': np.std([r['R0_MC'] for r in runs]) / np.sqrt(len(runs)),
            'V': np.mean([r['V_MC'] for r in runs]),
            'V_err': np.std([r['V_MC'] for r in runs]) / np.sqrt(len(runs)),
            'sigma2': np.mean([r['sigma2_MC'] for r in runs]),
            'sigma2_err': np.std([r['sigma2_MC'] for r in runs]) / np.sqrt(len(runs)),
            'mag': np.mean([r['mag'] for r in runs]),
            'mag_err': np.std([r['mag'] for r in runs]) / np.sqrt(len(runs)),
            'N': runs[0]['N'],
            'n_plaq': runs[0]['n_plaq'],
        }
        plaq_vals = [r['plaq_MC'] for r in runs if r['plaq_MC'] is not None]
        if plaq_vals:
            per_L[L]['plaq'] = np.mean(plaq_vals)
            per_L[L]['plaq_err'] = np.std(plaq_vals) / np.sqrt(len(plaq_vals))
        else:
            per_L[L]['plaq'] = None

        d = per_L[L]
        plaq_str = f"{d['plaq']:.6e}" if d['plaq'] is not None else "N/A"
        print(f"  {L:4d}  {d['N']:6d}  {d['R0']:.6f}  {d['V']:.6e}  {d['sigma2']:.4f}  "
              f"{d['mag']:.4f}  {plaq_str}")

    # --- Finite-size extrapolation ---
    print("\n" + "=" * 70)
    print("  FINITE-SIZE EXTRAPOLATION")
    print("=" * 70)

    Ls = np.array(L_vals, dtype=float)

    # R₀
    R0_vals = np.array([per_L[L]['R0'] for L in L_vals])
    R0_errs = np.array([per_L[L]['R0_err'] for L in L_vals])
    # Guard against zero errors
    R0_errs = np.maximum(R0_errs, 1e-8)
    extrap_R0, best_R0 = extrapolate(Ls, R0_vals, R0_errs, "R0")

    print(f"\n  R₀ extrapolation:")
    for p, r in extrap_R0.items():
        if 'error' not in r:
            print(f"    1/L^{p}: R₀(∞) = {r['a_inf']:.8f} ± {r['a_inf_err']:.2e}  "
                  f"(χ²/dof = {r['chi2_dof']:.2f})")
        else:
            print(f"    1/L^{p}: {r['error']}")
    if best_R0:
        print(f"  Best fit: 1/L^{best_R0}")
        print(f"  R₀(∞) = {extrap_R0[best_R0]['a_inf']:.8f}")
        print(f"  Star graph: {R0_ref:.8f}")
        delta_R0 = extrap_R0[best_R0]['a_inf'] - R0_ref
        print(f"  δR₀ = R₀(MC) - R₀(star) = {delta_R0:+.6e}")

    # V
    V_vals = np.array([per_L[L]['V'] for L in L_vals])
    V_errs = np.array([per_L[L]['V_err'] for L in L_vals])
    V_errs = np.maximum(V_errs, 1e-10)
    extrap_V, best_V = extrapolate(Ls, V_vals, V_errs, "V")

    print(f"\n  V extrapolation:")
    for p, r in extrap_V.items():
        if 'error' not in r:
            print(f"    1/L^{p}: V(∞) = {r['a_inf']:.8e} ± {r['a_inf_err']:.2e}  "
                  f"(χ²/dof = {r['chi2_dof']:.2f})")
        else:
            print(f"    1/L^{p}: {r['error']}")
    if best_V:
        V_inf = extrap_V[best_V]['a_inf']
        print(f"  Best fit: 1/L^{best_V}")
        print(f"  V(∞) = {V_inf:.8e}")
        print(f"  Star graph: {V_ref:.8e}")
        delta_V = V_inf - V_ref
        print(f"  δV = V(MC) - V(star) = {delta_V:+.6e}")
        print(f"  δV/V(star) = {delta_V / V_ref:+.6e}")

    # σ²
    s2_vals = np.array([per_L[L]['sigma2'] for L in L_vals])
    s2_errs = np.array([per_L[L]['sigma2_err'] for L in L_vals])
    s2_errs = np.maximum(s2_errs, 1e-8)
    extrap_sigma2, best_s2 = extrapolate(Ls, s2_vals, s2_errs, "sigma2")

    print(f"\n  σ² extrapolation:")
    for p, r in extrap_sigma2.items():
        if 'error' not in r:
            print(f"    1/L^{p}: σ²(∞) = {r['a_inf']:.6f} ± {r['a_inf_err']:.2e}  "
                  f"(χ²/dof = {r['chi2_dof']:.2f})")
        else:
            print(f"    1/L^{p}: {r['error']}")

    # --- Corrected alpha ---
    print("\n" + "=" * 70)
    print("  CORRECTED ALPHA")
    print("=" * 70)

    # Star graph alpha (reference)
    alpha_star, n_iter = solve_alpha_self_consistent(V_STAR)
    print(f"\n  Star graph:")
    print(f"    α = {alpha_star:.12e}")
    print(f"    1/α = {1/alpha_star:.6f}")

    alpha_results = {'alpha_star': float(alpha_star)}

    # MC vertex correction
    if best_V and 'a_inf' in extrap_V[best_V]:
        V_inf = extrap_V[best_V]['a_inf']
        alpha_V, n_iter = solve_alpha_self_consistent(V_inf)
        print(f"\n  MC vertex correction (V_MC → ∞):")
        print(f"    V(∞) = {V_inf:.8e} (star: {V_ref:.8e})")
        print(f"    α = {alpha_V:.12e}")
        print(f"    1/α = {1/alpha_V:.6f}")
        print(f"    Shift: {1/alpha_V - 1/alpha_star:+.6f}")
        direction = "CORRECT (toward CODATA)" if 1/alpha_V > 1/alpha_star else "WRONG (away from CODATA)"
        if abs(1/alpha_V - 1/alpha_star) < 1e-6:
            direction = "NEGLIGIBLE"
        print(f"    Direction: {direction}")
        alpha_results['alpha_corrected_V'] = float(alpha_V)
        alpha_results['V_inf'] = float(V_inf)

    # MC vertex + σ² correction
    if best_V and best_s2 and 'a_inf' in extrap_V[best_V] and 'a_inf' in extrap_sigma2[best_s2]:
        V_inf = extrap_V[best_V]['a_inf']
        s2_inf = extrap_sigma2[best_s2]['a_inf']
        alpha_both, n_dw_corr, n_iter = solve_alpha_with_sigma2(V_inf, s2_inf)
        print(f"\n  MC vertex + σ² correction:")
        print(f"    V(∞) = {V_inf:.8e}")
        print(f"    σ²(∞) = {s2_inf:.6f} (theory: 0.5)")
        print(f"    n_DW corrected = {n_dw_corr:.8f} (theory: {N_DW:.8f})")
        print(f"    α = {alpha_both:.12e}")
        print(f"    1/α = {1/alpha_both:.6f}")
        print(f"    Shift from star: {1/alpha_both - 1/alpha_star:+.6f}")
        alpha_results['alpha_corrected_both'] = float(alpha_both)
        alpha_results['sigma2_inf'] = float(s2_inf)
        alpha_results['n_dw_corrected'] = float(n_dw_corr)

    # Gap analysis
    print(f"\n  Gap analysis:")
    print(f"    CODATA: 1/α = {1/ALPHA_CODATA:.6f}")
    print(f"    Star:   1/α = {1/alpha_star:.6f}  (gap = {1/ALPHA_CODATA - 1/alpha_star:+.6f})")
    if 'alpha_corrected_V' in alpha_results:
        gap_V = 1/ALPHA_CODATA - 1/alpha_results['alpha_corrected_V']
        gap_star = 1/ALPHA_CODATA - 1/alpha_star
        print(f"    MC V:   1/α = {1/alpha_results['alpha_corrected_V']:.6f}  (gap = {gap_V:+.6f})")
        if abs(gap_star) > 0:
            frac_closed = 1.0 - abs(gap_V) / abs(gap_star)
            print(f"    Fraction of gap closed: {frac_closed:.4f} ({frac_closed*100:.2f}%)")
    if 'alpha_corrected_both' in alpha_results:
        gap_both = 1/ALPHA_CODATA - 1/alpha_results['alpha_corrected_both']
        print(f"    MC V+σ²: 1/α = {1/alpha_results['alpha_corrected_both']:.6f}  (gap = {gap_both:+.6f})")

    alpha_results['gap_star_ppm'] = float(abs(1 - alpha_star/ALPHA_CODATA) * 1e6)

    # --- Analytical cross-check ---
    print("\n" + "=" * 70)
    print("  ANALYTICAL CROSS-CHECK")
    print("=" * 70)

    # Count hex plaquettes per bond from MC data
    # On diamond lattice: each bond participates in n_hex plaquettes
    # For L large enough, n_hex_per_bond = n_plaq * 6 / (2 * n_bonds) per bond
    # (each plaquette has 6 bonds, factor 2 from counting both directions)
    ref_L = min(L for L in L_vals if per_L[L]['n_plaq'] > 0) if any(per_L[L]['n_plaq'] > 0 for L in L_vals) else None
    if ref_L is not None:
        n_plaq = per_L[ref_L]['n_plaq']
        n_bonds = [r for r in all_results if r['L'] == ref_L][0]['n_bonds']
        n_hex_per_bond = n_plaq * 6.0 / (2.0 * n_bonds)
        print(f"\n  From L={ref_L}: n_plaq={n_plaq}, n_bonds={n_bonds}")
        print(f"  n_hex_per_bond = {n_hex_per_bond:.2f}")

        ana = analytical_plaquette_correction(n_hex_per_bond, R0_ref)
        print(f"\n  Linked-cluster estimate:")
        print(f"    δR₀/R₀ ~ n_hex × R₀^4 = {ana['delta_R0_frac']:.6e}")
        print(f"    δR₀ = {ana['delta_R0_abs']:.6e}")
        print(f"    δV/V ~ 4 × n_hex × R₀^4 = {ana['delta_V_frac']:.6e}")
        print(f"    δV = {ana['delta_V_abs']:.6e}")

        if best_V and 'a_inf' in extrap_V[best_V]:
            delta_V_mc = extrap_V[best_V]['a_inf'] - V_ref
            print(f"\n  Comparison:")
            print(f"    δV (MC)        = {delta_V_mc:+.6e}")
            print(f"    δV (analytical) = {ana['delta_V_abs']:+.6e}")
            if abs(ana['delta_V_abs']) > 0:
                print(f"    Ratio MC/analytical = {delta_V_mc / ana['delta_V_abs']:.2f}")
    else:
        print("  No plaquette data available for analytical cross-check.")

    # --- Verification checks ---
    print("\n" + "=" * 70)
    print("  VERIFICATION CHECKS")
    print("=" * 70)

    checks = []

    # R₀ check
    if best_R0 and 'a_inf' in extrap_R0[best_R0]:
        R0_inf = extrap_R0[best_R0]['a_inf']
        check = abs(R0_inf - R0_ref) < 0.01
        checks.append(('R₀(∞) ≈ R₀(star)', f'{R0_inf:.6f} vs {R0_ref:.6f}', check))
    else:
        R0_inf = R0_vals[-1]  # use largest L
        check = abs(R0_inf - R0_ref) < 0.02
        checks.append(('R₀(Lmax) ≈ R₀(star)', f'{R0_inf:.6f} vs {R0_ref:.6f}', check))

    # V check
    if best_V and 'a_inf' in extrap_V[best_V]:
        V_inf_val = extrap_V[best_V]['a_inf']
        check = abs(V_inf_val - V_ref) < 1e-3
        checks.append(('V(∞) ≈ R₀⁴', f'{V_inf_val:.6e} vs {V_ref:.6e}', check))

    # σ² check
    if best_s2 and 'a_inf' in extrap_sigma2[best_s2]:
        s2_inf = extrap_sigma2[best_s2]['a_inf']
        check = abs(s2_inf - 0.5) < 0.05
        checks.append(('σ²(∞) ≈ 0.5', f'{s2_inf:.4f} vs 0.5', check))

    for name, detail, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"""
  +-------------------+---------------+---------------+---------------+
  |    Quantity        |  Star Graph   |   MC (L→∞)    |   CODATA      |
  +-------------------+---------------+---------------+---------------+""")

    R0_inf_display = extrap_R0[best_R0]['a_inf'] if best_R0 and 'a_inf' in extrap_R0.get(best_R0, {}) else R0_vals[-1]
    V_inf_display = extrap_V[best_V]['a_inf'] if best_V and 'a_inf' in extrap_V.get(best_V, {}) else V_vals[-1]
    s2_inf_display = extrap_sigma2[best_s2]['a_inf'] if best_s2 and 'a_inf' in extrap_sigma2.get(best_s2, {}) else s2_vals[-1]

    print(f"  | R₀              | {R0_ref:13.8f} | {R0_inf_display:13.8f} |      —        |")
    print(f"  | V = vertex      | {V_ref:13.8e} | {V_inf_display:13.8e} |      —        |")
    print(f"  | σ²              | {'0.50000000':>13s} | {s2_inf_display:13.8f} |      —        |")

    inv_alpha_star = 1.0 / alpha_star
    inv_alpha_mc = 1.0 / alpha_results.get('alpha_corrected_V', alpha_star)
    inv_alpha_cod = 1.0 / ALPHA_CODATA

    print(f"  | 1/α             | {inv_alpha_star:13.6f} | {inv_alpha_mc:13.6f} | {inv_alpha_cod:13.6f} |")
    print(f"  +-------------------+---------------+---------------+---------------+")

    # Direction assessment
    if 'alpha_corrected_V' in alpha_results:
        delta_V_mc = V_inf_display - V_ref
        if abs(delta_V_mc) < V_errs[-1]:
            assessment = "NEGLIGIBLE: Plaquette correction is within MC noise"
        elif delta_V_mc < 0:
            assessment = "CORRECT DIRECTION: V_MC < R₀⁴ → 1/α pushed toward 137.036"
        else:
            assessment = "WRONG DIRECTION: V_MC > R₀⁴ → 1/α pushed away from 137.036"
    else:
        assessment = "INCONCLUSIVE: extrapolation failed"

    print(f"\n  Assessment: {assessment}")

    gap_ppm = abs(1 - alpha_star / ALPHA_CODATA) * 1e6
    if 'alpha_corrected_V' in alpha_results:
        gap_corrected_ppm = abs(1 - alpha_results['alpha_corrected_V'] / ALPHA_CODATA) * 1e6
        print(f"  Gap: {gap_ppm:.1f} ppm (star) → {gap_corrected_ppm:.1f} ppm (MC corrected)")
    else:
        print(f"  Gap: {gap_ppm:.1f} ppm (star)")

    # Save alpha results
    alpha_json = os.path.join(OUTPUT_DIR, 'plaquette_correction_alpha_results.json')
    with open(alpha_json, 'w') as f:
        json.dump(alpha_results, f, indent=2)
    print(f"\n  Alpha results saved: {alpha_json}")

    # --- Plot ---
    if not args.no_plot:
        make_plots(all_results, extrap_R0, extrap_V, extrap_sigma2, alpha_results)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
