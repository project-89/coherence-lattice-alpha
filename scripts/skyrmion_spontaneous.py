#!/usr/bin/env python3
"""
Spontaneous Skyrmion Generation via Coupled CLR-MC

The hand-placed Skyrmion experiment (skyrmion_frame_soliton.py, EXP-LT-191)
proved that CLR stabilizes a hedgehog texture: SKYRMION_STABILIZED at L=8 and
L=12 with clear core K_F depression and positive mass. But that experiment
hard-coded the topology.

This experiment asks: Can the lattice spontaneously generate frame-sector
topology when R_i and K_F co-evolve?

Three protocols:
  A — Thermal scan (MC only): map the frame-sector phase diagram
  B — Hot-start CLR-MC: random frames + CLR → spontaneous topology?
  C — Localized kick + CLR-MC: inject a frame perturbation → nucleation?

Architecture: 3-diamond lattice with bipartite vectorized Metropolis.
Convention: Shannon CLR with Fiedler feedback + dead bond pruning (184b).

Depends: LT-43, LT-58, LT-81, LT-84, LT-121
Prior: EXP-LT-191 (hand-placed Skyrmion)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import iv as I_bessel
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components

SCRIPT = "skyrmion_spontaneous.py"
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Utilities (from 184b / skyrmion_frame_soliton.py)
# =====================================================================

def jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (complex, np.complexfloating)):
        return {"re": float(np.real(obj)), "im": float(np.imag(obj))}
    return obj


# =====================================================================
# 3D Diamond Lattice Builder with Neighbor Tables
# =====================================================================

def make_simplex_deltas(d):
    """Construct d+1 vertices of regular d-simplex in R^d."""
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


def make_d_diamond_with_neighbors(d, L):
    """Build d-diamond adjacency on L^d supercell with PBC.

    Returns standard lattice PLUS per-site neighbor/bond tables:
      sublat:    (N,) array, 0=A, 1=B
      site_nbr:  (N, z) neighbor site indices per site
      site_bond: (N, z) bond index per neighbor

    The diamond lattice is bipartite: every A-site neighbors only B-sites
    and vice versa, enabling sublattice-parallel Metropolis.
    """
    N_cells = L ** d
    N = 2 * N_cells
    z = d + 1  # coordination number
    bonds = []
    positions = np.zeros((N, d))
    sublat = np.zeros(N, dtype=np.int32)
    deltas = make_simplex_deltas(d)
    A_mat = np.array([deltas[0] - deltas[i] for i in range(1, d + 1)])

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
        r_cell = sum(n[k] * A_mat[k] for k in range(d))
        iA = 2 * flat
        iB_same = 2 * flat + 1
        positions[iA] = r_cell
        positions[iB_same] = r_cell + deltas[0]
        sublat[iA] = 0
        sublat[iB_same] = 1
        bonds.append((iA, iB_same))
        for dim in range(d):
            n_shifted = list(n)
            n_shifted[dim] = (n[dim] - 1) % L
            iB = 2 * cell_idx(tuple(n_shifted)) + 1
            bonds.append((iA, iB))

    ei = np.array([b[0] for b in bonds], dtype=np.int32)
    ej = np.array([b[1] for b in bonds], dtype=np.int32)
    n_bonds = len(ei)

    # Build per-site neighbor and bond-index tables
    site_nbr_list = [[] for _ in range(N)]
    site_bond_list = [[] for _ in range(N)]
    for b_idx in range(n_bonds):
        i, j = int(ei[b_idx]), int(ej[b_idx])
        site_nbr_list[i].append(j)
        site_bond_list[i].append(b_idx)
        site_nbr_list[j].append(i)
        site_bond_list[j].append(b_idx)

    # Verify uniform coordination
    degs = [len(site_nbr_list[i]) for i in range(N)]
    assert all(deg == z for deg in degs), \
        f"Non-uniform degree: min={min(degs)}, max={max(degs)}, expected={z}"

    site_nbr = np.array(site_nbr_list, dtype=np.int32)   # (N, z)
    site_bond = np.array(site_bond_list, dtype=np.int32)  # (N, z)

    A_sites = np.where(sublat == 0)[0]
    B_sites = np.where(sublat == 1)[0]

    return (positions, N, ei, ej, bonds, sublat,
            site_nbr, site_bond, A_sites, B_sites)


# =====================================================================
# SO(3) Bessel Functions (from skyrmion_frame_soliton.py)
# =====================================================================

def a_l_so3(K, l):
    """SO(3) character coefficient a_l(K) = e^K [I_l(2K) - I_{l+1}(2K)]."""
    K = np.asarray(K, dtype=float)
    safe = np.where(K > 1e-10, K, 1.0)
    result = np.exp(safe) * (I_bessel(l, 2 * safe) - I_bessel(l + 1, 2 * safe))
    if l == 0:
        result = np.where(K > 1e-10, result, 1.0)
    else:
        result = np.where(K > 1e-10, result, 0.0)
    return result


def R0_so3(K):
    """SO(3) order parameter: R_0^F(K) = a_1/a_0."""
    K = np.asarray(K, dtype=float)
    a0 = a_l_so3(K, 0)
    a1 = a_l_so3(K, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(a0 > 1e-30, a1 / a0, 0.0)
    return r


def R0_so3_scalar(K):
    """Scalar version for single K value."""
    if K < 1e-10:
        return 0.0
    a0 = float(np.exp(K) * (I_bessel(0, 2 * K) - I_bessel(1, 2 * K)))
    a1 = float(np.exp(K) * (I_bessel(1, 2 * K) - I_bessel(2, 2 * K)))
    if a0 < 1e-30:
        return 0.0
    return a1 / a0


class _R0_SO3_LookupTable:
    """Precomputed lookup table for R0_so3(K)."""

    def __init__(self, K_max=150.0, n_points=16384):
        self.K_max = K_max
        self.n_points = n_points
        self.dK = K_max / n_points
        K_grid = np.linspace(0, K_max, n_points + 1)
        self.R0_grid = np.array([R0_so3_scalar(k) for k in K_grid])

    def evaluate(self, K_arr):
        result = np.zeros_like(K_arr, dtype=float)
        mask = K_arr > 1e-15
        if not np.any(mask):
            return result
        K_clipped = np.clip(K_arr[mask], 0.0, self.K_max - 1e-10)
        idx_f = K_clipped / self.dK
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        result[mask] = (1 - frac) * self.R0_grid[idx] + \
            frac * self.R0_grid[idx + 1]
        return result


_R0_SO3_LUT = _R0_SO3_LookupTable()


def R0_so3_vec(K_arr):
    """Vectorized frame order parameter via LUT."""
    return _R0_SO3_LUT.evaluate(K_arr)


def C_channel_so3(K):
    """SO(3) coherence capital: c(K) = -log(a_0(K)) + K * R_0^F(K)."""
    K = np.asarray(K, dtype=float)
    a0 = a_l_so3(K, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_a0 = np.where(a0 > 0, np.log(a0), 0.0)
    return -log_a0 + K * R0_so3(K)


# =====================================================================
# Frame Bond Alignment (from skyrmion_frame_soliton.py)
# =====================================================================

def frame_bond_alignment(R, ei, ej):
    """Compute frame alignment per bond: cos_frame = tr(R_i^T R_j) / 3."""
    return np.einsum('bac,bac->b', R[ei], R[ej]) / 3.0


# =====================================================================
# Frame CLR Fixed Point Solver (from skyrmion_frame_soliton.py)
# =====================================================================

def solve_clr_frame(cos_frame, r_F, K_max=50.0):
    """Solve frame-sector CLR fixed point: K = (r_F/2) * cos_frame * R0_so3(K)."""
    if cos_frame <= 0 or r_F <= 0:
        return 0.0
    threshold = 2.0 / r_F if r_F > 2.0 else 1.0
    if cos_frame <= threshold + 1e-12:
        return 0.0

    def F(K):
        return K - (r_F / 2.0) * cos_frame * R0_so3_scalar(K)

    eps = 1e-10
    if F(eps) >= 0:
        return 0.0
    if F(K_max) <= 0:
        K_max = 200.0
    try:
        return brentq(F, eps, K_max, xtol=1e-14, rtol=1e-14)
    except ValueError:
        return 0.0


# =====================================================================
# Graph Laplacian & Fiedler Vector (from skyrmion_frame_soliton.py)
# =====================================================================

def _build_graph_laplacian(ei, ej, w, degree, N):
    """Build sparse graph Laplacian L = D - A from bond weights."""
    row_off = np.concatenate([ei, ej])
    col_off = np.concatenate([ej, ei])
    data_off = np.concatenate([-w, -w])
    row_all = np.concatenate([row_off, np.arange(N)])
    col_all = np.concatenate([col_off, np.arange(N)])
    data_all = np.concatenate([data_off, degree])
    return csr_matrix((data_all, (row_all, col_all)), shape=(N, N))


_fiedler_cache = {'X0': None, 'N': 0}


def sparse_laplacian_and_fiedler(ei, ej, K_arr, N):
    """Compute graph Laplacian and Fiedler vector."""
    from scipy.sparse.linalg import lobpcg

    w = np.maximum(K_arr, 0.0)
    degree = np.zeros(N)
    np.add.at(degree, ei, w)
    np.add.at(degree, ej, w)

    live = w > 1e-12
    if np.sum(live) == 0:
        v_2 = np.random.randn(N)
        v_2 -= v_2.mean()
        v_2 /= max(np.linalg.norm(v_2), 1e-15)
        return 0.0, v_2, degree

    adj_row = np.concatenate([ei[live], ej[live]])
    adj_col = np.concatenate([ej[live], ei[live]])
    adj_data = np.ones(2 * int(np.sum(live)))
    A_sp = csr_matrix((adj_data, (adj_row, adj_col)), shape=(N, N))
    n_comp, labels = connected_components(A_sp, directed=False)
    if n_comp > 1:
        v_2 = labels.astype(float)
        v_2 -= v_2.mean()
        v_2 /= max(np.linalg.norm(v_2), 1e-15)
        _fiedler_cache['X0'] = None
        return 0.0, v_2, degree

    L_sp = _build_graph_laplacian(ei, ej, w, degree, N)

    cache = _fiedler_cache
    if cache['X0'] is not None and cache['N'] == N:
        try:
            import warnings, io, contextlib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _devnull = io.StringIO()
                with contextlib.redirect_stderr(_devnull):
                    evals, evecs = lobpcg(
                        L_sp, cache['X0'], largest=False,
                        maxiter=60, tol=1e-5, verbosityLevel=0)
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]
            lambda_2 = float(max(evals[1], 0.0))
            v_2 = evecs[:, 1]
            cache['X0'] = evecs.copy()
            return lambda_2, v_2, degree
        except Exception:
            pass

    try:
        evals, evecs = eigsh(L_sp, k=3, sigma=0, which='LM',
                             maxiter=5000, tol=1e-10)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        lambda_2 = float(max(evals[1], 0.0))
        v_2 = evecs[:, 1]
        cache['X0'] = evecs.copy()
        cache['N'] = N
    except Exception:
        try:
            evals, evecs = eigsh(L_sp, k=min(6, N - 1), which='SM',
                                 maxiter=5000, tol=1e-8)
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]
            lambda_2 = float(max(evals[1], 0.0))
            v_2 = evecs[:, 1]
            cache['X0'] = evecs[:, :3].copy()
            cache['N'] = N
        except Exception:
            lambda_2 = 0.0
            v_2 = np.random.randn(N)
            v_2 -= v_2.mean()
            v_2 /= max(np.linalg.norm(v_2), 1e-15)
    return lambda_2, v_2, degree


# =====================================================================
# Frame Shannon CLR K̇ (from skyrmion_frame_soliton.py)
# =====================================================================

def frame_shannon_clr_Kdot(K_F, cos_frame, ei, ej, N,
                           v_2, lambda_2, degree, eta, lam,
                           fiedler_sens=None, dead_mask=None,
                           struct_weight=1.0):
    """Frame-sector Shannon CLR time derivative."""
    R0_arr = R0_so3_vec(K_F)
    K_total = np.sum(K_F)
    I_frame = float(np.sum(K_F * R0_arr * cos_frame) / max(K_total, 1e-15))

    K_dot_clr = eta * (R0_arr * cos_frame - 2 * lam * K_F)

    active = K_F > 1e-12
    N_active = float(np.sum(active))
    E_active = max(N_active, 1.0)
    K_active = K_F[active]
    K_mean = float(K_active.mean()) if len(K_active) > 0 else 0.0
    d_max = max(np.max(degree), 1e-12)
    B_fb = lambda_2 / d_max
    if fiedler_sens is None:
        fiedler_sens = (v_2[ei] - v_2[ej]) ** 2
    grad_rho = N_active * (B_fb / E_active + K_mean * fiedler_sens / d_max)
    S = grad_rho - np.mean(grad_rho)
    K_dot_struct = struct_weight * eta * I_frame * S
    K_dot = K_dot_clr + K_dot_struct

    if dead_mask is not None:
        K_dot[dead_mask] = 0.0
    return K_dot


# =====================================================================
# Coherence Capital & Mass (from skyrmion_frame_soliton.py)
# =====================================================================

def frame_coherence_capital(K_F):
    """Per-bond coherence capital in frame sector."""
    K_arr = np.asarray(K_F, dtype=float)
    a0 = a_l_so3(K_arr, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_a0 = np.where(a0 > 1e-30, np.log(a0), 0.0)
    R0 = R0_so3_vec(K_arr)
    return -log_a0 + K_arr * R0


def measure_skyrmion_mass(K_F, dead_mask):
    """Compute Skyrmion mass from coherence capital deficit."""
    alive = ~dead_mask
    c_bonds = frame_coherence_capital(K_F)
    M_total = float(np.sum(c_bonds))

    alive_K = K_F[alive]
    if len(alive_K) > 0:
        K_sorted = np.sort(alive_K)
        n_top = max(len(K_sorted) // 2, 1)
        K_star = float(np.mean(K_sorted[-n_top:]))
    else:
        K_star = 0.0

    n_bonds_total = len(K_F)
    c_vacuum_per_bond = float(frame_coherence_capital(np.array([K_star]))[0])
    M_vacuum = n_bonds_total * c_vacuum_per_bond
    M_skyrmion = M_vacuum - M_total

    return {
        'M_total': M_total,
        'M_vacuum': M_vacuum,
        'M_skyrmion': M_skyrmion,
        'K_star': K_star,
        'c_vacuum_per_bond': c_vacuum_per_bond,
        'n_alive': int(np.sum(alive)),
        'n_dead': int(np.sum(dead_mask)),
    }


# =====================================================================
# Skyrmion Profile & Hedgehog Rotation (for Protocol C)
# =====================================================================

def skyrmion_profile(r, R_sky, profile_type='linear'):
    """Skyrmion profile function f(r). f(0)=π, f(R_sky)=0."""
    if profile_type == 'linear':
        return np.pi * np.clip(1.0 - r / R_sky, 0.0, 1.0)
    elif profile_type == 'rational':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(r > 1e-15, 2.0 * np.arctan(R_sky**2 / r**2), np.pi)
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")


def hedgehog_rotation(r_vec, f_val):
    """SO(3) rotation via Rodrigues: axis=r̂, angle=2f."""
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-15:
        return np.eye(3)
    r_hat = r_vec / r_norm
    angle = 2.0 * f_val
    Kx = np.array([[0, -r_hat[2], r_hat[1]],
                    [r_hat[2], 0, -r_hat[0]],
                    [-r_hat[1], r_hat[0], 0]])
    R = np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * (Kx @ Kx)
    return R


# =====================================================================
# Radial K_F Profile (from skyrmion_frame_soliton.py)
# =====================================================================

def radial_K_profile(K_F, ei, ej, positions, center, n_bins=10):
    """Compute radial profile of K_F (bond midpoints vs distance from center)."""
    mid = (positions[ei] + positions[ej]) / 2.0
    dists = np.linalg.norm(mid - center, axis=1)
    r_max = float(np.max(dists))
    bin_edges = np.linspace(0, r_max + 1e-10, n_bins + 1)
    profile = []
    for b in range(n_bins):
        mask = (dists >= bin_edges[b]) & (dists < bin_edges[b + 1])
        if np.any(mask):
            profile.append({
                'r_mid': float((bin_edges[b] + bin_edges[b + 1]) / 2),
                'K_mean': float(np.mean(K_F[mask])),
                'K_std': float(np.std(K_F[mask])),
                'n_bonds': int(np.sum(mask)),
            })
    return profile


# =====================================================================
# NEW: Thermal CLR — Self-Consistent Initialization
# =====================================================================

def thermal_self_consistent_K(R, ei, ej, N, site_nbr, site_bond,
                               A_sites, B_sites, r_F, beta,
                               n_iter=3, n_mc_sweeps=200, rng=None,
                               step_size=0.3, verbose=False):
    """Self-consistent thermal K initialization.

    Iterate:
      1. Run MC at current K_F to thermalize
      2. Measure <cos_frame>
      3. Compute K_new = solve_clr_frame(<cos_frame>, r_F) per bond
    until K converges.

    Returns K_F (per-bond), cos_frame_eq (per-bond EMA from final MC).
    """
    n_bonds = len(ei)
    K_F_star = solve_clr_frame(1.0, r_F)
    K_F = np.full(n_bonds, K_F_star)

    if rng is None:
        rng = np.random.RandomState(0)

    for it in range(n_iter):
        # MC thermalization at current K_F
        for _ in range(n_mc_sweeps):
            bipartite_mc_sweep_so3(
                R, K_F, site_nbr, site_bond,
                A_sites, B_sites, step_size, beta, rng)

        # Measure cos_frame (average over a few sweeps for noise reduction)
        cos_accum = np.zeros(n_bonds)
        n_samples = 20
        for _ in range(n_samples):
            bipartite_mc_sweep_so3(
                R, K_F, site_nbr, site_bond,
                A_sites, B_sites, step_size, beta, rng)
            cos_accum += frame_bond_alignment(R, ei, ej)
        cos_avg = cos_accum / n_samples

        # Update K_F from fixed-point per bond
        K_F_new = np.zeros(n_bonds)
        for b in range(n_bonds):
            K_F_new[b] = solve_clr_frame(cos_avg[b], r_F)

        delta = float(np.max(np.abs(K_F_new - K_F)))
        if verbose:
            print(f"    thermal init iter {it}: K_mean={np.mean(K_F_new):.4f}, "
                  f"<cos>={np.mean(cos_avg):.4f}, |dK|_inf={delta:.4f}")
        K_F = K_F_new

        if delta < 0.05:
            break

    return K_F, cos_avg


# =====================================================================
# NEW: Random Rotation Generators
# =====================================================================

def random_rotation_uniform_batch(n, rng):
    """Uniform SO(3) via QR decomposition. Returns (n, 3, 3)."""
    R_out = np.zeros((n, 3, 3))
    for i in range(n):
        M = rng.randn(3, 3)
        Q, Rr = np.linalg.qr(M)
        # Ensure det(Q) = +1 (proper rotation)
        Q *= np.sign(np.linalg.det(Q))
        R_out[i] = Q
    return R_out


def random_rotation_small_batch(n, step_size, rng):
    """Vectorized small-angle SO(3) rotation via Rodrigues.

    Random axis on S², uniform angle in [-step_size, step_size].
    R = cos(a)·I + (1-cos(a))·n⊗n + sin(a)·[n]×
    All computed with broadcasting, no Python loop over n.
    """
    # Random axes: sample Gaussian, normalize
    raw = rng.randn(n, 3)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    axes = raw / norms  # (n, 3)

    # Random angles
    angles = rng.uniform(-step_size, step_size, size=n)  # (n,)

    cos_a = np.cos(angles)  # (n,)
    sin_a = np.sin(angles)  # (n,)
    one_minus_cos = 1.0 - cos_a  # (n,)

    # Outer product: n⊗n, shape (n, 3, 3)
    nn = axes[:, :, None] * axes[:, None, :]  # (n, 3, 3)

    # Cross-product matrix [n]×, shape (n, 3, 3)
    # [n]× = [[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]]
    Kx = np.zeros((n, 3, 3))
    Kx[:, 0, 1] = -axes[:, 2]
    Kx[:, 0, 2] = axes[:, 1]
    Kx[:, 1, 0] = axes[:, 2]
    Kx[:, 1, 2] = -axes[:, 0]
    Kx[:, 2, 0] = -axes[:, 1]
    Kx[:, 2, 1] = axes[:, 0]

    # Identity batch
    I_batch = np.eye(3)[None, :, :] * np.ones((n, 1, 1))

    # Rodrigues: R = cos(a)·I + (1-cos(a))·n⊗n + sin(a)·[n]×
    R = (cos_a[:, None, None] * I_batch +
         one_minus_cos[:, None, None] * nn +
         sin_a[:, None, None] * Kx)

    return R


# =====================================================================
# NEW: Bipartite MC Sweep for SO(3) Frame Field
# =====================================================================

def bipartite_mc_sweep_so3(R, K_F, site_nbr, site_bond,
                           A_sites, B_sites, step_size, beta, rng):
    """Sublattice-parallel Metropolis sweep for SO(3) frames.

    For each sublattice (A then B):
      1. Generate n_sub proposals via random_rotation_small_batch
      2. Compute dE for all proposals in parallel
      3. Accept/reject vectorized
      4. Apply accepted moves

    Energy: E = -Σ_bonds K_F[b] · tr(R_i^T R_j) / 3
    dE_i = -Σ_{j∈nbr(i)} K_F[b_ij] · [tr(R_new_i^T R_j) - tr(R_old_i^T R_j)] / 3

    No Python loop over sites. Returns number of accepted moves.
    """
    z = site_nbr.shape[1]  # coordination number
    n_accepted = 0

    for sites in [A_sites, B_sites]:
        n_sub = len(sites)

        # Generate proposals: R_new = dR @ R_old
        dR = random_rotation_small_batch(n_sub, step_size, rng)
        R_old = R[sites]                      # (n_sub, 3, 3)
        R_new = np.einsum('nab,nbc->nac', dR, R_old)  # (n_sub, 3, 3)

        # Compute dE for each site
        dE = np.zeros(n_sub)
        for k in range(z):
            nbr_sites = site_nbr[sites, k]     # (n_sub,) neighbor indices
            bond_idxs = site_bond[sites, k]     # (n_sub,) bond indices
            R_nbr = R[nbr_sites]                # (n_sub, 3, 3)
            K_b = K_F[bond_idxs]                # (n_sub,)

            # tr(R_new^T R_nbr) - tr(R_old^T R_nbr) per site
            tr_new = np.einsum('nac,nac->n', R_new, R_nbr)   # (n_sub,)
            tr_old = np.einsum('nac,nac->n', R_old, R_nbr)   # (n_sub,)
            dE += -K_b * (tr_new - tr_old) / 3.0

        # Metropolis accept/reject
        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            rand_vals = rng.uniform(size=n_sub)
            accept |= (rand_vals < boltzmann)
        # else: beta=inf → only accept dE <= 0 (ground state search)

        # Apply accepted moves
        accept_idx = np.where(accept)[0]
        R[sites[accept_idx]] = R_new[accept_idx]
        n_accepted += len(accept_idx)

    return n_accepted


# =====================================================================
# Adiabatic CLR: K*(cos) Lookup Table & MC Sweep
# =====================================================================

class _KstarCosLUT:
    """Precomputed lookup table for K*(cos_frame) — the adiabatic CLR fixed point.

    The adiabatic CLR sets K_F = K*(cos_frame) instantly on every bond,
    giving the MC an effective nonlinear potential E_eff = -K*(cos)*cos.

    This contains the Skyrme-like quartic term that emerges from the CLR's
    nonlinear response: d²E_eff/dcos² is CONFINING near cos=1.
    """

    def __init__(self, r_F=6.0, n_points=4096, cos_min=-0.4, cos_max=1.0):
        self.r_F = r_F
        self.n_points = n_points
        self.cos_min = cos_min
        self.cos_max = cos_max
        self.dcos = (cos_max - cos_min) / n_points
        self.cos_grid = np.linspace(cos_min, cos_max, n_points + 1)
        # Precompute K*(cos) for each grid point
        self.Kstar_grid = np.array([
            solve_clr_frame(c, r_F) for c in self.cos_grid
        ])
        # Also precompute E_eff = -K*(cos)*cos for energy evaluation
        self.Eeff_grid = -self.Kstar_grid * self.cos_grid

    def evaluate_K(self, cos_arr):
        """Interpolate K*(cos) for array of cos values."""
        result = np.zeros_like(cos_arr, dtype=float)
        # Clip to LUT range
        c_clipped = np.clip(cos_arr, self.cos_min, self.cos_max - 1e-10)
        idx_f = (c_clipped - self.cos_min) / self.dcos
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        result = (1 - frac) * self.Kstar_grid[idx] + frac * self.Kstar_grid[idx + 1]
        return np.maximum(result, 0.0)

    def evaluate_Eeff(self, cos_arr):
        """Interpolate E_eff(cos) = -K*(cos)*cos for array of cos values."""
        c_clipped = np.clip(cos_arr, self.cos_min, self.cos_max - 1e-10)
        idx_f = (c_clipped - self.cos_min) / self.dcos
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        return (1 - frac) * self.Eeff_grid[idx] + frac * self.Eeff_grid[idx + 1]


def adiabatic_mc_sweep_so3(R, K_F, kstar_lut, site_nbr, site_bond, ei, ej,
                            A_sites, B_sites, step_size, beta, rng):
    """Sublattice-parallel Metropolis with ADIABATIC CLR potential.

    Key difference from standard MC: the energy per bond is
        E_bond = -K*(cos_frame) * cos_frame  (nonlinear in cos)
    not the standard -K * cos. K_F is updated to K*(cos) after each
    sublattice sweep.

    The energy change for proposing R_i → R_i' is:
        dE_i = Σ_{j∈nbr(i)} [E_eff(cos_new_ij) - E_eff(cos_old_ij)]

    where E_eff(cos) = -K*(cos) * cos is the adiabatic effective potential.

    Returns: n_accepted, and K_F is updated in-place to K*(cos_frame).
    """
    z = site_nbr.shape[1]
    n_accepted = 0

    for sites in [A_sites, B_sites]:
        n_sub = len(sites)

        # Generate proposals
        dR = random_rotation_small_batch(n_sub, step_size, rng)
        R_old = R[sites]
        R_new = np.einsum('nab,nbc->nac', dR, R_old)

        # Compute dE using the nonlinear effective potential
        dE = np.zeros(n_sub)
        for k in range(z):
            nbr_sites = site_nbr[sites, k]
            bond_idxs = site_bond[sites, k]
            R_nbr = R[nbr_sites]

            # cos_frame = tr(R^T R_nbr) / 3
            cos_new = np.einsum('nac,nac->n', R_new, R_nbr) / 3.0
            cos_old = np.einsum('nac,nac->n', R_old, R_nbr) / 3.0

            # E_eff(cos) = -K*(cos) * cos  (from LUT)
            E_new = kstar_lut.evaluate_Eeff(cos_new)
            E_old = kstar_lut.evaluate_Eeff(cos_old)
            dE += (E_new - E_old)

        # Metropolis accept/reject
        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            rand_vals = rng.uniform(size=n_sub)
            accept |= (rand_vals < boltzmann)

        accept_idx = np.where(accept)[0]
        R[sites[accept_idx]] = R_new[accept_idx]
        n_accepted += len(accept_idx)

    # Update K_F to K*(cos_frame) for all bonds (adiabatic condition)
    cos_frame = frame_bond_alignment(R, ei, ej)
    K_F[:] = kstar_lut.evaluate_K(cos_frame)

    return n_accepted


# =====================================================================
# NEW: Site Defect Proxy
# =====================================================================

def site_defect_proxy(R, site_nbr):
    """Per-site defect proxy: min(cos_frame) over all z neighbors.

    Low values indicate defect cores where all neighbors are misaligned.
    Returns (N,) array.
    """
    N, z = site_nbr.shape
    min_cos = np.ones(N)
    for k in range(z):
        nbr = site_nbr[:, k]  # (N,)
        # cos_frame = tr(R_i^T R_j) / 3
        cos_k = np.einsum('iac,iac->i', R, R[nbr]) / 3.0  # (N,)
        min_cos = np.minimum(min_cos, cos_k)
    return min_cos


# =====================================================================
# NEW: Dead Bond Cluster Analysis
# =====================================================================

def identify_dead_bond_clusters(K_F, ei, ej, N, K_thresh=1e-4):
    """Find connected components of dead-bond subgraph.

    Dead bonds = bonds with K_F < K_thresh.
    Topology detector: clusters of dead bonds = topological scars.
    """
    dead_mask = K_F < K_thresh
    n_dead = int(np.sum(dead_mask))

    if n_dead == 0:
        return dead_mask, 0, [], []

    # Build subgraph of dead bonds
    dead_ei = ei[dead_mask]
    dead_ej = ej[dead_mask]

    # Need to work with the sites touched by dead bonds
    dead_sites = np.unique(np.concatenate([dead_ei, dead_ej]))
    n_dead_sites = len(dead_sites)

    if n_dead_sites == 0:
        return dead_mask, 0, [], []

    # Build adjacency among dead-bond sites
    row = np.concatenate([dead_ei, dead_ej])
    col = np.concatenate([dead_ej, dead_ei])
    data = np.ones(len(row))
    A_dead = csr_matrix((data, (row, col)), shape=(N, N))

    n_clusters, labels = connected_components(A_dead, directed=False)

    # Collect cluster info (only clusters that actually contain dead-bond sites)
    cluster_sizes = []
    cluster_bond_lists = []
    dead_bond_indices = np.where(dead_mask)[0]

    # Only count clusters that contain dead-bond sites
    active_labels = set(labels[dead_sites])
    for lab in sorted(active_labels):
        sites_in = np.where(labels == lab)[0]
        # Count dead bonds in this cluster
        bonds_in = []
        for b_idx in dead_bond_indices:
            if labels[ei[b_idx]] == lab:
                bonds_in.append(int(b_idx))
        if len(bonds_in) > 0:
            cluster_sizes.append(len(bonds_in))
            cluster_bond_lists.append(bonds_in)

    return dead_mask, len(cluster_sizes), cluster_sizes, cluster_bond_lists


# =====================================================================
# Protocol A: Thermal Scan (MC only)
# =====================================================================

def protocol_a_thermal_scan(L, seed, K_min=0.5, K_max=8.0, n_K=16,
                            n_therm=2000, n_measure=2000,
                            sample_interval=10, verbose=True):
    """Map the frame-sector phase diagram with MC-only thermal scan.

    For each K_F value (uniform, no CLR):
    1. Init R = random uniform SO(3)
    2. Thermalize: n_therm bipartite MC sweeps
    3. Measure: cos_frame_mean, frame magnetization m², defect fraction
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol A: Thermal Scan — L={L}, seed={seed}")
        print(f"  K_F range: [{K_min}, {K_max}], n_K={n_K}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    z = 4  # d+1 for d=3

    K_vals = np.linspace(K_min, K_max, n_K)
    scan_data = []

    for idx_K, K_val in enumerate(K_vals):
        K_F_uniform = np.full(n_bonds, K_val)
        beta = 1.0  # temperature absorbed into K_F

        # Fresh random start
        R = random_rotation_uniform_batch(N, rng)

        # Adaptive step size
        step_size = 0.5 if K_val < 2.0 else 0.3 if K_val < 5.0 else 0.15

        # Thermalize
        for sweep in range(n_therm):
            bipartite_mc_sweep_so3(R, K_F_uniform, site_nbr, site_bond,
                                   A_sites, B_sites, step_size, beta, rng)

        # Measure
        cos_samples = []
        mag_samples = []
        defect_samples = []
        n_samples = n_measure // sample_interval

        for sweep in range(n_measure):
            n_acc = bipartite_mc_sweep_so3(R, K_F_uniform, site_nbr, site_bond,
                                           A_sites, B_sites, step_size, beta, rng)
            if sweep % sample_interval == 0:
                cos_frame = frame_bond_alignment(R, ei, ej)
                cos_samples.append(float(np.mean(cos_frame)))

                # Frame magnetization: M = (1/N) Σ R_i, m² = tr(M^T M)/9
                M_frame = np.mean(R, axis=0)  # (3,3)
                m2 = float(np.sum(M_frame * M_frame) / 9.0)
                mag_samples.append(m2)

                # Defect fraction
                defect_proxy = site_defect_proxy(R, site_nbr)
                defect_frac = float(np.mean(defect_proxy < 0.3))
                defect_samples.append(defect_frac)

        cos_arr = np.array(cos_samples)
        mag_arr = np.array(mag_samples)
        defect_arr = np.array(defect_samples)

        # MC acceptance rate from last sweep
        accept_rate = n_acc / N

        entry = {
            'K_F': float(K_val),
            'cos_frame_mean': float(np.mean(cos_arr)),
            'cos_frame_std': float(np.std(cos_arr)),
            'm2_mean': float(np.mean(mag_arr)),
            'm2_std': float(np.std(mag_arr)),
            'defect_fraction_mean': float(np.mean(defect_arr)),
            'defect_fraction_std': float(np.std(defect_arr)),
            'accept_rate': float(accept_rate),
            'step_size': float(step_size),
        }
        scan_data.append(entry)

        if verbose:
            print(f"  K_F={K_val:.2f}: <cos>={entry['cos_frame_mean']:.4f}, "
                  f"m²={entry['m2_mean']:.4f}, "
                  f"defect={entry['defect_fraction_mean']:.3f}, "
                  f"acc={entry['accept_rate']:.2f}")

    # Estimate K_F_c from steepest magnetization rise
    m2_vals = np.array([d['m2_mean'] for d in scan_data])
    dm2 = np.diff(m2_vals)
    dK = np.diff(K_vals)
    slope = dm2 / dK
    i_max = np.argmax(slope)
    K_F_c_est = float((K_vals[i_max] + K_vals[i_max + 1]) / 2.0)

    t_elapsed = time.time() - t0

    results = {
        'script': SCRIPT,
        'protocol': 'A',
        'L': L,
        'seed': seed,
        'N': N,
        'n_bonds': n_bonds,
        'K_min': K_min,
        'K_max': K_max,
        'n_K': n_K,
        'n_therm': n_therm,
        'n_measure': n_measure,
        'scan_data': scan_data,
        'K_F_c_estimate': K_F_c_est,
        't_elapsed': t_elapsed,
    }

    if verbose:
        print(f"\n  K_F_c estimate (steepest m² rise): {K_F_c_est:.2f}")
        print(f"  Elapsed: {t_elapsed:.1f}s")

    return results


# =====================================================================
# Protocol B: Hot-Start CLR-MC
# =====================================================================

def protocol_b_hotstart_clr_mc(L, seed, r_F=6.0,
                                n_mc_per_clr=10, n_clr_steps=2000,
                                n_anneal_steps=10, n_anneal_sweeps=500,
                                eta=1.0, dt=0.005, struct_weight=0.1,
                                record_interval=10,
                                prune_after=500, K_dead_thresh=1e-6,
                                target_accept=(0.35, 0.45),
                                n_smooth=50, beta=1.0,
                                verbose=True):
    """Hot-start CLR-MC: random frames + CLR → spontaneous topology?

    Two-phase protocol:
    Phase 1 — Annealing: Start from random frames with K_F=0 (infinite T).
      Gradually increase uniform K_F through n_anneal_steps, running MC at
      each level to let the frame field partially order. This is a slow quench
      from the disordered phase into the ordered phase. Any topological defects
      that survive the quench are candidates for Skyrmion nucleation.
    Phase 2 — CLR takeover: Initialize K_F from the CLR fixed-point solution
      of the measured cos_frame per bond. Then run coupled CLR-MC where both
      R_i and K_F co-evolve. The CLR will either confine the surviving defects
      (ordered uniform) or stabilize them (defects survived).
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol B: Hot-Start CLR-MC — L={L}, seed={seed}, r_F={r_F}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)
    lam = 1.0 / r_F

    # Initialize: random frames
    R = random_rotation_uniform_batch(N, rng)

    if verbose:
        print(f"  N={N}, n_bonds={n_bonds}")
        print(f"  Phase 1: Annealing {n_anneal_steps} steps × {n_anneal_sweeps} MC sweeps")

    # ── Phase 1: Annealing — gradual quench from T=∞ to T_final ──
    # Compute K_F* (bulk equilibrium) as the upper target
    K_F_star = solve_clr_frame(1.0, r_F)
    K_anneal_vals = np.linspace(0.5, K_F_star, n_anneal_steps)

    step_size = 0.5
    beta = 1.0
    anneal_timeline = []

    for anneal_idx, K_val in enumerate(K_anneal_vals):
        K_F_uniform = np.full(n_bonds, K_val)

        for sweep in range(n_anneal_sweeps):
            n_acc = bipartite_mc_sweep_so3(
                R, K_F_uniform, site_nbr, site_bond,
                A_sites, B_sites, step_size, beta, rng)
            accept_rate = n_acc / N
            if accept_rate < target_accept[0]:
                step_size *= 0.95
            elif accept_rate > target_accept[1]:
                step_size *= 1.05
            step_size = np.clip(step_size, 0.01, 2.0)

        cos_frame = frame_bond_alignment(R, ei, ej)
        defect_proxy = site_defect_proxy(R, site_nbr)
        defect_frac = float(np.mean(defect_proxy < 0.3))

        anneal_snap = {
            'K_anneal': float(K_val),
            'cos_frame_mean': float(np.mean(cos_frame)),
            'defect_fraction': defect_frac,
            'accept_rate': float(accept_rate),
            'step_size': float(step_size),
        }
        anneal_timeline.append(anneal_snap)

        if verbose:
            print(f"    anneal {anneal_idx}/{n_anneal_steps}: K={K_val:.2f}, "
                  f"<cos>={anneal_snap['cos_frame_mean']:.4f}, "
                  f"defect={defect_frac:.3f}, acc={accept_rate:.2f}")

    # ── Phase 2: CLR takeover — initialize K_F from fixed-point ──
    cos_frame = frame_bond_alignment(R, ei, ej)
    K_F = np.zeros(n_bonds)
    for b in range(n_bonds):
        K_F[b] = solve_clr_frame(cos_frame[b], r_F)

    dead_mask = np.zeros(n_bonds, dtype=bool)
    # Pre-mark frustrated bonds
    frustrated = cos_frame < 0
    dead_mask[frustrated] = True
    K_F[dead_mask] = 0.0

    # K_seed: seed near-zero alive bonds (same role as 184b's K_seed)
    K_seed = 0.01
    dead_seedable = (K_F < 1e-12) & (~dead_mask)
    K_F[dead_seedable] = K_seed

    alive_init = ~dead_mask
    if verbose:
        alive_K = K_F[alive_init]
        n_frust = int(np.sum(frustrated))
        print(f"\n  Phase 2: CLR-MC — {n_clr_steps} steps × {n_mc_per_clr} MC sweeps")
        print(f"  K_F from fixed-point: mean={np.mean(alive_K):.4f}, "
              f"std={np.std(alive_K):.4f}")
        print(f"  Frustrated bonds (dead): {n_frust}/{n_bonds}")
        print(f"  <cos_frame>={np.mean(cos_frame):.4f}")

    # Fiedler init
    _fiedler_cache['X0'] = None
    lambda_2, v_2, degree = sparse_laplacian_and_fiedler(ei, ej, K_F, N)
    fiedler_ema = (v_2[ei] - v_2[ej]) ** 2

    # Thermal CLR: per-bond cos_frame EMA
    alpha_ema = 1.0 / max(n_smooth, 1)
    cos_frame_ema = cos_frame.copy()  # init from current measurement

    timeline = []
    dt_cur = float(dt)
    dt_min = max(dt * 0.05, 1e-5)
    dt_max = min(dt * 3.0, 0.05)

    # Convergence tracking (184b pattern)
    burn_in = 200
    avg_window = 500
    K_sum = np.zeros(n_bonds)
    K_count = 0
    K_avg_prev = None
    converged = False
    conv_tol = 0.01

    if verbose:
        print(f"  Thermal CLR: n_smooth={n_smooth}, alpha_ema={alpha_ema:.4f}")

    for clr_step in range(n_clr_steps):
        # ── MC: evolve frame field ──
        total_acc = 0
        for _ in range(n_mc_per_clr):
            n_acc = bipartite_mc_sweep_so3(
                R, K_F, site_nbr, site_bond,
                A_sites, B_sites, step_size, beta, rng)
            total_acc += n_acc
        accept_rate = total_acc / (n_mc_per_clr * N)

        if accept_rate < target_accept[0]:
            step_size *= 0.95
        elif accept_rate > target_accept[1]:
            step_size *= 1.05
        step_size = np.clip(step_size, 0.01, 2.0)

        # ── Re-measure cos_frame and update EMA ──
        cos_frame = frame_bond_alignment(R, ei, ej)
        cos_frame_ema = (1.0 - alpha_ema) * cos_frame_ema + alpha_ema * cos_frame

        # ── CLR: evolve K_F using noise-averaged cos_frame ──
        if clr_step % 20 == 0:
            lambda_2, v_2, degree = sparse_laplacian_and_fiedler(ei, ej, K_F, N)
            f_new = (v_2[ei] - v_2[ej]) ** 2
            fiedler_ema = 0.9 * fiedler_ema + 0.1 * f_new

        K_dot = frame_shannon_clr_Kdot(
            K_F, cos_frame_ema, ei, ej, N,
            v_2, lambda_2, degree, eta, lam,
            fiedler_sens=fiedler_ema, dead_mask=dead_mask,
            struct_weight=struct_weight)

        # Adaptive dt
        alive = ~dead_mask
        if np.any(alive):
            kdot_inf = float(np.max(np.abs(K_dot[alive])))
        else:
            kdot_inf = 0.0
        if kdot_inf > 5.0:
            dt_cur = max(dt_min, dt_cur * 0.7)
        elif kdot_inf < 0.2:
            dt_cur = min(dt_max, dt_cur * 1.02)

        K_F = np.clip(K_F + dt_cur * K_dot, 0.0, 100.0)
        K_F[dead_mask] = 0.0

        # ── Dead bond pruning (prune_after=500, matching 184b convention) ──
        if clr_step >= prune_after:
            newly_dead = (~dead_mask) & (K_F < K_dead_thresh) & (cos_frame < 0)
            if np.any(newly_dead):
                dead_mask |= newly_dead
                K_F[dead_mask] = 0.0

        # ── Convergence check (184b pattern) ──
        if clr_step >= burn_in:
            K_sum += K_F
            K_count += 1
            if K_count > 0 and K_count % avg_window == 0:
                K_avg = K_sum / K_count
                if K_avg_prev is not None:
                    alive = ~dead_mask
                    if np.any(alive):
                        delta = float(np.max(np.abs(
                            K_avg[alive] - K_avg_prev[alive])))
                    else:
                        delta = 0.0
                    if delta < conv_tol:
                        converged = True
                        if verbose:
                            print(f"    CLR converged at step {clr_step} "
                                  f"(|dK_avg|_inf = {delta:.2e})")
                        break
                K_avg_prev = K_avg.copy()
                K_sum = np.zeros(n_bonds)
                K_count = 0

        # ── Record observables ──
        if clr_step % record_interval == 0:
            alive = ~dead_mask
            alive_K = K_F[alive] if np.any(alive) else np.array([0.0])
            defect_proxy = site_defect_proxy(R, site_nbr)
            defect_frac = float(np.mean(defect_proxy < 0.3))

            _, n_clusters, cluster_sizes, _ = identify_dead_bond_clusters(
                K_F, ei, ej, N, K_thresh=K_dead_thresh)

            snap = {
                'clr_step': clr_step,
                'K_F_mean': float(np.mean(alive_K)),
                'K_F_std': float(np.std(alive_K)),
                'cos_frame_mean': float(np.mean(cos_frame)),
                'n_dead': int(np.sum(dead_mask)),
                'n_dead_clusters': n_clusters,
                'dead_cluster_sizes': cluster_sizes[:10],  # top 10
                'defect_fraction': defect_frac,
                'mc_accept_rate': float(accept_rate),
                'lambda_2': float(lambda_2),
                'step_size': float(step_size),
                'dt': float(dt_cur),
            }
            timeline.append(snap)

            if verbose and clr_step % (record_interval * 10) == 0:
                print(f"  step {clr_step}: K_F={snap['K_F_mean']:.4f}, "
                      f"<cos>={snap['cos_frame_mean']:.4f}, "
                      f"dead={snap['n_dead']}, "
                      f"clusters={n_clusters}, "
                      f"defect={defect_frac:.3f}, "
                      f"acc={accept_rate:.2f}")

    # ── Use time-averaged K_F if we accumulated enough ──
    if K_count > 0:
        K_F_eq = K_sum / K_count
        K_F_eq[dead_mask] = 0.0
    else:
        K_F_eq = K_F.copy()

    # ── Final analysis ──
    cos_frame_final = frame_bond_alignment(R, ei, ej)
    _, n_clusters_final, cluster_sizes_final, _ = identify_dead_bond_clusters(
        K_F_eq, ei, ej, N, K_thresh=K_dead_thresh)
    defect_proxy_final = site_defect_proxy(R, site_nbr)
    defect_frac_final = float(np.mean(defect_proxy_final < 0.3))

    alive = ~dead_mask
    alive_K = K_F_eq[alive] if np.any(alive) else np.array([0.0])

    # Mass measurement
    mass = measure_skyrmion_mass(K_F_eq, dead_mask)

    # Radial K_F profile
    profile = radial_K_profile(K_F_eq, ei, ej, positions, center, n_bins=12)

    # Verdict
    n_dead_final = int(np.sum(dead_mask))
    if n_dead_final > 0 and n_clusters_final > 0:
        verdict = 'DEFECTS_SURVIVED'
    elif float(np.mean(alive_K)) > 1.0 and n_dead_final == 0:
        verdict = 'ORDERED_UNIFORM'
    elif not converged:
        verdict = 'NOT_CONVERGED'
    else:
        verdict = 'ORDERED_UNIFORM'

    t_elapsed = time.time() - t0

    results = {
        'script': SCRIPT,
        'protocol': 'B',
        'L': L,
        'seed': seed,
        'r_F': r_F,
        'N': N,
        'n_bonds': n_bonds,
        'n_anneal_steps': n_anneal_steps,
        'n_anneal_sweeps': n_anneal_sweeps,
        'n_mc_per_clr': n_mc_per_clr,
        'n_clr_steps': n_clr_steps,
        'converged': converged,
        'n_smooth': n_smooth,
        'beta': beta,
        'eta': eta,
        'dt': dt,
        'struct_weight': struct_weight,
        'anneal_timeline': anneal_timeline,
        'K_F_mean_final': float(np.mean(alive_K)),
        'K_F_std_final': float(np.std(alive_K)),
        'cos_frame_mean_final': float(np.mean(cos_frame_final)),
        'n_dead_final': n_dead_final,
        'n_dead_clusters_final': n_clusters_final,
        'dead_cluster_sizes_final': cluster_sizes_final[:10],
        'defect_fraction_final': defect_frac_final,
        'mass': mass,
        'radial_profile': profile,
        'timeline': timeline,
        'verdict': verdict,
        't_elapsed': t_elapsed,
    }

    if verbose:
        print(f"\n  Final: K_F_mean={results['K_F_mean_final']:.4f}, "
              f"<cos>={results['cos_frame_mean_final']:.4f}")
        print(f"  Dead bonds: {n_dead_final}, clusters: {n_clusters_final}")
        if cluster_sizes_final:
            print(f"  Cluster sizes: {cluster_sizes_final[:10]}")
        print(f"  Defect fraction: {defect_frac_final:.3f}")
        print(f"  M_skyrmion: {mass['M_skyrmion']:.4f}")
        print(f"  Verdict: {verdict}")
        print(f"  Elapsed: {t_elapsed:.1f}s")

    return results


# =====================================================================
# Protocol C: Localized Kick + CLR-MC
# =====================================================================

def protocol_c_localized_kick(L, seed, r_F=6.0, kick_strength=1.0,
                               kick_radius=None, n_equil_mc=2000,
                               n_mc_per_clr=10, n_clr_steps=2000,
                               eta=1.0, dt=0.005, struct_weight=0.1,
                               record_interval=10,
                               prune_after=500, K_dead_thresh=1e-6,
                               target_accept=(0.35, 0.45),
                               n_smooth=50, beta=1.0,
                               thermal_init=True,
                               verbose=True):
    """Localized kick + CLR-MC: inject frame perturbation → nucleation?

    1. Init ordered: R_i = I, K_F = K_F*(r_F) or K*_thermal
    2. Pre-equilibrate: MC sweeps near ordered state
    3. Apply hedgehog kick at lattice center
    4. Run coupled CLR-MC
    5. Diagnose: persist (SKYRMION_FORMED) or dissipate (DISSIPATED)?
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol C: Localized Kick — L={L}, seed={seed}, "
              f"r_F={r_F}, kick={kick_strength}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)
    deltas = make_simplex_deltas(3)
    nn_dist = float(np.linalg.norm(deltas[0]))
    lam = 1.0 / r_F

    if kick_radius is None:
        kick_radius = L * nn_dist / 4.0

    # Initialize: ordered state
    R = np.tile(np.eye(3), (N, 1, 1))  # All identity

    # Compute bulk K_F* from CLR fixed point at cos_frame=1
    K_F_star = solve_clr_frame(1.0, r_F)
    K_F = np.full(n_bonds, K_F_star)
    dead_mask = np.zeros(n_bonds, dtype=bool)

    if verbose:
        print(f"  N={N}, n_bonds={n_bonds}")
        print(f"  K_F*(r_F={r_F}) = {K_F_star:.4f}")
        print(f"  kick_radius={kick_radius:.4f}, kick_strength={kick_strength}")

    # ── Pre-equilibrate with MC near ordered state ──
    step_size = 0.15

    if verbose:
        print(f"  Pre-equilibrating: {n_equil_mc} MC sweeps...")

    for sweep in range(n_equil_mc):
        n_acc = bipartite_mc_sweep_so3(
            R, K_F, site_nbr, site_bond,
            A_sites, B_sites, step_size, beta, rng)
        # Adapt step size
        accept_rate = n_acc / N
        if accept_rate < target_accept[0]:
            step_size *= 0.95
        elif accept_rate > target_accept[1]:
            step_size *= 1.05
        step_size = np.clip(step_size, 0.01, 2.0)

    cos_frame_pre = frame_bond_alignment(R, ei, ej)
    if verbose:
        print(f"  Pre-kick: <cos_frame>={np.mean(cos_frame_pre):.4f}, "
              f"acc={accept_rate:.2f}, step={step_size:.3f}")

    # ── Thermal self-consistent K initialization ──
    if thermal_init:
        if verbose:
            print(f"  Thermal init: self-consistent K at beta={beta}...")
        K_F, cos_eq = thermal_self_consistent_K(
            R, ei, ej, N, site_nbr, site_bond,
            A_sites, B_sites, r_F, beta,
            n_iter=3, n_mc_sweeps=200, rng=rng,
            step_size=step_size, verbose=verbose)
        K_F_thermal = float(np.mean(K_F))
        if verbose:
            print(f"  K*_thermal = {K_F_thermal:.4f} "
                  f"(vs K*_T0 = {K_F_star:.4f}, ratio = {K_F_thermal/K_F_star:.4f})")
    else:
        cos_eq = cos_frame_pre.copy()

    cos_frame_pre = frame_bond_alignment(R, ei, ej)
    if verbose:
        print(f"  Pre-kick (post thermal init): "
              f"<cos_frame>={np.mean(cos_frame_pre):.4f}")

    # ── Apply hedgehog kick ──
    dists = np.linalg.norm(positions - center, axis=1)
    kick_mask = dists < kick_radius
    n_kicked = int(np.sum(kick_mask))

    for i in np.where(kick_mask)[0]:
        r_vec = positions[i] - center
        r = np.linalg.norm(r_vec)
        # Profile: angle decreases with distance from center
        f_val = kick_strength * np.pi * max(1.0 - r / kick_radius, 0.0)
        R_kick = hedgehog_rotation(r_vec, f_val)
        R[i] = R_kick @ R[i]

    cos_frame_post_kick = frame_bond_alignment(R, ei, ej)
    if verbose:
        print(f"  Post-kick: {n_kicked} sites kicked, "
              f"<cos_frame>={np.mean(cos_frame_post_kick):.4f}")

    # ── Coupled CLR-MC evolution ──
    _fiedler_cache['X0'] = None
    lambda_2, v_2, degree = sparse_laplacian_and_fiedler(ei, ej, K_F, N)
    fiedler_ema = (v_2[ei] - v_2[ej]) ** 2

    # Thermal CLR: per-bond cos_frame EMA
    alpha_ema = 1.0 / max(n_smooth, 1)
    cos_frame_ema = cos_eq.copy()  # init from thermal equilibrium

    dt_cur = float(dt)
    dt_min = max(dt * 0.05, 1e-5)
    dt_max = min(dt * 3.0, 0.05)

    timeline = []

    # Convergence tracking (184b pattern)
    burn_in = 200
    avg_window = 500
    K_sum_conv = np.zeros(n_bonds)
    K_count_conv = 0
    K_avg_prev_conv = None
    converged = False
    conv_tol = 0.01

    if verbose:
        print(f"  Running {n_clr_steps} CLR-MC steps "
              f"(thermal CLR: n_smooth={n_smooth}, beta={beta})...")

    for clr_step in range(n_clr_steps):
        # MC sweeps
        total_acc = 0
        for _ in range(n_mc_per_clr):
            n_acc = bipartite_mc_sweep_so3(
                R, K_F, site_nbr, site_bond,
                A_sites, B_sites, step_size, beta, rng)
            total_acc += n_acc
        accept_rate = total_acc / (n_mc_per_clr * N)

        if accept_rate < target_accept[0]:
            step_size *= 0.95
        elif accept_rate > target_accept[1]:
            step_size *= 1.05
        step_size = np.clip(step_size, 0.01, 2.0)

        # Re-measure cos_frame and update EMA
        cos_frame = frame_bond_alignment(R, ei, ej)
        cos_frame_ema = (1.0 - alpha_ema) * cos_frame_ema + alpha_ema * cos_frame

        # CLR step using noise-averaged cos_frame
        if clr_step % 20 == 0:
            lambda_2, v_2, degree = sparse_laplacian_and_fiedler(ei, ej, K_F, N)
            f_new = (v_2[ei] - v_2[ej]) ** 2
            fiedler_ema = 0.9 * fiedler_ema + 0.1 * f_new

        K_dot = frame_shannon_clr_Kdot(
            K_F, cos_frame_ema, ei, ej, N,
            v_2, lambda_2, degree, eta, lam,
            fiedler_sens=fiedler_ema, dead_mask=dead_mask,
            struct_weight=struct_weight)

        alive = ~dead_mask
        if np.any(alive):
            kdot_inf = float(np.max(np.abs(K_dot[alive])))
        else:
            kdot_inf = 0.0
        if kdot_inf > 5.0:
            dt_cur = max(dt_min, dt_cur * 0.7)
        elif kdot_inf < 0.2:
            dt_cur = min(dt_max, dt_cur * 1.02)

        K_F = np.clip(K_F + dt_cur * K_dot, 0.0, 100.0)
        K_F[dead_mask] = 0.0

        # Dead bond pruning
        if clr_step >= prune_after:
            newly_dead = (~dead_mask) & (K_F < K_dead_thresh) & (cos_frame < 0)
            if np.any(newly_dead):
                dead_mask |= newly_dead
                K_F[dead_mask] = 0.0

        # Convergence check
        if clr_step >= burn_in:
            K_sum_conv += K_F
            K_count_conv += 1
            if K_count_conv > 0 and K_count_conv % avg_window == 0:
                K_avg = K_sum_conv / K_count_conv
                if K_avg_prev_conv is not None:
                    alive = ~dead_mask
                    if np.any(alive):
                        delta = float(np.max(np.abs(
                            K_avg[alive] - K_avg_prev_conv[alive])))
                    else:
                        delta = 0.0
                    if delta < conv_tol:
                        converged = True
                        if verbose:
                            print(f"    CLR converged at step {clr_step} "
                                  f"(|dK_avg|_inf = {delta:.2e})")
                        break
                K_avg_prev_conv = K_avg.copy()
                K_sum_conv = np.zeros(n_bonds)
                K_count_conv = 0

        # Record
        if clr_step % record_interval == 0:
            alive = ~dead_mask
            alive_K = K_F[alive] if np.any(alive) else np.array([0.0])
            defect_proxy = site_defect_proxy(R, site_nbr)
            defect_frac = float(np.mean(defect_proxy < 0.3))

            _, n_clusters, cluster_sizes, _ = identify_dead_bond_clusters(
                K_F, ei, ej, N, K_thresh=K_dead_thresh)

            snap = {
                'clr_step': clr_step,
                'K_F_mean': float(np.mean(alive_K)),
                'K_F_std': float(np.std(alive_K)),
                'cos_frame_mean': float(np.mean(cos_frame)),
                'n_dead': int(np.sum(dead_mask)),
                'n_dead_clusters': n_clusters,
                'dead_cluster_sizes': cluster_sizes[:10],
                'defect_fraction': defect_frac,
                'mc_accept_rate': float(accept_rate),
                'lambda_2': float(lambda_2),
                'step_size': float(step_size),
                'dt': float(dt_cur),
            }
            timeline.append(snap)

            if verbose and clr_step % (record_interval * 10) == 0:
                print(f"  step {clr_step}: K_F={snap['K_F_mean']:.4f}, "
                      f"<cos>={snap['cos_frame_mean']:.4f}, "
                      f"dead={snap['n_dead']}, "
                      f"clusters={n_clusters}, "
                      f"acc={accept_rate:.2f}")

    # ── Use time-averaged K_F if we accumulated enough (184b pattern) ──
    if K_count_conv > 0:
        K_F_eq = K_sum_conv / K_count_conv
        K_F_eq[dead_mask] = 0.0
    else:
        K_F_eq = K_F.copy()

    # ── Final analysis ──
    cos_frame_final = frame_bond_alignment(R, ei, ej)
    _, n_clusters_final, cluster_sizes_final, _ = identify_dead_bond_clusters(
        K_F_eq, ei, ej, N, K_thresh=K_dead_thresh)
    defect_proxy_final = site_defect_proxy(R, site_nbr)
    defect_frac_final = float(np.mean(defect_proxy_final < 0.3))

    alive = ~dead_mask
    alive_K = K_F_eq[alive] if np.any(alive) else np.array([0.0])
    n_dead_final = int(np.sum(dead_mask))

    mass = measure_skyrmion_mass(K_F_eq, dead_mask)
    profile = radial_K_profile(K_F_eq, ei, ej, positions, center, n_bins=12)

    # Verdict
    # Check for core depression near kick site
    has_core_depression = False
    if profile and len(profile) > 1:
        K_inner = profile[0]['K_mean']
        K_outer_mean = np.mean([p['K_mean'] for p in profile[-3:]])
        if K_outer_mean > 0:
            has_core_depression = K_inner < 0.8 * K_outer_mean

    if n_dead_final > 0 and has_core_depression:
        verdict = 'SKYRMION_FORMED'
    elif n_dead_final > 0 and not has_core_depression:
        verdict = 'DEFECT_STABLE'
    elif n_dead_final == 0 and float(np.mean(alive_K)) > 0.5 * K_F_star:
        # Topology gone: K_F high and uniform, no dead bonds
        verdict = 'DISSIPATED'
    elif not converged:
        verdict = 'NOT_CONVERGED'
    else:
        verdict = 'DISSIPATED'

    t_elapsed = time.time() - t0

    results = {
        'script': SCRIPT,
        'protocol': 'C',
        'L': L,
        'seed': seed,
        'r_F': r_F,
        'kick_strength': kick_strength,
        'kick_radius': kick_radius,
        'N': N,
        'n_bonds': n_bonds,
        'n_equil_mc': n_equil_mc,
        'n_mc_per_clr': n_mc_per_clr,
        'n_clr_steps': n_clr_steps,
        'K_F_star': K_F_star,
        'n_smooth': n_smooth,
        'beta': beta,
        'thermal_init': thermal_init,
        'converged': converged,
        'n_kicked': n_kicked,
        'cos_frame_pre_kick': float(np.mean(cos_frame_pre)),
        'cos_frame_post_kick': float(np.mean(cos_frame_post_kick)),
        'K_F_mean_final': float(np.mean(alive_K)),
        'K_F_std_final': float(np.std(alive_K)),
        'cos_frame_mean_final': float(np.mean(cos_frame_final)),
        'n_dead_final': n_dead_final,
        'n_dead_clusters_final': n_clusters_final,
        'dead_cluster_sizes_final': cluster_sizes_final[:10],
        'defect_fraction_final': defect_frac_final,
        'mass': mass,
        'radial_profile': profile,
        'timeline': timeline,
        'verdict': verdict,
        't_elapsed': t_elapsed,
    }

    if verbose:
        print(f"\n  Final: K_F_mean={results['K_F_mean_final']:.4f}, "
              f"<cos>={results['cos_frame_mean_final']:.4f}")
        print(f"  Dead bonds: {n_dead_final}, clusters: {n_clusters_final}")
        if cluster_sizes_final:
            print(f"  Cluster sizes: {cluster_sizes_final[:10]}")
        print(f"  Core depression: {has_core_depression}")
        print(f"  M_skyrmion: {mass['M_skyrmion']:.4f}")
        print(f"  Verdict: {verdict}")
        print(f"  Elapsed: {t_elapsed:.1f}s")

    return results


# =====================================================================
# Protocol D: Adiabatic CLR + Localized Kick
# =====================================================================

def protocol_d_adiabatic_kick(L, seed, r_F=6.0, kick_strength=1.0,
                               kick_radius=None, n_equil_mc=2000,
                               n_mc_steps=5000, beta=1.0,
                               record_interval=50,
                               target_accept=(0.35, 0.45),
                               verbose=True):
    """Adiabatic CLR + localized kick: K_F = K*(cos_frame) at every MC step.

    The CLR is infinitely fast — K_F tracks cos_frame instantaneously via
    the fixed-point equation K = (r_F/2)*cos*R_0(K). This gives the MC
    a nonlinear effective potential E_eff = -K*(cos)*cos that contains a
    Skyrme-like quartic term from the CLR's nonlinear response.

    No separate CLR time-stepping. No EMA. No thermal init needed (K always
    tracks cos). The topology detector is the K_F field itself — depressions
    mark defect cores, dead bonds mark topological scars.

    Protocol:
    1. Build lattice, init R = I (ordered)
    2. Build K*(cos) LUT for given r_F
    3. Pre-equilibrate: MC sweeps with adiabatic K_F
    4. Apply hedgehog kick at lattice center
    5. Run adiabatic MC: each sweep updates R AND K_F
    6. Diagnose: persist (SKYRMION_FORMED) or dissipate (DISSIPATED)?
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol D: Adiabatic CLR + Kick — L={L}, seed={seed}, "
              f"r_F={r_F}, kick={kick_strength}, beta={beta}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)
    deltas = make_simplex_deltas(3)
    nn_dist = float(np.linalg.norm(deltas[0]))

    if kick_radius is None:
        kick_radius = L * nn_dist / 4.0

    # Build K*(cos) lookup table
    if verbose:
        print(f"  Building K*(cos) LUT for r_F={r_F}...")
    kstar_lut = _KstarCosLUT(r_F=r_F)

    # Initialize: ordered state
    R = np.tile(np.eye(3), (N, 1, 1))
    K_F_star = solve_clr_frame(1.0, r_F)

    # Set K_F from adiabatic condition
    cos_frame = frame_bond_alignment(R, ei, ej)
    K_F = kstar_lut.evaluate_K(cos_frame)

    if verbose:
        print(f"  N={N}, n_bonds={n_bonds}")
        print(f"  K_F*(cos=1) = {K_F_star:.4f}")
        print(f"  Adiabatic K_F init: mean={np.mean(K_F):.4f}")
        print(f"  kick_radius={kick_radius:.4f}, kick_strength={kick_strength}")

    # ── Pre-equilibrate with adiabatic MC ──
    step_size = 0.15

    if verbose:
        print(f"  Pre-equilibrating: {n_equil_mc} adiabatic MC sweeps...")

    for sweep in range(n_equil_mc):
        n_acc = adiabatic_mc_sweep_so3(
            R, K_F, kstar_lut, site_nbr, site_bond, ei, ej,
            A_sites, B_sites, step_size, beta, rng)
        accept_rate = n_acc / N
        if accept_rate < target_accept[0]:
            step_size *= 0.95
        elif accept_rate > target_accept[1]:
            step_size *= 1.05
        step_size = np.clip(step_size, 0.01, 2.0)

    cos_frame_pre = frame_bond_alignment(R, ei, ej)
    if verbose:
        print(f"  Pre-kick: <cos_frame>={np.mean(cos_frame_pre):.4f}, "
              f"K_F_mean={np.mean(K_F):.4f}, acc={accept_rate:.2f}")

    # ── Apply hedgehog kick ──
    dists = np.linalg.norm(positions - center, axis=1)
    kick_mask = dists < kick_radius
    n_kicked = int(np.sum(kick_mask))

    for i in np.where(kick_mask)[0]:
        r_vec = positions[i] - center
        r = np.linalg.norm(r_vec)
        f_val = kick_strength * np.pi * max(1.0 - r / kick_radius, 0.0)
        R_kick = hedgehog_rotation(r_vec, f_val)
        R[i] = R_kick @ R[i]

    # Update K_F after kick (adiabatic condition)
    cos_frame_post_kick = frame_bond_alignment(R, ei, ej)
    K_F[:] = kstar_lut.evaluate_K(cos_frame_post_kick)

    if verbose:
        print(f"  Post-kick: {n_kicked} sites kicked, "
              f"<cos_frame>={np.mean(cos_frame_post_kick):.4f}, "
              f"K_F_mean={np.mean(K_F):.4f}")
        # Show K_F at kick core
        bond_center = (positions[ei] + positions[ej]) / 2.0
        bond_dists = np.linalg.norm(bond_center - center, axis=1)
        core_mask = bond_dists < kick_radius * 0.5
        if np.any(core_mask):
            print(f"  Core K_F (r < R_kick/2): mean={np.mean(K_F[core_mask]):.4f}, "
                  f"min={np.min(K_F[core_mask]):.4f}")

    # ── Adiabatic MC evolution ──
    timeline = []

    if verbose:
        print(f"  Running {n_mc_steps} adiabatic MC sweeps...")

    for mc_step in range(n_mc_steps):
        n_acc = adiabatic_mc_sweep_so3(
            R, K_F, kstar_lut, site_nbr, site_bond, ei, ej,
            A_sites, B_sites, step_size, beta, rng)
        accept_rate = n_acc / N

        if accept_rate < target_accept[0]:
            step_size *= 0.95
        elif accept_rate > target_accept[1]:
            step_size *= 1.05
        step_size = np.clip(step_size, 0.01, 2.0)

        # Record
        if mc_step % record_interval == 0:
            cos_frame = frame_bond_alignment(R, ei, ej)
            dead_mask_snap = K_F < 1e-4
            n_dead = int(np.sum(dead_mask_snap))
            alive = ~dead_mask_snap
            alive_K = K_F[alive] if np.any(alive) else np.array([0.0])

            _, n_clusters, cluster_sizes, _ = identify_dead_bond_clusters(
                K_F, ei, ej, N, K_thresh=1e-4)

            defect_proxy = site_defect_proxy(R, site_nbr)
            defect_frac = float(np.mean(defect_proxy < 0.3))

            snap = {
                'mc_step': mc_step,
                'K_F_mean': float(np.mean(alive_K)),
                'K_F_std': float(np.std(alive_K)),
                'K_F_min': float(np.min(K_F)),
                'cos_frame_mean': float(np.mean(cos_frame)),
                'n_dead': n_dead,
                'n_dead_clusters': n_clusters,
                'dead_cluster_sizes': cluster_sizes[:10],
                'defect_fraction': defect_frac,
                'mc_accept_rate': float(accept_rate),
                'step_size': float(step_size),
            }
            timeline.append(snap)

            if verbose and mc_step % (record_interval * 10) == 0:
                print(f"  step {mc_step}: K_F={snap['K_F_mean']:.4f}±{snap['K_F_std']:.4f}, "
                      f"<cos>={snap['cos_frame_mean']:.4f}, "
                      f"dead={n_dead}, clusters={n_clusters}, "
                      f"acc={accept_rate:.2f}")

    # ── Final analysis ──
    cos_frame_final = frame_bond_alignment(R, ei, ej)
    K_F_final = kstar_lut.evaluate_K(cos_frame_final)  # ensure adiabatic
    dead_mask = K_F_final < 1e-4

    _, n_clusters_final, cluster_sizes_final, _ = identify_dead_bond_clusters(
        K_F_final, ei, ej, N, K_thresh=1e-4)
    defect_proxy_final = site_defect_proxy(R, site_nbr)
    defect_frac_final = float(np.mean(defect_proxy_final < 0.3))

    alive = ~dead_mask
    alive_K = K_F_final[alive] if np.any(alive) else np.array([0.0])
    n_dead_final = int(np.sum(dead_mask))

    mass = measure_skyrmion_mass(K_F_final, dead_mask)
    profile = radial_K_profile(K_F_final, ei, ej, positions, center, n_bins=12)

    # Verdict
    has_core_depression = False
    if profile and len(profile) > 1:
        K_inner = profile[0]['K_mean']
        K_outer_mean = np.mean([p['K_mean'] for p in profile[-3:]])
        if K_outer_mean > 0:
            has_core_depression = K_inner < 0.8 * K_outer_mean

    if n_dead_final > 0 and has_core_depression:
        verdict = 'SKYRMION_FORMED'
    elif n_dead_final > 0 and not has_core_depression:
        verdict = 'DEFECT_STABLE'
    elif has_core_depression and float(np.mean(alive_K)) > 0.3 * K_F_star:
        verdict = 'SKYRMION_PARTIAL'  # depression but no dead bonds
    elif n_dead_final == 0 and float(np.mean(alive_K)) > 0.5 * K_F_star:
        verdict = 'DISSIPATED'
    else:
        verdict = 'DISORDERED'

    t_elapsed = time.time() - t0

    results = {
        'script': SCRIPT,
        'protocol': 'D',
        'L': L,
        'seed': seed,
        'r_F': r_F,
        'kick_strength': kick_strength,
        'kick_radius': kick_radius,
        'beta': beta,
        'N': N,
        'n_bonds': n_bonds,
        'n_equil_mc': n_equil_mc,
        'n_mc_steps': n_mc_steps,
        'K_F_star': K_F_star,
        'n_kicked': n_kicked,
        'cos_frame_pre_kick': float(np.mean(cos_frame_pre)),
        'cos_frame_post_kick': float(np.mean(cos_frame_post_kick)),
        'K_F_mean_final': float(np.mean(alive_K)),
        'K_F_std_final': float(np.std(alive_K)),
        'cos_frame_mean_final': float(np.mean(cos_frame_final)),
        'n_dead_final': n_dead_final,
        'n_dead_clusters_final': n_clusters_final,
        'dead_cluster_sizes_final': cluster_sizes_final[:10],
        'defect_fraction_final': defect_frac_final,
        'mass': mass,
        'radial_profile': profile,
        'timeline': timeline,
        'verdict': verdict,
        't_elapsed': t_elapsed,
    }

    if verbose:
        print(f"\n  Final: K_F_mean={results['K_F_mean_final']:.4f}±{results['K_F_std_final']:.4f}, "
              f"<cos>={results['cos_frame_mean_final']:.4f}")
        print(f"  Dead bonds: {n_dead_final}, clusters: {n_clusters_final}")
        if cluster_sizes_final:
            print(f"  Cluster sizes: {cluster_sizes_final[:10]}")
        print(f"  Core depression: {has_core_depression}")
        print(f"  M_skyrmion: {mass['M_skyrmion']:.4f}")
        print(f"  Verdict: {verdict}")
        print(f"  Elapsed: {t_elapsed:.1f}s")

    return results


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Spontaneous Skyrmion Generation via Coupled CLR-MC')
    parser.add_argument('--protocol', type=str, default='all',
                        choices=['A', 'B', 'C', 'D', 'all'],
                        help='Protocol to run (default: all)')
    parser.add_argument('--L', type=int, default=8,
                        help='Lattice size (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--r_F', type=float, default=6.0,
                        help='Frame CLR coupling r_F (default: 6.0)')
    parser.add_argument('--n_clr_steps', type=int, default=2000,
                        help='CLR steps for Protocol B/C (default: 2000)')
    parser.add_argument('--kick_strength', type=float, default=1.0,
                        help='Protocol C kick strength (default: 1.0)')
    parser.add_argument('--n_mc_per_clr', type=int, default=10,
                        help='MC sweeps per CLR step (default: 10)')
    parser.add_argument('--n_therm', type=int, default=2000,
                        help='Protocol A thermalization sweeps (default: 2000)')
    parser.add_argument('--n_measure', type=int, default=2000,
                        help='Protocol A measurement sweeps (default: 2000)')
    parser.add_argument('--n_equil_mc', type=int, default=2000,
                        help='Protocol C pre-equilibration sweeps (default: 2000)')
    parser.add_argument('--n_anneal_steps', type=int, default=10,
                        help='Protocol B annealing K levels (default: 10)')
    parser.add_argument('--n_anneal_sweeps', type=int, default=500,
                        help='Protocol B MC sweeps per anneal level (default: 500)')
    parser.add_argument('--n_smooth', type=int, default=50,
                        help='Thermal CLR: EMA window for cos_frame (default: 50)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Inverse temperature for MC (default: 1.0)')
    parser.add_argument('--no_thermal_init', action='store_true',
                        help='Disable self-consistent thermal K init (Protocol C)')
    parser.add_argument('--n_mc_steps', type=int, default=5000,
                        help='Protocol D: total MC sweeps (default: 5000)')
    parser.add_argument('--kick_radius', type=float, default=None,
                        help='Protocol C/D: kick radius (default: L*nn/4)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    verbose = not args.quiet
    protocols = ['A', 'B', 'C', 'D'] if args.protocol == 'all' else [args.protocol]

    all_results = {}

    for proto in protocols:
        if proto == 'A':
            res = protocol_a_thermal_scan(
                L=args.L, seed=args.seed,
                n_therm=args.n_therm, n_measure=args.n_measure,
                verbose=verbose)
        elif proto == 'B':
            res = protocol_b_hotstart_clr_mc(
                L=args.L, seed=args.seed, r_F=args.r_F,
                n_mc_per_clr=args.n_mc_per_clr,
                n_clr_steps=args.n_clr_steps,
                n_anneal_steps=args.n_anneal_steps,
                n_anneal_sweeps=args.n_anneal_sweeps,
                n_smooth=args.n_smooth, beta=args.beta,
                verbose=verbose)
        elif proto == 'C':
            res = protocol_c_localized_kick(
                L=args.L, seed=args.seed, r_F=args.r_F,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                n_equil_mc=args.n_equil_mc,
                n_mc_per_clr=args.n_mc_per_clr,
                n_clr_steps=args.n_clr_steps,
                n_smooth=args.n_smooth, beta=args.beta,
                thermal_init=not args.no_thermal_init,
                verbose=verbose)
        elif proto == 'D':
            res = protocol_d_adiabatic_kick(
                L=args.L, seed=args.seed, r_F=args.r_F,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                n_equil_mc=args.n_equil_mc,
                n_mc_steps=args.n_mc_steps,
                beta=args.beta,
                verbose=verbose)

        out_file = os.path.join(
            OUTPUT_DIR,
            f"skyrmion_spontaneous_{proto}_L{args.L}_s{args.seed}.json")
        with open(out_file, 'w') as f:
            json.dump(jsonify(res), f, indent=2)
        if verbose:
            print(f"\n  Saved: {out_file}")

        all_results[proto] = res

    # Summary
    if len(protocols) > 1 and verbose:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        if 'A' in all_results:
            print(f"  Protocol A: K_F_c ≈ {all_results['A']['K_F_c_estimate']:.2f}")
        if 'B' in all_results:
            print(f"  Protocol B: {all_results['B']['verdict']} "
                  f"(dead={all_results['B']['n_dead_final']}, "
                  f"clusters={all_results['B']['n_dead_clusters_final']})")
        if 'C' in all_results:
            print(f"  Protocol C: {all_results['C']['verdict']} "
                  f"(dead={all_results['C']['n_dead_final']}, "
                  f"M={all_results['C']['mass']['M_skyrmion']:.2f})")
        if 'D' in all_results:
            print(f"  Protocol D: {all_results['D']['verdict']} "
                  f"(dead={all_results['D']['n_dead_final']}, "
                  f"M={all_results['D']['mass']['M_skyrmion']:.2f})")


if __name__ == '__main__':
    main()
