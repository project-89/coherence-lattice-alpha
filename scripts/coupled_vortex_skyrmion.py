#!/usr/bin/env python3
"""
Coupled Vortex-Skyrmion Topological Stabilization (EXP-LT-195)

The isolated frame-sector Skyrmion (EXP-LT-192, Protocol D) is metastable on T^3
— surviving ~7000 MC sweeps at beta=5 before unwinding (pi_3(T^3)=0 provides no
absolute topological protection).

Hypothesis: coupling the Skyrmion (frame sector, pi_3(SU(2))) to a phase vortex
(phase sector, pi_1(S^1)) through the cross-sector "seagull" vertex provides mutual
topological stabilization. The vortex is topologically protected (pi_1 stable on
torus), and through the coupling lambda*cos(dtheta)*cos_frame, the Skyrmion becomes
pinned to the vortex core.

Test: If the coupled defect survives 20,000 MC sweeps where the isolated Skyrmion
unwound at 7,000 — that's the answer.

Four protocols:
  E1 — Isolated Skyrmion control (lambda=0, no vortex) — reproduce Protocol D baseline
  E2 — Isolated vortex control (lambda=0, no Skyrmion) — verify pi_1 stability
  E3 — Coupled vortex + Skyrmion (lambda>0) — THE TEST
  E4 — Lambda scan — find critical coupling for stabilization

Architecture: Coupled adiabatic MC on 3-diamond lattice.
Two fields per site: phase theta_i in [0,2pi), frame R_i in SO(3).
Two K-fields per bond: K_phi (phase CLR), K_F (frame CLR).
Cross-sector coupling: -lambda * cos(dtheta) * cos_frame per bond.

Critical: r_F = r_phi = 5.892665 from LT-191 exact coexistence calibration.

Depends: LT-70, LT-83, LT-121, LT-191, LT-193, LT-194
Prior: EXP-LT-191 (r* calibration), EXP-LT-192 (metastable Skyrmion baseline)
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

SCRIPT = "coupled_vortex_skyrmion.py"
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LT-191 calibrated coexistence point
R_STAR_DEFAULT = 5.892665


# =====================================================================
# Utilities
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
# SO(3) Bessel Functions (frame sector)
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
# U(1) Phase-Sector Bessel Functions
# =====================================================================

class _R0_Phase_LookupTable:
    """Precomputed lookup table for R0_phi(K) = I_1(K)/I_0(K).
    Smooth monotonic function [0,1), linear interpolation is accurate."""

    def __init__(self, K_max=150.0, n_points=16384):
        self.K_max = K_max
        self.n_points = n_points
        self.dK = K_max / n_points
        K_grid = np.linspace(0, K_max, n_points + 1)
        R0_grid = np.zeros(n_points + 1)
        mask = K_grid > 1e-15
        R0_grid[mask] = I_bessel(1, K_grid[mask]) / I_bessel(0, K_grid[mask])
        self.R0_grid = R0_grid

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


_R0_PHASE_LUT = _R0_Phase_LookupTable()


def R0_phase_vec(K_arr):
    """Vectorized phase order parameter R0(K) = I_1(K)/I_0(K)."""
    return _R0_PHASE_LUT.evaluate(K_arr)


def R0_phase_scalar(K):
    """Scalar phase order parameter."""
    if K < 1e-15:
        return 0.0
    return float(I_bessel(1, K) / I_bessel(0, K))


def solve_clr_phase(cos_dth, r, K_max=50.0):
    """Solve phase-sector CLR fixed point: K = (r/2) * cos(dtheta) * R0_phi(K).

    Phase CLR: K_phi = (r_phi/2) * cos(dtheta) * I_1(K_phi)/I_0(K_phi)
    Threshold: 4/r_phi (bifurcation point).
    """
    if cos_dth <= 0 or r <= 0:
        return 0.0
    threshold = 4.0 / r if r > 4.0 else 1.0
    if cos_dth <= threshold + 1e-12:
        return 0.0

    def F(K):
        return K - (r / 2.0) * cos_dth * R0_phase_scalar(K)

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
# Frame Bond Alignment
# =====================================================================

def frame_bond_alignment(R, ei, ej):
    """Compute frame alignment per bond: cos_frame = tr(R_i^T R_j) / 3."""
    return np.einsum('bac,bac->b', R[ei], R[ej]) / 3.0


# =====================================================================
# Frame CLR Fixed Point Solver
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
# Graph Laplacian & Fiedler Vector
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


def sparse_laplacian_and_fiedler(ei, ej, K_arr, N):
    """Compute graph Laplacian and Fiedler vector (simplified — no warm start)."""
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
        return 0.0, v_2, degree

    L_sp = _build_graph_laplacian(ei, ej, w, degree, N)

    try:
        evals, evecs = eigsh(L_sp, k=3, sigma=0, which='LM',
                             maxiter=5000, tol=1e-10)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        lambda_2 = float(max(evals[1], 0.0))
        v_2 = evecs[:, 1]
    except Exception:
        lambda_2 = 0.0
        v_2 = np.random.randn(N)
        v_2 -= v_2.mean()
        v_2 /= max(np.linalg.norm(v_2), 1e-15)
    return lambda_2, v_2, degree


# =====================================================================
# Coherence Capital & Mass (frame sector)
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
# Skyrmion Profile & Hedgehog Rotation
# =====================================================================

def hedgehog_rotation(r_vec, f_val):
    """SO(3) rotation via Rodrigues: axis=r_hat, angle=2f."""
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
# Random Rotation Generators
# =====================================================================

def random_rotation_uniform_batch(n, rng):
    """Uniform SO(3) via QR decomposition. Returns (n, 3, 3)."""
    R_out = np.zeros((n, 3, 3))
    for i in range(n):
        M = rng.randn(3, 3)
        Q, Rr = np.linalg.qr(M)
        Q *= np.sign(np.linalg.det(Q))
        R_out[i] = Q
    return R_out


def random_rotation_small_batch(n, step_size, rng):
    """Vectorized small-angle SO(3) rotation via Rodrigues.

    Random axis on S^2, uniform angle in [-step_size, step_size].
    R = cos(a)*I + (1-cos(a))*n*n^T + sin(a)*[n]x
    """
    raw = rng.randn(n, 3)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    axes = raw / norms

    angles = rng.uniform(-step_size, step_size, size=n)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    one_minus_cos = 1.0 - cos_a

    nn = axes[:, :, None] * axes[:, None, :]
    Kx = np.zeros((n, 3, 3))
    Kx[:, 0, 1] = -axes[:, 2]
    Kx[:, 0, 2] = axes[:, 1]
    Kx[:, 1, 0] = axes[:, 2]
    Kx[:, 1, 2] = -axes[:, 0]
    Kx[:, 2, 0] = -axes[:, 1]
    Kx[:, 2, 1] = axes[:, 0]

    I_batch = np.eye(3)[None, :, :] * np.ones((n, 1, 1))
    R = (cos_a[:, None, None] * I_batch +
         one_minus_cos[:, None, None] * nn +
         sin_a[:, None, None] * Kx)
    return R


# =====================================================================
# Dead Bond Cluster Analysis
# =====================================================================

def site_defect_proxy(R, site_nbr):
    """Per-site defect proxy: min(cos_frame) over all z neighbors."""
    N, z = site_nbr.shape
    min_cos = np.ones(N)
    for k in range(z):
        nbr = site_nbr[:, k]
        cos_k = np.einsum('iac,iac->i', R, R[nbr]) / 3.0
        min_cos = np.minimum(min_cos, cos_k)
    return min_cos


def identify_dead_bond_clusters(K_arr, ei, ej, N, K_thresh=1e-4):
    """Find connected components of dead-bond subgraph."""
    dead_mask = K_arr < K_thresh
    n_dead = int(np.sum(dead_mask))

    if n_dead == 0:
        return dead_mask, 0, [], []

    dead_ei = ei[dead_mask]
    dead_ej = ej[dead_mask]
    dead_sites = np.unique(np.concatenate([dead_ei, dead_ej]))

    if len(dead_sites) == 0:
        return dead_mask, 0, [], []

    row = np.concatenate([dead_ei, dead_ej])
    col = np.concatenate([dead_ej, dead_ei])
    data = np.ones(len(row))
    A_dead = csr_matrix((data, (row, col)), shape=(N, N))
    n_clusters, labels = connected_components(A_dead, directed=False)

    cluster_sizes = []
    cluster_bond_lists = []
    dead_bond_indices = np.where(dead_mask)[0]
    active_labels = set(labels[dead_sites])

    for lab in sorted(active_labels):
        bonds_in = []
        for b_idx in dead_bond_indices:
            if labels[ei[b_idx]] == lab:
                bonds_in.append(int(b_idx))
        if len(bonds_in) > 0:
            cluster_sizes.append(len(bonds_in))
            cluster_bond_lists.append(bonds_in)

    return dead_mask, len(cluster_sizes), cluster_sizes, cluster_bond_lists


# =====================================================================
# NEW: Phase-sector adiabatic CLR LUT
# =====================================================================

class _KstarCosLUT_frame:
    """Precomputed K*(cos_frame) for the frame sector — adiabatic CLR."""

    def __init__(self, r_F=R_STAR_DEFAULT, n_points=4096,
                 cos_min=-0.4, cos_max=1.0):
        self.r_F = r_F
        self.n_points = n_points
        self.cos_min = cos_min
        self.cos_max = cos_max
        self.dcos = (cos_max - cos_min) / n_points
        self.cos_grid = np.linspace(cos_min, cos_max, n_points + 1)
        self.Kstar_grid = np.array([
            solve_clr_frame(c, r_F) for c in self.cos_grid
        ])
        self.Eeff_grid = -self.Kstar_grid * self.cos_grid

    def evaluate_K(self, cos_arr):
        c_clipped = np.clip(cos_arr, self.cos_min, self.cos_max - 1e-10)
        idx_f = (c_clipped - self.cos_min) / self.dcos
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        result = (1 - frac) * self.Kstar_grid[idx] + frac * self.Kstar_grid[idx + 1]
        return np.maximum(result, 0.0)

    def evaluate_Eeff(self, cos_arr):
        c_clipped = np.clip(cos_arr, self.cos_min, self.cos_max - 1e-10)
        idx_f = (c_clipped - self.cos_min) / self.dcos
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        return (1 - frac) * self.Eeff_grid[idx] + frac * self.Eeff_grid[idx + 1]


class _KstarCosLUT_phase:
    """Precomputed K*(cos(dtheta)) for the phase sector — adiabatic CLR.

    Uses solve_clr_phase (threshold = 4/r_phi) + R0_phase_scalar (I_1/I_0).
    Same API as frame sector LUT.
    """

    def __init__(self, r_phi=R_STAR_DEFAULT, n_points=4096,
                 cos_min=-0.4, cos_max=1.0):
        self.r_phi = r_phi
        self.n_points = n_points
        self.cos_min = cos_min
        self.cos_max = cos_max
        self.dcos = (cos_max - cos_min) / n_points
        self.cos_grid = np.linspace(cos_min, cos_max, n_points + 1)
        self.Kstar_grid = np.array([
            solve_clr_phase(c, r_phi) for c in self.cos_grid
        ])
        self.Eeff_grid = -self.Kstar_grid * self.cos_grid

    def evaluate_K(self, cos_arr):
        c_clipped = np.clip(cos_arr, self.cos_min, self.cos_max - 1e-10)
        idx_f = (c_clipped - self.cos_min) / self.dcos
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        result = (1 - frac) * self.Kstar_grid[idx] + frac * self.Kstar_grid[idx + 1]
        return np.maximum(result, 0.0)

    def evaluate_Eeff(self, cos_arr):
        c_clipped = np.clip(cos_arr, self.cos_min, self.cos_max - 1e-10)
        idx_f = (c_clipped - self.cos_min) / self.dcos
        idx = np.minimum(idx_f.astype(np.intp), self.n_points - 1)
        frac = idx_f - idx
        return (1 - frac) * self.Eeff_grid[idx] + frac * self.Eeff_grid[idx + 1]


# =====================================================================
# NEW: Phase Bond Alignment
# =====================================================================

def phase_bond_alignment(theta, ei, ej):
    """Compute phase alignment per bond: cos(theta[ei] - theta[ej])."""
    return np.cos(theta[ei] - theta[ej])


# =====================================================================
# NEW: Phase Coherence Capital
# =====================================================================

def phase_coherence_capital(K_phi):
    """Per-bond coherence capital in phase sector.

    c(K) = -log(I_0(K)) + K * I_1(K)/I_0(K)
    """
    K_arr = np.asarray(K_phi, dtype=float)
    I0 = I_bessel(0, K_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_I0 = np.where(I0 > 1e-30, np.log(I0), 0.0)
    R0 = R0_phase_vec(K_arr)
    return -log_I0 + K_arr * R0


# =====================================================================
# NEW: Vortex Line Initialization
# =====================================================================

def init_vortex_line_phase(positions, center, axis=2):
    """Assign theta[i] = atan2(y - y_c, x - x_c) where (x,y) are the two
    axes perpendicular to `axis`. Creates a vortex line along the specified
    axis through the lattice center. On T^3 with PBC this is a topologically
    protected pi_1 defect (winding number +1).
    """
    N = positions.shape[0]
    theta = np.zeros(N)
    # Axes perpendicular to the vortex line
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp[0], perp[1]
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    theta = np.arctan2(dy, dx)
    # Map to [0, 2pi)
    theta = theta % (2 * np.pi)
    return theta


# =====================================================================
# NEW: Winding Number Measurement
# =====================================================================

def measure_winding_number(theta, positions, center, axis=2, radius_max=None):
    """Measure the pi_1 winding number of a phase vortex.

    Select sites in an annulus around the vortex core in the plane
    perpendicular to `axis`. Sort by angle, sum wrap(dtheta) around
    the loop. Returns integer winding number (should be +/-1 for
    intact vortex, 0 if dissipated).
    """
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp[0], perp[1]
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    rho = np.sqrt(dx**2 + dy**2)

    if radius_max is None:
        radius_max = float(np.max(rho)) * 0.8

    # Select annulus: exclude core (rho < 0.5) and far region
    r_min = 0.5
    mask = (rho > r_min) & (rho < radius_max)

    if np.sum(mask) < 4:
        return 0

    # Sort by geometric angle in perp plane
    angles_geo = np.arctan2(dy[mask], dx[mask])
    order = np.argsort(angles_geo)
    theta_sorted = theta[np.where(mask)[0][order]]

    # Compute winding
    dtheta = np.diff(theta_sorted)
    # Wrap to [-pi, pi)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    # Close the loop
    dtheta_close = (theta_sorted[0] - theta_sorted[-1] + np.pi) % (2 * np.pi) - np.pi
    total = float(np.sum(dtheta) + dtheta_close)
    winding = int(np.round(total / (2 * np.pi)))
    return winding


# =====================================================================
# NEW: Coupled Adiabatic MC Sweep
# =====================================================================

def coupled_adiabatic_mc_sweep(theta, R, K_phi, K_F,
                                kstar_lut_phase, kstar_lut_frame,
                                site_nbr, site_bond, ei, ej,
                                A_sites, B_sites,
                                step_size_theta, step_size_R,
                                beta, lam, rng):
    """Coupled adiabatic MC sweep: phase sub-sweep + frame sub-sweep.

    Energy per bond:
        E = E_eff_phi(cos dtheta) + E_eff_F(cos_frame) - lam*cos(dtheta)*cos_frame

    Two sub-sweeps per call (phase then frame), each sublattice-parallel.
    K-fields updated to adiabatic values after both sub-sweeps.
    Returns (n_accepted_phase, n_accepted_frame).
    """
    z = site_nbr.shape[1]
    n_accepted_phase = 0
    n_accepted_frame = 0

    # --- Phase sub-sweep ---
    # Precompute current cos_frame for cross-term
    cos_frame_all = frame_bond_alignment(R, ei, ej)

    for sites in [A_sites, B_sites]:
        n_sub = len(sites)

        # Propose: theta_new = theta_old + uniform noise
        theta_old = theta[sites].copy()
        dth = rng.uniform(-step_size_theta, step_size_theta, size=n_sub)
        theta_new = (theta_old + dth) % (2 * np.pi)

        dE = np.zeros(n_sub)
        for k in range(z):
            nbr_sites = site_nbr[sites, k]
            bond_idxs = site_bond[sites, k]
            theta_nbr = theta[nbr_sites]

            cos_dth_new = np.cos(theta_new - theta_nbr)
            cos_dth_old = np.cos(theta_old - theta_nbr)

            # Adiabatic phase potential
            E_eff_new = kstar_lut_phase.evaluate_Eeff(cos_dth_new)
            E_eff_old = kstar_lut_phase.evaluate_Eeff(cos_dth_old)
            dE += (E_eff_new - E_eff_old)

            # Cross-sector coupling
            if lam > 0:
                cf = cos_frame_all[bond_idxs]
                dE += -lam * (cos_dth_new - cos_dth_old) * cf

        # Metropolis
        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            accept |= (rng.uniform(size=n_sub) < boltzmann)
        accept_idx = np.where(accept)[0]
        theta[sites[accept_idx]] = theta_new[accept_idx]
        n_accepted_phase += len(accept_idx)

    # --- Frame sub-sweep ---
    # Precompute current cos_dth for cross-term
    cos_dth_all = phase_bond_alignment(theta, ei, ej)

    for sites in [A_sites, B_sites]:
        n_sub = len(sites)

        dR = random_rotation_small_batch(n_sub, step_size_R, rng)
        R_old = R[sites]
        R_new = np.einsum('nab,nbc->nac', dR, R_old)

        dE = np.zeros(n_sub)
        for k in range(z):
            nbr_sites = site_nbr[sites, k]
            bond_idxs = site_bond[sites, k]
            R_nbr = R[nbr_sites]

            cos_frame_new = np.einsum('nac,nac->n', R_new, R_nbr) / 3.0
            cos_frame_old = np.einsum('nac,nac->n', R_old, R_nbr) / 3.0

            # Adiabatic frame potential
            E_eff_new = kstar_lut_frame.evaluate_Eeff(cos_frame_new)
            E_eff_old = kstar_lut_frame.evaluate_Eeff(cos_frame_old)
            dE += (E_eff_new - E_eff_old)

            # Cross-sector coupling
            if lam > 0:
                cd = cos_dth_all[bond_idxs]
                dE += -lam * cd * (cos_frame_new - cos_frame_old)

        # Metropolis
        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            accept |= (rng.uniform(size=n_sub) < boltzmann)
        accept_idx = np.where(accept)[0]
        R[sites[accept_idx]] = R_new[accept_idx]
        n_accepted_frame += len(accept_idx)

    # --- Adiabatic K update ---
    cos_dth_final = phase_bond_alignment(theta, ei, ej)
    cos_frame_final = frame_bond_alignment(R, ei, ej)
    K_phi[:] = kstar_lut_phase.evaluate_K(cos_dth_final)
    K_F[:] = kstar_lut_frame.evaluate_K(cos_frame_final)

    return n_accepted_phase, n_accepted_frame


# =====================================================================
# NEW: Defect Proximity
# =====================================================================

def defect_proximity(K_F, K_phi, ei, ej, positions, center, K_thresh=1e-4):
    """Compute spatial overlap of frame-dead and phase-dead bond regions.

    Returns dict with overlap_fraction, center-of-mass positions,
    and separation distance.
    """
    dead_frame = K_F < K_thresh
    dead_phase = K_phi < K_thresh
    n_dead_frame = int(np.sum(dead_frame))
    n_dead_phase = int(np.sum(dead_phase))
    overlap = dead_frame & dead_phase
    n_overlap = int(np.sum(overlap))

    mid = (positions[ei] + positions[ej]) / 2.0

    def com(mask):
        if np.sum(mask) == 0:
            return center.copy()
        return np.mean(mid[mask], axis=0)

    com_frame = com(dead_frame)
    com_phase = com(dead_phase)
    sep = float(np.linalg.norm(com_frame - com_phase))

    denom = max(n_dead_frame, n_dead_phase, 1)
    overlap_frac = n_overlap / denom

    return {
        'n_dead_frame': n_dead_frame,
        'n_dead_phase': n_dead_phase,
        'n_overlap': n_overlap,
        'overlap_fraction': overlap_frac,
        'frame_dead_center': com_frame.tolist(),
        'phase_dead_center': com_phase.tolist(),
        'separation_distance': sep,
    }


# =====================================================================
# Record Observables
# =====================================================================

def record_observables(sweep, theta, R, K_phi, K_F,
                       ei, ej, positions, center, N,
                       n_accepted_phase, n_accepted_frame, lam):
    """Record all observables for one snapshot."""
    cos_dth = phase_bond_alignment(theta, ei, ej)
    cos_frame = frame_bond_alignment(R, ei, ej)
    n_bonds = len(ei)

    alive_F = K_F > 1e-4
    alive_phi = K_phi > 1e-4

    K_F_mean = float(np.mean(K_F[alive_F])) if np.any(alive_F) else 0.0
    K_phi_mean = float(np.mean(K_phi[alive_phi])) if np.any(alive_phi) else 0.0

    dead_mask_F, n_clusters_F, sizes_F, _ = identify_dead_bond_clusters(
        K_F, ei, ej, N, K_thresh=1e-4)
    dead_mask_phi, n_clusters_phi, sizes_phi, _ = identify_dead_bond_clusters(
        K_phi, ei, ej, N, K_thresh=1e-4)

    mass_info = measure_skyrmion_mass(K_F, dead_mask_F)

    winding = measure_winding_number(theta, positions, center)

    # Cross-sector energy
    if lam > 0:
        E_cross = -lam * cos_dth * cos_frame
        E_cross_mean = float(np.mean(E_cross))
    else:
        E_cross_mean = 0.0

    prox = defect_proximity(K_F, K_phi, ei, ej, positions, center)

    # Total coherence capital
    C_frame = float(np.sum(frame_coherence_capital(K_F)))
    C_phase = float(np.sum(phase_coherence_capital(K_phi)))
    C_coupled = C_frame + C_phase

    obs = {
        'sweep': sweep,
        'K_F_mean': K_F_mean,
        'K_phi_mean': K_phi_mean,
        'cos_frame_mean': float(np.mean(cos_frame)),
        'cos_dth_mean': float(np.mean(cos_dth)),
        'n_dead_frame': int(np.sum(~alive_F)),
        'n_dead_phase': int(np.sum(~alive_phi)),
        'n_dead_clusters_frame': n_clusters_F,
        'M_skyrmion': mass_info['M_skyrmion'],
        'winding_number': winding,
        'E_cross_mean': E_cross_mean,
        'defect_overlap': prox['overlap_fraction'],
        'defect_separation': prox['separation_distance'],
        'mc_accept_phase': n_accepted_phase,
        'mc_accept_frame': n_accepted_frame,
        'C_coupled': C_coupled,
        'C_frame': C_frame,
        'C_phase': C_phase,
    }
    return obs


# =====================================================================
# Protocol E1: Isolated Skyrmion Control (lambda=0, no vortex)
# =====================================================================

def protocol_e1(L, seed, r_F=R_STAR_DEFAULT, beta=5.0,
                n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                record_interval=100, verbose=True):
    """Isolated Skyrmion control — reproduce Protocol D baseline with lambda=0."""
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol E1: Isolated Skyrmion Control")
        print(f"  L={L}, seed={seed}, beta={beta}, r_F={r_F}")
        print(f"  n_mc_steps={n_mc_steps}, n_equil={n_equil}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)

    # Match Protocol D kick radius: L * nn_dist / 4
    deltas = make_simplex_deltas(3)
    nn_dist = float(np.linalg.norm(deltas[0]))
    if kick_radius is None:
        kick_radius = L * nn_dist / 4.0

    # Build LUT
    kstar_lut = _KstarCosLUT_frame(r_F=r_F)

    # Ordered start: R_i = I_3
    R = np.tile(np.eye(3), (N, 1, 1))

    # Initial K from ordered state
    cos_frame = frame_bond_alignment(R, ei, ej)
    K_F = kstar_lut.evaluate_K(cos_frame)

    # Match Protocol D step size (0.15 initial, adaptive)
    step_size = 0.15

    if verbose:
        print(f"  Pre-equilibrating {n_equil} sweeps (frame only, lambda=0)...")

    # Pre-equilibrate (frame sector only, using adiabatic MC)
    for s in range(n_equil):
        # Pure frame adiabatic MC sweep (reuse coupled sweep with theta=0, lam=0)
        dR = random_rotation_small_batch(len(A_sites), step_size, rng)
        R_old_A = R[A_sites]
        R_new_A = np.einsum('nab,nbc->nac', dR, R_old_A)
        dE = np.zeros(len(A_sites))
        z = site_nbr.shape[1]
        for k in range(z):
            nbr_sites = site_nbr[A_sites, k]
            R_nbr = R[nbr_sites]
            cos_new = np.einsum('nac,nac->n', R_new_A, R_nbr) / 3.0
            cos_old = np.einsum('nac,nac->n', R_old_A, R_nbr) / 3.0
            dE += kstar_lut.evaluate_Eeff(cos_new) - kstar_lut.evaluate_Eeff(cos_old)
        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            accept |= (rng.uniform(size=len(A_sites)) < boltzmann)
        R[A_sites[np.where(accept)[0]]] = R_new_A[np.where(accept)[0]]

        dR = random_rotation_small_batch(len(B_sites), step_size, rng)
        R_old_B = R[B_sites]
        R_new_B = np.einsum('nab,nbc->nac', dR, R_old_B)
        dE = np.zeros(len(B_sites))
        for k in range(z):
            nbr_sites = site_nbr[B_sites, k]
            R_nbr = R[nbr_sites]
            cos_new = np.einsum('nac,nac->n', R_new_B, R_nbr) / 3.0
            cos_old = np.einsum('nac,nac->n', R_old_B, R_nbr) / 3.0
            dE += kstar_lut.evaluate_Eeff(cos_new) - kstar_lut.evaluate_Eeff(cos_old)
        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            accept |= (rng.uniform(size=len(B_sites)) < boltzmann)
        R[B_sites[np.where(accept)[0]]] = R_new_B[np.where(accept)[0]]

        cos_frame = frame_bond_alignment(R, ei, ej)
        K_F[:] = kstar_lut.evaluate_K(cos_frame)

    if verbose:
        print(f"  Post-equil: K_F_mean={np.mean(K_F):.4f}, "
              f"cos_frame_mean={np.mean(cos_frame):.4f}")

    # Inject hedgehog Skyrmion
    if verbose:
        print(f"  Injecting hedgehog kick: strength={kick_strength}, "
              f"radius={kick_radius:.2f}")
    for i in range(N):
        r_vec = positions[i] - center
        r_dist = np.linalg.norm(r_vec)
        if r_dist < kick_radius:
            f_val = kick_strength * np.pi * max(1.0 - r_dist / kick_radius, 0.0)
            R[i] = hedgehog_rotation(r_vec, f_val) @ R[i]

    # Update K after injection
    cos_frame = frame_bond_alignment(R, ei, ej)
    K_F[:] = kstar_lut.evaluate_K(cos_frame)

    dead_mask, n_clust, sizes, _ = identify_dead_bond_clusters(K_F, ei, ej, N)
    if verbose:
        print(f"  Post-kick: K_F_mean={np.mean(K_F):.4f}, "
              f"n_dead={np.sum(dead_mask)}, n_clusters={n_clust}")

    # Run MC
    history = []
    dissipation_sweep = None

    # Use a dummy theta (all zeros, no phase sector)
    theta = np.zeros(N)
    K_phi = np.zeros(n_bonds)

    for s in range(n_mc_steps):
        # Frame-only adiabatic MC
        n_acc = 0
        for sites in [A_sites, B_sites]:
            n_sub = len(sites)
            dR = random_rotation_small_batch(n_sub, step_size, rng)
            R_old = R[sites]
            R_new = np.einsum('nab,nbc->nac', dR, R_old)
            dE = np.zeros(n_sub)
            for k in range(z):
                nbr_sites = site_nbr[sites, k]
                R_nbr = R[nbr_sites]
                cos_new = np.einsum('nac,nac->n', R_new, R_nbr) / 3.0
                cos_old = np.einsum('nac,nac->n', R_old, R_nbr) / 3.0
                dE += kstar_lut.evaluate_Eeff(cos_new) - kstar_lut.evaluate_Eeff(cos_old)
            accept = dE <= 0
            if beta < float('inf'):
                boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
                accept |= (rng.uniform(size=n_sub) < boltzmann)
            acc_idx = np.where(accept)[0]
            R[sites[acc_idx]] = R_new[acc_idx]
            n_acc += len(acc_idx)

        cos_frame = frame_bond_alignment(R, ei, ej)
        K_F[:] = kstar_lut.evaluate_K(cos_frame)

        if (s + 1) % record_interval == 0:
            obs = record_observables(
                s + 1, theta, R, K_phi, K_F,
                ei, ej, positions, center, N,
                0, n_acc, lam=0.0)
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: K_F={obs['K_F_mean']:.4f}, "
                      f"n_dead_F={obs['n_dead_frame']}, "
                      f"M_sky={obs['M_skyrmion']:.1f}, "
                      f"clusters={obs['n_dead_clusters_frame']}")

            # Check for dissipation
            if obs['n_dead_frame'] == 0 and dissipation_sweep is None and s > 0:
                dissipation_sweep = s + 1

    # Verdict
    final = history[-1] if history else {}
    n_dead_final = final.get('n_dead_frame', 0)
    if n_dead_final > 0:
        verdict = "SKYRMION_SURVIVED"
    else:
        verdict = "SKYRMION_DISSIPATED"

    elapsed = time.time() - t0
    result = {
        'protocol': 'E1',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'seed': seed, 'beta': beta, 'r_F': r_F,
        'lam': 0.0,
        'n_mc_steps': n_mc_steps,
        'n_equil': n_equil,
        'kick_strength': kick_strength,
        'kick_radius': kick_radius,
        'dissipation_sweep': dissipation_sweep,
        'final_n_dead_frame': n_dead_final,
        'final_M_skyrmion': final.get('M_skyrmion', 0),
        'elapsed_sec': elapsed,
        'history': history,
    }

    if verbose:
        print(f"\n  VERDICT: {verdict}")
        if dissipation_sweep is not None:
            print(f"  Dissipation at sweep ~{dissipation_sweep}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Protocol E2: Isolated Vortex Control (lambda=0, no Skyrmion)
# =====================================================================

def protocol_e2(L, seed, r_phi=R_STAR_DEFAULT, beta=5.0,
                n_mc_steps=20000, n_equil=2000,
                record_interval=100, verbose=True):
    """Isolated vortex control — verify pi_1 stability."""
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol E2: Isolated Vortex Control")
        print(f"  L={L}, seed={seed}, beta={beta}, r_phi={r_phi}")
        print(f"  n_mc_steps={n_mc_steps}, n_equil={n_equil}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)

    # Build LUT
    kstar_lut = _KstarCosLUT_phase(r_phi=r_phi)

    # Ordered start: theta_i = 0
    theta = np.zeros(N)

    # Initial K from ordered state
    cos_dth = phase_bond_alignment(theta, ei, ej)
    K_phi = kstar_lut.evaluate_K(cos_dth)

    step_size = 0.15

    if verbose:
        print(f"  Pre-equilibrating {n_equil} sweeps (phase only, lambda=0)...")

    # Pre-equilibrate
    for s in range(n_equil):
        for sites in [A_sites, B_sites]:
            n_sub = len(sites)
            theta_old = theta[sites].copy()
            dth = rng.uniform(-step_size, step_size, size=n_sub)
            theta_new = (theta_old + dth) % (2 * np.pi)
            dE = np.zeros(n_sub)
            z = site_nbr.shape[1]
            for k in range(z):
                nbr = site_nbr[sites, k]
                cos_new = np.cos(theta_new - theta[nbr])
                cos_old = np.cos(theta_old - theta[nbr])
                dE += kstar_lut.evaluate_Eeff(cos_new) - kstar_lut.evaluate_Eeff(cos_old)
            accept = dE <= 0
            if beta < float('inf'):
                boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
                accept |= (rng.uniform(size=n_sub) < boltzmann)
            theta[sites[np.where(accept)[0]]] = theta_new[np.where(accept)[0]]

        cos_dth = phase_bond_alignment(theta, ei, ej)
        K_phi[:] = kstar_lut.evaluate_K(cos_dth)

    if verbose:
        print(f"  Post-equil: K_phi_mean={np.mean(K_phi):.4f}, "
              f"cos_dth_mean={np.mean(cos_dth):.4f}")

    # Inject vortex line
    if verbose:
        print(f"  Injecting vortex line along z-axis through center...")
    theta = init_vortex_line_phase(positions, center, axis=2)

    # Update K after injection
    cos_dth = phase_bond_alignment(theta, ei, ej)
    K_phi[:] = kstar_lut.evaluate_K(cos_dth)

    dead_mask, n_clust, sizes, _ = identify_dead_bond_clusters(K_phi, ei, ej, N)
    w0 = measure_winding_number(theta, positions, center)
    if verbose:
        print(f"  Post-injection: K_phi_mean={np.mean(K_phi):.4f}, "
              f"n_dead={np.sum(dead_mask)}, W={w0}")

    # Run MC
    history = []
    R = np.tile(np.eye(3), (N, 1, 1))  # dummy
    K_F = np.zeros(n_bonds)

    for s in range(n_mc_steps):
        # Phase-only adiabatic MC
        n_acc = 0
        for sites in [A_sites, B_sites]:
            n_sub = len(sites)
            theta_old = theta[sites].copy()
            dth = rng.uniform(-step_size, step_size, size=n_sub)
            theta_new = (theta_old + dth) % (2 * np.pi)
            dE = np.zeros(n_sub)
            z = site_nbr.shape[1]
            for k in range(z):
                nbr = site_nbr[sites, k]
                cos_new = np.cos(theta_new - theta[nbr])
                cos_old = np.cos(theta_old - theta[nbr])
                dE += kstar_lut.evaluate_Eeff(cos_new) - kstar_lut.evaluate_Eeff(cos_old)
            accept = dE <= 0
            if beta < float('inf'):
                boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
                accept |= (rng.uniform(size=n_sub) < boltzmann)
            acc_idx = np.where(accept)[0]
            theta[sites[acc_idx]] = theta_new[acc_idx]
            n_acc += len(acc_idx)

        cos_dth = phase_bond_alignment(theta, ei, ej)
        K_phi[:] = kstar_lut.evaluate_K(cos_dth)

        if (s + 1) % record_interval == 0:
            obs = record_observables(
                s + 1, theta, R, K_phi, K_F,
                ei, ej, positions, center, N,
                n_acc, 0, lam=0.0)
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: K_phi={obs['K_phi_mean']:.4f}, "
                      f"n_dead_phi={obs['n_dead_phase']}, "
                      f"W={obs['winding_number']}")

    # Verdict
    final = history[-1] if history else {}
    w_final = final.get('winding_number', 0)
    if abs(w_final) >= 1:
        verdict = "VORTEX_STABLE"
    else:
        verdict = "VORTEX_DISSIPATED"

    elapsed = time.time() - t0
    result = {
        'protocol': 'E2',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'seed': seed, 'beta': beta, 'r_phi': r_phi,
        'lam': 0.0,
        'n_mc_steps': n_mc_steps,
        'n_equil': n_equil,
        'initial_winding': w0,
        'final_winding': w_final,
        'final_n_dead_phase': final.get('n_dead_phase', 0),
        'elapsed_sec': elapsed,
        'history': history,
    }

    if verbose:
        print(f"\n  VERDICT: {verdict}")
        print(f"  Final winding: {w_final}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Protocol E3: Coupled Vortex + Skyrmion (lambda>0) — THE TEST
# =====================================================================

def protocol_e3(L, seed, r_F=R_STAR_DEFAULT, r_phi=R_STAR_DEFAULT,
                beta=5.0, lam=0.5,
                n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                record_interval=100, verbose=True):
    """Coupled vortex + Skyrmion — the main test."""
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol E3: Coupled Vortex + Skyrmion")
        print(f"  L={L}, seed={seed}, beta={beta}, lam={lam}")
        print(f"  r_F={r_F}, r_phi={r_phi}")
        print(f"  n_mc_steps={n_mc_steps}, n_equil={n_equil}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)

    # Match Protocol D kick radius
    deltas = make_simplex_deltas(3)
    nn_dist = float(np.linalg.norm(deltas[0]))
    if kick_radius is None:
        kick_radius = L * nn_dist / 4.0

    # Build LUTs
    kstar_lut_frame = _KstarCosLUT_frame(r_F=r_F)
    kstar_lut_phase = _KstarCosLUT_phase(r_phi=r_phi)

    # Ordered start
    R = np.tile(np.eye(3), (N, 1, 1))
    theta = np.zeros(N)

    # Initial K
    cos_frame = frame_bond_alignment(R, ei, ej)
    cos_dth = phase_bond_alignment(theta, ei, ej)
    K_F = kstar_lut_frame.evaluate_K(cos_frame)
    K_phi = kstar_lut_phase.evaluate_K(cos_dth)

    # Conservative step sizes (match Protocol D)
    step_size_R = 0.15
    step_size_theta = 0.15

    if verbose:
        print(f"  Pre-equilibrating {n_equil} coupled sweeps...")

    # Pre-equilibrate with coupled adiabatic MC
    for s in range(n_equil):
        coupled_adiabatic_mc_sweep(
            theta, R, K_phi, K_F,
            kstar_lut_phase, kstar_lut_frame,
            site_nbr, site_bond, ei, ej,
            A_sites, B_sites,
            step_size_theta, step_size_R,
            beta, lam, rng)

    cos_frame = frame_bond_alignment(R, ei, ej)
    cos_dth = phase_bond_alignment(theta, ei, ej)
    if verbose:
        print(f"  Post-equil: K_F_mean={np.mean(K_F):.4f}, "
              f"K_phi_mean={np.mean(K_phi):.4f}")
        print(f"  Post-equil: cos_frame={np.mean(cos_frame):.4f}, "
              f"cos_dth={np.mean(cos_dth):.4f}")

    # Inject both defects co-located at center
    # 1. Hedgehog Skyrmion
    if verbose:
        print(f"  Injecting hedgehog kick: strength={kick_strength}, "
              f"radius={kick_radius:.2f}")
    for i in range(N):
        r_vec = positions[i] - center
        r_dist = np.linalg.norm(r_vec)
        if r_dist < kick_radius:
            f_val = kick_strength * np.pi * max(1.0 - r_dist / kick_radius, 0.0)
            R[i] = hedgehog_rotation(r_vec, f_val) @ R[i]

    # 2. Phase vortex line
    if verbose:
        print(f"  Injecting vortex line along z-axis...")
    theta = init_vortex_line_phase(positions, center, axis=2)

    # Update K after injection
    cos_frame = frame_bond_alignment(R, ei, ej)
    cos_dth = phase_bond_alignment(theta, ei, ej)
    K_F[:] = kstar_lut_frame.evaluate_K(cos_frame)
    K_phi[:] = kstar_lut_phase.evaluate_K(cos_dth)

    dead_mask_F, n_clust_F, _, _ = identify_dead_bond_clusters(K_F, ei, ej, N)
    dead_mask_phi, n_clust_phi, _, _ = identify_dead_bond_clusters(K_phi, ei, ej, N)
    w0 = measure_winding_number(theta, positions, center)

    if verbose:
        print(f"  Post-injection: n_dead_F={np.sum(dead_mask_F)}, "
              f"n_dead_phi={np.sum(dead_mask_phi)}, W={w0}")
        prox0 = defect_proximity(K_F, K_phi, ei, ej, positions, center)
        print(f"  Initial defect overlap: {prox0['overlap_fraction']:.3f}, "
              f"sep={prox0['separation_distance']:.3f}")

    # Run coupled MC
    history = []
    skyrmion_dissipation_sweep = None

    for s in range(n_mc_steps):
        n_acc_phase, n_acc_frame = coupled_adiabatic_mc_sweep(
            theta, R, K_phi, K_F,
            kstar_lut_phase, kstar_lut_frame,
            site_nbr, site_bond, ei, ej,
            A_sites, B_sites,
            step_size_theta, step_size_R,
            beta, lam, rng)

        if (s + 1) % record_interval == 0:
            obs = record_observables(
                s + 1, theta, R, K_phi, K_F,
                ei, ej, positions, center, N,
                n_acc_phase, n_acc_frame, lam)
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: K_F={obs['K_F_mean']:.4f}, "
                      f"K_phi={obs['K_phi_mean']:.4f}, "
                      f"n_dead_F={obs['n_dead_frame']}, "
                      f"n_dead_phi={obs['n_dead_phase']}, "
                      f"W={obs['winding_number']}, "
                      f"M_sky={obs['M_skyrmion']:.1f}, "
                      f"overlap={obs['defect_overlap']:.3f}")

            if obs['n_dead_frame'] == 0 and skyrmion_dissipation_sweep is None and s > 0:
                skyrmion_dissipation_sweep = s + 1

    # Verdict
    final = history[-1] if history else {}
    n_dead_F = final.get('n_dead_frame', 0)
    n_dead_phi = final.get('n_dead_phase', 0)
    w_final = final.get('winding_number', 0)
    skyrmion_alive = n_dead_F > 0
    vortex_alive = abs(w_final) >= 1

    if skyrmion_alive and vortex_alive:
        verdict = "COUPLED_STABLE"
    elif not skyrmion_alive and vortex_alive:
        verdict = "SKYRMION_DISSIPATED"
    elif not skyrmion_alive and not vortex_alive:
        verdict = "BOTH_DISSIPATED"
    elif skyrmion_alive and not vortex_alive:
        verdict = "DECOUPLED"
    else:
        verdict = "UNKNOWN"

    elapsed = time.time() - t0
    result = {
        'protocol': 'E3',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'seed': seed, 'beta': beta, 'lam': lam,
        'r_F': r_F, 'r_phi': r_phi,
        'n_mc_steps': n_mc_steps,
        'n_equil': n_equil,
        'kick_strength': kick_strength,
        'kick_radius': kick_radius,
        'initial_winding': w0,
        'final_winding': w_final,
        'skyrmion_dissipation_sweep': skyrmion_dissipation_sweep,
        'final_n_dead_frame': n_dead_F,
        'final_n_dead_phase': n_dead_phi,
        'final_M_skyrmion': final.get('M_skyrmion', 0),
        'final_defect_overlap': final.get('defect_overlap', 0),
        'elapsed_sec': elapsed,
        'history': history,
    }

    if verbose:
        print(f"\n  VERDICT: {verdict}")
        print(f"  Final winding: {w_final}, "
              f"n_dead_F={n_dead_F}, n_dead_phi={n_dead_phi}")
        if skyrmion_dissipation_sweep:
            print(f"  Skyrmion dissipation at sweep ~{skyrmion_dissipation_sweep}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Protocol E4: Lambda Scan
# =====================================================================

def protocol_e4(L, seed, r_F=R_STAR_DEFAULT, r_phi=R_STAR_DEFAULT,
                beta=5.0, n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                record_interval=100,
                lam_values=None, verbose=True):
    """Lambda scan — find critical coupling for stabilization."""
    t0 = time.time()

    if lam_values is None:
        lam_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol E4: Lambda Scan")
        print(f"  L={L}, seed={seed}, beta={beta}")
        print(f"  r_F={r_F}, r_phi={r_phi}")
        print(f"  Lambda values: {lam_values}")
        print(f"{'='*70}")

    scan_results = []

    for lam_val in lam_values:
        if verbose:
            print(f"\n  --- Lambda = {lam_val} ---")

        res = protocol_e3(
            L=L, seed=seed, r_F=r_F, r_phi=r_phi,
            beta=beta, lam=lam_val,
            n_mc_steps=n_mc_steps, n_equil=n_equil,
            kick_strength=kick_strength, kick_radius=kick_radius,
            record_interval=record_interval, verbose=verbose)

        # Find skyrmion survival time (sweeps until n_dead_frame hits 0)
        survival_time = n_mc_steps  # default: survived whole run
        for obs in res['history']:
            if obs['n_dead_frame'] == 0:
                survival_time = obs['sweep']
                break

        scan_results.append({
            'lam': lam_val,
            'verdict': res['verdict'],
            'survival_time': survival_time,
            'final_winding': res['final_winding'],
            'final_n_dead_frame': res['final_n_dead_frame'],
            'final_M_skyrmion': res['final_M_skyrmion'],
        })

    # Find lambda_critical
    lam_critical = None
    for sr in scan_results:
        if sr['verdict'] == 'COUPLED_STABLE':
            lam_critical = sr['lam']
            break

    # Check monotonicity
    survival_times = [sr['survival_time'] for sr in scan_results]
    monotonic = all(survival_times[i] <= survival_times[i+1]
                    for i in range(len(survival_times) - 1))

    elapsed = time.time() - t0
    result = {
        'protocol': 'E4',
        'L': L, 'N': 2 * L**3, 'seed': seed,
        'beta': beta, 'r_F': r_F, 'r_phi': r_phi,
        'n_mc_steps': n_mc_steps,
        'lam_values': lam_values,
        'scan_results': scan_results,
        'lam_critical': lam_critical,
        'survival_monotonic': monotonic,
        'elapsed_sec': elapsed,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  E4 Summary:")
        print(f"  {'lam':>6s}  {'verdict':>25s}  {'survival':>10s}  "
              f"{'W_final':>7s}  {'n_dead_F':>8s}")
        for sr in scan_results:
            print(f"  {sr['lam']:6.2f}  {sr['verdict']:>25s}  "
                  f"{sr['survival_time']:10d}  "
                  f"{sr['final_winding']:7d}  "
                  f"{sr['final_n_dead_frame']:8d}")
        print(f"  Lambda_critical: {lam_critical}")
        print(f"  Survival monotonic: {monotonic}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Main / CLI
# =====================================================================

# =====================================================================
# Protocol E5: Vacuum Polarization — Skyrmion + Thermal Phase (no vortex)
# =====================================================================

def protocol_e5(L, seed, r_F=R_STAR_DEFAULT, r_phi=R_STAR_DEFAULT,
                beta=5.0, lam=0.5,
                n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                record_interval=100, verbose=True):
    """Vacuum polarization: Skyrmion coupled to thermal phase field, no injected vortex.

    The phase field starts ordered and equilibrates thermally. No vortex is
    injected — virtual vortex-antivortex pairs arise from thermal fluctuations.
    Only the hedgehog kick (frame sector) is applied.

    This tests whether distributed, gentle phase-sector coupling stabilizes
    the Skyrmion, unlike E3 where a massive coherent vortex destabilized it.
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol E5: Vacuum Polarization (Skyrmion + thermal phase)")
        print(f"  L={L}, seed={seed}, beta={beta}, lam={lam}")
        print(f"  r_F={r_F}, r_phi={r_phi}")
        print(f"  n_mc_steps={n_mc_steps}, n_equil={n_equil}")
        print(f"  NO injected vortex — phase starts ordered, thermalizes")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)
    center = np.mean(positions, axis=0)

    # Match Protocol D kick radius
    deltas = make_simplex_deltas(3)
    nn_dist = float(np.linalg.norm(deltas[0]))
    if kick_radius is None:
        kick_radius = L * nn_dist / 4.0

    # Build LUTs
    kstar_lut_frame = _KstarCosLUT_frame(r_F=r_F)
    kstar_lut_phase = _KstarCosLUT_phase(r_phi=r_phi)

    # Ordered start: both fields ordered
    R = np.tile(np.eye(3), (N, 1, 1))
    theta = np.zeros(N)

    # Initial K
    cos_frame = frame_bond_alignment(R, ei, ej)
    cos_dth = phase_bond_alignment(theta, ei, ej)
    K_F = kstar_lut_frame.evaluate_K(cos_frame)
    K_phi = kstar_lut_phase.evaluate_K(cos_dth)

    step_size_R = 0.15
    step_size_theta = 0.15

    if verbose:
        print(f"  Pre-equilibrating {n_equil} coupled sweeps "
              f"(both sectors, ordered start)...")

    # Pre-equilibrate with coupled adiabatic MC — both sectors thermalize together
    for s in range(n_equil):
        coupled_adiabatic_mc_sweep(
            theta, R, K_phi, K_F,
            kstar_lut_phase, kstar_lut_frame,
            site_nbr, site_bond, ei, ej,
            A_sites, B_sites,
            step_size_theta, step_size_R,
            beta, lam, rng)

    cos_frame = frame_bond_alignment(R, ei, ej)
    cos_dth = phase_bond_alignment(theta, ei, ej)
    w_pre = measure_winding_number(theta, positions, center)
    if verbose:
        print(f"  Post-equil: K_F_mean={np.mean(K_F):.4f}, "
              f"K_phi_mean={np.mean(K_phi):.4f}")
        print(f"  Post-equil: cos_frame={np.mean(cos_frame):.4f}, "
              f"cos_dth={np.mean(cos_dth):.4f}, W={w_pre}")
        print(f"  Phase dead bonds: {np.sum(K_phi < 1e-4)}, "
              f"Frame dead bonds: {np.sum(K_F < 1e-4)}")

    # Inject ONLY Skyrmion — NO vortex
    if verbose:
        print(f"  Injecting hedgehog kick ONLY: strength={kick_strength}, "
              f"radius={kick_radius:.2f}")
        print(f"  NO vortex injection — phase field remains thermal vacuum")
    for i in range(N):
        r_vec = positions[i] - center
        r_dist = np.linalg.norm(r_vec)
        if r_dist < kick_radius:
            f_val = kick_strength * np.pi * max(1.0 - r_dist / kick_radius, 0.0)
            R[i] = hedgehog_rotation(r_vec, f_val) @ R[i]

    # Update K after kick (frame sector changes, phase sector untouched)
    cos_frame = frame_bond_alignment(R, ei, ej)
    cos_dth = phase_bond_alignment(theta, ei, ej)
    K_F[:] = kstar_lut_frame.evaluate_K(cos_frame)
    K_phi[:] = kstar_lut_phase.evaluate_K(cos_dth)

    dead_mask_F, n_clust_F, _, _ = identify_dead_bond_clusters(K_F, ei, ej, N)
    dead_mask_phi, n_clust_phi, _, _ = identify_dead_bond_clusters(K_phi, ei, ej, N)
    w0 = measure_winding_number(theta, positions, center)

    if verbose:
        print(f"  Post-kick: n_dead_F={np.sum(dead_mask_F)}, "
              f"n_dead_phi={np.sum(dead_mask_phi)}, W={w0}")
        prox0 = defect_proximity(K_F, K_phi, ei, ej, positions, center)
        print(f"  Defect overlap: {prox0['overlap_fraction']:.3f}, "
              f"sep={prox0['separation_distance']:.3f}")

    # Run coupled MC
    history = []
    skyrmion_dissipation_sweep = None

    for s in range(n_mc_steps):
        n_acc_phase, n_acc_frame = coupled_adiabatic_mc_sweep(
            theta, R, K_phi, K_F,
            kstar_lut_phase, kstar_lut_frame,
            site_nbr, site_bond, ei, ej,
            A_sites, B_sites,
            step_size_theta, step_size_R,
            beta, lam, rng)

        if (s + 1) % record_interval == 0:
            obs = record_observables(
                s + 1, theta, R, K_phi, K_F,
                ei, ej, positions, center, N,
                n_acc_phase, n_acc_frame, lam)
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: K_F={obs['K_F_mean']:.4f}, "
                      f"K_phi={obs['K_phi_mean']:.4f}, "
                      f"n_dead_F={obs['n_dead_frame']}, "
                      f"n_dead_phi={obs['n_dead_phase']}, "
                      f"W={obs['winding_number']}, "
                      f"M_sky={obs['M_skyrmion']:.1f}")

            if obs['n_dead_frame'] == 0 and skyrmion_dissipation_sweep is None and s > 0:
                skyrmion_dissipation_sweep = s + 1

    # Verdict
    final = history[-1] if history else {}
    n_dead_F = final.get('n_dead_frame', 0)
    skyrmion_alive = n_dead_F > 0

    if skyrmion_alive:
        verdict = "SKYRMION_STABILIZED"
    else:
        verdict = "SKYRMION_DISSIPATED"

    elapsed = time.time() - t0
    result = {
        'protocol': 'E5',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'seed': seed, 'beta': beta, 'lam': lam,
        'r_F': r_F, 'r_phi': r_phi,
        'n_mc_steps': n_mc_steps,
        'n_equil': n_equil,
        'kick_strength': kick_strength,
        'kick_radius': kick_radius,
        'skyrmion_dissipation_sweep': skyrmion_dissipation_sweep,
        'final_n_dead_frame': n_dead_F,
        'final_n_dead_phase': final.get('n_dead_phase', 0),
        'final_M_skyrmion': final.get('M_skyrmion', 0),
        'final_winding': final.get('winding_number', 0),
        'elapsed_sec': elapsed,
        'history': history,
    }

    if verbose:
        print(f"\n  VERDICT: {verdict}")
        print(f"  Final: n_dead_F={n_dead_F}, "
              f"M_sky={final.get('M_skyrmion', 0):.1f}")
        if skyrmion_dissipation_sweep:
            print(f"  Skyrmion dissipation at sweep ~{skyrmion_dissipation_sweep}")
        else:
            print(f"  Skyrmion SURVIVED all {n_mc_steps} sweeps!")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Protocol E5-scan: Lambda scan with vacuum polarization
# =====================================================================

def protocol_e5_scan(L, seed, r_F=R_STAR_DEFAULT, r_phi=R_STAR_DEFAULT,
                     beta=5.0, n_mc_steps=20000, n_equil=2000,
                     kick_strength=1.0, kick_radius=None,
                     record_interval=100,
                     lam_values=None, verbose=True):
    """Lambda scan for E5 (vacuum polarization)."""
    t0 = time.time()

    if lam_values is None:
        lam_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol E5-scan: Vacuum Polarization Lambda Scan")
        print(f"  L={L}, seed={seed}, beta={beta}")
        print(f"  Lambda values: {lam_values}")
        print(f"{'='*70}")

    scan_results = []

    for lam_val in lam_values:
        if verbose:
            print(f"\n  --- Lambda = {lam_val} ---")

        res = protocol_e5(
            L=L, seed=seed, r_F=r_F, r_phi=r_phi,
            beta=beta, lam=lam_val,
            n_mc_steps=n_mc_steps, n_equil=n_equil,
            kick_strength=kick_strength, kick_radius=kick_radius,
            record_interval=record_interval, verbose=verbose)

        survival_time = n_mc_steps
        for obs in res['history']:
            if obs['n_dead_frame'] == 0:
                survival_time = obs['sweep']
                break

        scan_results.append({
            'lam': lam_val,
            'verdict': res['verdict'],
            'survival_time': survival_time,
            'final_n_dead_frame': res['final_n_dead_frame'],
            'final_M_skyrmion': res['final_M_skyrmion'],
        })

    elapsed = time.time() - t0
    result = {
        'protocol': 'E5-scan',
        'L': L, 'N': 2 * L**3, 'seed': seed,
        'beta': beta, 'r_F': r_F, 'r_phi': r_phi,
        'n_mc_steps': n_mc_steps,
        'lam_values': lam_values,
        'scan_results': scan_results,
        'elapsed_sec': elapsed,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  E5-scan Summary:")
        print(f"  {'lam':>6s}  {'verdict':>25s}  {'survival':>10s}  "
              f"{'n_dead_F':>8s}  {'M_sky':>8s}")
        for sr in scan_results:
            print(f"  {sr['lam']:6.2f}  {sr['verdict']:>25s}  "
                  f"{sr['survival_time']:10d}  "
                  f"{sr['final_n_dead_frame']:8d}  "
                  f"{sr['final_M_skyrmion']:8.1f}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="EXP-LT-195: Coupled Vortex-Skyrmion Topological Stabilization")
    parser.add_argument('--protocol', type=str, default='E3',
                        choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E5-scan', 'all'],
                        help='Protocol to run')
    parser.add_argument('--L', type=int, default=12,
                        help='Lattice size (default: 12)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--beta', type=float, default=5.0,
                        help='Inverse temperature (default: 5.0)')
    parser.add_argument('--r_F', type=float, default=R_STAR_DEFAULT,
                        help=f'Frame CLR coupling (default: {R_STAR_DEFAULT})')
    parser.add_argument('--r_phi', type=float, default=R_STAR_DEFAULT,
                        help=f'Phase CLR coupling (default: {R_STAR_DEFAULT})')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Cross-sector coupling lambda (default: 0.5)')
    parser.add_argument('--n_mc_steps', type=int, default=20000,
                        help='MC sweeps (default: 20000)')
    parser.add_argument('--n_equil', type=int, default=2000,
                        help='Equilibration sweeps (default: 2000)')
    parser.add_argument('--kick_strength', type=float, default=1.0,
                        help='Hedgehog kick strength (default: 1.0)')
    parser.add_argument('--kick_radius', type=float, default=None,
                        help='Hedgehog kick radius (default: L*0.3)')
    parser.add_argument('--record_interval', type=int, default=100,
                        help='Recording interval (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    verbose = not args.quiet
    protocols = []

    if args.protocol == 'all':
        protocols = ['E1', 'E2', 'E3', 'E4', 'E5']
    else:
        protocols = [args.protocol]

    all_results = {}

    for proto in protocols:
        if proto == 'E1':
            res = protocol_e1(
                L=args.L, seed=args.seed, r_F=args.r_F,
                beta=args.beta, n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'E2':
            res = protocol_e2(
                L=args.L, seed=args.seed, r_phi=args.r_phi,
                beta=args.beta, n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'E3':
            res = protocol_e3(
                L=args.L, seed=args.seed,
                r_F=args.r_F, r_phi=args.r_phi,
                beta=args.beta, lam=args.lam,
                n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'E4':
            res = protocol_e4(
                L=args.L, seed=args.seed,
                r_F=args.r_F, r_phi=args.r_phi,
                beta=args.beta, n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'E5':
            res = protocol_e5(
                L=args.L, seed=args.seed,
                r_F=args.r_F, r_phi=args.r_phi,
                beta=args.beta, lam=args.lam,
                n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'E5-scan':
            res = protocol_e5_scan(
                L=args.L, seed=args.seed,
                r_F=args.r_F, r_phi=args.r_phi,
                beta=args.beta, n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                record_interval=args.record_interval,
                verbose=verbose)
        else:
            continue

        all_results[proto] = res

        # Save individual result
        fname = (f"{OUTPUT_DIR}/coupled_vortex_skyrmion_{proto}"
                 f"_L{args.L}_s{args.seed}.json")
        with open(fname, 'w') as f:
            json.dump(jsonify(res), f, indent=2)
        if verbose:
            print(f"  Saved: {fname}")

    # Summary table
    if verbose and len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        for proto, res in all_results.items():
            print(f"  {proto}: {res.get('verdict', 'N/A')}")


if __name__ == "__main__":
    main()
