#!/usr/bin/env python3
"""
SU(2) Wilson Plaquette Gauge Confinement for Skyrmion Stabilization (EXP-LT-197)

EXP-LT-195 conclusively showed that the seagull cross-sector coupling DESTABILIZES
Skyrmions at ALL lambda values (lambda>0: repulsion, lambda<0: attraction but still
worse than baseline). The seagull is a bulk-ordered coupling that vanishes at defect
cores where stabilization is needed.

The correct physics is GAUGE CONFINEMENT: in QCD the proton (= Skyrmion in the Skyrme
model) is stabilized by non-abelian gauge dynamics. The lattice theory derives:
  LT-36: SO(3) Yang-Mills from frame coupling
  LT-118: Cubic vertex V_3 ~ eps^{abc} (su(2) structure constants)
  LT-52: Non-perturbative area law (confinement)
  LT-63: String tension sigma = -log(R0^F/3) > 0 for all finite K
  LT-121: Mixed-phase window (U(1) deconfined + SU(2) confined)
  LT-183: SU(2) transporter U = cos(alpha/2)*I2 + i*sin(alpha/2)*(delta_hat.sigma)

Critical insight (LT-118 Prop 4.2): The current sigma model has U_P = I identically
(flat connection). True gauge confinement requires INDEPENDENT link DOFs V_ij in SU(2)
that can fluctuate away from the sigma-model connection.

Architecture: Gauge-matter coupled adiabatic MC on 3-diamond lattice.
  Site DOF: Frame R_i in SO(3) (3x3 rotation matrix)
  Link DOF: Gauge link V_ij in SU(2) (unit quaternion)
  Bond DOF: K_F per bond (frame sector adiabatic CLR)

Four protocols:
  G1 -- Sigma model control (beta_g=inf, V=I frozen) -- reproduce E1 baseline
  G2 -- Gauge confinement test (beta_g=2.0) -- THE TEST
  G3 -- beta_g scan (0.5,1.0,2.0,4.0,8.0,16.0) -- phase diagram
  G4 -- String tension measurement (pure gauge, R=I frozen)

Depends: LT-36, LT-52, LT-63, LT-118, LT-121, LT-183, LT-193, LT-195
Prior: EXP-LT-192 (Skyrmion baseline), EXP-LT-195 (seagull refutation)
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
from scipy.sparse.csgraph import connected_components
import numba
from numba import njit, prange

SCRIPT = "gauge_skyrmion.py"
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
    """
    N_cells = L ** d
    N = 2 * N_cells
    z = d + 1
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

    site_nbr_list = [[] for _ in range(N)]
    site_bond_list = [[] for _ in range(N)]
    for b_idx in range(n_bonds):
        i, j = int(ei[b_idx]), int(ej[b_idx])
        site_nbr_list[i].append(j)
        site_bond_list[i].append(b_idx)
        site_nbr_list[j].append(i)
        site_bond_list[j].append(b_idx)

    degs = [len(site_nbr_list[i]) for i in range(N)]
    assert all(deg == z for deg in degs), \
        f"Non-uniform degree: min={min(degs)}, max={max(degs)}, expected={z}"

    site_nbr = np.array(site_nbr_list, dtype=np.int32)
    site_bond = np.array(site_bond_list, dtype=np.int32)

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
# Frame Bond Alignment & CLR
# =====================================================================

def frame_bond_alignment(R, ei, ej):
    """Compute frame alignment per bond: cos_frame = tr(R_i^T R_j) / 3."""
    return np.einsum('bac,bac->b', R[ei], R[ej]) / 3.0


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


class _KstarCosLUT_frame:
    """Precomputed K*(cos_frame) for the frame sector -- adiabatic CLR."""

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
    """Vectorized small-angle SO(3) rotation via Rodrigues."""
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
# SU(2) Quaternion Core
# =====================================================================

def quat_multiply(a, b):
    """Hamilton product of unit quaternions. a, b: (..., 4). Returns (..., 4).

    q = (q0, q1, q2, q3) represents q0 + q1*i + q2*j + q3*k.
    Product: (a0*b0 - a.b, a0*b + b0*a + a x b).
    """
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3
    r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2
    r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1
    r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0

    return np.stack([r0, r1, r2, r3], axis=-1)


def quat_conjugate(q):
    """Conjugate of quaternion: (q0, -q1, -q2, -q3). Also the inverse for unit q."""
    result = q.copy()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_to_so3(q):
    """Convert unit quaternion to 3x3 SO(3) adjoint representation.

    adj(q)_ij = (2q0^2 - 1)*delta_ij + 2*qi*qj + 2*q0*eps_ijk*qk

    q: (..., 4) -> returns (..., 3, 3).
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    # Diagonal: (2q0^2 - 1) + 2*qi^2
    d = 2 * q0 * q0 - 1.0
    shape = q.shape[:-1]
    R = np.zeros(shape + (3, 3))

    R[..., 0, 0] = d + 2*q1*q1
    R[..., 1, 1] = d + 2*q2*q2
    R[..., 2, 2] = d + 2*q3*q3

    # Off-diagonal symmetric: 2*qi*qj
    R[..., 0, 1] = 2*(q1*q2 - q0*q3)
    R[..., 1, 0] = 2*(q1*q2 + q0*q3)
    R[..., 0, 2] = 2*(q1*q3 + q0*q2)
    R[..., 2, 0] = 2*(q1*q3 - q0*q2)
    R[..., 1, 2] = 2*(q2*q3 - q0*q1)
    R[..., 2, 1] = 2*(q2*q3 + q0*q1)

    return R


def random_su2_uniform(n, rng):
    """Uniform SU(2) via Marsaglia method. Returns (n, 4) unit quaternions."""
    result = np.zeros((n, 4))
    remaining = n
    idx = 0
    while remaining > 0:
        batch = remaining * 2  # oversample
        x = rng.uniform(-1, 1, size=(batch, 4))
        s1 = x[:, 0]**2 + x[:, 1]**2
        s2 = x[:, 2]**2 + x[:, 3]**2
        valid = (s1 < 1.0) & (s2 < 1.0)
        x_valid = x[valid]
        s1_valid = s1[valid]
        s2_valid = s2[valid]

        take = min(len(x_valid), remaining)
        if take == 0:
            continue

        f = np.sqrt((1 - s1_valid[:take]) / s2_valid[:take])
        result[idx:idx+take, 0] = x_valid[:take, 0]
        result[idx:idx+take, 1] = x_valid[:take, 1]
        result[idx:idx+take, 2] = x_valid[:take, 2] * f
        result[idx:idx+take, 3] = x_valid[:take, 3] * f
        idx += take
        remaining -= take

    return result


def random_su2_small(n, step, rng):
    """Small SU(2) perturbation around identity.

    q = (cos(eps/2), sin(eps/2)*n_hat) where eps ~ uniform(0, step), n_hat ~ S^2.
    Returns (n, 4) unit quaternions.
    """
    eps = rng.uniform(0, step, size=n)
    # Random axis on S^2 via Gaussian projection
    raw = rng.randn(n, 3)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    n_hat = raw / norms

    half_eps = eps / 2.0
    q = np.zeros((n, 4))
    q[:, 0] = np.cos(half_eps)
    sin_half = np.sin(half_eps)
    q[:, 1] = sin_half * n_hat[:, 0]
    q[:, 2] = sin_half * n_hat[:, 1]
    q[:, 3] = sin_half * n_hat[:, 2]
    return q


# =====================================================================
# Plaquette Enumeration on 3-Diamond
# =====================================================================

def enumerate_hexagonal_plaquettes(ei, ej, site_nbr, site_bond, N, N_bonds):
    """Find all 6-bond hexagonal "chair" rings on the 3-diamond lattice.

    Algorithm: For each directed bond (i->j), walk 5 more steps through the
    neighbor table looking for closed 6-cycles. Deduplicate by storing
    canonical sorted site tuples.

    Returns:
        plaq_bonds: (n_plaq, 6) int array of bond indices
        plaq_signs: (n_plaq, 6) int array of orientations (+1 forward, -1 conjugate)
        bond_to_plaq: list of lists, bond_to_plaq[b] = list of plaquette indices
                      containing bond b
    """
    # Build directed adjacency: for each site, its neighbors and the bond indices
    # site_nbr[i, k] = neighbor site, site_bond[i, k] = bond index
    z = site_nbr.shape[1]

    # For each bond, determine orientation: +1 if bond goes i->j (ei[b]=i, ej[b]=j),
    # -1 if reversed
    # Build a fast lookup: given (site_i, site_j), return (bond_idx, sign)
    bond_lookup = {}
    for b in range(N_bonds):
        i, j = int(ei[b]), int(ej[b])
        bond_lookup[(i, j)] = (b, +1)
        bond_lookup[(j, i)] = (b, -1)

    seen = set()
    plaq_list = []
    plaq_sign_list = []

    # For each site s0, for each neighbor direction, attempt to close a 6-ring
    for s0 in range(N):
        for k1 in range(z):
            s1 = int(site_nbr[s0, k1])
            for k2 in range(z):
                s2 = int(site_nbr[s1, k2])
                if s2 == s0:
                    continue
                for k3 in range(z):
                    s3 = int(site_nbr[s2, k3])
                    if s3 == s1 or s3 == s0:
                        continue
                    for k4 in range(z):
                        s4 = int(site_nbr[s3, k4])
                        if s4 == s2 or s4 == s1 or s4 == s0:
                            continue
                        for k5 in range(z):
                            s5 = int(site_nbr[s4, k5])
                            if s5 == s3 or s5 == s2 or s5 == s1:
                                continue
                            # Check if s5 is a neighbor of s0
                            if (s5, s0) not in bond_lookup:
                                continue
                            # Found a 6-ring: s0->s1->s2->s3->s4->s5->s0
                            sites = (s0, s1, s2, s3, s4, s5)
                            # Canonical form: smallest starting site, then
                            # compare forward vs reverse
                            canon = _canonical_hex(sites)
                            if canon in seen:
                                continue
                            seen.add(canon)

                            # Record bond indices and signs
                            bonds_6 = []
                            signs_6 = []
                            ring = [s0, s1, s2, s3, s4, s5]
                            for step in range(6):
                                sa = ring[step]
                                sb = ring[(step + 1) % 6]
                                b_idx, sign = bond_lookup[(sa, sb)]
                                bonds_6.append(b_idx)
                                signs_6.append(sign)

                            plaq_list.append(bonds_6)
                            plaq_sign_list.append(signs_6)

    n_plaq = len(plaq_list)
    if n_plaq == 0:
        plaq_bonds = np.zeros((0, 6), dtype=np.int32)
        plaq_signs = np.zeros((0, 6), dtype=np.int32)
    else:
        plaq_bonds = np.array(plaq_list, dtype=np.int32)
        plaq_signs = np.array(plaq_sign_list, dtype=np.int32)

    # Build bond -> plaquette mapping
    bond_to_plaq = [[] for _ in range(N_bonds)]
    for p in range(n_plaq):
        for k in range(6):
            b = int(plaq_bonds[p, k])
            bond_to_plaq[b].append(p)

    return plaq_bonds, plaq_signs, bond_to_plaq


def pad_bond_to_plaq(bond_to_plaq, n_bonds):
    """Convert ragged list-of-lists to padded arrays for Numba.

    Returns:
        bond_to_plaq_arr: (n_bonds, max_plaq_per_bond) int32 array
        bond_to_plaq_count: (n_bonds,) int32 array of actual counts
    """
    max_ppb = max(len(bond_to_plaq[b]) for b in range(n_bonds)) if n_bonds > 0 else 0
    arr = np.full((n_bonds, max_ppb), -1, dtype=np.int32)
    count = np.zeros(n_bonds, dtype=np.int32)
    for b in range(n_bonds):
        pp = bond_to_plaq[b]
        count[b] = len(pp)
        for k, p in enumerate(pp):
            arr[b, k] = p
    return arr, count


def _canonical_hex(sites):
    """Canonical form for a 6-site ring: rotate to smallest start, pick
    lexicographically smaller of forward and reverse."""
    n = len(sites)
    # Find all positions of the minimum site
    min_s = min(sites)
    starts = [i for i, s in enumerate(sites) if s == min_s]

    candidates = []
    for st in starts:
        fwd = tuple(sites[(st + i) % n] for i in range(n))
        rev = tuple(sites[(st - i) % n] for i in range(n))
        candidates.append(fwd)
        candidates.append(rev)

    return min(candidates)


def wilson_loop_trace(V_quat, plaq_bonds, plaq_signs):
    """Compute Re(tr(V_P))/2 for all plaquettes.

    V_P = V_{b0}^{s0} * V_{b1}^{s1} * ... * V_{b5}^{s5}
    where V^{+1} = V and V^{-1} = V^dagger (conjugate quaternion).

    Returns (n_plaq,) array.
    """
    n_plaq = plaq_bonds.shape[0]
    if n_plaq == 0:
        return np.array([])

    # Start with identity quaternion
    result = np.zeros((n_plaq, 4))
    result[:, 0] = 1.0  # identity

    for k in range(6):
        bond_idx = plaq_bonds[:, k]
        sign = plaq_signs[:, k]

        q_link = V_quat[bond_idx].copy()
        # For sign=-1, conjugate the quaternion
        neg_mask = sign < 0
        q_link[neg_mask, 1:] = -q_link[neg_mask, 1:]

        result = quat_multiply(result, q_link)

    # Re(tr(V_P))/2 = q0 (the scalar part)
    # Because tr(V) = 2*q0 for SU(2) in fundamental rep, so Re(tr)/2 = q0
    return result[:, 0]


# =====================================================================
# Gauge-Matter Coupling
# =====================================================================

def gauge_dressed_frame_alignment(R, V_so3, ei, ej):
    """Compute gauge-dressed frame alignment per bond.

    cos_dressed(i,j) = tr(R_i^T . V_so3_ij . R_j) / 3

    R: (N, 3, 3) frame rotations
    V_so3: (N_bonds, 3, 3) adjoint representation of gauge links
    ei, ej: (N_bonds,) bond endpoints
    """
    # V_so3[b] @ R[ej[b]] -> dressed neighbor frame
    R_dressed = np.einsum('bik,bkj->bij', V_so3, R[ej])
    return np.einsum('bji,bij->b', R[ei], R_dressed) / 3.0


def update_adjoint_cache(V_quat, bond_mask=None):
    """Compute SO(3) adjoint representation from quaternions.

    V_quat: (N_bonds, 4) unit quaternions
    bond_mask: optional boolean mask, only update these bonds

    Returns (N_bonds, 3, 3) SO(3) matrices.
    """
    if bond_mask is not None:
        result = np.zeros((len(V_quat), 3, 3))
        result[bond_mask] = quat_to_so3(V_quat[bond_mask])
        return result
    return quat_to_so3(V_quat)


# =====================================================================
# MC Sweeps
# =====================================================================

def frame_mc_sweep_gauge(R, K_F, V_so3, kstar_lut, site_nbr, site_bond,
                         ei, ej, A_sites, B_sites, step_size, beta, rng):
    """Frame Metropolis with gauge-dressed energy.

    Energy per bond: E_eff(cos_dressed) where cos_dressed = tr(R^T V R')/3.
    Sublattice-parallel bipartite Metropolis.
    Returns n_accepted.
    """
    z = site_nbr.shape[1]
    n_accepted = 0

    for sites in [A_sites, B_sites]:
        n_sub = len(sites)
        dR = random_rotation_small_batch(n_sub, step_size, rng)
        R_old = R[sites]
        R_new = np.einsum('nab,nbc->nac', dR, R_old)

        dE = np.zeros(n_sub)
        for k in range(z):
            nbr_sites = site_nbr[sites, k]
            bond_idxs = site_bond[sites, k]

            # Gauge-dressed alignment with V_so3
            V_b = V_so3[bond_idxs]  # (n_sub, 3, 3)
            R_nbr = R[nbr_sites]    # (n_sub, 3, 3)

            # Need to determine orientation: if site is ei[bond], V goes forward
            # if site is ej[bond], V goes backward (use V^T = V^{-1} in SO(3))
            is_forward = (ei[bond_idxs] == sites)
            # Forward: cos = tr(R_site^T @ V @ R_nbr) / 3
            # Backward: cos = tr(R_site^T @ V^T @ R_nbr) / 3

            # Compute dressed neighbor for new and old R
            # Forward case: V @ R_nbr
            R_dressed_fwd = np.einsum('nik,nkj->nij', V_b, R_nbr)
            # Backward case: V^T @ R_nbr
            R_dressed_bwd = np.einsum('nki,nkj->nij', V_b, R_nbr)

            R_dressed = np.where(is_forward[:, None, None], R_dressed_fwd, R_dressed_bwd)

            cos_new = np.einsum('nji,nij->n', R_new, R_dressed) / 3.0
            cos_old = np.einsum('nji,nij->n', R_old, R_dressed) / 3.0

            E_eff_new = kstar_lut.evaluate_Eeff(cos_new)
            E_eff_old = kstar_lut.evaluate_Eeff(cos_old)
            dE += (E_eff_new - E_eff_old)

        accept = dE <= 0
        if beta < float('inf'):
            boltzmann = np.exp(-beta * np.clip(dE, 0, 500))
            accept |= (rng.uniform(size=n_sub) < boltzmann)
        acc_idx = np.where(accept)[0]
        R[sites[acc_idx]] = R_new[acc_idx]
        n_accepted += len(acc_idx)

    return n_accepted


def gauge_mc_sweep(V_quat, V_so3, R, K_F, kstar_lut, ei, ej,
                   plaq_bonds, plaq_signs, bond_to_plaq_arr, bond_to_plaq_count,
                   beta, beta_g, step_gauge, rng):
    """Gauge link Metropolis sweep (Numba-accelerated inner loop).

    For each bond, propose V_new = random_small @ V_old.
    dE = dE_matter + dE_gauge
    Returns n_accepted.
    """
    n_bonds = len(ei)

    # Precompute current plaquette traces
    plaq_trace = wilson_loop_trace(V_quat, plaq_bonds, plaq_signs)

    # Pre-generate all random numbers for the sweep
    rand_eps = rng.uniform(0, step_gauge, size=n_bonds)
    rand_axes_raw = rng.randn(n_bonds, 3)
    rand_uniform = rng.uniform(size=n_bonds)

    # Get LUT arrays for inline evaluation
    lut_Eeff = kstar_lut.Eeff_grid
    lut_cos_min = kstar_lut.cos_min
    lut_dcos = kstar_lut.dcos
    lut_npoints = kstar_lut.n_points

    n_accepted = _gauge_mc_sweep_numba(
        V_quat, V_so3, R, ei, ej,
        plaq_bonds, plaq_signs, plaq_trace,
        bond_to_plaq_arr, bond_to_plaq_count,
        beta, beta_g,
        rand_eps, rand_axes_raw, rand_uniform,
        lut_Eeff, lut_cos_min, lut_dcos, lut_npoints)

    return n_accepted


@njit(cache=True)
def _gauge_mc_sweep_numba(V_quat, V_so3, R, ei, ej,
                          plaq_bonds, plaq_signs, plaq_trace,
                          bond_to_plaq_arr, bond_to_plaq_count,
                          beta, beta_g,
                          rand_eps, rand_axes_raw, rand_uniform,
                          lut_Eeff, lut_cos_min, lut_dcos, lut_npoints):
    """Numba-JIT gauge sweep inner loop."""
    n_bonds = len(ei)
    n_accepted = 0

    for b in range(n_bonds):
        i = ei[b]
        j = ej[b]

        # --- Generate small SU(2) perturbation ---
        eps = rand_eps[b]
        ax0 = rand_axes_raw[b, 0]
        ax1 = rand_axes_raw[b, 1]
        ax2 = rand_axes_raw[b, 2]
        ax_norm = (ax0*ax0 + ax1*ax1 + ax2*ax2) ** 0.5
        if ax_norm < 1e-15:
            ax_norm = 1.0
        ax0 /= ax_norm
        ax1 /= ax_norm
        ax2 /= ax_norm

        half_eps = eps * 0.5
        c_half = np.cos(half_eps)
        s_half = np.sin(half_eps)

        dq0 = c_half
        dq1 = s_half * ax0
        dq2 = s_half * ax1
        dq3 = s_half * ax2

        # Hamilton product: V_new = dq * V_old
        v0 = V_quat[b, 0]
        v1 = V_quat[b, 1]
        v2 = V_quat[b, 2]
        v3 = V_quat[b, 3]

        nq0 = dq0*v0 - dq1*v1 - dq2*v2 - dq3*v3
        nq1 = dq0*v1 + dq1*v0 + dq2*v3 - dq3*v2
        nq2 = dq0*v2 - dq1*v3 + dq2*v0 + dq3*v1
        nq3 = dq0*v3 + dq1*v2 - dq2*v1 + dq3*v0

        # Normalize
        nq_norm = (nq0*nq0 + nq1*nq1 + nq2*nq2 + nq3*nq3) ** 0.5
        if nq_norm < 1e-15:
            nq_norm = 1.0
        nq0 /= nq_norm
        nq1 /= nq_norm
        nq2 /= nq_norm
        nq3 /= nq_norm

        # --- Compute new SO(3) adjoint ---
        d = 2.0 * nq0 * nq0 - 1.0
        V_new_so3 = np.empty((3, 3))
        V_new_so3[0, 0] = d + 2*nq1*nq1
        V_new_so3[1, 1] = d + 2*nq2*nq2
        V_new_so3[2, 2] = d + 2*nq3*nq3
        V_new_so3[0, 1] = 2*(nq1*nq2 - nq0*nq3)
        V_new_so3[1, 0] = 2*(nq1*nq2 + nq0*nq3)
        V_new_so3[0, 2] = 2*(nq1*nq3 + nq0*nq2)
        V_new_so3[2, 0] = 2*(nq1*nq3 - nq0*nq2)
        V_new_so3[1, 2] = 2*(nq2*nq3 - nq0*nq1)
        V_new_so3[2, 1] = 2*(nq2*nq3 + nq0*nq1)

        # --- Matter energy change: cos_dressed = tr(R_i^T V R_j)/3 ---
        # tr(R_i^T V R_j) = sum_{a,b,c} R_i[b,a] * V[b,c] * R_j[c,a]
        cos_old = 0.0
        cos_new = 0.0
        for a in range(3):
            for bb in range(3):
                Ri_ba = R[i, bb, a]
                for c in range(3):
                    Rj_ca = R[j, c, a]
                    cos_old += Ri_ba * V_so3[b, bb, c] * Rj_ca
                    cos_new += Ri_ba * V_new_so3[bb, c] * Rj_ca
        cos_old /= 3.0
        cos_new /= 3.0

        # LUT evaluation (inline)
        dE_matter = _lut_eeff(cos_new, lut_Eeff, lut_cos_min, lut_dcos, lut_npoints) \
                  - _lut_eeff(cos_old, lut_Eeff, lut_cos_min, lut_dcos, lut_npoints)

        # --- Gauge (plaquette) energy change ---
        dE_gauge = 0.0
        n_affected = bond_to_plaq_count[b]
        for ap in range(n_affected):
            p_idx = bond_to_plaq_arr[b, ap]
            old_trace = plaq_trace[p_idx]

            # Compute new plaquette trace with proposed link
            new_trace = _plaq_trace_with_replacement(
                V_quat, plaq_bonds[p_idx], plaq_signs[p_idx],
                b, nq0, nq1, nq2, nq3)
            dE_gauge += -(new_trace - old_trace)

        dE_gauge *= beta_g
        # Total action change: beta * dE_matter (raw energy) + dE_gauge (already has beta_g)
        dS = beta * dE_matter + dE_gauge

        # Metropolis
        accept = False
        if dS <= 0:
            accept = True
        else:
            exp_arg = -min(dS, 500.0)
            if rand_uniform[b] < np.exp(exp_arg):
                accept = True

        if accept:
            V_quat[b, 0] = nq0
            V_quat[b, 1] = nq1
            V_quat[b, 2] = nq2
            V_quat[b, 3] = nq3
            for a in range(3):
                for c in range(3):
                    V_so3[b, a, c] = V_new_so3[a, c]
            n_accepted += 1
            # Update affected plaquette traces
            for ap in range(n_affected):
                p_idx = bond_to_plaq_arr[b, ap]
                plaq_trace[p_idx] = _compute_plaq_trace(
                    V_quat, plaq_bonds[p_idx], plaq_signs[p_idx])

    return n_accepted


@njit(cache=True)
def _lut_eeff(cos_val, Eeff_grid, cos_min, dcos, n_points):
    """Inline LUT evaluation for E_eff."""
    c = min(max(cos_val, cos_min), cos_min + dcos * n_points - 1e-10)
    idx_f = (c - cos_min) / dcos
    idx = min(int(idx_f), n_points - 1)
    frac = idx_f - idx
    return (1.0 - frac) * Eeff_grid[idx] + frac * Eeff_grid[idx + 1]


@njit(cache=True)
def _plaq_trace_with_replacement(V_quat, bonds_6, signs_6,
                                  replace_bond, rq0, rq1, rq2, rq3):
    """Compute plaquette trace, substituting one bond's quaternion."""
    q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0
    for k in range(6):
        bb = bonds_6[k]
        s = signs_6[k]
        if bb == replace_bond:
            lq0, lq1, lq2, lq3 = rq0, rq1, rq2, rq3
        else:
            lq0 = V_quat[bb, 0]
            lq1 = V_quat[bb, 1]
            lq2 = V_quat[bb, 2]
            lq3 = V_quat[bb, 3]
        if s < 0:
            lq1 = -lq1
            lq2 = -lq2
            lq3 = -lq3
        # Hamilton product: q = q * lq
        nq0 = q0*lq0 - q1*lq1 - q2*lq2 - q3*lq3
        nq1 = q0*lq1 + q1*lq0 + q2*lq3 - q3*lq2
        nq2 = q0*lq2 - q1*lq3 + q2*lq0 + q3*lq1
        nq3 = q0*lq3 + q1*lq2 - q2*lq1 + q3*lq0
        q0, q1, q2, q3 = nq0, nq1, nq2, nq3
    return q0


@njit(cache=True)
def _compute_plaq_trace(V_quat, bonds_6, signs_6):
    """Compute Re(tr(V_P))/2 for a single plaquette (Numba)."""
    q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0
    for k in range(6):
        bb = bonds_6[k]
        s = signs_6[k]
        lq0 = V_quat[bb, 0]
        lq1 = V_quat[bb, 1]
        lq2 = V_quat[bb, 2]
        lq3 = V_quat[bb, 3]
        if s < 0:
            lq1 = -lq1
            lq2 = -lq2
            lq3 = -lq3
        nq0 = q0*lq0 - q1*lq1 - q2*lq2 - q3*lq3
        nq1 = q0*lq1 + q1*lq0 + q2*lq3 - q3*lq2
        nq2 = q0*lq2 - q1*lq3 + q2*lq0 + q3*lq1
        nq3 = q0*lq3 + q1*lq2 - q2*lq1 + q3*lq0
        q0, q1, q2, q3 = nq0, nq1, nq2, nq3
    return q0


def _single_plaq_trace(V_quat, bonds_6, signs_6):
    """Compute Re(tr(V_P))/2 for a single plaquette (Python fallback)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(6):
        b = int(bonds_6[k])
        s = int(signs_6[k])
        q_link = V_quat[b].copy()
        if s < 0:
            q_link[1:] = -q_link[1:]
        q = quat_multiply(q[None], q_link[None])[0]
    return q[0]


# =====================================================================
# Observables
# =====================================================================

def measure_wilson_loops(V_quat, plaq_bonds, plaq_signs):
    """Measure Wilson loop observables.

    Returns dict with mean_plaquette, plaquette_std.
    """
    traces = wilson_loop_trace(V_quat, plaq_bonds, plaq_signs)
    n_plaq = len(traces)
    if n_plaq == 0:
        return {'mean_plaquette': 1.0, 'plaquette_std': 0.0, 'n_plaq': 0}

    return {
        'mean_plaquette': float(np.mean(traces)),
        'plaquette_std': float(np.std(traces)),
        'n_plaq': n_plaq,
    }


def measure_polyakov_loop(V_quat, ei, ej, positions, L, axis=0):
    """Measure Polyakov loop along specified axis.

    For simplicity, trace product of gauge links along a straight line
    in the given axis direction. Returns mean |P|.
    """
    # On diamond lattice, Polyakov loops are non-trivial to define.
    # Use a proxy: average of plaquette traces is the gauge order parameter.
    # Full Polyakov loop implementation deferred — mean plaquette is
    # sufficient for confinement signal in 3D SU(2).
    return 0.0  # placeholder


def measure_gauge_skyrmion(K_F, R, V_so3, ei, ej, positions, center):
    """Measure Skyrmion mass + gauge flux profile."""
    dead_mask = K_F < 1e-4
    mass_info = measure_skyrmion_mass(K_F, dead_mask)

    # Gauge flux at Skyrmion core vs bulk
    mid = (positions[ei] + positions[ej]) / 2.0
    dists = np.linalg.norm(mid - center, axis=1)
    r_core = 3.0  # approximate Skyrmion core radius

    core_mask = dists < r_core
    bulk_mask = dists >= r_core

    # V_so3 deviation from identity: ||V - I||_F
    V_dev = np.zeros(len(ei))
    for b in range(len(ei)):
        V_dev[b] = np.linalg.norm(V_so3[b] - np.eye(3))

    core_flux = float(np.mean(V_dev[core_mask])) if np.any(core_mask) else 0.0
    bulk_flux = float(np.mean(V_dev[bulk_mask])) if np.any(bulk_mask) else 0.0

    mass_info['core_gauge_flux'] = core_flux
    mass_info['bulk_gauge_flux'] = bulk_flux
    mass_info['flux_ratio'] = core_flux / max(bulk_flux, 1e-15)

    return mass_info


# =====================================================================
# Record Observables
# =====================================================================

def record_observables(sweep, R, K_F, V_quat, V_so3,
                       ei, ej, positions, center, N,
                       plaq_bonds, plaq_signs,
                       n_acc_frame, n_acc_gauge, beta_g):
    """Record all observables for one snapshot."""
    n_bonds = len(ei)

    # Frame sector
    cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
    alive_F = K_F > 1e-4
    K_F_mean = float(np.mean(K_F[alive_F])) if np.any(alive_F) else 0.0

    dead_mask_F, n_clusters_F, sizes_F, _ = identify_dead_bond_clusters(
        K_F, ei, ej, N, K_thresh=1e-4)
    mass_info = measure_gauge_skyrmion(K_F, R, V_so3, ei, ej, positions, center)

    # Gauge sector
    wilson_info = measure_wilson_loops(V_quat, plaq_bonds, plaq_signs)

    # Coherence capital
    C_frame = float(np.sum(frame_coherence_capital(K_F)))

    obs = {
        'sweep': sweep,
        'K_F_mean': K_F_mean,
        'cos_dressed_mean': float(np.mean(cos_dressed)),
        'n_dead_frame': int(np.sum(~alive_F)),
        'n_dead_clusters': n_clusters_F,
        'M_skyrmion': mass_info['M_skyrmion'],
        'mean_plaquette': wilson_info['mean_plaquette'],
        'plaquette_std': wilson_info['plaquette_std'],
        'core_gauge_flux': mass_info.get('core_gauge_flux', 0.0),
        'bulk_gauge_flux': mass_info.get('bulk_gauge_flux', 0.0),
        'mc_accept_frame': n_acc_frame,
        'mc_accept_gauge': n_acc_gauge,
        'C_frame': C_frame,
    }
    return obs


# =====================================================================
# Protocol G1: Sigma Model Control (beta_g = infinity, V = I)
# =====================================================================

def protocol_g1(L, seed, r_F=6.0, beta=5.0,
                n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                record_interval=100, verbose=True):
    """Sigma model control: V_ij = I for all bonds (frozen).
    Reproduce E1 Skyrmion lifetime (~11,750 sweeps at r_F=6.0).
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol G1: Sigma Model Control (beta_g = inf)")
        print(f"  L={L}, seed={seed}, beta={beta}, r_F={r_F}")
        print(f"  n_mc_steps={n_mc_steps}, n_equil={n_equil}")
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

    # Build LUT
    kstar_lut = _KstarCosLUT_frame(r_F=r_F)

    # Frozen gauge links: V = I (identity quaternion)
    V_quat = np.zeros((n_bonds, 4))
    V_quat[:, 0] = 1.0
    V_so3 = np.tile(np.eye(3), (n_bonds, 1, 1))

    # No plaquettes needed (V frozen)
    plaq_bonds = np.zeros((0, 6), dtype=np.int32)
    plaq_signs = np.zeros((0, 6), dtype=np.int32)

    # Ordered start: R_i = I_3
    R = np.tile(np.eye(3), (N, 1, 1))
    cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
    K_F = kstar_lut.evaluate_K(cos_dressed)

    step_size = 0.15

    if verbose:
        print(f"  Pre-equilibrating {n_equil} sweeps (frame only)...")

    # Pre-equilibrate
    for s in range(n_equil):
        frame_mc_sweep_gauge(R, K_F, V_so3, kstar_lut, site_nbr, site_bond,
                             ei, ej, A_sites, B_sites, step_size, beta, rng)
        cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
        K_F[:] = kstar_lut.evaluate_K(cos_dressed)

    if verbose:
        print(f"  Post-equil: K_F_mean={np.mean(K_F):.4f}")

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

    cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
    K_F[:] = kstar_lut.evaluate_K(cos_dressed)

    dead_mask, n_clust, sizes, _ = identify_dead_bond_clusters(K_F, ei, ej, N)
    if verbose:
        print(f"  Post-kick: n_dead={np.sum(dead_mask)}, n_clusters={n_clust}")

    # Run MC
    history = []
    dissipation_sweep = None

    for s in range(n_mc_steps):
        n_acc_frame = frame_mc_sweep_gauge(
            R, K_F, V_so3, kstar_lut, site_nbr, site_bond,
            ei, ej, A_sites, B_sites, step_size, beta, rng)
        cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
        K_F[:] = kstar_lut.evaluate_K(cos_dressed)

        if (s + 1) % record_interval == 0:
            obs = record_observables(
                s + 1, R, K_F, V_quat, V_so3,
                ei, ej, positions, center, N,
                plaq_bonds, plaq_signs,
                n_acc_frame, 0, beta_g=float('inf'))
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: K_F={obs['K_F_mean']:.4f}, "
                      f"n_dead={obs['n_dead_frame']}, "
                      f"M_sky={obs['M_skyrmion']:.1f}")

            if obs['n_dead_frame'] == 0 and dissipation_sweep is None and s > 0:
                dissipation_sweep = s + 1

    # Verdict
    final = history[-1] if history else {}
    n_dead_final = final.get('n_dead_frame', 0)
    verdict = "SKYRMION_SURVIVED" if n_dead_final > 0 else "SKYRMION_DISSIPATED"

    elapsed = time.time() - t0
    result = {
        'protocol': 'G1',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'seed': seed, 'beta': beta, 'beta_g': float('inf'),
        'r_F': r_F,
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
# Protocol G2: Gauge Confinement Test (beta_g = 2.0)
# =====================================================================

def protocol_g2(L, seed, r_F=6.0, beta=5.0, beta_g=2.0,
                n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                step_gauge=0.5, record_interval=100, verbose=True):
    """THE TEST: Independent SU(2) gauge links with Wilson plaquette action.
    Does SU(2) confinement stabilize the Skyrmion beyond 20k sweeps?
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol G2: Gauge Confinement Test")
        print(f"  L={L}, seed={seed}, beta={beta}, beta_g={beta_g}")
        print(f"  step_gauge={step_gauge}")
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

    # Build LUT
    kstar_lut = _KstarCosLUT_frame(r_F=r_F)

    # Enumerate plaquettes
    if verbose:
        print(f"  Enumerating hexagonal plaquettes...")
        t_plaq = time.time()
    plaq_bonds, plaq_signs, bond_to_plaq = enumerate_hexagonal_plaquettes(
        ei, ej, site_nbr, site_bond, N, n_bonds)
    n_plaq = len(plaq_bonds)
    bond_to_plaq_arr, bond_to_plaq_count = pad_bond_to_plaq(bond_to_plaq, n_bonds)
    if verbose:
        print(f"  Found {n_plaq} plaquettes in {time.time()-t_plaq:.1f}s")

    # Initialize gauge links to identity
    V_quat = np.zeros((n_bonds, 4))
    V_quat[:, 0] = 1.0
    V_so3 = np.tile(np.eye(3), (n_bonds, 1, 1))

    # Ordered start: R_i = I_3
    R = np.tile(np.eye(3), (N, 1, 1))
    cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
    K_F = kstar_lut.evaluate_K(cos_dressed)

    step_size = 0.15

    # Pre-equilibrate (frame + gauge)
    if verbose:
        print(f"  Pre-equilibrating {n_equil} sweeps (frame + gauge)...")
    for s in range(n_equil):
        frame_mc_sweep_gauge(R, K_F, V_so3, kstar_lut, site_nbr, site_bond,
                             ei, ej, A_sites, B_sites, step_size, beta, rng)
        gauge_mc_sweep(V_quat, V_so3, R, K_F, kstar_lut, ei, ej,
                       plaq_bonds, plaq_signs, bond_to_plaq_arr, bond_to_plaq_count,
                       beta, beta_g, step_gauge, rng)
        cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
        K_F[:] = kstar_lut.evaluate_K(cos_dressed)

    if verbose:
        plaq_mean = float(np.mean(wilson_loop_trace(V_quat, plaq_bonds, plaq_signs)))
        print(f"  Post-equil: K_F_mean={np.mean(K_F):.4f}, "
              f"mean_plaq={plaq_mean:.4f}")

    # Inject hedgehog Skyrmion
    if verbose:
        print(f"  Injecting hedgehog kick...")
    for i in range(N):
        r_vec = positions[i] - center
        r_dist = np.linalg.norm(r_vec)
        if r_dist < kick_radius:
            f_val = kick_strength * np.pi * max(1.0 - r_dist / kick_radius, 0.0)
            R[i] = hedgehog_rotation(r_vec, f_val) @ R[i]

    cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
    K_F[:] = kstar_lut.evaluate_K(cos_dressed)

    dead_mask, n_clust, sizes, _ = identify_dead_bond_clusters(K_F, ei, ej, N)
    if verbose:
        print(f"  Post-kick: n_dead={np.sum(dead_mask)}, n_clusters={n_clust}")

    # Run MC
    history = []
    dissipation_sweep = None

    for s in range(n_mc_steps):
        n_acc_frame = frame_mc_sweep_gauge(
            R, K_F, V_so3, kstar_lut, site_nbr, site_bond,
            ei, ej, A_sites, B_sites, step_size, beta, rng)
        n_acc_gauge = gauge_mc_sweep(
            V_quat, V_so3, R, K_F, kstar_lut, ei, ej,
            plaq_bonds, plaq_signs, bond_to_plaq_arr, bond_to_plaq_count,
            beta, beta_g, step_gauge, rng)
        cos_dressed = gauge_dressed_frame_alignment(R, V_so3, ei, ej)
        K_F[:] = kstar_lut.evaluate_K(cos_dressed)

        if (s + 1) % record_interval == 0:
            obs = record_observables(
                s + 1, R, K_F, V_quat, V_so3,
                ei, ej, positions, center, N,
                plaq_bonds, plaq_signs,
                n_acc_frame, n_acc_gauge, beta_g)
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: K_F={obs['K_F_mean']:.4f}, "
                      f"n_dead={obs['n_dead_frame']}, "
                      f"M_sky={obs['M_skyrmion']:.1f}, "
                      f"plaq={obs['mean_plaquette']:.4f}, "
                      f"acc_g={n_acc_gauge}/{n_bonds}")

            if obs['n_dead_frame'] == 0 and dissipation_sweep is None and s > 0:
                dissipation_sweep = s + 1

    # Verdict
    final = history[-1] if history else {}
    n_dead_final = final.get('n_dead_frame', 0)
    verdict = "SKYRMION_SURVIVED" if n_dead_final > 0 else "SKYRMION_DISSIPATED"

    elapsed = time.time() - t0
    result = {
        'protocol': 'G2',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'n_plaq': n_plaq,
        'seed': seed, 'beta': beta, 'beta_g': beta_g,
        'r_F': r_F,
        'step_gauge': step_gauge,
        'n_mc_steps': n_mc_steps,
        'n_equil': n_equil,
        'kick_strength': kick_strength,
        'kick_radius': kick_radius,
        'dissipation_sweep': dissipation_sweep,
        'final_n_dead_frame': n_dead_final,
        'final_M_skyrmion': final.get('M_skyrmion', 0),
        'final_mean_plaquette': final.get('mean_plaquette', 0),
        'elapsed_sec': elapsed,
        'history': history,
    }

    if verbose:
        print(f"\n  VERDICT: {verdict}")
        if dissipation_sweep is not None:
            print(f"  Dissipation at sweep ~{dissipation_sweep}")
        else:
            print(f"  Skyrmion SURVIVED all {n_mc_steps} sweeps!")
        print(f"  Final mean plaquette: {final.get('mean_plaquette', 0):.4f}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Protocol G3: beta_g Scan
# =====================================================================

def protocol_g3(L, seed, r_F=6.0, beta=5.0,
                n_mc_steps=20000, n_equil=2000,
                kick_strength=1.0, kick_radius=None,
                step_gauge=0.5, record_interval=100,
                beta_g_values=None, verbose=True):
    """Map Skyrmion lifetime vs gauge coupling strength."""
    t0 = time.time()

    if beta_g_values is None:
        beta_g_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol G3: beta_g Scan")
        print(f"  L={L}, seed={seed}, beta={beta}")
        print(f"  beta_g values: {beta_g_values}")
        print(f"{'='*70}")

    scan_results = []

    for bg in beta_g_values:
        if verbose:
            print(f"\n  --- beta_g = {bg} ---")

        res = protocol_g2(
            L=L, seed=seed, r_F=r_F, beta=beta, beta_g=bg,
            n_mc_steps=n_mc_steps, n_equil=n_equil,
            kick_strength=kick_strength, kick_radius=kick_radius,
            step_gauge=step_gauge, record_interval=record_interval,
            verbose=verbose)

        scan_results.append({
            'beta_g': bg,
            'verdict': res['verdict'],
            'dissipation_sweep': res['dissipation_sweep'],
            'final_n_dead_frame': res['final_n_dead_frame'],
            'final_M_skyrmion': res['final_M_skyrmion'],
            'final_mean_plaquette': res.get('final_mean_plaquette', 0),
        })

        # Save individual result
        fname = (f"{OUTPUT_DIR}/gauge_skyrmion_G2_L{L}_bg{bg}_s{seed}.json")
        with open(fname, 'w') as f:
            json.dump(jsonify(res), f, indent=2)

    elapsed = time.time() - t0
    result = {
        'protocol': 'G3',
        'L': L, 'N': 2 * L**3, 'seed': seed,
        'beta': beta, 'r_F': r_F,
        'beta_g_values': beta_g_values,
        'n_mc_steps': n_mc_steps,
        'scan_results': scan_results,
        'elapsed_sec': elapsed,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  G3 Scan Summary:")
        print(f"  {'beta_g':>8s}  {'verdict':>25s}  {'dissip':>8s}  "
              f"{'n_dead':>6s}  {'plaq':>8s}")
        for sr in scan_results:
            ds = str(sr['dissipation_sweep']) if sr['dissipation_sweep'] else 'ALIVE'
            print(f"  {sr['beta_g']:8.1f}  {sr['verdict']:>25s}  {ds:>8s}  "
                  f"{sr['final_n_dead_frame']:6d}  "
                  f"{sr['final_mean_plaquette']:8.4f}")
        print(f"  Total elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Protocol G4: String Tension Measurement (pure gauge)
# =====================================================================

def protocol_g4(L, seed, beta_g=2.0,
                n_mc_steps=10000, n_equil=1000,
                step_gauge=0.5, record_interval=100, verbose=True):
    """Pure gauge sector: R frozen to identity, measure string tension.

    String tension from Wilson loops: sigma > 0 confirms confinement.
    """
    t0 = time.time()
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Protocol G4: String Tension (Pure Gauge)")
        print(f"  L={L}, seed={seed}, beta_g={beta_g}")
        print(f"  n_mc_steps={n_mc_steps}")
        print(f"{'='*70}")

    # Build lattice
    (positions, N, ei, ej, bonds, sublat,
     site_nbr, site_bond, A_sites, B_sites) = make_d_diamond_with_neighbors(3, L)
    n_bonds = len(ei)

    # Build frame LUT (needed for energy computation but frames are frozen)
    r_F = 6.0
    kstar_lut = _KstarCosLUT_frame(r_F=r_F)

    # Enumerate plaquettes
    if verbose:
        print(f"  Enumerating hexagonal plaquettes...")
        t_plaq = time.time()
    plaq_bonds, plaq_signs, bond_to_plaq = enumerate_hexagonal_plaquettes(
        ei, ej, site_nbr, site_bond, N, n_bonds)
    n_plaq = len(plaq_bonds)
    bond_to_plaq_arr, bond_to_plaq_count = pad_bond_to_plaq(bond_to_plaq, n_bonds)
    if verbose:
        print(f"  Found {n_plaq} plaquettes in {time.time()-t_plaq:.1f}s")

    # Frozen frames: R = I
    R = np.tile(np.eye(3), (N, 1, 1))

    # Initialize gauge links: random SU(2) (hot start for pure gauge)
    V_quat = random_su2_uniform(n_bonds, rng)
    V_so3 = quat_to_so3(V_quat)

    K_F = np.zeros(n_bonds)  # Frame K not meaningful when R is frozen

    if verbose:
        plaq_mean = float(np.mean(wilson_loop_trace(V_quat, plaq_bonds, plaq_signs)))
        print(f"  Initial mean plaquette (hot start): {plaq_mean:.4f}")

    # Equilibrate gauge sector
    if verbose:
        print(f"  Equilibrating {n_equil} gauge sweeps...")
    for s in range(n_equil):
        gauge_mc_sweep(V_quat, V_so3, R, K_F, kstar_lut, ei, ej,
                       plaq_bonds, plaq_signs, bond_to_plaq_arr, bond_to_plaq_count,
                       1.0, beta_g, step_gauge, rng)

    # Measure
    history = []
    plaq_accumulator = []

    for s in range(n_mc_steps):
        n_acc = gauge_mc_sweep(V_quat, V_so3, R, K_F, kstar_lut, ei, ej,
                               plaq_bonds, plaq_signs, bond_to_plaq_arr, bond_to_plaq_count,
                               1.0, beta_g, step_gauge, rng)

        if (s + 1) % record_interval == 0:
            traces = wilson_loop_trace(V_quat, plaq_bonds, plaq_signs)
            mean_plaq = float(np.mean(traces))
            plaq_accumulator.append(mean_plaq)

            obs = {
                'sweep': s + 1,
                'mean_plaquette': mean_plaq,
                'plaquette_std': float(np.std(traces)),
                'mc_accept_gauge': n_acc,
            }
            history.append(obs)

            if verbose and (s + 1) % (record_interval * 10) == 0:
                print(f"    sweep {s+1}: plaq={mean_plaq:.4f}, "
                      f"acc={n_acc}/{n_bonds}")

    # String tension estimate
    # For 3D SU(2), string tension from mean plaquette:
    # In strong coupling expansion: <P> ~ (beta_g/4)^6 for hexagonal plaquettes
    # sigma = -log(<W(C)>) / Area(C)
    # Simple proxy: sigma_proxy = -log(max(<plaq>, 1e-15))
    mean_plaq_final = float(np.mean(plaq_accumulator)) if plaq_accumulator else 0.0
    sigma_proxy = -np.log(max(mean_plaq_final, 1e-15))

    elapsed = time.time() - t0
    result = {
        'protocol': 'G4',
        'L': L, 'N': N, 'n_bonds': n_bonds,
        'n_plaq': n_plaq,
        'seed': seed, 'beta_g': beta_g,
        'n_mc_steps': n_mc_steps,
        'n_equil': n_equil,
        'step_gauge': step_gauge,
        'mean_plaquette': mean_plaq_final,
        'sigma_proxy': sigma_proxy,
        'confined': sigma_proxy > 0,
        'elapsed_sec': elapsed,
        'history': history,
    }

    if verbose:
        print(f"\n  Results:")
        print(f"  Mean plaquette: {mean_plaq_final:.6f}")
        print(f"  String tension proxy: {sigma_proxy:.4f}")
        print(f"  Confined: {sigma_proxy > 0}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-LT-197: SU(2) Wilson Plaquette Gauge Confinement "
                    "for Skyrmion Stabilization")
    parser.add_argument('--protocol', type=str, default='G2',
                        choices=['G1', 'G2', 'G3', 'G4', 'all'],
                        help='Protocol to run (default: G2)')
    parser.add_argument('--L', type=int, default=12,
                        help='Lattice size (default: 12)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--beta', type=float, default=5.0,
                        help='Inverse temperature (default: 5.0)')
    parser.add_argument('--beta_g', type=float, default=2.0,
                        help='Gauge coupling (default: 2.0)')
    parser.add_argument('--r_F', type=float, default=6.0,
                        help='Frame CLR coupling (default: 6.0)')
    parser.add_argument('--n_mc_steps', type=int, default=20000,
                        help='MC sweeps (default: 20000)')
    parser.add_argument('--n_equil', type=int, default=2000,
                        help='Equilibration sweeps (default: 2000)')
    parser.add_argument('--kick_strength', type=float, default=1.0,
                        help='Hedgehog kick strength (default: 1.0)')
    parser.add_argument('--kick_radius', type=float, default=None,
                        help='Hedgehog kick radius (default: L*nn_dist/4)')
    parser.add_argument('--step_gauge', type=float, default=0.5,
                        help='Gauge link step size (default: 0.5)')
    parser.add_argument('--record_interval', type=int, default=100,
                        help='Recording interval (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    verbose = not args.quiet
    protocols = []

    if args.protocol == 'all':
        protocols = ['G1', 'G2', 'G3', 'G4']
    else:
        protocols = [args.protocol]

    all_results = {}

    for proto in protocols:
        if proto == 'G1':
            res = protocol_g1(
                L=args.L, seed=args.seed, r_F=args.r_F,
                beta=args.beta, n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'G2':
            res = protocol_g2(
                L=args.L, seed=args.seed, r_F=args.r_F,
                beta=args.beta, beta_g=args.beta_g,
                n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                step_gauge=args.step_gauge,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'G3':
            res = protocol_g3(
                L=args.L, seed=args.seed, r_F=args.r_F,
                beta=args.beta, n_mc_steps=args.n_mc_steps,
                n_equil=args.n_equil,
                kick_strength=args.kick_strength,
                kick_radius=args.kick_radius,
                step_gauge=args.step_gauge,
                record_interval=args.record_interval,
                verbose=verbose)
        elif proto == 'G4':
            res = protocol_g4(
                L=args.L, seed=args.seed, beta_g=args.beta_g,
                n_mc_steps=args.n_mc_steps,
                n_equil=min(args.n_equil, 1000),
                step_gauge=args.step_gauge,
                record_interval=args.record_interval,
                verbose=verbose)
        else:
            continue

        all_results[proto] = res

        # Save individual result
        bg_str = f"bg{args.beta_g}" if proto != 'G1' else "bginf"
        if proto == 'G3':
            bg_str = "bgscan"
        fname = (f"{OUTPUT_DIR}/gauge_skyrmion_{proto}_L{args.L}_{bg_str}"
                 f"_s{args.seed}.json")
        with open(fname, 'w') as f:
            json.dump(jsonify(res), f, indent=2)
        if verbose:
            print(f"  Saved: {fname}")

    # Summary
    if verbose and len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        for proto, res in all_results.items():
            print(f"  {proto}: {res.get('verdict', 'N/A')}")


if __name__ == "__main__":
    main()
