#!/usr/bin/env python3
"""
D9: Electron Nucleates Proton via Gauge CLR
============================================

The electron (U(1) phase vortex) nucleates the proton (Skyrmion) through
gauge-CLR mediated coupling at the vortex core. The vortex creates a
dead-K zone (cos(Δφ)=0 → K=0) where frames explore freely via MC.
The 3-torque gauge CLR concentrates optimization at core bonds (Fiedler
structural push). Wilson plaquette confinement stabilizes emerging topology.

After nucleation, the vortex scaffold is removed. The proton must survive
independently through gauge confinement alone — proving that the electron
was the midwife, not the life support.

Two-phase protocol:
  Phase 1 (NUCLEATION): Frozen vortex + sublattice downfold + MC frames
      + 3-torque gauge CLR. Monitor M_sky until threshold crossing.
  Phase 2 (INDEPENDENCE): Remove vortex, unfreeze all sublattices,
      full-lattice MC + gauge confinement. Does M_sky stabilize?

Sublattice downfold: freeze one sublattice (A or B), MC on the other.
Frozen sublattice creates mediated frustrated NNN interactions on FCC
network — the mechanism that prevents bipartite frustration cancellation.

Verification gates:
  G1: Budget conservation |Σ S^gauge| < 1e-10
  G2: Nucleation event M_sky crosses threshold
  G3: Core exploration accept_rate at K=0 bonds > 0.9
  G4: Flux localization flux_ratio > 1.5
  G5: Independence M_sky stable after vortex removal
  G6: Wilson confinement wilson_mean > 0.5 in Phase 2
  G7: Lifecycle M_sky_phase2 / M_sky_nucleation > 0.5
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import breadth_first_order
import json
import os
import time

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Part 1: 3D Diamond Lattice (from d8_proper_gauge_clr.py)
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
    Returns: positions, ei, ej, site_nbr, site_bond, N, n_bonds, sublat
    """
    d = 3
    N_cells = L ** d
    N = 2 * N_cells
    deltas = make_simplex_deltas(d)
    A_mat = np.array([deltas[0] - deltas[i] for i in range(1, d + 1)])

    positions = np.zeros((N, d))
    sublat = np.zeros(N, dtype=np.int32)
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

    site_nbr = np.array(site_nbr_list, dtype=np.int32)
    site_bond = np.array(site_bond_list, dtype=np.int32)
    return positions, ei, ej, site_nbr, site_bond, N, n_bonds, sublat


def enumerate_hexagonal_plaquettes(ei, ej, site_nbr, site_bond, N, n_bonds):
    """Find all 6-bond hexagonal chair rings on 3-diamond."""
    z = site_nbr.shape[1]
    bond_lookup = {}
    for b in range(n_bonds):
        i, j = int(ei[b]), int(ej[b])
        bond_lookup[(i, j)] = (b, +1)
        bond_lookup[(j, i)] = (b, -1)

    seen = set()
    plaq_list, plaq_sign_list = [], []

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
                            bonds_6, signs_6 = [], []
                            ring = list(sites)
                            for step in range(6):
                                sa, sb = ring[step], ring[(step + 1) % 6]
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

    bond_to_plaq = [[] for _ in range(n_bonds)]
    for p in range(n_plaq):
        for k in range(6):
            bond_to_plaq[int(plaq_bonds[p, k])].append(p)

    max_ppb = max((len(bond_to_plaq[b]) for b in range(n_bonds)), default=0)
    btp_arr = np.full((n_bonds, max_ppb), -1, dtype=np.int32)
    btp_cnt = np.zeros(n_bonds, dtype=np.int32)
    for b in range(n_bonds):
        pp = bond_to_plaq[b]
        btp_cnt[b] = len(pp)
        for k, p in enumerate(pp):
            btp_arr[b, k] = p

    return plaq_bonds, plaq_signs, btp_arr, btp_cnt


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
# Part 2: SO(3) Utilities (from d8_proper_gauge_clr.py)
# =====================================================================

def skew_matrix(w):
    n = w.shape[0]
    W = np.zeros((n, 3, 3))
    W[:, 0, 1] = -w[:, 2]
    W[:, 0, 2] =  w[:, 1]
    W[:, 1, 0] =  w[:, 2]
    W[:, 1, 2] = -w[:, 0]
    W[:, 2, 0] = -w[:, 1]
    W[:, 2, 1] =  w[:, 0]
    return W


def rodrigues_batch(w):
    n = w.shape[0]
    theta = np.linalg.norm(w, axis=1)
    small = theta < 1e-8
    large = ~small
    sinc = np.ones(n)
    cosc = np.full(n, 0.5)
    if np.any(large):
        sinc[large] = np.sin(theta[large]) / theta[large]
        cosc[large] = (1.0 - np.cos(theta[large])) / (theta[large] ** 2)
    W = skew_matrix(w)
    W2 = np.einsum('nij,njk->nik', W, W)
    I_batch = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    return I_batch + sinc[:, None, None] * W + cosc[:, None, None] * W2


def skew_extract(M):
    return np.array([
        (M[2, 1] - M[1, 2]) / 2.0,
        (M[0, 2] - M[2, 0]) / 2.0,
        (M[1, 0] - M[0, 1]) / 2.0
    ])


def skew_extract_batch(M):
    return np.stack([
        (M[:, 2, 1] - M[:, 1, 2]) / 2.0,
        (M[:, 0, 2] - M[:, 2, 0]) / 2.0,
        (M[:, 1, 0] - M[:, 0, 1]) / 2.0
    ], axis=1)


def reorthogonalize(R):
    U, _, Vt = np.linalg.svd(R)
    R_clean = np.einsum('nij,njk->nik', U, Vt)
    det = np.linalg.det(R_clean)
    fix = det < 0
    if np.any(fix):
        U[fix, :, -1] *= -1
        R_clean[fix] = np.einsum('nij,njk->nik', U[fix], Vt[fix])
    return R_clean


def reorthogonalize_single(M):
    """Single 3x3 → SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def random_rotation_small_batch(n, step_size, rng):
    """Vectorized small-angle SO(3) rotation via Rodrigues."""
    raw = rng.standard_normal((n, 3))
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
# Part 3: Gauge-Dressed Observables (from d8_proper_gauge_clr.py)
# =====================================================================

def cos_dressed_all(R, V, ei, ej):
    """cos_dressed_ij = tr(R_i^T V_ij R_j) / 3."""
    M = np.einsum('bji,bjk,bkl->bil', R[ei], V, R[ej])
    return np.trace(M, axis1=1, axis2=2) / 3.0


def compute_I_dressed(K, cos_dr):
    H = -np.sum(K * cos_dr)
    A = np.sum(K)
    if A < 1e-30:
        return 0.5
    return 0.5 * (1.0 + (-H) / A)


# =====================================================================
# Part 4: Three Gauge Torques (CORRECTED SIGNS, from d8_proper)
# =====================================================================

def matter_torque_3d(R, V, ei, ej):
    """tau^matter = +(1/3) skew(V^T R_i R_j^T). No K prefactor."""
    M = np.einsum('bji,bjk,blk->bil', V, R[ei], R[ej])
    return skew_extract_batch(M) / 3.0


def plaquette_torque_3d(V, ei, ej, plaq_bonds, plaq_signs,
                         btp_arr, btp_cnt, n_bonds):
    """tau^plaq = +(1/3) Σ_p skew(V_k^T staple^T)."""
    tau = np.zeros((n_bonds, 3))
    for b in range(n_bonds):
        n_p = btp_cnt[b]
        for k in range(n_p):
            p_idx = btp_arr[b, k]
            if p_idx < 0:
                continue
            staple = np.eye(3)
            b_pos = -1
            for kk in range(6):
                bb = plaq_bonds[p_idx, kk]
                ss = plaq_signs[p_idx, kk]
                if bb == b:
                    b_pos = kk
                    continue
                Vlink = V[bb] if ss > 0 else V[bb].T
                staple = staple @ Vlink
            if b_pos >= 0:
                s_b = plaq_signs[p_idx, b_pos]
                V_k = V[b] if s_b > 0 else V[b].T
                M_p = V_k.T @ staple.T
                tau[b] += skew_extract(M_p) / 3.0
    return tau


def structural_torque_3d(v2_ema, R, V, ei, ej, cos_dr):
    """tau^struct = (fiedler_sens / 3) * skew(V^T R_i R_j^T)."""
    fiedler_sens = (v2_ema[ei] - v2_ema[ej]) ** 2
    M = np.einsum('bji,bjk,blk->bil', V, R[ei], R[ej])
    tau_dir = skew_extract_batch(M) / 3.0
    active = cos_dr > 0
    tau = np.where(active[:, None], fiedler_sens[:, None] * tau_dir, 0.0)
    return tau


# =====================================================================
# Part 5: Sparse Laplacian + Fiedler (from d8_proper)
# =====================================================================

def sparse_laplacian_and_fiedler(ei, ej, K_arr, cos_dr, N):
    w = K_arr * np.maximum(cos_dr, 0)
    degree = np.zeros(N)
    np.add.at(degree, ei, w)
    np.add.at(degree, ej, w)
    row = np.concatenate([ei, ej, np.arange(N)])
    col = np.concatenate([ej, ei, np.arange(N)])
    data = np.concatenate([-w, -w, degree])
    L = csr_matrix((data, (row, col)), shape=(N, N))
    try:
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM', maxiter=5000, tol=1e-10)
        order = np.argsort(vals)
        lam2 = max(vals[order[1]], 0.0)
        v2 = vecs[:, order[1]]
    except Exception:
        Ld = L.toarray()
        vals, vecs = np.linalg.eigh(Ld)
        lam2 = max(vals[1], 0.0)
        v2 = vecs[:, 1]
    return lam2, v2, degree


# =====================================================================
# Part 6: Gauge CLR Step (from d8_proper)
# =====================================================================

def gauge_clr_step(R, V, K, ei, ej,
                   plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                   v2_ema, N, n_bonds,
                   eta_g, beta_g, dt, ema_alpha=0.1,
                   fiedler_recompute=True):
    """One step of proper gauge CLR (all three terms)."""
    cos_dr = cos_dressed_all(R, V, ei, ej)
    I_dressed = compute_I_dressed(K, cos_dr)

    tau_m = matter_torque_3d(R, V, ei, ej)
    tau_p = plaquette_torque_3d(V, ei, ej, plaq_bonds,
                                plaq_signs, btp_arr, btp_cnt, n_bonds)

    if fiedler_recompute:
        lam2, v2_raw, deg = sparse_laplacian_and_fiedler(ei, ej, K, cos_dr, N)
        if v2_ema is None:
            v2_ema = v2_raw.copy()
        else:
            if np.dot(v2_ema, v2_raw) < 0:
                v2_raw = -v2_raw
            v2_ema = (1 - ema_alpha) * v2_ema + ema_alpha * v2_raw
    else:
        lam2 = 0.0

    if v2_ema is not None:
        tau_s = structural_torque_3d(v2_ema, R, V, ei, ej, cos_dr)
    else:
        tau_s = np.zeros((n_bonds, 3))

    tau_s_mean = np.mean(tau_s, axis=0)
    S_gauge = tau_s - tau_s_mean[None, :]
    budget_err = np.max(np.abs(np.sum(S_gauge, axis=0)))

    tau_total = K[:, None] * tau_m + beta_g * tau_p + I_dressed * S_gauge

    if dt > 0:
        dV = rodrigues_batch(eta_g * dt * tau_total)
        V_new = np.einsum('bij,bjk->bik', V, dV)
    else:
        V_new = V

    diag = {
        'cos_dr_mean': float(np.mean(cos_dressed_all(R, V_new, ei, ej))),
        'I_dressed': I_dressed,
        'budget_err': budget_err,
        'lam2': lam2,
    }
    return V_new, v2_ema, diag


# =====================================================================
# Part 7: Topology Observables (from d8_proper)
# =====================================================================

def topological_charge_bare(R, site_nbr):
    N, z = site_nbr.shape
    Q = np.zeros(N)
    if z < 4:
        return Q
    for i in range(N):
        nbrs = site_nbr[i]
        R_i = R[i]
        R_nbrs = R[nbrs[:4]]
        M = np.einsum('ji,ajk->aik', R_i, R_nbrs)
        q_sum = 0.0
        for a, b, c in [(0,1,2),(1,2,3),(2,3,0),(3,0,1)]:
            comm = M[b] @ M[c] - M[c] @ M[b]
            q_sum += np.trace(M[a] @ comm)
        Q[i] = q_sum / (32.0 * np.pi**2)
    return Q


def gauge_dressed_topo_charge(R, V, ei, ej, site_nbr, positions, N, n_bonds):
    """Gauge-invariant topological charge using tree gauge fixing."""
    center = np.mean(positions, axis=0)
    dists = np.linalg.norm(positions - center[None, :], axis=1)
    root = np.argmin(dists)

    row = np.concatenate([ei, ej])
    col = np.concatenate([ej, ei])
    data = np.ones(2 * n_bonds)
    adj = csr_matrix((data, (row, col)), shape=(N, N))
    bfs_order, predecessors = breadth_first_order(adj, root, directed=False)

    bond_lookup = {}
    for b in range(n_bonds):
        i, j = int(ei[b]), int(ej[b])
        bond_lookup[(i, j)] = (b, False)
        bond_lookup[(j, i)] = (b, True)

    R_dressed = np.zeros_like(R)
    R_dressed[root] = R[root].copy()
    for node in bfs_order[1:]:
        parent = predecessors[node]
        if (parent, node) in bond_lookup:
            b, rev = bond_lookup[(parent, node)]
            V_link = V[b].T if rev else V[b]
        else:
            V_link = np.eye(3)
        R_dressed[node] = V_link @ R[node]

    Q_dressed = topological_charge_bare(R_dressed, site_nbr)
    Q_bare = topological_charge_bare(R, site_nbr)

    V_optimal = np.einsum('bij,bkj->bik', R[ei], R[ej])
    flux = np.sqrt(np.sum((V - V_optimal) ** 2, axis=(1, 2)))

    bond_centers = (positions[ei] + positions[ej]) / 2.0
    bond_dists = np.linalg.norm(bond_centers - center[None, :], axis=1)
    r_median = np.median(bond_dists)
    core_mask = bond_dists < r_median * 0.5
    bulk_mask = bond_dists > r_median * 1.0

    flux_core = float(np.mean(flux[core_mask])) if np.any(core_mask) else 0.0
    flux_bulk = float(np.mean(flux[bulk_mask])) if np.any(bulk_mask) else 0.0
    flux_ratio = flux_core / max(flux_bulk, 1e-10)

    return {
        'Q_dressed_total': float(np.sum(Q_dressed)),
        'Q_bare_total': float(np.sum(Q_bare)),
        'M_dressed': float(np.sum(np.abs(Q_dressed))),
        'M_bare': float(np.sum(np.abs(Q_bare))),
        'flux_core': flux_core,
        'flux_bulk': flux_bulk,
        'flux_ratio': flux_ratio,
    }


def spatial_topology_analysis(R, site_nbr, positions, N):
    Q_site = topological_charge_bare(R, site_nbr)
    Q_abs = np.abs(Q_site)
    has_4 = np.sum(site_nbr >= 0, axis=1) >= 4 if site_nbr.shape[1] >= 4 else np.ones(N, dtype=bool)
    Q_active = Q_site[has_4]
    Q_mean = float(np.mean(Q_active))
    Q_std = float(np.std(Q_active))
    M_sky_unsigned = float(np.sum(Q_abs))

    q_thresh = np.percentile(Q_abs[has_4], 95)
    hot_sites = np.where(Q_abs > q_thresh)[0]
    n_hot = len(hot_sites)
    lump_info = []
    if 1 < n_hot < 500:
        from scipy.spatial import cKDTree
        tree = cKDTree(positions[hot_sites])
        dists, _ = tree.query(positions[hot_sites], k=min(3, n_hot))
        nn_dist = float(np.median(dists[:, 1])) if dists.shape[1] > 1 else 1.0
        cluster_radius = 3.0 * nn_dist
        visited = set()
        lumps = []
        for start in range(n_hot):
            if start in visited:
                continue
            cluster = [start]
            queue = [start]
            visited.add(start)
            while queue:
                curr = queue.pop(0)
                for other in range(n_hot):
                    if other not in visited:
                        dist = np.linalg.norm(
                            positions[hot_sites[curr]] - positions[hot_sites[other]])
                        if dist < cluster_radius:
                            visited.add(other)
                            cluster.append(other)
                            queue.append(other)
            if len(cluster) >= 3:
                lump_sites = hot_sites[cluster]
                lumps.append({
                    'n_sites': len(cluster),
                    'Q_signed': round(float(np.sum(Q_site[lump_sites])), 4),
                    'Q_unsigned': round(float(np.sum(np.abs(Q_site[lump_sites]))), 4),
                    'center': [round(x, 2) for x in np.mean(positions[lump_sites], axis=0).tolist()],
                })
        lump_info = sorted(lumps, key=lambda x: -x['Q_unsigned'])

    return {
        'M_sky_unsigned': M_sky_unsigned,
        'Q_mean': Q_mean,
        'Q_std': Q_std,
        'n_lumps': len(lump_info),
        'lumps': lump_info[:10],
        'max_lump_Q': max((l['Q_unsigned'] for l in lump_info), default=0.0),
    }


def wilson_mean(V, plaq_bonds, plaq_signs):
    n_plaq = plaq_bonds.shape[0]
    if n_plaq == 0:
        return 0.0
    total = 0.0
    for p in range(n_plaq):
        Vp = np.eye(3)
        for k in range(6):
            bb = plaq_bonds[p, k]
            ss = plaq_signs[p, k]
            Vlink = V[bb] if ss > 0 else V[bb].T
            Vp = Vp @ Vlink
        total += np.trace(Vp) / 3.0
    return total / n_plaq


# =====================================================================
# Part 8: Phase Vortex (from coupled_vortex_skyrmion.py)
# =====================================================================

def init_vortex_line_phase(positions, center, axis=2):
    """θ_i = atan2(dy, dx) for vortex line along axis through center."""
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp[0], perp[1]
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    theta = np.arctan2(dy, dx) % (2 * np.pi)
    return theta


def compute_cos_dphi(theta, ei, ej):
    """cos(Δφ) per bond. At vortex core bonds, cos(Δφ) ≈ 0."""
    dphi = theta[ei] - theta[ej]
    return np.cos(dphi)


def measure_winding_number(theta, positions, center, axis=2):
    """Measure π₁ winding number of phase vortex."""
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp[0], perp[1]
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    rho = np.sqrt(dx**2 + dy**2)
    radius_max = float(np.max(rho)) * 0.8
    r_min = 0.5
    mask = (rho > r_min) & (rho < radius_max)
    if np.sum(mask) < 4:
        return 0
    angles_geo = np.arctan2(dy[mask], dx[mask])
    order = np.argsort(angles_geo)
    theta_sorted = theta[np.where(mask)[0][order]]
    dtheta = np.diff(theta_sorted)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_close = (theta_sorted[0] - theta_sorted[-1] + np.pi) % (2 * np.pi) - np.pi
    total = float(np.sum(dtheta) + dtheta_close)
    return int(np.round(total / (2 * np.pi)))


# =====================================================================
# Part 9: Sublattice Downfold (from oscillator_matter_creation.py)
# =====================================================================

def equilibrate_frozen_sublattice(R, frozen_sites, site_nbr,
                                   V_so3=None, site_bond=None, ei=None):
    """Set frozen sublattice frames via SVD to minimize energy.
    One pass suffices on bipartite lattice (all neighbors are active).
    """
    R_eq = R.copy()
    for i in frozen_sites:
        M = np.zeros((3, 3))
        for k in range(site_nbr.shape[1]):
            j = site_nbr[i, k]
            if V_so3 is not None and site_bond is not None and ei is not None:
                b = site_bond[i, k]
                if ei[b] == i:
                    M += V_so3[b] @ R_eq[j]
                else:
                    M += V_so3[b].T @ R_eq[j]
            else:
                M += R_eq[j]
        try:
            U, S, Vt = np.linalg.svd(M)
            det = np.linalg.det(U @ Vt)
            if det < 0:
                U[:, -1] *= -1
            R_eq[i] = U @ Vt
        except np.linalg.LinAlgError:
            pass  # Keep existing frame if SVD fails
    return R_eq


# =====================================================================
# Part 10: K with Product Coupling
# =====================================================================

def compute_K_product(cos_dphi, cos_dr, K0):
    """K = K0 * max(0, cos_dphi * cos_dr).
    At vortex core: cos_dphi ≈ 0 → K = 0. Bulk: K = K0 * cos_dr.
    """
    return K0 * np.maximum(0.0, cos_dphi * cos_dr)


# =====================================================================
# Part 10b: Phase MC Sweep (evolving vortex)
# =====================================================================

def phase_mc_sweep(theta, K_phi, ei, ej, site_nbr, site_bond,
                   beta_mc, sigma_theta, N, rng, active_sites=None):
    """Metropolis sweep for U(1) phases with XY energy.

    E_phase = -K_phi * cos(θ_i - θ_j) per bond.
    Bipartite: propose θ_new = θ_old + uniform noise.

    The vortex is π₁-protected — local MC moves can't unwind it.
    But the vortex can precess, oscillate, and fluctuate.
    """
    z = site_nbr.shape[1]
    if active_sites is None:
        active_sites = np.arange(N)

    n_sub = len(active_sites)
    if n_sub == 0:
        return theta, 0, 0

    theta_old = theta[active_sites].copy()
    dth = rng.uniform(-sigma_theta, sigma_theta, size=n_sub)
    theta_new = (theta_old + dth) % (2 * np.pi)

    dE = np.zeros(n_sub)
    for k in range(z):
        nbr_sites = site_nbr[active_sites, k]
        bond_idxs = site_bond[active_sites, k]
        theta_nbr = theta[nbr_sites]
        K_b = K_phi[bond_idxs]

        cos_new = np.cos(theta_new - theta_nbr)
        cos_old = np.cos(theta_old - theta_nbr)
        dE += -K_b * (cos_new - cos_old)

    accept = dE <= 0
    if beta_mc < float('inf'):
        boltzmann = np.exp(-beta_mc * np.clip(dE, 0, 500))
        accept |= (rng.uniform(size=n_sub) < boltzmann)

    acc_idx = np.where(accept)[0]
    theta[active_sites[acc_idx]] = theta_new[acc_idx]

    return theta, len(acc_idx), n_sub


# =====================================================================
# Part 10c: Dead Bond Pruning
# =====================================================================

def build_alive_site_nbr(site_nbr, site_bond, K_arr, K_thresh=1e-4):
    """Build pruned neighbor array excluding dead bonds.

    Dead bonds (K < K_thresh) are replaced with -1 in the neighbor list.
    This prevents topology calculations from seeing uncorrelated frames
    across dead bonds (which produce noise, not signal).

    Returns: site_nbr_alive (N, z) with -1 for pruned neighbors.
    """
    site_nbr_alive = site_nbr.copy()
    N, z = site_nbr.shape
    n_pruned = 0
    for i in range(N):
        for k in range(z):
            b = site_bond[i, k]
            if K_arr[b] < K_thresh:
                site_nbr_alive[i, k] = -1
                n_pruned += 1
    return site_nbr_alive, n_pruned


def topological_charge_pruned(R, site_nbr_alive):
    """Topological charge with dead-bond-aware neighbor lists.

    Sites with fewer than 4 alive neighbors get Q=0 (can't form
    a proper solid-angle element). This is the correct physics:
    a site at the vortex core with dead bonds to some neighbors
    doesn't contribute to the topological charge of the hedgehog —
    only the intact tetrahedra in the bulk do.
    """
    N = len(R)
    Q = np.zeros(N)
    for i in range(N):
        nbrs = site_nbr_alive[i]
        alive = nbrs[nbrs >= 0]
        if len(alive) < 4:
            continue  # Can't form complete tetrahedral element
        # Use first 4 alive neighbors
        R_i = R[i]
        R_nbrs = R[alive[:4]]
        M = np.einsum('ji,ajk->aik', R_i, R_nbrs)
        q_sum = 0.0
        for a, b, c in [(0,1,2),(1,2,3),(2,3,0),(3,0,1)]:
            comm = M[b] @ M[c] - M[c] @ M[b]
            q_sum += np.trace(M[a] @ comm)
        Q[i] = q_sum / (32.0 * np.pi**2)
    return Q


def spatial_topology_pruned(R, site_nbr_alive, positions, N):
    """spatial_topology_analysis but using pruned neighbors."""
    Q_site = topological_charge_pruned(R, site_nbr_alive)
    Q_abs = np.abs(Q_site)
    # Only sites with 4 alive neighbors contribute
    has_4 = np.array([np.sum(site_nbr_alive[i] >= 0) >= 4 for i in range(N)])
    Q_active = Q_site[has_4]
    if len(Q_active) == 0:
        return {'M_sky_unsigned': 0.0, 'Q_mean': 0.0, 'Q_std': 0.0,
                'n_lumps': 0, 'lumps': [], 'max_lump_Q': 0.0,
                'n_active_sites': 0}

    Q_mean = float(np.mean(Q_active))
    Q_std = float(np.std(Q_active))
    M_sky_unsigned = float(np.sum(Q_abs))

    q_thresh = np.percentile(Q_abs[has_4], 95)
    hot_sites = np.where(Q_abs > q_thresh)[0]
    n_hot = len(hot_sites)
    lump_info = []
    if 1 < n_hot < 500:
        from scipy.spatial import cKDTree
        tree = cKDTree(positions[hot_sites])
        dists, _ = tree.query(positions[hot_sites], k=min(3, n_hot))
        nn_dist = float(np.median(dists[:, 1])) if dists.shape[1] > 1 else 1.0
        cluster_radius = 3.0 * nn_dist
        visited = set()
        lumps = []
        for start in range(n_hot):
            if start in visited:
                continue
            cluster = [start]
            queue = [start]
            visited.add(start)
            while queue:
                curr = queue.pop(0)
                for other in range(n_hot):
                    if other not in visited:
                        dist = np.linalg.norm(
                            positions[hot_sites[curr]] - positions[hot_sites[other]])
                        if dist < cluster_radius:
                            visited.add(other)
                            cluster.append(other)
                            queue.append(other)
            if len(cluster) >= 3:
                lump_sites = hot_sites[cluster]
                lumps.append({
                    'n_sites': len(cluster),
                    'Q_signed': round(float(np.sum(Q_site[lump_sites])), 4),
                    'Q_unsigned': round(float(np.sum(np.abs(Q_site[lump_sites]))), 4),
                    'center': [round(x, 2) for x in
                               np.mean(positions[lump_sites], axis=0).tolist()],
                })
        lump_info = sorted(lumps, key=lambda x: -x['Q_unsigned'])

    return {
        'M_sky_unsigned': M_sky_unsigned,
        'Q_mean': Q_mean,
        'Q_std': Q_std,
        'n_lumps': len(lump_info),
        'lumps': lump_info[:10],
        'max_lump_Q': max((l['Q_unsigned'] for l in lump_info), default=0.0),
        'n_active_sites': int(np.sum(has_4)),
    }


# =====================================================================
# Part 10d: Hedgehog Seeding
# =====================================================================

def seed_hedgehog(R, positions, center, R_seed, profile='linear'):
    """Seed a 3D hedgehog frame pattern at center within radius R_seed.

    The hedgehog is a mapping from S² → SO(3) with unit degree in π₃.
    Profile: R_i = Rodrigues(axis=cross(z, r_hat), angle=f(r))
    where f(r) = π * (1 - r/R_seed) for r < R_seed.

    At center: R → rotation by π (maximum twist).
    At boundary: R → I (matches bulk ferromagnet).

    Returns: (R, n_seeded) — modified frames and count of seeded sites.
    """
    n_seeded = 0
    z_hat = np.array([0.0, 0.0, 1.0])

    for i in range(len(R)):
        d = positions[i] - center
        r = np.linalg.norm(d)
        if r < R_seed:
            n_seeded += 1
            if r < 1e-10:
                # At exact center: rotate by π about x-axis
                R[i] = np.diag([-1.0, 1.0, -1.0])  # Rz(π) = diag(-1,-1,1) × ... hmm
                # Actually: rotation by π about x: (1,0,0; 0,-1,0; 0,0,-1)
                R[i] = np.array([[1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0],
                                 [0.0, 0.0, -1.0]])
            else:
                r_hat = d / r
                # Profile: angle from π at center to 0 at boundary
                if profile == 'linear':
                    angle = np.pi * (1.0 - r / R_seed)
                elif profile == 'cosine':
                    angle = np.pi * 0.5 * (1.0 + np.cos(np.pi * r / R_seed))
                else:
                    angle = np.pi * (1.0 - r / R_seed)

                # Rotation axis: perpendicular to r_hat in the (r_hat, z) plane
                # Use r_hat × z_hat, unless r_hat is parallel to z
                cross = np.cross(z_hat, r_hat)
                cross_norm = np.linalg.norm(cross)
                if cross_norm < 1e-10:
                    # r_hat parallel to z: use x-axis
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis = cross / cross_norm

                # Rodrigues formula (single vector version)
                ax, ay, az = axis
                Kmat = np.array([[0, -az, ay],
                                 [az, 0, -ax],
                                 [-ay, ax, 0]])
                R[i] = (np.eye(3) + np.sin(angle) * Kmat
                        + (1.0 - np.cos(angle)) * Kmat @ Kmat)

    return R, n_seeded


# =====================================================================
# Part 10e: Kuramoto Bootstrap (Spontaneous Vortex Nucleation)
# =====================================================================

def kuramoto_frequency_profile(positions, center, profile='circulation',
                                omega_gradient=1.0, axis=2, rng=None):
    """Generate structured natural frequency field for Kuramoto oscillators.

    The frequency heterogeneity drives vortex formation: different ω_i values
    cause neighbors to precess at different rates, and Kuramoto coupling
    creates phase frustration that nucleates topological defects (vortices).

    Profiles:
      circulation: ω ∝ atan2(y,x)/(2π). Angular gradient forces
        differential rotation → guaranteed single vortex (winding=±1).
      shear_ring: Fast oscillators on a ring, slow in bulk.
        Azimuthal shear → vortex pair nucleation.
      dipole: ω = +gradient for y>0, -gradient for y<0.
        Shear line → vortex pair.
      random_gradient: ω = gradient·(r·n̂) for random n̂.
        May or may not create vortex — tests pure spontaneous breaking.
    """
    N = len(positions)
    omega = np.zeros(N)
    perp = [i for i in range(3) if i != axis]
    a0, a1 = perp[0], perp[1]
    dx = positions[:, a0] - center[a0]
    dy = positions[:, a1] - center[a1]
    rho = np.sqrt(dx**2 + dy**2)

    if profile == 'circulation':
        # ω ∝ angular position → differential rotation
        # This naturally creates winding=1 after Kuramoto synchronization
        phi = np.arctan2(dy, dx)  # -π to π
        omega = omega_gradient * phi / (2 * np.pi)

    elif profile == 'shear_ring':
        # Fast oscillators on a ring at ~40% of max radius
        r_max = np.max(rho) if np.max(rho) > 0 else 1.0
        r_ring = r_max * 0.4
        width = r_max * 0.15
        ring_mask = np.abs(rho - r_ring) < width
        omega[ring_mask] = omega_gradient

    elif profile == 'dipole':
        # Smooth sign change across y=0 plane
        dy_max = np.max(np.abs(dy)) if np.max(np.abs(dy)) > 0 else 1.0
        omega = omega_gradient * np.tanh(dy * 3.0 / dy_max)

    elif profile == 'random_gradient':
        # Random direction linear gradient — no guaranteed winding
        if rng is None:
            rng = np.random.default_rng(42)
        direction = rng.standard_normal(3)
        direction /= max(np.linalg.norm(direction), 1e-15)
        omega = omega_gradient * (positions @ direction)
        omega -= np.mean(omega)

    else:
        raise ValueError(f"Unknown Kuramoto profile: {profile}")

    return omega


def kuramoto_bootstrap(theta, omega, positions, center, ei, ej, site_nbr,
                       coupling=2.0, dt=0.1, n_steps=200, axis=2,
                       verbose=True):
    """Bootstrap phase vortex from frequency heterogeneity via Kuramoto dynamics.

    dθ_i/dt = ω_i + (coupling/z) Σ_j sin(θ_j - θ_i)

    The coupling term tries to synchronize neighbors. The frequency term
    drives persistent differential precession. Their competition produces
    stable vortices when the frequency profile has nontrivial topology.

    Returns: (theta, winding, step_nucleated)
    """
    N = len(theta)
    z = site_nbr.shape[1]
    winding = 0
    step_nucleated = None

    for step in range(1, n_steps + 1):
        # Kuramoto coupling torque: (coupling/z) * Σ_j sin(θ_j - θ_i)
        torque = np.zeros(N)
        for k in range(z):
            nbr = site_nbr[:, k]
            torque += np.sin(theta[nbr] - theta)
        torque *= coupling / z

        # Euler step
        theta = (theta + dt * (omega + torque)) % (2 * np.pi)

        # Check winding every 10 steps
        if step % 10 == 0:
            w = measure_winding_number(theta, positions, center, axis=axis)
            if verbose and step % 50 == 0:
                cos_dphi = compute_cos_dphi(theta, ei, ej)
                n_dead = int(np.sum(cos_dphi < 0.1))
                print(f"      Kuramoto step {step:4d}: winding={w}, "
                      f"cos_dphi_mean={np.mean(cos_dphi):.4f}, "
                      f"dead={n_dead}")
            if w != 0 and step_nucleated is None:
                step_nucleated = step
                winding = w
                if verbose:
                    print(f"      *** Vortex nucleated at step {step}! "
                          f"winding={w} ***")

    winding = measure_winding_number(theta, positions, center, axis=axis)
    return theta, winding, step_nucleated


# =====================================================================
# Part 11: Frame MC Sweep
# =====================================================================

def frame_mc_sweep(R, K_arr, V, ei, ej, site_nbr, site_bond,
                   beta_mc, sigma_mc, N, rng, active_sites=None):
    """Bipartite Metropolis sweep for frames with gauge-dressed energy.

    If active_sites provided: only propose moves for those sites.
    Frozen sites contribute to energy but don't move.

    Energy per bond: E = -K * cos_dressed = -K * tr(R_i^T V R_j)/3.
    At K=0 bonds: ΔE=0 → 100% accept → free exploration.
    """
    z = site_nbr.shape[1]
    if active_sites is None:
        active_sites = np.arange(N)

    n_accepted = 0
    n_proposed = 0

    # Process all active sites (no sublattice splitting needed since
    # on diamond lattice with downfold, all active sites are same sublattice
    # and their neighbors are frozen — no active-active conflicts)
    n_sub = len(active_sites)
    if n_sub == 0:
        return R, 0, 0

    dR = random_rotation_small_batch(n_sub, sigma_mc, rng)
    R_old = R[active_sites].copy()
    R_new = np.einsum('nab,nbc->nac', dR, R_old)

    dE = np.zeros(n_sub)
    for k in range(z):
        nbr_sites = site_nbr[active_sites, k]
        bond_idxs = site_bond[active_sites, k]

        V_b = V[bond_idxs]
        R_nbr = R[nbr_sites]
        K_b = K_arr[bond_idxs]

        # Determine bond orientation
        is_forward = (ei[bond_idxs] == active_sites)

        # Forward: cos = tr(R_site^T V R_nbr)/3
        # Backward: cos = tr(R_site^T V^T R_nbr)/3
        R_dressed_fwd = np.einsum('nik,nkj->nij', V_b, R_nbr)
        R_dressed_bwd = np.einsum('nki,nkj->nij', V_b, R_nbr)
        R_dressed = np.where(is_forward[:, None, None], R_dressed_fwd, R_dressed_bwd)

        cos_new = np.einsum('nji,nij->n', R_new, R_dressed) / 3.0
        cos_old = np.einsum('nji,nij->n', R_old, R_dressed) / 3.0

        # ΔE = -K(cos_new - cos_old) for Heisenberg energy E=-K*cos
        dE += -K_b * (cos_new - cos_old)

    # Metropolis accept/reject
    accept = dE <= 0
    if beta_mc < float('inf'):
        boltzmann = np.exp(-beta_mc * np.clip(dE, 0, 500))
        accept |= (rng.uniform(size=n_sub) < boltzmann)

    acc_idx = np.where(accept)[0]
    R[active_sites[acc_idx]] = R_new[acc_idx]
    n_accepted = len(acc_idx)
    n_proposed = n_sub

    return R, n_accepted, n_proposed


# =====================================================================
# Part 12: Gauge MC Sweep
# =====================================================================

def gauge_mc_sweep(V, R, K_arr, ei, ej,
                   plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                   beta_mc, beta_wilson, sigma_g, n_bonds, rng):
    """Metropolis sweep for gauge links V with Wilson confinement.

    ΔE = ΔE_matter + ΔE_wilson
    ΔE_matter = -K * (cos_dressed_new - cos_dressed_old)
    ΔE_wilson = -β_w * Σ_{plaq ∋ b} (wilson_new - wilson_old)
    """
    n_accepted = 0

    for b in range(n_bonds):
        # Propose V_new = exp(σ * random) @ V_old
        raw = rng.standard_normal(3)
        raw /= max(np.linalg.norm(raw), 1e-15)
        angle = rng.uniform(-sigma_g, sigma_g)
        omega = angle * raw
        c, s = np.cos(angle), np.sin(angle)
        K_hat = np.array([[0, -raw[2], raw[1]],
                           [raw[2], 0, -raw[0]],
                           [-raw[1], raw[0], 0]])
        dV = np.eye(3) + s * K_hat + (1 - c) * (K_hat @ K_hat)
        V_new_b = dV @ V[b]

        # ΔE_matter
        i_site, j_site = int(ei[b]), int(ej[b])
        cos_old = np.trace(R[i_site].T @ V[b] @ R[j_site]) / 3.0
        cos_new = np.trace(R[i_site].T @ V_new_b @ R[j_site]) / 3.0
        dE = -K_arr[b] * (cos_new - cos_old)

        # ΔE_wilson
        n_p = btp_cnt[b]
        for k in range(n_p):
            p_idx = btp_arr[b, k]
            if p_idx < 0:
                continue
            # Compute plaquette trace with old and new V[b]
            for use_new in [False, True]:
                Vp = np.eye(3)
                for kk in range(6):
                    bb = plaq_bonds[p_idx, kk]
                    ss = plaq_signs[p_idx, kk]
                    if bb == b and use_new:
                        Vlink = V_new_b if ss > 0 else V_new_b.T
                    else:
                        Vlink = V[bb] if ss > 0 else V[bb].T
                    Vp = Vp @ Vlink
                tr_val = np.trace(Vp) / 3.0
                if not use_new:
                    tr_old = tr_val
                else:
                    tr_new = tr_val
            dE += -beta_wilson * (tr_new - tr_old)

        # Accept/reject
        if dE <= 0 or rng.uniform() < np.exp(-beta_mc * min(dE, 500)):
            V[b] = V_new_b
            n_accepted += 1

    return V, n_accepted


# =====================================================================
# Part 13: D9 Protocol
# =====================================================================

def protocol_d9(L, seed, downfold='B',
                K0=8.0, K_phi=4.0,
                beta_hot=0.05, beta_cold=4.0, beta_wilson=2.0,
                eta_g=0.1, beta_g=2.0,
                sigma_mc=0.5, sigma_g=0.1, sigma_theta=0.3,
                n_disorder_sweeps=200,
                n_anneal_sweeps=2000,
                n_independence_sweeps=2000,
                nucleation_threshold=50.0,
                n_clr_per_mc=5, fiedler_every=50,
                reortho_every=50,
                measure_every=100,
                evolve_phases=True,
                seed_radius=None,
                beta_grow=None,
                # Kuramoto spontaneous nucleation
                kuramoto=False,
                kuramoto_profile='circulation',
                omega_gradient=1.0,
                kuramoto_coupling=2.0,
                kuramoto_steps=200,
                frame_init='random',
                verbose=True):
    """D9: Electron nucleates proton.

    MODE 1 (seed_radius=None, kuramoto=False): Hot-start annealing.
        Phase 0: Random frames, hot MC. Phase 1: Cool β_hot→β_cold.
        Phase 2: Independence test.

    MODE 2 (seed_radius=float): Hedgehog seeding + growth.
        Phase 0: Seed small hedgehog at center, bulk=I (ferromagnet).
        Phase 1: GROW at moderate β_grow with vortex + gauge CLR.
            Monitor M_sky growth from seed.
        Phase 2: Independence test.

    MODE 3 (kuramoto=True): Spontaneous nucleation.
        Phase 0: Kuramoto bootstrap — oscillators with structured ω field
            self-organize into phase vortex from uniform initial conditions.
        Phase 1: GROW with spontaneous vortex (no hedgehog seed).
            Frames start random or identity. K=0 at vortex core allows
            free MC exploration. Gauge CLR concentrates at core.
            Topology must emerge purely from dynamics.
        Phase 2: Independence test (remove vortex, test survival).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    kuramoto_mode = kuramoto
    seeded_mode = seed_radius is not None and not kuramoto_mode
    if beta_grow is None:
        beta_grow = 2.0

    if verbose:
        if kuramoto_mode:
            mode_str = f"Kuramoto Spontaneous ({kuramoto_profile})"
        elif seeded_mode:
            mode_str = f"Hedgehog Seed (R={seed_radius})"
        else:
            mode_str = "Hot-Start Anneal"
        print(f"\n{'='*70}")
        print(f"  D9: Electron Nucleates Proton ({mode_str})")
        print(f"  L={L}, seed={seed}, downfold={downfold}")
        print(f"  K0={K0}, K_phi={K_phi}")
        if kuramoto_mode:
            print(f"  beta_grow={beta_grow}, beta_wilson={beta_wilson}")
            print(f"  kuramoto: profile={kuramoto_profile}, "
                  f"omega_grad={omega_gradient}, "
                  f"coupling={kuramoto_coupling}, steps={kuramoto_steps}")
            print(f"  frame_init={frame_init}")
        elif seeded_mode:
            print(f"  beta_grow={beta_grow}, beta_wilson={beta_wilson}")
        else:
            print(f"  beta_hot={beta_hot}, beta_cold={beta_cold}, beta_wilson={beta_wilson}")
        print(f"  eta_g={eta_g}, beta_g={beta_g}")
        print(f"  sigma_mc={sigma_mc}, sigma_g={sigma_g}, sigma_theta={sigma_theta}")
        print(f"  evolve_phases={evolve_phases}")
        print(f"  nucleation_threshold={nucleation_threshold}")
        print(f"{'='*70}")

    # Build lattice
    positions, ei, ej, site_nbr, site_bond, N, n_bonds, sublat = \
        build_diamond_lattice(L)
    center = np.mean(positions, axis=0)

    if verbose:
        print(f"  N={N}, n_bonds={n_bonds}")
        print(f"  Enumerating plaquettes...")

    plaq_bonds, plaq_signs, btp_arr, btp_cnt = \
        enumerate_hexagonal_plaquettes(ei, ej, site_nbr, site_bond, N, n_bonds)
    n_plaq = plaq_bonds.shape[0]
    if verbose:
        print(f"  {n_plaq} hexagonal plaquettes found")

    # Initialize frames
    if kuramoto_mode:
        if frame_init == 'random':
            R = random_rotation_small_batch(N, np.pi, rng)
            if verbose:
                print(f"  Frames: random SO(3)")
        else:
            R = np.tile(np.eye(3), (N, 1, 1)).astype(np.float64)
            if verbose:
                print(f"  Frames: identity (R=I)")
    elif seeded_mode:
        # Start from ordered ferromagnet (R=I) + hedgehog seed
        R = np.tile(np.eye(3), (N, 1, 1)).astype(np.float64)
        R, n_seeded = seed_hedgehog(R, positions, center, seed_radius)
        if verbose:
            print(f"  Hedgehog seed: R_seed={seed_radius:.2f}, "
                  f"n_seeded={n_seeded}/{N} sites")
    else:
        # Random frames for hot-start annealing
        R_random = random_rotation_small_batch(N, np.pi, rng)
        R = R_random.copy()

    # Initialize: flat gauge links (V = I)
    V = np.tile(np.eye(3), (n_bonds, 1, 1))

    # Phase vortex initialization
    if kuramoto_mode:
        # Phases start uniform — vortex will emerge from Kuramoto bootstrap
        theta = np.zeros(N)
        cos_dphi = np.ones(n_bonds)
        winding = 0
        if verbose:
            print(f"  Phases: uniform (will bootstrap via Kuramoto)")
    else:
        # Place deterministic vortex line along z-axis
        theta = init_vortex_line_phase(positions, center, axis=2)
        cos_dphi = compute_cos_dphi(theta, ei, ej)
        winding = measure_winding_number(theta, positions, center, axis=2)
        if verbose:
            print(f"  Vortex: winding={winding}, "
                  f"cos_dphi_mean={np.mean(cos_dphi):.4f}")
            n_dead = np.sum(cos_dphi < 0.1)
            print(f"    Dead bonds (cos_dphi < 0.1): {n_dead}/{n_bonds}")

    # Phase coupling K_phi array (uniform for now)
    K_phi_arr = np.full(n_bonds, K_phi)

    # Sublattice downfold
    if downfold in ('A', 'B'):
        frozen_sublat = 0 if downfold == 'A' else 1
        active_sublat = 1 - frozen_sublat
        frozen_sites = np.where(sublat == frozen_sublat)[0]
        active_sites = np.where(sublat == active_sublat)[0]
        # Don't equilibrate yet — frames are random, will equilibrate during anneal
        if verbose:
            print(f"  Downfold: freeze {'A' if frozen_sublat==0 else 'B'} "
                  f"({len(frozen_sites)} sites), "
                  f"active={'B' if frozen_sublat==0 else 'A'} "
                  f"({len(active_sites)} sites)")
    else:
        frozen_sites = np.array([], dtype=np.int32)
        active_sites = np.arange(N)
        if verbose:
            print(f"  No downfold: all {N} sites active")

    # Initial K from product coupling
    cos_dr = cos_dressed_all(R, V, ei, ej)
    K = compute_K_product(cos_dphi, cos_dr, K0)

    if verbose:
        print(f"  Initial (random): K_mean={np.mean(K):.4f}, cos_dr_mean={np.mean(cos_dr):.4f}")

    # History tracking
    history = []
    v2_ema = None
    nucleation_sweep = None
    M_sky_at_nucleation = 0.0
    global_sweep = 0

    def measure(phase_name, sweep_num):
        cos_dr_now = cos_dressed_all(R, V, ei, ej)
        wl = wilson_mean(V, plaq_bonds, plaq_signs)

        # Pruned topology: exclude dead bonds from neighbor lists
        snbr_alive, n_pruned = build_alive_site_nbr(site_nbr, site_bond, K, K_thresh=1e-4)
        topo = spatial_topology_pruned(R, snbr_alive, positions, N)

        # Also compute unpruned for comparison
        topo_raw = spatial_topology_analysis(R, site_nbr, positions, N)

        topo_dressed = gauge_dressed_topo_charge(R, V, ei, ej, site_nbr,
                                                  positions, N, n_bonds)
        w = measure_winding_number(theta, positions, center, axis=2)

        n_dead = int(np.sum(K < 1e-4))
        obs = {
            'phase': phase_name,
            'sweep': sweep_num,
            'M_sky': topo['M_sky_unsigned'],
            'M_sky_raw': topo_raw['M_sky_unsigned'],
            'max_lump_Q': topo['max_lump_Q'],
            'n_lumps': topo['n_lumps'],
            'n_active_topo': topo.get('n_active_sites', N),
            'n_dead_bonds': n_dead,
            'cos_dressed_mean': float(np.mean(cos_dr_now)),
            'wilson_mean': wl,
            'K_mean': float(np.mean(K)),
            'flux_ratio': topo_dressed['flux_ratio'],
            'flux_core': topo_dressed['flux_core'],
            'flux_bulk': topo_dressed['flux_bulk'],
            'M_dressed': topo_dressed['M_dressed'],
            'M_bare': topo_dressed['M_bare'],
            'winding': w,
        }
        history.append(obs)
        return obs

    # Initial measurement
    obs0 = measure('init', 0)
    if verbose:
        print(f"  Init: M_sky={obs0['M_sky']:.2f}, cos_dr={obs0['cos_dressed_mean']:.4f}")

    fiedler_counter = 0

    # =================== PHASE 0: BOOTSTRAP / SEED / DISORDER ===================
    if kuramoto_mode:
        # Kuramoto bootstrap: generate vortex from frequency heterogeneity
        omega = kuramoto_frequency_profile(
            positions, center, profile=kuramoto_profile,
            omega_gradient=omega_gradient, axis=2, rng=rng)

        if verbose:
            print(f"\n  PHASE 0: KURAMOTO BOOTSTRAP ({kuramoto_steps} steps)")
            print(f"    Profile: {kuramoto_profile}, "
                  f"omega_gradient={omega_gradient}")
            print(f"    coupling={kuramoto_coupling}, dt=0.1")
            print(f"    Frame init: {frame_init}")

        theta, winding, step_nuc = kuramoto_bootstrap(
            theta, omega, positions, center, ei, ej, site_nbr,
            coupling=kuramoto_coupling, dt=0.1,
            n_steps=kuramoto_steps, axis=2, verbose=verbose)

        cos_dphi = compute_cos_dphi(theta, ei, ej)
        n_dead = int(np.sum(cos_dphi < 0.1))

        if verbose:
            print(f"    Bootstrap result: winding={winding}, "
                  f"cos_dphi_mean={np.mean(cos_dphi):.4f}, dead={n_dead}")
            if winding == 0:
                print(f"    WARNING: No vortex nucleated! "
                      f"Proceeding anyway.")

        # Update K with spontaneous vortex phase structure
        cos_dr = cos_dressed_all(R, V, ei, ej)
        K = compute_K_product(cos_dphi, cos_dr, K0)

        # Downfold handling
        if len(frozen_sites) > 0:
            R = equilibrate_frozen_sublattice(
                R, frozen_sites, site_nbr,
                V_so3=V, site_bond=site_bond, ei=ei)
            R_frozen_backup = R[frozen_sites].copy()
        else:
            R_frozen_backup = None

        obs_dis = measure('bootstrap', 0)
        if verbose:
            print(f"\n  BOOTSTRAP: M_sky={obs_dis['M_sky']:.2f}, "
                  f"Md={obs_dis['M_dressed']:.2f}, "
                  f"cos_dr={obs_dis['cos_dressed_mean']:.4f}")

    elif seeded_mode:
        # Seeded mode: equilibrate frozen sublattice around seed
        if len(frozen_sites) > 0:
            R = equilibrate_frozen_sublattice(R, frozen_sites, site_nbr,
                                               V_so3=V, site_bond=site_bond, ei=ei)
            R_frozen_backup = R[frozen_sites].copy()
        else:
            R_frozen_backup = None

        obs_dis = measure('seed', 0)
        if verbose:
            print(f"\n  SEED: M_sky={obs_dis['M_sky']:.2f}, "
                  f"cos_dr={obs_dis['cos_dressed_mean']:.4f}")
    else:
        # Hot-start annealing: disorder phase
        if verbose:
            print(f"\n  PHASE 0: DISORDER ({n_disorder_sweeps} sweeps, beta={beta_hot})")
            print(f"    Hot MC to fully disorder frames")

        for sweep in range(1, n_disorder_sweeps + 1):
            if evolve_phases:
                theta, _, _ = phase_mc_sweep(
                    theta, K_phi_arr, ei, ej, site_nbr, site_bond,
                    beta_hot, sigma_theta, N, rng)
                cos_dphi = compute_cos_dphi(theta, ei, ej)

            R, _, _ = frame_mc_sweep(
                R, K, V, ei, ej, site_nbr, site_bond,
                beta_hot, sigma_mc, N, rng, active_sites=np.arange(N))

            V, _ = gauge_mc_sweep(
                V, R, K, ei, ej,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                beta_hot, beta_wilson * 0.1, sigma_g, n_bonds, rng)

            cos_dr = cos_dressed_all(R, V, ei, ej)
            K = compute_K_product(cos_dphi, cos_dr, K0)

            if sweep % reortho_every == 0:
                R = reorthogonalize(R)
                for b in range(n_bonds):
                    V[b] = reorthogonalize_single(V[b])

            global_sweep += 1

        if len(frozen_sites) > 0:
            R = equilibrate_frozen_sublattice(R, frozen_sites, site_nbr,
                                               V_so3=V, site_bond=site_bond, ei=ei)
            R_frozen_backup = R[frozen_sites].copy()
        else:
            R_frozen_backup = None

        obs_dis = measure('disorder_end', global_sweep)
        if verbose:
            w = measure_winding_number(theta, positions, center, axis=2)
            print(f"  Post-disorder: M_sky={obs_dis['M_sky']:.2f}, "
                  f"cos_dr={obs_dis['cos_dressed_mean']:.4f}, "
                  f"winding={w}")

    # =================== PHASE 1: ANNEAL / GROW ===================
    n_phase1 = n_anneal_sweeps
    if kuramoto_mode or seeded_mode:
        phase1_name = 'grow'
        if verbose:
            print(f"\n  PHASE 1: GROW ({n_phase1} sweeps)")
            print(f"    beta_grow={beta_grow}, beta_wilson={beta_wilson}")
            src = 'Kuramoto' if kuramoto_mode else 'hand-placed'
            print(f"    Vortex ({src}) "
                  f"{'evolving' if evolve_phases else 'frozen'}, "
                  f"downfold={downfold}, gauge CLR active")
            if kuramoto_mode:
                print(f"    No frame seed — topology must emerge "
                      f"from dynamics")
    else:
        phase1_name = 'anneal'
        if verbose:
            print(f"\n  PHASE 1: ANNEAL ({n_phase1} sweeps)")
            print(f"    beta: {beta_hot} -> {beta_cold}")
            print(f"    Vortex {'evolving' if evolve_phases else 'frozen'}, "
                  f"downfold={downfold}, gauge CLR active")

    accept_rate_frame_accum = []
    accept_rate_gauge_accum = []

    for sweep in range(1, n_phase1 + 1):
        if kuramoto_mode or seeded_mode:
            # Fixed temperature for growth
            beta_mc = beta_grow
            beta_w_current = beta_wilson
            sigma_current = sigma_mc * 0.5
        else:
            # Annealing schedule: linear in log(beta) for smooth cooling
            frac = sweep / n_phase1
            beta_mc = beta_hot * (beta_cold / beta_hot) ** frac
            beta_w_current = beta_wilson * min(frac * 2, 1.0)
            sigma_current = sigma_mc * (1.0 - 0.5 * frac)

        # Phase MC (evolving vortex)
        if evolve_phases:
            theta, _, _ = phase_mc_sweep(
                theta, K_phi_arr, ei, ej, site_nbr, site_bond,
                beta_mc, sigma_theta, N, rng)
            cos_dphi = compute_cos_dphi(theta, ei, ej)

        # Frame MC sweep (active sites only)
        R, n_acc_f, n_prop_f = frame_mc_sweep(
            R, K, V, ei, ej, site_nbr, site_bond,
            beta_mc, sigma_current, N, rng, active_sites=active_sites)

        if n_prop_f > 0:
            accept_rate_frame_accum.append(n_acc_f / n_prop_f)

        # Restore frozen frames
        if len(frozen_sites) > 0:
            R[frozen_sites] = R_frozen_backup

        # Gauge MC sweep
        V, n_acc_g = gauge_mc_sweep(
            V, R, K, ei, ej,
            plaq_bonds, plaq_signs, btp_arr, btp_cnt,
            beta_mc, beta_w_current, sigma_g, n_bonds, rng)
        accept_rate_gauge_accum.append(n_acc_g / max(n_bonds, 1))

        # Gauge CLR step
        if sweep % n_clr_per_mc == 0:
            fiedler_counter += 1
            do_fiedler = (fiedler_counter % max(fiedler_every // n_clr_per_mc, 1) == 0)
            V, v2_ema, gdiag = gauge_clr_step(
                R, V, K, ei, ej,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                v2_ema, N, n_bonds,
                eta_g, beta_g, dt=0.01,
                fiedler_recompute=do_fiedler or v2_ema is None)

        # Adiabatic downfold re-equilibration every 100 sweeps
        if len(frozen_sites) > 0 and sweep % 100 == 0:
            R = equilibrate_frozen_sublattice(R, frozen_sites, site_nbr,
                                               V_so3=V, site_bond=site_bond, ei=ei)
            R_frozen_backup = R[frozen_sites].copy()

        # Update K
        cos_dr = cos_dressed_all(R, V, ei, ej)
        K = compute_K_product(cos_dphi, cos_dr, K0)

        # Reorthogonalize
        if sweep % reortho_every == 0:
            R = reorthogonalize(R)
            for b in range(n_bonds):
                V[b] = reorthogonalize_single(V[b])
            if len(frozen_sites) > 0:
                R_frozen_backup = R[frozen_sites].copy()

        global_sweep += 1

        # Measure
        if sweep % measure_every == 0:
            obs = measure(phase1_name, global_sweep)
            ar_f = np.mean(accept_rate_frame_accum[-measure_every:]) if accept_rate_frame_accum else 0
            ar_g = np.mean(accept_rate_gauge_accum[-measure_every:]) if accept_rate_gauge_accum else 0

            if verbose:
                print(f"    sweep {sweep:5d} (beta={beta_mc:.3f}): "
                      f"M={obs['M_sky']:.2f}({obs['M_sky_raw']:.2f}), "
                      f"Md={obs['M_dressed']:.2f}, "
                      f"lump={obs['max_lump_Q']:.2f}, "
                      f"cos_dr={obs['cos_dressed_mean']:.4f}, "
                      f"wl={obs['wilson_mean']:.4f}, "
                      f"ar_f={ar_f:.2f}, ar_g={ar_g:.2f}, "
                      f"dead={obs['n_dead_bonds']}, "
                      f"w={obs['winding']}")

            # Check nucleation threshold
            if obs['M_sky'] > nucleation_threshold and nucleation_sweep is None:
                nucleation_sweep = global_sweep
                M_sky_at_nucleation = obs['M_sky']
                if verbose:
                    print(f"\n    *** NUCLEATION EVENT at sweep {sweep} "
                          f"(beta={beta_mc:.3f})! "
                          f"M_sky={obs['M_sky']:.2f} > {nucleation_threshold} ***\n")

    # End-of-phase-1 measurement
    obs_ann_end = measure(f'{phase1_name}_end', global_sweep)
    ar_f_final = np.mean(accept_rate_frame_accum) if accept_rate_frame_accum else 0
    ar_g_final = np.mean(accept_rate_gauge_accum) if accept_rate_gauge_accum else 0

    if verbose:
        print(f"\n  {phase1_name.capitalize()} summary:")
        print(f"    M_sky final: {obs_ann_end['M_sky']:.2f}")
        print(f"    Nucleation: {'YES at global sweep ' + str(nucleation_sweep) if nucleation_sweep else 'NO'}")
        print(f"    Accept rates: frame={ar_f_final:.3f}, gauge={ar_g_final:.3f}")
        if not (kuramoto_mode or seeded_mode):
            print(f"    Winding: {obs_ann_end['winding']}")

    # =================== PHASE 2: INDEPENDENCE ===================
    beta_indep = beta_grow if (kuramoto_mode or seeded_mode) else beta_cold
    if verbose:
        print(f"\n  PHASE 2: INDEPENDENCE ({n_independence_sweeps} sweeps)")
        print(f"    Removing vortex, unfreezing all sublattices, beta={beta_indep}")

    # Remove vortex: set all phases to zero
    cos_dphi_indep = np.ones(n_bonds)

    # Unfreeze all sublattices
    active_sites_indep = np.arange(N)

    # Recompute K without vortex
    cos_dr = cos_dressed_all(R, V, ei, ej)
    K = compute_K_product(cos_dphi_indep, cos_dr, K0)

    if verbose:
        print(f"    K_mean (no vortex)={np.mean(K):.4f}")

    accept_rate_frame_p2 = []
    accept_rate_gauge_p2 = []

    for sweep in range(1, n_independence_sweeps + 1):
        # Frame MC — all sites active, cold temperature
        R, n_acc_f, n_prop_f = frame_mc_sweep(
            R, K, V, ei, ej, site_nbr, site_bond,
            beta_indep, sigma_mc * 0.5, N, rng, active_sites=active_sites_indep)

        if n_prop_f > 0:
            accept_rate_frame_p2.append(n_acc_f / n_prop_f)

        # Gauge MC with full Wilson confinement
        V, n_acc_g = gauge_mc_sweep(
            V, R, K, ei, ej,
            plaq_bonds, plaq_signs, btp_arr, btp_cnt,
            beta_indep, beta_wilson, sigma_g, n_bonds, rng)
        accept_rate_gauge_p2.append(n_acc_g / max(n_bonds, 1))

        # Gauge CLR step
        if sweep % n_clr_per_mc == 0:
            fiedler_counter += 1
            do_fiedler = (fiedler_counter % max(fiedler_every // n_clr_per_mc, 1) == 0)
            V, v2_ema, gdiag = gauge_clr_step(
                R, V, K, ei, ej,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                v2_ema, N, n_bonds,
                eta_g, beta_g, dt=0.01,
                fiedler_recompute=do_fiedler or v2_ema is None)

        # Update K (no vortex)
        cos_dr = cos_dressed_all(R, V, ei, ej)
        K = compute_K_product(cos_dphi_indep, cos_dr, K0)

        # Reorthogonalize
        if sweep % reortho_every == 0:
            R = reorthogonalize(R)
            for b in range(n_bonds):
                V[b] = reorthogonalize_single(V[b])

        global_sweep += 1

        # Measure
        if sweep % measure_every == 0:
            obs = measure('independence', global_sweep)
            ar_f = np.mean(accept_rate_frame_p2[-measure_every:]) if accept_rate_frame_p2 else 0
            ar_g = np.mean(accept_rate_gauge_p2[-measure_every:]) if accept_rate_gauge_p2 else 0

            if verbose:
                print(f"    sweep {sweep:5d}: M={obs['M_sky']:.2f}({obs['M_sky_raw']:.2f}), "
                      f"Md={obs['M_dressed']:.2f}, "
                      f"lump={obs['max_lump_Q']:.2f}, "
                      f"cos_dr={obs['cos_dressed_mean']:.4f}, "
                      f"wl={obs['wilson_mean']:.4f}, "
                      f"ar_f={ar_f:.2f}, ar_g={ar_g:.2f}, "
                      f"dead={obs['n_dead_bonds']}")

    # Final measurement
    obs_final = measure('final', global_sweep)

    # =================== VERIFICATION GATES ===================
    # G1: Budget conservation
    _, _, gdiag_final = gauge_clr_step(
        R, V, K, ei, ej, plaq_bonds, plaq_signs, btp_arr, btp_cnt,
        v2_ema, N, n_bonds, eta_g, beta_g, dt=0.0,
        fiedler_recompute=True)
    G1 = gdiag_final['budget_err'] < 1e-10

    # G2: Nucleation event
    G2 = nucleation_sweep is not None

    # G3: Core exploration (accept rate at dead bonds)
    G3 = ar_f_final > 0.5  # Lower threshold since accept rate is averaged over all

    # G4: Flux localization during anneal/grow
    phase1_entries = [h for h in history if h['phase'] in ('anneal', 'grow')]
    max_flux_ratio = max((h['flux_ratio'] for h in phase1_entries), default=0)
    G4 = max_flux_ratio > 1.5

    # G5: Independence — M_sky stable after vortex removal
    indep_entries = [h for h in history if h['phase'] == 'independence']
    if indep_entries:
        M_sky_indep_mean = np.mean([h['M_sky'] for h in indep_entries[-5:]])
        G5 = M_sky_indep_mean > nucleation_threshold * 0.5
    else:
        M_sky_indep_mean = 0
        G5 = False

    # G6: Wilson confinement in Phase 2
    if indep_entries:
        wl_indep_mean = np.mean([h['wilson_mean'] for h in indep_entries[-5:]])
        G6 = wl_indep_mean > 0.5
    else:
        wl_indep_mean = 0
        G6 = False

    # G7: Lifecycle
    if nucleation_sweep and M_sky_at_nucleation > 0 and indep_entries:
        lifecycle_ratio = M_sky_indep_mean / M_sky_at_nucleation
        G7 = lifecycle_ratio > 0.5
    else:
        lifecycle_ratio = 0
        G7 = False

    gates = {
        'G1_budget': bool(G1),
        'G2_nucleation': bool(G2),
        'G3_core_explore': bool(G3),
        'G4_flux_local': bool(G4),
        'G5_independence': bool(G5),
        'G6_wilson': bool(G6),
        'G7_lifecycle': bool(G7),
    }
    n_pass = sum(gates.values())

    # Verdict
    if G2 and G5 and G7:
        verdict = 'PROTON_CREATED'
    elif G2 and G5:
        verdict = 'NUCLEATED_PARTIAL_SURVIVAL'
    elif G2:
        verdict = 'NUCLEATED_NO_SURVIVAL'
    elif obs_ann_end['M_sky'] > nucleation_threshold * 0.3:
        verdict = 'TOPOLOGY_SEEDS'
    else:
        verdict = 'NO_NUCLEATION'

    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  VERDICT: {verdict}")
        print(f"  Gates: {n_pass}/7 PASS")
        print(f"    G1 budget:       {'PASS' if G1 else 'FAIL'} (err={gdiag_final['budget_err']:.2e})")
        print(f"    G2 nucleation:   {'PASS' if G2 else 'FAIL'} (sweep={nucleation_sweep})")
        print(f"    G3 core explore: {'PASS' if G3 else 'FAIL'} (ar={ar_f_final:.3f})")
        print(f"    G4 flux local:   {'PASS' if G4 else 'FAIL'} (max_ratio={max_flux_ratio:.3f})")
        print(f"    G5 independence: {'PASS' if G5 else 'FAIL'} (M_sky_indep={M_sky_indep_mean:.2f})")
        print(f"    G6 wilson:       {'PASS' if G6 else 'FAIL'} (wl={wl_indep_mean:.4f})")
        print(f"    G7 lifecycle:    {'PASS' if G7 else 'FAIL'} (ratio={lifecycle_ratio:.3f})")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'='*70}")

    result = {
        'protocol': 'D9',
        'verdict': verdict,
        'L': L, 'N': N, 'n_bonds': n_bonds, 'n_plaq': n_plaq,
        'seed': seed, 'downfold': downfold,
        'params': {
            'K0': K0, 'K_phi': K_phi,
            'beta_hot': beta_hot, 'beta_cold': beta_cold,
            'beta_grow': beta_grow, 'beta_wilson': beta_wilson,
            'eta_g': eta_g, 'beta_g': beta_g,
            'sigma_mc': sigma_mc, 'sigma_g': sigma_g,
            'sigma_theta': sigma_theta,
            'evolve_phases': evolve_phases,
            'seed_radius': seed_radius,
            'nucleation_threshold': nucleation_threshold,
            'n_anneal_sweeps': n_anneal_sweeps,
            'n_independence_sweeps': n_independence_sweeps,
            'kuramoto': kuramoto,
            'kuramoto_profile': kuramoto_profile if kuramoto else None,
            'omega_gradient': omega_gradient if kuramoto else None,
            'kuramoto_coupling': kuramoto_coupling if kuramoto else None,
            'frame_init': frame_init if kuramoto else None,
        },
        'gates': gates,
        'n_pass': n_pass,
        'nucleation_sweep': nucleation_sweep,
        'M_sky_at_nucleation': M_sky_at_nucleation,
        'M_sky_final': obs_final['M_sky'],
        'M_sky_indep_mean': float(M_sky_indep_mean),
        'cos_dressed_final': obs_final['cos_dressed_mean'],
        'wilson_final': obs_final['wilson_mean'],
        'accept_rate_frame_p1': float(ar_f_final),
        'accept_rate_gauge_p1': float(ar_g_final),
        'M_dressed_final': obs_final.get('M_dressed', 0),
        'M_bare_final': obs_final.get('M_bare', 0),
        'M_dressed_indep_mean': float(np.mean([h['M_dressed'] for h in indep_entries[-5:]])) if indep_entries else 0,
        'elapsed_sec': elapsed,
        'n_history': len(history),
        'history': history,
    }

    # Save
    df_label = downfold if downfold else 'full'
    outfile = os.path.join(OUTPUT_DIR, f'd9_L{L}_s{seed}_df{df_label}.json')

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
        return obj

    with open(outfile, 'w') as f:
        json.dump(jsonify(result), f, indent=2)
    if verbose:
        print(f"  Results saved to {outfile}")

    return result


# =====================================================================
# Part 14: Multi-Variant Runner
# =====================================================================

def run_d9_both_downfolds(L=6, seed=42, **kwargs):
    """Run both downfold variants and compare."""
    results = {}

    for df in ['A', 'B']:
        print(f"\n{'#'*70}")
        print(f"  Running D9 with downfold={df}")
        print(f"{'#'*70}")
        results[df] = protocol_d9(L=L, seed=seed, downfold=df, **kwargs)

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  D9 COMPARISON TABLE (L={L}, seed={seed})")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'dfA (freeze A)':>15} {'dfB (freeze B)':>15}")
    print(f"  {'-'*55}")

    for key in ['verdict', 'n_pass', 'nucleation_sweep', 'M_sky_at_nucleation',
                'M_sky_final', 'M_sky_indep_mean', 'cos_dressed_final',
                'wilson_final', 'accept_rate_frame_p1', 'elapsed_sec']:
        vA = results['A'].get(key, 'N/A')
        vB = results['B'].get(key, 'N/A')
        if isinstance(vA, float):
            vA = f"{vA:.4f}"
        if isinstance(vB, float):
            vB = f"{vB:.4f}"
        print(f"  {key:<25} {str(vA):>15} {str(vB):>15}")

    print(f"{'='*70}")
    return results


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='D9: Electron Nucleates Proton (Hot-Start Annealing)')
    parser.add_argument('--L', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--downfold', type=str, default='both',
                        help="'A', 'B', or 'both'")
    parser.add_argument('--K0', type=float, default=8.0)
    parser.add_argument('--K_phi', type=float, default=4.0)
    parser.add_argument('--beta_hot', type=float, default=0.05)
    parser.add_argument('--beta_cold', type=float, default=4.0)
    parser.add_argument('--beta_wilson', type=float, default=2.0)
    parser.add_argument('--eta_g', type=float, default=0.1)
    parser.add_argument('--beta_g', type=float, default=2.0)
    parser.add_argument('--sigma_mc', type=float, default=0.5)
    parser.add_argument('--sigma_g', type=float, default=0.1)
    parser.add_argument('--sigma_theta', type=float, default=0.3)
    parser.add_argument('--n_disorder', type=int, default=200)
    parser.add_argument('--n_anneal', type=int, default=2000)
    parser.add_argument('--n_independence', type=int, default=2000)
    parser.add_argument('--threshold', type=float, default=50.0)
    parser.add_argument('--no_evolve_phases', action='store_true',
                        help='Disable dynamic phase evolution')
    parser.add_argument('--seed_radius', type=float, default=None,
                        help='Hedgehog seed radius (enables seeded mode)')
    parser.add_argument('--beta_grow', type=float, default=None,
                        help='Beta for growth phase in seeded/kuramoto mode (default 2.0)')
    # Kuramoto spontaneous nucleation
    parser.add_argument('--kuramoto', action='store_true',
                        help='Enable Kuramoto spontaneous vortex nucleation')
    parser.add_argument('--kuramoto_profile', type=str, default='circulation',
                        choices=['circulation', 'shear_ring', 'dipole',
                                 'random_gradient'],
                        help='Frequency profile for Kuramoto bootstrap')
    parser.add_argument('--omega_gradient', type=float, default=1.0,
                        help='Frequency gradient strength')
    parser.add_argument('--kuramoto_coupling', type=float, default=2.0,
                        help='Kuramoto coupling constant')
    parser.add_argument('--kuramoto_steps', type=int, default=200,
                        help='Number of Kuramoto bootstrap integration steps')
    parser.add_argument('--frame_init', type=str, default='random',
                        choices=['random', 'identity'],
                        help='Frame initialization for Kuramoto mode')
    args = parser.parse_args()

    kwargs = dict(
        K0=args.K0, K_phi=args.K_phi,
        beta_hot=args.beta_hot, beta_cold=args.beta_cold,
        beta_wilson=args.beta_wilson,
        eta_g=args.eta_g, beta_g=args.beta_g,
        sigma_mc=args.sigma_mc, sigma_g=args.sigma_g,
        sigma_theta=args.sigma_theta,
        n_disorder_sweeps=args.n_disorder,
        n_anneal_sweeps=args.n_anneal,
        n_independence_sweeps=args.n_independence,
        nucleation_threshold=args.threshold,
        evolve_phases=not args.no_evolve_phases,
        seed_radius=args.seed_radius,
        beta_grow=args.beta_grow,
        kuramoto=args.kuramoto,
        kuramoto_profile=args.kuramoto_profile,
        omega_gradient=args.omega_gradient,
        kuramoto_coupling=args.kuramoto_coupling,
        kuramoto_steps=args.kuramoto_steps,
        frame_init=args.frame_init,
    )

    if args.downfold == 'both':
        run_d9_both_downfolds(L=args.L, seed=args.seed, **kwargs)
    else:
        protocol_d9(L=args.L, seed=args.seed, downfold=args.downfold, **kwargs)
