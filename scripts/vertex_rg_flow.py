#!/usr/bin/env python3
"""
Vertex Renormalization RG Flow on the 3-Diamond

Computes how BKT renormalization affects the vertex structure, specifically
the ratio of Coulomb (sigma_z) vs Peierls (v_D * sigma_{x,y}) vertex
renormalization factors Z_0 / Z_1.

Three independent calculations confirmed F_2 < 0 (diamagnetic) at the bare
lattice. The question: does the BKT critical point renormalize the vertex
weights toward QED equality (Lorentz invariance)?

Parts:
  1. Three-channel vertex renormalization (Z_0 vs Z_1) at K=1 and K=K_BKT
  2. Momentum-shell decomposition of F_2 — where does the sign come from?
  3. K-scan from K_BKT to K=10 — vertex ratio evolution
  4. (Stretch) BKT-renormalized propagator with running K_ren(|q|)

Key physics:
  - Coulomb channel: sz x T x sz -> F_2^C > 0 (paramagnetic, ALWAYS)
  - Peierls channel: v_D^2 * sum_i si x T x si -> F_2^P < 0 (diamagnetic)
  - Total: F_2 = F_2^C * (1 - 2*v_D^2)
  - If BKT suppresses effective Peierls coupling -> Coulomb dominates -> g > 2

Reuses lattice setup from valley_vertex_d3.py (same directory).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize
import itertools

# =====================================================================
# Pauli matrices
# =====================================================================
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# =====================================================================
# Lattice setup (from valley_vertex_d3.py)
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


def f_d_diamond(k, deltas):
    return sum(np.exp(1j * np.dot(k, deltas[i])) for i in range(len(deltas)))


def make_bravais_reciprocal(deltas, d):
    A_mat = np.array([deltas[0] - deltas[i] for i in range(1, d + 1)])
    B_mat = 2 * np.pi * la.inv(A_mat).T
    return A_mat, B_mat


def find_f_zero(d, deltas, B_mat, nk_scan=20):
    best_t = np.zeros(d)
    best_val = float('inf')
    nk = min(nk_scan, max(8, int(20 * (3.0 / d))))
    for idx in itertools.product(range(nk), repeat=d):
        t = np.array(idx, float) / nk
        k = t @ B_mat
        val = abs(f_d_diamond(k, deltas))
        if val < best_val:
            best_val = val
            best_t = t.copy()
    def obj(t):
        k = t @ B_mat
        return abs(f_d_diamond(k, deltas)) ** 2
    res = minimize(obj, best_t, method='Powell',
                   options={'maxiter': 20000, 'ftol': 1e-30, 'xtol': 1e-15})
    k_zero = res.x @ B_mat
    return k_zero, abs(f_d_diamond(k_zero, deltas))


def compute_f_jacobian(k, deltas, eps=1e-7):
    d = len(k)
    J = np.zeros((2, d))
    for j in range(d):
        kp, km = k.copy(), k.copy()
        kp[j] += eps
        km[j] -= eps
        df = (f_d_diamond(kp, deltas) - f_d_diamond(km, deltas)) / (2 * eps)
        J[0, j] = df.real
        J[1, j] = df.imag
    return J


def get_normal_plane(J):
    v1 = J[0, :]
    v2 = J[1, :]
    e1 = v1 / la.norm(v1)
    v2p = v2 - np.dot(v2, e1) * e1
    e2 = v2p / la.norm(v2p)
    return e1, e2


def lattice_laplacian(qx, qy, e1, e2, deltas):
    q_vec = qx * e1 + qy * e2
    return sum(4.0 * np.sin(np.dot(q_vec, deltas[j]) / 2.0)**2
               for j in range(len(deltas)))


def freq_integral_3pole_J0(E1, E2, Omega):
    eps = 1e-12
    if abs(E1 - E2) < eps: E2 += eps
    if abs(E1 - Omega) < eps: Omega += eps
    if abs(E2 - Omega) < eps: Omega += eps
    a, b, c = E1, E2, Omega
    return (1.0/(2*a*(b*b - a*a)*(c*c - a*a))
          + 1.0/(2*b*(a*a - b*b)*(c*c - b*b))
          + 1.0/(2*c*(a*a - c*c)*(b*b - c*c)))


def freq_integral_3pole_J2(E1, E2, Omega):
    eps = 1e-12
    if abs(E1 - E2) < eps: E2 += eps
    if abs(E1 - Omega) < eps: Omega += eps
    if abs(E2 - Omega) < eps: Omega += eps
    a, b, c = E1, E2, Omega
    return (-a/(2*(b*b - a*a)*(c*c - a*a))
          + -b/(2*(a*a - b*b)*(c*c - b*b))
          + -c/(2*(a*a - c*c)*(b*b - c*c)))


def get_lattice_setup(d):
    deltas = make_simplex_deltas(d)
    _, B_mat = make_bravais_reciprocal(deltas, d)
    k0, fval = find_f_zero(d, deltas, B_mat)
    J = compute_f_jacobian(k0, deltas)
    e1, e2 = get_normal_plane(J)
    v_D = np.sqrt((d + 1) / 2.0)
    return deltas, e1, e2, v_D


# =====================================================================
# New functions for vertex RG flow
# =====================================================================

def pauli_decompose(M):
    """Decompose 2x2 matrix M = a0*I + ax*sx + ay*sy + az*sz."""
    a0 = np.trace(M) / 2
    ax = np.trace(sx @ M) / 2
    ay = np.trace(sy @ M) / 2
    az = np.trace(sz @ M) / 2
    return {'I': complex(a0), 'x': complex(ax), 'y': complex(ay), 'z': complex(az)}


def vertex_correction_3channel(px, py, ppx, ppy, v_D, m, e_eff,
                                e1, e2, deltas, K, I_b, n_grid,
                                q_max_frac=None):
    """
    Three-channel vertex correction: Coulomb (sz) + Peierls (v_D*sx, v_D*sy).

    Returns dict with keys 'coulomb', 'peierls', 'full', each containing
    a list of 3 matrices [Lambda_0, Lambda_1, Lambda_2] corresponding to
    the three vertex channels [sz, v_D*sx, v_D*sy].

    If q_max_frac is set (0..1), only include modes with |q|/q_BZ < q_max_frac.
    """
    gamma_mu_list = [sz, v_D * sx, v_D * sy]
    n_channels = 3

    shapes = {k: [np.zeros((2, 2), dtype=complex) for _ in range(n_channels)]
              for k in ['coulomb', 'peierls', 'full']}
    N_BZ = n_grid**2

    # Compute max |q| for radial cutoff
    # The BZ goes from 0 to 2*pi, so max radius ~ pi*sqrt(2)
    q_bz_max = np.pi * np.sqrt(2)

    for ix in range(n_grid):
        for iy in range(n_grid):
            qx = (ix + 0.5) / n_grid * 2 * np.pi
            qy = (iy + 0.5) / n_grid * 2 * np.pi

            # Fold to [-pi, pi] for radial distance
            qx_fold = qx - 2 * np.pi * round(qx / (2 * np.pi))
            qy_fold = qy - 2 * np.pi * round(qy / (2 * np.pi))
            q_rad = np.sqrt(qx_fold**2 + qy_fold**2)

            # Radial cutoff
            if q_max_frac is not None and q_rad > q_max_frac * q_bz_max:
                continue

            lap = lattice_laplacian(qx, qy, e1, e2, deltas)
            if lap < 1e-30:
                continue
            Omega = np.sqrt(K * lap / I_b)

            k1x, k1y = px - qx, py - qy
            k2x, k2y = ppx - qx, ppy - qy

            E1 = np.sqrt(v_D**2 * (k1x**2 + k1y**2) + m**2)
            E2 = np.sqrt(v_D**2 * (k2x**2 + k2y**2) + m**2)

            A1 = v_D * (sx * k1x + sy * k1y) + m * I2
            A2 = v_D * (sx * k2x + sy * k2y) + m * I2

            J0 = freq_integral_3pole_J0(E1, E2, Omega)
            J2 = freq_integral_3pole_J2(E1, E2, Omega)

            for mu in range(n_channels):
                gm = gamma_mu_list[mu]
                T1 = A1 @ gm @ A2 * J0
                T2 = -(sz @ gm @ sz) * J2
                T = T1 + T2

                coul = sz @ T @ sz
                peir = v_D**2 * (sx @ T @ sx + sy @ T @ sy)
                full = coul + peir

                shapes['coulomb'][mu] += coul
                shapes['peierls'][mu] += peir
                shapes['full'][mu] += full

    scale = e_eff**2 / (I_b * N_BZ)
    for k in shapes:
        for mu in range(n_channels):
            shapes[k][mu] *= scale

    return shapes


def vertex_correction_3channel_bkt(px, py, ppx, ppy, v_D, m, e_eff,
                                    e1, e2, deltas, K_bkt, I_b, n_grid,
                                    c_bkt):
    """
    Vertex correction with BKT-renormalized propagator: K_ren(|q|).

    K_ren(q) = K_bkt / (1 + c * ln(q_max / |q|))
    where c = pi*K_bkt/2 - 1 is the BKT flow coefficient.
    """
    gamma_mu_list = [sz, v_D * sx, v_D * sy]
    n_channels = 3

    shapes = {k: [np.zeros((2, 2), dtype=complex) for _ in range(n_channels)]
              for k in ['coulomb', 'peierls', 'full']}
    N_BZ = n_grid**2

    q_bz_max = np.pi * np.sqrt(2)

    for ix in range(n_grid):
        for iy in range(n_grid):
            qx = (ix + 0.5) / n_grid * 2 * np.pi
            qy = (iy + 0.5) / n_grid * 2 * np.pi

            qx_fold = qx - 2 * np.pi * round(qx / (2 * np.pi))
            qy_fold = qy - 2 * np.pi * round(qy / (2 * np.pi))
            q_rad = np.sqrt(qx_fold**2 + qy_fold**2)

            # BKT running K
            if q_rad < 1e-10:
                K_run = K_bkt  # IR limit
            else:
                log_ratio = np.log(q_bz_max / q_rad)
                K_run = K_bkt / (1.0 + c_bkt * log_ratio)
                if K_run < 0.01:
                    K_run = 0.01  # floor to avoid negative/zero

            lap = lattice_laplacian(qx, qy, e1, e2, deltas)
            if lap < 1e-30:
                continue
            Omega = np.sqrt(K_run * lap / I_b)

            k1x, k1y = px - qx, py - qy
            k2x, k2y = ppx - qx, ppy - qy

            E1 = np.sqrt(v_D**2 * (k1x**2 + k1y**2) + m**2)
            E2 = np.sqrt(v_D**2 * (k2x**2 + k2y**2) + m**2)

            A1 = v_D * (sx * k1x + sy * k1y) + m * I2
            A2 = v_D * (sx * k2x + sy * k2y) + m * I2

            J0 = freq_integral_3pole_J0(E1, E2, Omega)
            J2 = freq_integral_3pole_J2(E1, E2, Omega)

            for mu in range(n_channels):
                gm = gamma_mu_list[mu]
                T1 = A1 @ gm @ A2 * J0
                T2 = -(sz @ gm @ sz) * J2
                T = T1 + T2

                coul = sz @ T @ sz
                peir = v_D**2 * (sx @ T @ sx + sy @ T @ sy)
                full = coul + peir

                shapes['coulomb'][mu] += coul
                shapes['peierls'][mu] += peir
                shapes['full'][mu] += full

    scale = e_eff**2 / (I_b * N_BZ)
    for k in shapes:
        for mu in range(n_channels):
            shapes[k][mu] *= scale

    return shapes


def extract_Z_factors(decomp_0, v_D):
    """
    Extract vertex renormalization Z0 (Coulomb) and Z1 (Peierls)
    from the 3-channel vertex correction at q=0.

    The bare vertex is [sz, v_D*sx, v_D*sy].
    The dressed vertex is bare + Lambda_mu(0,0).
    Z_mu = 1 + (correction projected onto bare channel) / (bare amplitude).

    For mu=0 (Coulomb): bare = sz, project Lambda_0 onto sz -> Z0 = 1 + az/1
    For mu=1 (Peierls): bare = v_D*sx, project Lambda_1 onto sx -> Z1 = 1 + ax/v_D
    """
    # Channel 0: Coulomb (gamma_0 = sz)
    # Lambda_0 at q=0 for the full dressed vertex
    Lam0_full = decomp_0['full'][0]
    dec0 = pauli_decompose(Lam0_full)
    # The sz component of Lambda_0 renormalizes the sz vertex
    # Dressed vertex_0 = sz + Lambda_0 = (1 + az)*sz + ...
    Z0 = 1.0 + dec0['z'].real  # Real part (imaginary should be small)

    # Channel 1: Peierls-x (gamma_1 = v_D * sx)
    Lam1_full = decomp_0['full'][1]
    dec1 = pauli_decompose(Lam1_full)
    # The sx component of Lambda_1 renormalizes the v_D*sx vertex
    # Dressed vertex_1 = v_D*sx + Lambda_1 = (v_D + ax)*sx + ...
    # Z1 = (v_D + ax) / v_D = 1 + ax/v_D
    Z1 = 1.0 + dec1['x'].real / v_D

    # Channel 2: Peierls-y (gamma_2 = v_D * sy) — should match Z1 by symmetry
    Lam2_full = decomp_0['full'][2]
    dec2 = pauli_decompose(Lam2_full)
    Z2 = 1.0 + dec2['y'].real / v_D  # should == Z1

    return Z0, Z1, Z2, dec0, dec1, dec2


def extract_F2_3ch(decomp_0, decomp_q, v_D, m, dq):
    """
    Extract F2 from 3-channel vertex correction.

    F2 = m * Im[Tr(sz * dLambda_y/dq_x)] / dq
    where Lambda_y is channel 2 (Peierls-y, i.e. the v_D*sigma_y vertex).
    The momentum transfer q = (dq, 0) is in x, and the Gordon decomposition
    gives sigma_{yx} * q_x = sz * q_x, so Tr(sz * dLambda_y/dq_x) ~ F2.

    In 3-channel ordering [sz, v_D*sx, v_D*sy], sy is at index 2.
    """
    results = {}
    for key in ['coulomb', 'peierls', 'full']:
        # Channel 2 = v_D*sy vertex, momentum transfer in x
        diff = decomp_q[key][2] - decomp_0[key][2]
        trace_val = np.trace(sz @ diff)
        F2 = m * np.imag(trace_val) / dq
        results[key] = F2

    return results


def compute_F2_3ch(v_D, m, e_eff, e1, e2, deltas, K, I_b, n_grid,
                   dq=0.03, q_max_frac=None):
    """Compute F2 using 3-channel vertex with optional radial cutoff."""
    decomp_0 = vertex_correction_3channel(
        0, 0, 0, 0, v_D, m, e_eff, e1, e2, deltas, K, I_b, n_grid,
        q_max_frac=q_max_frac)
    decomp_q = vertex_correction_3channel(
        dq/2, 0, -dq/2, 0, v_D, m, e_eff, e1, e2, deltas, K, I_b, n_grid,
        q_max_frac=q_max_frac)

    F2 = extract_F2_3ch(decomp_0, decomp_q, v_D, m, dq)
    Z0, Z1, Z2, dec0, dec1, dec2 = extract_Z_factors(decomp_0, v_D)

    return F2, Z0, Z1, Z2, decomp_0


def compute_F2_3ch_bkt(v_D, m, e_eff, e1, e2, deltas, K_bkt, I_b, n_grid,
                        c_bkt, dq=0.03):
    """Compute F2 using BKT-renormalized propagator."""
    decomp_0 = vertex_correction_3channel_bkt(
        0, 0, 0, 0, v_D, m, e_eff, e1, e2, deltas, K_bkt, I_b, n_grid,
        c_bkt)
    decomp_q = vertex_correction_3channel_bkt(
        dq/2, 0, -dq/2, 0, v_D, m, e_eff, e1, e2, deltas, K_bkt, I_b, n_grid,
        c_bkt)

    F2 = extract_F2_3ch(decomp_0, decomp_q, v_D, m, dq)
    Z0, Z1, Z2, dec0, dec1, dec2 = extract_Z_factors(decomp_0, v_D)

    return F2, Z0, Z1, Z2


# =====================================================================
# Main computation
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vertex Renormalization RG Flow on 3-Diamond")
    parser.add_argument("--n_grid", type=int, default=35,
                        help="Momentum grid per direction (default 35)")
    parser.add_argument("--n_shells", type=int, default=20,
                        help="Radial momentum shells (default 20)")
    parser.add_argument("--skip_part4", action="store_true",
                        help="Skip BKT-renormalized propagator (Part 4)")
    args = parser.parse_args()

    n_grid = args.n_grid
    n_shells = args.n_shells

    print("=" * 72)
    print("  VERTEX RENORMALIZATION RG FLOW ON THE 3-DIAMOND")
    print("=" * 72)

    # ---- Lattice setup ----
    d = 3
    deltas, e1, e2, v_D_bare = get_lattice_setup(d)

    m = 0.5
    e_eff = 0.3
    I_b = 1.0
    K_BKT = 2.0 / np.pi  # 0.6366...
    dq = 0.03

    alpha_ref = e_eff**2 / (4 * np.pi * v_D_bare)

    print(f"\n  d = {d}")
    print(f"  v_D (bare) = sqrt(2) = {v_D_bare:.6f}")
    print(f"  v_D^2 = {v_D_bare**2:.4f}")
    print(f"  Bare contraction = 1 - 2*v_D^2 = {1 - 2*v_D_bare**2:.1f}")
    print(f"  m = {m}, e_eff = {e_eff}, n_grid = {n_grid}")
    print(f"  K_BKT = 2/pi = {K_BKT:.6f}")
    print(f"  alpha_ref = {alpha_ref:.6f}")

    results_all = {}
    t_start = time.time()

    # ==================================================================
    # PART 1: Three-Channel Vertex Renormalization
    # ==================================================================
    print(f"\n{'='*72}")
    print(f"  PART 1: THREE-CHANNEL VERTEX RENORMALIZATION (Z_0 vs Z_1)")
    print(f"{'='*72}")

    K_values_p1 = [1.0, K_BKT]
    part1_results = []

    for K in K_values_p1:
        print(f"\n  --- K = {K:.4f} ---", flush=True)
        t0 = time.time()

        F2, Z0, Z1, Z2, decomp_0 = compute_F2_3ch(
            v_D_bare, m, e_eff, e1, e2, deltas, K, I_b, n_grid, dq=dq)

        dt_comp = time.time() - t0
        alpha_v = e_eff**2 / (4 * np.pi * v_D_bare)

        # Pauli decomposition of each channel
        print(f"  Computation time: {dt_comp:.1f}s")
        print(f"\n  Vertex renormalization factors:")
        print(f"    Z_0 (Coulomb, sz)  = {Z0:.8f}")
        print(f"    Z_1 (Peierls, sx)  = {Z1:.8f}")
        print(f"    Z_2 (Peierls, sy)  = {Z2:.8f}")
        print(f"    Z_0 / Z_1          = {Z0/Z1:.8f}")
        print(f"    Z_1 / Z_2          = {Z1/Z2:.8f} (should be ~1)")

        print(f"\n  F_2 decomposition:")
        print(f"    F_2^Coulomb = {F2['coulomb']:+.6e}")
        print(f"    F_2^Peierls = {F2['peierls']:+.6e}")
        print(f"    F_2^Full    = {F2['full']:+.6e}")
        if abs(F2['coulomb']) > 1e-30:
            ratio = F2['full'] / F2['coulomb']
            print(f"    F_2^Full / F_2^C = {ratio:.4f} (expected {1-2*v_D_bare**2:.1f})")

        # Pauli decomposition details
        print(f"\n  Pauli decomposition of Lambda_mu(0,0) [full]:")
        for mu, label in enumerate(['sz (Coulomb)', 'v_D*sx (Peierls-x)', 'v_D*sy (Peierls-y)']):
            dec = pauli_decompose(decomp_0['full'][mu])
            print(f"    Channel {mu} ({label}):")
            print(f"      I:  {dec['I'].real:+.6e} {dec['I'].imag:+.6e}j")
            print(f"      sx: {dec['x'].real:+.6e} {dec['x'].imag:+.6e}j")
            print(f"      sy: {dec['y'].real:+.6e} {dec['y'].imag:+.6e}j")
            print(f"      sz: {dec['z'].real:+.6e} {dec['z'].imag:+.6e}j")

        r = {
            'K': K,
            'Z0': Z0, 'Z1': Z1, 'Z2': Z2,
            'Z0_over_Z1': Z0 / Z1,
            'F2_coulomb': F2['coulomb'],
            'F2_peierls': F2['peierls'],
            'F2_full': F2['full'],
            'time_s': dt_comp,
        }
        part1_results.append(r)

    results_all['part1'] = part1_results

    print(f"\n  --- Part 1 Summary ---")
    print(f"  {'K':>8s}  {'Z0':>12s}  {'Z1':>12s}  {'Z0/Z1':>12s}  {'F2_C':>12s}  {'F2_Full':>12s}")
    for r in part1_results:
        print(f"  {r['K']:8.4f}  {r['Z0']:12.8f}  {r['Z1']:12.8f}  "
              f"{r['Z0_over_Z1']:12.8f}  {r['F2_coulomb']:+12.6e}  {r['F2_full']:+12.6e}")

    # Interpretation
    z_ratio_1 = part1_results[0]['Z0_over_Z1']
    z_ratio_bkt = part1_results[1]['Z0_over_Z1']
    print(f"\n  Vertex ratio Z0/Z1 at K=1:     {z_ratio_1:.8f}")
    print(f"  Vertex ratio Z0/Z1 at K=K_BKT: {z_ratio_bkt:.8f}")
    if z_ratio_bkt < z_ratio_1:
        print(f"  -> Peierls grows FASTER at BKT -> flows TOWARD QED equality")
    elif z_ratio_bkt > z_ratio_1:
        print(f"  -> Coulomb grows FASTER at BKT -> flows AWAY from QED")
    else:
        print(f"  -> Ratio PRESERVED (Ward identity protects)")

    # ==================================================================
    # PART 2: Momentum-Shell Decomposition of F_2
    # ==================================================================
    print(f"\n{'='*72}")
    print(f"  PART 2: MOMENTUM-SHELL DECOMPOSITION OF F_2")
    print(f"{'='*72}")

    K_shell = 1.0
    shell_fracs = np.linspace(1.0 / n_shells, 1.0, n_shells)

    print(f"\n  K = {K_shell}, n_shells = {n_shells}")
    print(f"\n  {'shell':>5s}  {'q_max/q_BZ':>10s}  {'F2_C_cum':>12s}  {'F2_P_cum':>12s}  "
          f"{'F2_Full_cum':>12s}  {'Z0':>10s}  {'Z1':>10s}  {'Z0/Z1':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    part2_results = []
    t0_p2 = time.time()

    for i, frac in enumerate(shell_fracs):
        F2, Z0, Z1, Z2, _ = compute_F2_3ch(
            v_D_bare, m, e_eff, e1, e2, deltas, K_shell, I_b, n_grid,
            dq=dq, q_max_frac=frac)

        z_ratio = Z0 / Z1 if abs(Z1) > 1e-30 else float('nan')

        r = {
            'shell': i + 1,
            'q_max_frac': frac,
            'F2_coulomb': F2['coulomb'],
            'F2_peierls': F2['peierls'],
            'F2_full': F2['full'],
            'Z0': Z0, 'Z1': Z1, 'Z0_over_Z1': z_ratio,
        }
        part2_results.append(r)

        sign_C = '+' if F2['coulomb'] >= 0 else '-'
        sign_F = '+' if F2['full'] >= 0 else '-'
        print(f"  {i+1:5d}  {frac:10.4f}  {F2['coulomb']:+12.6e}  {F2['peierls']:+12.6e}  "
              f"{F2['full']:+12.6e}  {Z0:10.6f}  {Z1:10.6f}  {z_ratio:10.6f}",
              flush=True)

    dt_p2 = time.time() - t0_p2
    print(f"\n  Part 2 time: {dt_p2:.1f}s")

    results_all['part2'] = part2_results

    # Identify sign-change shell
    sign_changes = []
    for i in range(1, len(part2_results)):
        prev_F2 = part2_results[i-1]['F2_full']
        curr_F2 = part2_results[i]['F2_full']
        if prev_F2 * curr_F2 < 0:
            sign_changes.append(i)

    if sign_changes:
        print(f"\n  F_2^Full SIGN CHANGE at shell(s): {sign_changes}")
        for sc in sign_changes:
            f_prev = part2_results[sc-1]['q_max_frac']
            f_curr = part2_results[sc]['q_max_frac']
            print(f"    Between q_max/q_BZ = {f_prev:.3f} and {f_curr:.3f}")
        print(f"  -> Sign comes from UV modes above the crossover")
    else:
        final_sign = 'positive' if part2_results[-1]['F2_full'] > 0 else 'negative'
        print(f"\n  No sign change in F_2^Full — {final_sign} throughout")

    # ==================================================================
    # PART 3: K-Scan of Vertex Renormalization
    # ==================================================================
    print(f"\n{'='*72}")
    print(f"  PART 3: K-SCAN OF VERTEX RENORMALIZATION")
    print(f"{'='*72}")

    K_scan_values = [K_BKT, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    K_scan_values.sort()

    print(f"\n  K values: {[f'{k:.3f}' for k in K_scan_values]}")
    print(f"\n  {'K':>8s}  {'Z0':>12s}  {'Z1':>12s}  {'Z0/Z1':>12s}  "
          f"{'F2_C':>12s}  {'F2_Full':>12s}  {'slope_C':>10s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    part3_results = []
    t0_p3 = time.time()

    for K in K_scan_values:
        t0 = time.time()
        F2, Z0, Z1, Z2, _ = compute_F2_3ch(
            v_D_bare, m, e_eff, e1, e2, deltas, K, I_b, n_grid, dq=dq)
        dt_k = time.time() - t0

        z_ratio = Z0 / Z1 if abs(Z1) > 1e-30 else float('nan')
        slope_C = F2['coulomb'] / alpha_ref if alpha_ref > 0 else 0

        r = {
            'K': K,
            'Z0': Z0, 'Z1': Z1, 'Z0_over_Z1': z_ratio,
            'F2_coulomb': F2['coulomb'],
            'F2_peierls': F2['peierls'],
            'F2_full': F2['full'],
            'slope_C': slope_C,
            'time_s': dt_k,
        }
        part3_results.append(r)

        print(f"  {K:8.4f}  {Z0:12.8f}  {Z1:12.8f}  {z_ratio:12.8f}  "
              f"{F2['coulomb']:+12.6e}  {F2['full']:+12.6e}  {slope_C:10.4f}",
              flush=True)

    dt_p3 = time.time() - t0_p3
    print(f"\n  Part 3 time: {dt_p3:.1f}s")

    results_all['part3'] = part3_results

    # Analyze trend
    z_ratios = [r['Z0_over_Z1'] for r in part3_results]
    K_vals = [r['K'] for r in part3_results]

    # Monotonicity check
    increasing = all(z_ratios[i] <= z_ratios[i+1] for i in range(len(z_ratios)-1))
    decreasing = all(z_ratios[i] >= z_ratios[i+1] for i in range(len(z_ratios)-1))

    print(f"\n  Z0/Z1 trend with K:")
    print(f"    At K_BKT ({K_BKT:.3f}): {z_ratios[0]:.8f}")
    print(f"    At K=10:          {z_ratios[-1]:.8f}")
    print(f"    Monotonic increasing: {increasing}")
    print(f"    Monotonic decreasing: {decreasing}")

    if z_ratios[0] < z_ratios[-1]:
        print(f"    -> Z0/Z1 INCREASES with K (stiff limit has higher ratio)")
        print(f"    -> BKT critical point SUPPRESSES Coulomb relative to Peierls")
        print(f"    -> Flows TOWARD QED equality at K_BKT")
    elif z_ratios[0] > z_ratios[-1]:
        print(f"    -> Z0/Z1 DECREASES with K")
        print(f"    -> BKT critical point ENHANCES Coulomb relative to Peierls")
        print(f"    -> Flows AWAY from QED equality at K_BKT")
    else:
        print(f"    -> Z0/Z1 CONSTANT — Ward identity protects the ratio")

    # ==================================================================
    # PART 4: BKT-Renormalized Propagator (stretch goal)
    # ==================================================================
    if not args.skip_part4:
        print(f"\n{'='*72}")
        print(f"  PART 4: BKT-RENORMALIZED PROPAGATOR")
        print(f"{'='*72}")

        # BKT flow coefficient: c = pi*K_BKT/2 - 1
        c_bkt = np.pi * K_BKT / 2.0 - 1.0
        print(f"\n  K_BKT = {K_BKT:.6f}")
        print(f"  c_BKT = pi*K_BKT/2 - 1 = {c_bkt:.6f}")
        print(f"  (c > 0 means K flows to strong coupling in IR)")

        t0_p4 = time.time()

        # Constant-K result at K_BKT for comparison
        print(f"\n  Computing constant-K result at K_BKT...", flush=True)
        F2_const, Z0_const, Z1_const, Z2_const, _ = compute_F2_3ch(
            v_D_bare, m, e_eff, e1, e2, deltas, K_BKT, I_b, n_grid, dq=dq)

        # BKT-running result
        print(f"  Computing BKT-running K(q) result...", flush=True)
        F2_bkt, Z0_bkt, Z1_bkt, Z2_bkt = compute_F2_3ch_bkt(
            v_D_bare, m, e_eff, e1, e2, deltas, K_BKT, I_b, n_grid,
            c_bkt, dq=dq)

        dt_p4 = time.time() - t0_p4

        print(f"\n  Comparison: Constant-K vs BKT-running K(q)")
        print(f"  {'':>20s}  {'Constant K':>14s}  {'BKT K(q)':>14s}  {'Ratio':>10s}")
        print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*10}")

        for label, const_val, bkt_val in [
            ('Z0 (Coulomb)', Z0_const, Z0_bkt),
            ('Z1 (Peierls)', Z1_const, Z1_bkt),
            ('Z0/Z1', Z0_const/Z1_const, Z0_bkt/Z1_bkt),
            ('F2_Coulomb', F2_const['coulomb'], F2_bkt['coulomb']),
            ('F2_Peierls', F2_const['peierls'], F2_bkt['peierls']),
            ('F2_Full', F2_const['full'], F2_bkt['full']),
        ]:
            ratio = bkt_val / const_val if abs(const_val) > 1e-30 else float('nan')
            if isinstance(const_val, float) and abs(const_val) < 1e-4:
                print(f"  {label:>20s}  {const_val:+14.6e}  {bkt_val:+14.6e}  {ratio:10.4f}")
            else:
                print(f"  {label:>20s}  {const_val:14.8f}  {bkt_val:14.8f}  {ratio:10.4f}")

        print(f"\n  Part 4 time: {dt_p4:.1f}s")

        results_all['part4'] = {
            'c_bkt': c_bkt,
            'constant_K': {
                'Z0': Z0_const, 'Z1': Z1_const,
                'Z0_over_Z1': Z0_const / Z1_const,
                'F2_coulomb': F2_const['coulomb'],
                'F2_full': F2_const['full'],
            },
            'bkt_running': {
                'Z0': Z0_bkt, 'Z1': Z1_bkt,
                'Z0_over_Z1': Z0_bkt / Z1_bkt,
                'F2_coulomb': F2_bkt['coulomb'],
                'F2_full': F2_bkt['full'],
            },
        }

        # Interpretation
        if abs(F2_bkt['full']) < abs(F2_const['full']):
            print(f"\n  BKT running REDUCES |F_2| — UV suppression at work")
        else:
            print(f"\n  BKT running INCREASES |F_2| — IR enhancement dominates")

        if (F2_const['full'] < 0) and (F2_bkt['full'] > 0):
            print(f"  *** SIGN FLIP: BKT running restores PARAMAGNETIC F_2 ***")
        elif (F2_const['full'] < 0) and (F2_bkt['full'] < 0):
            print(f"  No sign flip — still diamagnetic with BKT running")

    # ==================================================================
    # SUMMARY & INTERPRETATION
    # ==================================================================
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")

    t_total = time.time() - t_start
    print(f"\n  Total computation time: {t_total:.1f}s")

    # Key numbers
    p1_K1 = part1_results[0]
    p1_bkt = part1_results[1]

    print(f"\n  KEY RESULTS:")
    print(f"  ============")
    print(f"  At K=1.0:     Z0/Z1 = {p1_K1['Z0_over_Z1']:.8f}")
    print(f"  At K_BKT:     Z0/Z1 = {p1_bkt['Z0_over_Z1']:.8f}")

    delta_ratio = p1_bkt['Z0_over_Z1'] - p1_K1['Z0_over_Z1']
    print(f"  Delta(Z0/Z1) = {delta_ratio:+.8f} (BKT - K=1)")

    print(f"\n  INTERPRETATION:")
    if abs(delta_ratio) < 1e-6:
        print(f"  Ward identity: Z0/Z1 ratio is PROTECTED by gauge symmetry.")
        print(f"  The BKT critical point does NOT change the vertex structure.")
        print(f"  The alpha -> QED mapping works because alpha captures ALL physics")
        print(f"  through BKT renormalization of K, and QED is the universal way")
        print(f"  to convert alpha -> g-2 regardless of UV vertex structure.")
    elif delta_ratio < 0:
        print(f"  Peierls grows faster at BKT -> vertex flows TOWARD QED equality.")
        print(f"  The BKT critical point produces emergent Lorentz invariance")
        print(f"  in the vertex structure.")
    else:
        print(f"  Coulomb grows faster at BKT -> vertex flows AWAY from QED.")
        print(f"  The alpha -> QED mapping works for a different reason:")
        print(f"  alpha already encodes vertex + coupling + propagator corrections.")

    # Momentum-shell analysis
    if part2_results:
        print(f"\n  MOMENTUM-SHELL ANALYSIS:")
        first_pos = None
        last_pos = None
        for r in part2_results:
            if r['F2_full'] > 0:
                if first_pos is None:
                    first_pos = r['q_max_frac']
                last_pos = r['q_max_frac']
        if first_pos is not None:
            print(f"  F_2 > 0 (paramagnetic) in range q_max/q_BZ ~ [{first_pos:.2f}, {last_pos:.2f}]")
        else:
            print(f"  F_2 < 0 (diamagnetic) at ALL momentum scales")

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "vertex_rg_flow_results.json")

    # Convert numpy types for JSON
    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {'re': obj.real, 'im': obj.imag}
        return obj

    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2, default=to_json)
    print(f"\n  Results saved to: {output_file}")

    print(f"\n{'='*72}")
    print(f"  DONE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
