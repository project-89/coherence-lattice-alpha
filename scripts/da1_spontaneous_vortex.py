#!/usr/bin/env python3
"""
Co-evolutionary Kuramoto + CLR spontaneous vortex -> da1 kernel.

The lattice decides its own vortex topology through co-evolution:
  - Phases evolve under Kuramoto with K-dependent coupling
  - K-field evolves under Shannon CLR with current cos(dtheta)
  - Small initial K -> phases evolve freely -> vortex forms
  - CLR selectively kills misaligned bonds -> sharp dead core
  - Positive feedback: weak K -> free phases -> misalignment -> K dies

This is the D9/oscillator_matter_creation co-evolution protocol adapted
for the da1 kernel measurement. No hand-placed geometry anywhere.

Protocol:
  Phase 0: Diamond lattice
  Phase 1: Co-evolution (Kuramoto phases + Shannon CLR K-field)
  Phase 2: Downfold -> bound state (core-localized ground state)
  Phase 3: Jacobian + delta_k_b
  Phase 4: E(B) scan -> da1_kernel
  Phase 5: Multi-L comparison
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components

# =====================================================================
# Module imports
# =====================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LT183B_PATH = os.path.join(THIS_DIR, "..", "canon", "v4_2026-02-14",
                           "184b_lt183_vortex_ring_g2_convergence.py")
spec = importlib.util.spec_from_file_location("lt183b", LT183B_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUTPUT_DIR = os.path.join(THIS_DIR, "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# Deterministic Fiedler (eliminates ARPACK randomness)
# =====================================================================

_det_cache = {'X0': None, 'N': 0}

def _deterministic_fiedler(ei, ej, K_arr, N):
    """Deterministic Fiedler with degree-based v0."""
    w = np.maximum(K_arr, 0.0)
    degree = np.zeros(N)
    np.add.at(degree, ei, w)
    np.add.at(degree, ej, w)

    live = w > 1e-12
    if np.sum(live) == 0:
        v_2 = np.ones(N) / np.sqrt(N)
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
        _det_cache['X0'] = None
        return 0.0, v_2, degree

    L_sp = mod._build_graph_laplacian(ei, ej, w, degree, N)

    cache = _det_cache
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

    v0 = degree.copy()
    v0 -= v0.mean()
    norm = np.linalg.norm(v0)
    if norm < 1e-15:
        v0 = np.arange(N, dtype=float)
        v0 -= v0.mean()
        norm = np.linalg.norm(v0)
    v0 /= norm

    try:
        evals, evecs = eigsh(L_sp, k=3, sigma=0, which='LM',
                             maxiter=5000, tol=1e-10, v0=v0)
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
                                 maxiter=5000, tol=1e-8, v0=v0)
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]
            lambda_2 = float(max(evals[1], 0.0))
            v_2 = evecs[:, 1]
            cache['X0'] = evecs[:, :3].copy()
            cache['N'] = N
        except Exception:
            lambda_2 = 0.0
            v_2 = v0.copy()

    return lambda_2, v_2, degree

# Monkey-patch
mod.sparse_laplacian_and_fiedler = _deterministic_fiedler
import types
mod_globals = vars(mod)
mod_globals['sparse_laplacian_and_fiedler'] = _deterministic_fiedler
print("Fiedler patched: deterministic v0 (degree-based)")

# =====================================================================
# Kuramoto phase step (K-coupled)
# =====================================================================

def step_kuramoto(theta, omega, K_arr, ei, ej, N, dt_phase):
    """One forward Euler step of Kuramoto with K-dependent coupling.

    dtheta_i/dt = omega_i + sum_j K_ij * sin(theta_j - theta_i)
    """
    sin_diff = np.sin(theta[ej] - theta[ei])
    torque = np.zeros(N)
    weighted = K_arr * sin_diff
    np.add.at(torque, ei, weighted)
    np.add.at(torque, ej, -weighted)
    torque += omega
    return (theta + dt_phase * torque) % (2.0 * np.pi)


def circulation_omega(positions, e1, e2, gradient=1.0):
    """Circulation frequency profile: omega = gradient * atan2(y,x)/(2pi)."""
    center = positions.mean(axis=0)
    dr = positions - center[np.newaxis, :]
    x_proj = dr @ e1
    y_proj = dr @ e2
    return gradient * np.arctan2(y_proj, x_proj) / (2.0 * np.pi)


def coevolve_kuramoto_clr(positions, ei, ej, N, n_bonds, e1, e2,
                           eta_clr, lam_clr, K_init=0.01,
                           omega_gradient=1.0, n_warmup=400,
                           n_steps=30000,
                           dt_phase=0.1, dt_K=0.01,
                           spec_interval=20, prune_after=5000,
                           K_dead_thresh=1e-6, report_interval=1000):
    """Co-evolutionary Kuramoto + Shannon CLR.

    Two stages following the D9 pattern:
      Stage A: Kuramoto warmup — phases establish vortex with weak K coupling
      Stage B: Co-evolution — K evolves under CLR, phases under Kuramoto

    theta initialized from circulation profile (atan2) so phase frustration
    is present from the start. Small K_init means CLR selectively grows
    aligned bonds and kills misaligned ones.
    """
    # Initialize theta from circulation profile (pre-established vortex)
    theta = circulation_omega(positions, e1, e2, gradient=1.0) * (2.0 * np.pi)
    # This gives theta = atan2(y,x), the same as the acoustic vortex

    K_arr = np.ones(n_bonds) * K_init
    omega = circulation_omega(positions, e1, e2, gradient=omega_gradient)
    dead_mask = np.zeros(n_bonds, dtype=bool)

    # Stage A: Kuramoto warmup (weak K coupling, no CLR dynamics)
    cos_dth_init = np.cos(theta[ej] - theta[ei])
    n_anti_init = int(np.sum(cos_dth_init < 0))
    print(f"    Initial phase pattern: n_anti={n_anti_init}/{n_bonds}")

    if n_warmup > 0:
        print(f"    Stage A: Kuramoto warmup ({n_warmup} steps, "
              f"K_coupling={K_init:.4f})")
        for step in range(n_warmup):
            theta = step_kuramoto(theta, omega, K_arr, ei, ej, N, dt_phase)

        cos_dth_warm = np.cos(theta[ej] - theta[ei])
        n_anti_warm = int(np.sum(cos_dth_warm < 0))
        print(f"    After warmup: n_anti={n_anti_warm}/{n_bonds}, "
              f"cos(dth) range [{cos_dth_warm.min():.3f}, "
              f"{cos_dth_warm.max():.3f}]")

    # Stage B: Co-evolution
    print(f"    Stage B: Co-evolution ({n_steps} steps)")

    # First Fiedler
    lambda_2, v_2, degree = mod.sparse_laplacian_and_fiedler(ei, ej, K_arr, N)
    fiedler_ema = (v_2[ei] - v_2[ej]) ** 2

    dt_K_cur = dt_K
    K_max_clip = 100.0

    # Convergence tracking
    K_avg_prev = None
    converged = False

    for step in range(n_steps):
        # --- Phase update: Kuramoto with K coupling ---
        theta = step_kuramoto(theta, omega, K_arr, ei, ej, N, dt_phase)

        # --- K update: Shannon CLR with current cos(dtheta) ---
        cos_dth = np.cos(theta[ej] - theta[ei])

        if step % spec_interval == 0:
            lambda_2, v_2, degree = mod.sparse_laplacian_and_fiedler(
                ei, ej, K_arr, N)
            f_new = (v_2[ei] - v_2[ej]) ** 2
            fiedler_ema = 0.9 * fiedler_ema + 0.1 * f_new

        K_dot = mod.shannon_clr_Kdot(
            K_arr, cos_dth, ei, ej, N, v_2, lambda_2, degree,
            eta_clr, lam_clr, fiedler_sens=fiedler_ema, dead_mask=dead_mask)

        # Adaptive dt for K
        alive = ~dead_mask
        kdot_inf = float(np.max(np.abs(K_dot[alive]))) if np.any(alive) else 0.0
        if kdot_inf > 4.0:
            dt_K_cur = max(1e-4, dt_K_cur * 0.7)
        elif kdot_inf < 0.15:
            dt_K_cur = min(0.02, dt_K_cur * 1.02)

        K_arr = np.clip(K_arr + dt_K_cur * K_dot, 0.0, K_max_clip)
        K_arr[dead_mask] = 0.0

        # --- Pruning ---
        if step >= prune_after:
            newly_dead = (~dead_mask) & (K_arr < K_dead_thresh) & (cos_dth < 0)
            dead_mask |= newly_dead
            K_arr[dead_mask] = 0.0

        # --- Convergence check (every 2000 steps after burn-in) ---
        if step >= 1000 and step % 2000 == 0:
            K_alive = K_arr[alive]
            K_avg = float(K_alive.mean()) if len(K_alive) > 0 else 0.0
            if K_avg_prev is not None:
                delta = abs(K_avg - K_avg_prev)
                if delta < 5e-3:
                    converged = True
            K_avg_prev = K_avg

        # --- Report ---
        if step % report_interval == 0:
            n_alive = int(np.sum(~dead_mask))
            n_dead = int(np.sum(dead_mask))
            K_alive_vals = K_arr[~dead_mask]
            K_mean = float(K_alive_vals.mean()) if len(K_alive_vals) > 0 else 0.0
            n_anti = int(np.sum(cos_dth < 0))
            print(f"    step {step:6d}/{n_steps}: K_mean={K_mean:.4f}, "
                  f"n_alive={n_alive}/{n_bonds}, n_dead={n_dead}, "
                  f"n_anti={n_anti}, |Kdot|={kdot_inf:.4f}, "
                  f"dt_K={dt_K_cur:.5f}")

        if converged and step >= prune_after + 2000:
            print(f"    Converged at step {step}")
            break

    # Final state
    dtheta = mod.wrap(theta[ej] - theta[ei])
    cos_dth0 = np.cos(dtheta)
    k_eq = K_arr.copy()

    return k_eq, cos_dth0, dtheta, theta, dead_mask, converged

# =====================================================================
# Ground state selection from K-field topology
# =====================================================================

def find_vortex_core(k_eq, positions, ei, ej, K_thresh=1e-3):
    """Find vortex core center and radius from dead bonds."""
    dead = k_eq < K_thresh
    if not np.any(dead):
        return None, 0.0, 0

    core_sites = set()
    dead_idx = np.where(dead)[0]
    for a in dead_idx:
        core_sites.add(int(ei[a]))
        core_sites.add(int(ej[a]))
    core_sites = np.array(sorted(core_sites))

    core_center = positions[core_sites].mean(axis=0)
    core_dists = np.linalg.norm(
        positions[core_sites] - core_center[np.newaxis, :], axis=1)
    core_radius = float(np.max(core_dists)) if len(core_dists) > 0 else 0.0

    return core_center, core_radius, int(np.sum(dead))


def select_ground_state_core(evals, evecs, positions, core_center, core_radius,
                              nn_dist, mass_min=0.01, mass_max=2.0,
                              loc_threshold=0.3):
    """Select lowest-E state localized near vortex core.

    Returns: (idx, loc, E) or (None, None, None)
    """
    r_cut = max(core_radius + 2.0 * nn_dist, 3.0 * nn_dist)
    dist = np.linalg.norm(positions - core_center[np.newaxis, :], axis=1)
    near_core = dist < r_cut

    abs_ev = np.abs(evals)
    candidates = []
    for idx in range(len(evals)):
        if abs_ev[idx] < mass_min or abs_ev[idx] > mass_max:
            continue
        prob = np.abs(evecs[:, idx]) ** 2
        loc = float(np.sum(prob[near_core]))
        if loc >= loc_threshold:
            candidates.append((idx, loc, evals[idx]))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[2])  # lowest E
    return candidates[0]


# =====================================================================
# Parameters
# =====================================================================

parser = argparse.ArgumentParser(
    description="Co-evolutionary spontaneous vortex + da1 kernel sweep")
parser.add_argument("--L_values", type=int, nargs="+", default=[6, 8, 10])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--r_clr", type=float, default=5.893)
parser.add_argument("--eta_clr", type=float, default=1.0)
parser.add_argument("--b_max", type=float, default=0.0006)
parser.add_argument("--n_b", type=int, default=7)
parser.add_argument("--eps_jac", type=float, default=1e-6)
parser.add_argument("--eps_b", type=float, default=1e-5)
parser.add_argument("--mass_min", type=float, default=0.01)
parser.add_argument("--mass_max", type=float, default=2.0)
parser.add_argument("--loc_threshold", type=float, default=0.3)
# Co-evolution parameters
parser.add_argument("--K_init", type=float, default=0.01)
parser.add_argument("--omega_gradient", type=float, default=1.0)
parser.add_argument("--n_coevolve", type=int, default=30000)
parser.add_argument("--dt_phase", type=float, default=0.1)
parser.add_argument("--dt_K", type=float, default=0.01)
parser.add_argument("--n_warmup", type=int, default=400)
parser.add_argument("--prune_after", type=int, default=5000)
parser.add_argument("--no_plot", action="store_true")
args = parser.parse_args()

r_clr = args.r_clr
seed_base = args.seed
eta_clr = args.eta_clr
lam_clr = eta_clr / r_clr
b_max = args.b_max
n_b = args.n_b
eps_jac = args.eps_jac
eps_b = args.eps_b
downfold_alpha_so = float(np.pi)
spec_interval = 20

SCHWINGER = 1.0 / (2.0 * np.pi)

print(f"{'='*72}")
print(f"da1 Co-evolutionary Spontaneous Vortex Sweep")
print(f"L values: {args.L_values}")
print(f"K_init={args.K_init}, omega_gradient={args.omega_gradient}")
print(f"n_coevolve={args.n_coevolve}, dt_phase={args.dt_phase}, dt_K={args.dt_K}")
print(f"prune_after={args.prune_after}")
print(f"mass window: [{args.mass_min}, {args.mass_max}], loc >= {args.loc_threshold}")
print(f"{'='*72}")

# =====================================================================
# Main loop
# =====================================================================

all_results = []

for L in args.L_values:
    np.random.seed(seed_base)
    _det_cache['X0'] = None
    _det_cache['N'] = 0
    if hasattr(mod, '_fiedler_cache'):
        mod._fiedler_cache = {'X0': None, 'N': 0}

    t0_L = time.time()
    print(f"\n{'='*72}")
    print(f"  L = {L}")
    print(f"{'='*72}")

    # --- Phase 0: Diamond lattice ---
    d = 3
    deltas = mod.make_simplex_deltas(d)
    positions, N, ei, ej, _ = mod.make_d_diamond_adjacency(d, L)
    n_bonds = len(ei)
    nn_dist = float(np.linalg.norm(deltas[0]))
    sublat = np.array([i % 2 for i in range(N)])
    e1, e2, e3, _, _ = mod.get_3d_frame(deltas)
    print(f"  N={N}, n_bonds={n_bonds}")

    # --- Phase 1: Co-evolutionary Kuramoto + CLR ---
    t0_birth = time.time()
    print(f"  Phase 1: Co-evolution (K_init={args.K_init}, "
          f"omega_grad={args.omega_gradient}, n_steps={args.n_coevolve})")

    k_eq, cos_dth0, dtheta, theta_final, dead_mask, converged = \
        coevolve_kuramoto_clr(
            positions, ei, ej, N, n_bonds, e1, e2,
            eta_clr, lam_clr,
            K_init=args.K_init,
            omega_gradient=args.omega_gradient,
            n_warmup=args.n_warmup,
            n_steps=args.n_coevolve,
            dt_phase=args.dt_phase,
            dt_K=args.dt_K,
            spec_interval=spec_interval,
            prune_after=args.prune_after,
            K_dead_thresh=1e-6,
            report_interval=2000)

    alive = k_eq > 1e-3
    bulk = k_eq > 0.1
    n_alive = int(np.sum(alive))
    n_dead = n_bonds - n_alive
    K_bulk = float(np.mean(k_eq[bulk])) if np.any(bulk) else 0.0
    t_birth = time.time() - t0_birth

    n_antialigned = int(np.sum(cos_dth0 < 0))
    print(f"  K_bulk={K_bulk:.4f}, n_alive={n_alive}/{n_bonds}, "
          f"n_dead={n_dead}, converged={converged}")
    print(f"  cos(dtheta) range: [{cos_dth0.min():.3f}, {cos_dth0.max():.3f}]")
    print(f"  anti-aligned bonds (cos<0): {n_antialigned}/{n_bonds}")
    print(f"  Co-evolution: {t_birth:.1f}s")

    if n_dead < 10:
        print(f"  WARNING: Only {n_dead} dead bonds — vortex may not have formed!")

    # --- Phase 2: Downfold + bound state ---
    # Find vortex core from K-field topology
    core_center, core_radius, n_core_dead = find_vortex_core(
        k_eq, positions, ei, ej)

    if core_center is None:
        # Fallback: use lattice center as core center
        print(f"  No dead bonds — using lattice center as core reference")
        core_center = positions.mean(axis=0)
        core_radius = 2.0 * nn_dist
        n_core_dead = 0

    print(f"  Vortex core: center=({core_center[0]:.2f}, {core_center[1]:.2f}, "
          f"{core_center[2]:.2f}), radius={core_radius:.3f}, n_dead={n_core_dead}")

    md_0, _ = mod.build_downfold_mass_diag(
        N, positions, ei, ej, k_eq, dtheta, sublat, alpha_so=downfold_alpha_so)
    H0 = mod.build_hamiltonian_3d(
        N, ei, ej, k_eq, dtheta, sublat=sublat, derived_mass_diag=md_0)
    evals0, evecs0 = np.linalg.eigh(H0)

    best_idx, best_loc, E_ground = select_ground_state_core(
        evals0, evecs0, positions, core_center, core_radius, nn_dist,
        mass_min=args.mass_min, mass_max=args.mass_max,
        loc_threshold=args.loc_threshold)

    if best_idx is None:
        print(f"  No localized ground state in [{args.mass_min}, {args.mass_max}]")
        # Report available localized states
        r_cut = max(core_radius + 2.0 * nn_dist, 3.0 * nn_dist)
        dist_core = np.linalg.norm(
            positions - core_center[np.newaxis, :], axis=1)
        near = dist_core < r_cut
        abs_ev = np.abs(evals0)
        print(f"  Core-localized states (loc > {args.loc_threshold}, "
              f"r_cut={r_cut:.2f}):")
        n_shown = 0
        for idx in range(len(evals0)):
            prob = np.abs(evecs0[:, idx]) ** 2
            loc = float(np.sum(prob[near]))
            if loc >= args.loc_threshold and n_shown < 30:
                print(f"    idx={idx}, E={evals0[idx]:+.6f}, "
                      f"m={abs_ev[idx]:.4f}, loc={loc:.4f}")
                n_shown += 1
        all_results.append({"L": L, "status": "no_ground_state"})
        continue

    E_bs = evals0[best_idx]
    psi = evecs0[:, best_idx]
    m_eff = abs(E_bs)
    prob = np.abs(psi) ** 2
    print(f"  Ground state: idx={best_idx}, E={E_bs:+.6f}, "
          f"m_eff={m_eff:.6f}, loc={best_loc:.4f}")

    # Report localized states in mass window
    r_cut = max(core_radius + 2.0 * nn_dist, 3.0 * nn_dist)
    dist_core = np.linalg.norm(
        positions - core_center[np.newaxis, :], axis=1)
    near = dist_core < r_cut
    abs_ev = np.abs(evals0)
    print(f"  All core-localized states (loc > {args.loc_threshold}, "
          f"m in [{args.mass_min}, {args.mass_max}]):")
    for idx in range(len(evals0)):
        if abs_ev[idx] < args.mass_min or abs_ev[idx] > args.mass_max:
            continue
        p = np.abs(evecs0[:, idx]) ** 2
        loc = float(np.sum(p[near]))
        if loc >= args.loc_threshold:
            marker = " <-- SELECTED" if idx == best_idx else ""
            print(f"    idx={idx}, E={evals0[idx]:+.6f}, "
                  f"m={abs_ev[idx]:.4f}, loc={loc:.4f}{marker}")

    # --- Phase 3: Jacobian + delta_k_b ---
    t0 = time.time()
    J = mod.compute_jacobian(k_eq, cos_dth0, ei, ej, N, eta_clr, lam_clr,
                              eps_jac=eps_jac, verbose=False)

    l2_eq, v2_eq, deg_eq = mod.sparse_laplacian_and_fiedler(ei, ej, k_eq, N)
    b_ph_plus = mod.peierls_phases_general(ei, ej, positions, (+eps_b) * e3)
    b_ph_minus = mod.peierls_phases_general(ei, ej, positions, (-eps_b) * e3)
    cos_plus = np.cos(dtheta + b_ph_plus)
    cos_minus = np.cos(dtheta + b_ph_minus)
    kdot_plus = mod.shannon_clr_Kdot(
        k_eq, cos_plus, ei, ej, N, v2_eq, l2_eq, deg_eq, eta_clr, lam_clr)
    kdot_minus = mod.shannon_clr_Kdot(
        k_eq, cos_minus, ei, ej, N, v2_eq, l2_eq, deg_eq, eta_clr, lam_clr)
    b_B = (kdot_plus - kdot_minus) / (2.0 * eps_b)
    delta_k_b = -np.linalg.solve(J, b_B)

    sv_J = np.linalg.svd(J, compute_uv=False)
    J_cond = float(sv_J[0] / sv_J[-1])
    t_jac = time.time() - t0
    print(f"  Jacobian: {t_jac:.1f}s, cond={J_cond:.2e}")

    # --- Phase 4: E(B) scan ---
    b_grid = np.linspace(-b_max, b_max, 2 * n_b + 1)
    E_frozen = np.zeros_like(b_grid)
    E_kernel = np.zeros_like(b_grid)
    i_center = n_b

    psi_ref = psi.copy()
    for ib in range(len(b_grid)):
        b_ext = b_grid[ib]
        b_ph = mod.peierls_phases_general(ei, ej, positions, b_ext * e3)
        dtheta_total = dtheta + b_ph
        md_B, _ = mod.build_downfold_mass_diag(
            N, positions, ei, ej, k_eq, dtheta_total, sublat,
            alpha_so=downfold_alpha_so)
        H_B = mod.build_hamiltonian_3d(
            N, ei, ej, k_eq, dtheta, b_ph, sublat=sublat,
            derived_mass_diag=md_B)
        ev_B, evec_B = np.linalg.eigh(H_B)
        ov = np.abs(evec_B.conj().T @ psi_ref) ** 2
        best_m = int(np.argmax(ov))
        E_frozen[ib] = float(ev_B[best_m])
        psi_ref = evec_B[:, best_m].copy()

    def scan_kernel_branch(branch_indices, psi_start):
        psi_r = psi_start.copy()
        for ib in branch_indices:
            b_ext = b_grid[ib]
            if abs(b_ext) < 1e-15:
                E_kernel[ib] = E_frozen[i_center]
                continue
            b_ph = mod.peierls_phases_general(
                ei, ej, positions, b_ext * e3)
            dtheta_total = dtheta + b_ph
            k_lr = np.clip(k_eq + b_ext * delta_k_b, 0.0, None)
            md_B, _ = mod.build_downfold_mass_diag(
                N, positions, ei, ej, k_lr, dtheta_total, sublat,
                alpha_so=downfold_alpha_so)
            H_B = mod.build_hamiltonian_3d(
                N, ei, ej, k_lr, dtheta, b_ph, sublat=sublat,
                derived_mass_diag=md_B)
            ev_B, evec_B = np.linalg.eigh(H_B)
            ov = np.abs(evec_B.conj().T @ psi_r) ** 2
            best_m = int(np.argmax(ov))
            E_kernel[ib] = float(ev_B[best_m])
            psi_r = evec_B[:, best_m].copy()

    scan_kernel_branch(list(range(i_center, len(b_grid))), psi.copy())
    scan_kernel_branch(list(range(i_center, -1, -1)), psi.copy())

    c_frozen = np.polyfit(b_grid, E_frozen, 2)
    c_kernel = np.polyfit(b_grid, E_kernel, 2)
    a1_frozen = float(c_frozen[-2])
    a1_kernel = float(c_kernel[-2])
    a2_frozen = float(c_frozen[-3])
    a2_kernel = float(c_kernel[-3])
    da1_kernel = a1_kernel - a1_frozen

    # Directional FD cross-check
    eps_dir = 1e-7
    K_fw = np.clip(k_eq + eps_dir * delta_k_b, 0.0, None)
    K_bw = np.clip(k_eq - eps_dir * delta_k_b, 0.0, None)
    md_fw, _ = mod.build_downfold_mass_diag(
        N, positions, ei, ej, K_fw, dtheta, sublat, alpha_so=downfold_alpha_so)
    md_bw, _ = mod.build_downfold_mass_diag(
        N, positions, ei, ej, K_bw, dtheta, sublat, alpha_so=downfold_alpha_so)
    H_fw = mod.build_hamiltonian_3d(
        N, ei, ej, K_fw, dtheta, sublat=sublat, derived_mass_diag=md_fw)
    H_bw = mod.build_hamiltonian_3d(
        N, ei, ej, K_bw, dtheta, sublat=sublat, derived_mass_diag=md_bw)
    ev_fw, evec_fw = np.linalg.eigh(H_fw)
    ev_bw, evec_bw = np.linalg.eigh(H_bw)
    ov_fw = np.abs(evec_fw.conj().T @ psi) ** 2
    ov_bw = np.abs(evec_bw.conj().T @ psi) ** 2
    da1_dir = (ev_fw[int(np.argmax(ov_fw))] -
               ev_bw[int(np.argmax(ov_bw))]) / (2 * eps_dir)

    print(f"  a1_frozen ={a1_frozen:+.8f}")
    print(f"  a1_kernel ={a1_kernel:+.8f}")
    print(f"  da1_kernel={da1_kernel:+.8f}")
    print(f"  da1_dir   ={da1_dir:+.8f}")
    print(f"  a2_frozen ={a2_frozen:+.4f}")
    print(f"  a2_kernel ={a2_kernel:+.4f}")

    # --- Dimensionless ratios ---
    dk_norm = float(np.linalg.norm(delta_k_b))

    ratios = {
        "da1*m":       da1_kernel * m_eff,
        "da1*m^2":     da1_kernel * m_eff ** 2,
        "da1/m":       da1_kernel / m_eff,
        "g_eff=2ma1":  2.0 * m_eff * a1_kernel,
        "da1/K_bulk":  da1_kernel / K_bulk if K_bulk > 0 else float('inf'),
        "da1*m/K":     da1_kernel * m_eff / K_bulk if K_bulk > 0 else float('inf'),
        "da1*m^2/K":   da1_kernel * m_eff ** 2 / K_bulk if K_bulk > 0 else float('inf'),
        "da1/n_alive": da1_kernel / n_alive if n_alive > 0 else float('inf'),
        "da1/|dk|":    da1_kernel / dk_norm if dk_norm > 0 else float('inf'),
        "da1*m/|dk|":  da1_kernel * m_eff / dk_norm if dk_norm > 0 else float('inf'),
    }

    result = {
        "L": L, "N": N, "n_bonds": n_bonds, "n_alive": n_alive,
        "n_dead": n_dead, "K_bulk": K_bulk, "converged": converged,
        "E_bs": float(E_bs), "m_eff": m_eff,
        "best_idx": int(best_idx), "best_loc": best_loc,
        "core_center": core_center.tolist(),
        "core_radius": core_radius,
        "a1_frozen": a1_frozen, "a1_kernel": a1_kernel,
        "a2_frozen": a2_frozen, "a2_kernel": a2_kernel,
        "da1_kernel": da1_kernel, "da1_dir": float(da1_dir),
        "J_cond": J_cond, "dk_norm": dk_norm,
        "ratios": ratios, "status": "ok",
    }
    all_results.append(result)

    t_L = time.time() - t0_L
    print(f"\n  Ratios:")
    for name, val in ratios.items():
        print(f"    {name:20s} = {val:+.8f}")
    print(f"  L={L} done in {t_L:.1f}s")

# =====================================================================
# Cross-L comparison
# =====================================================================
print(f"\n{'='*72}")
print(f"CROSS-L COMPARISON")
print(f"{'='*72}")

ok = [r for r in all_results if r.get("status") == "ok"]

if len(ok) >= 2:
    print(f"\n  {'L':>4s} {'N':>6s} {'m_eff':>8s} {'K_bulk':>8s} {'E_bs':>10s} "
          f"{'da1_kern':>12s} {'da1_dir':>12s} {'loc':>6s} {'n_dead':>7s}")
    for r in ok:
        print(f"  {r['L']:4d} {r['N']:6d} {r['m_eff']:8.4f} {r['K_bulk']:8.4f} "
              f"{r['E_bs']:+10.6f} {r['da1_kernel']:+12.6f} {r['da1_dir']:+12.6f} "
              f"{r['best_loc']:6.4f} {r['n_dead']:7d}")

    # Ratio stability
    ratio_names = list(ok[0]["ratios"].keys())
    print(f"\n  Ratio stability:")
    print(f"  {'ratio':>20s}", end="")
    for r in ok:
        print(f"  {'L='+str(r['L']):>12s}", end="")
    print(f"  {'CV':>8s}  {'mean':>12s}")

    ratio_stats = []
    for name in ratio_names:
        vals = [r["ratios"][name] for r in ok]
        if any(abs(v) == float('inf') for v in vals):
            continue
        mean = np.mean(vals)
        std = np.std(vals)
        cv = abs(std / mean) if abs(mean) > 1e-30 else float('inf')
        ratio_stats.append((name, cv, mean, vals))

    ratio_stats.sort(key=lambda x: x[1])
    for name, cv, mean, vals in ratio_stats:
        print(f"  {name:>20s}", end="")
        for v in vals:
            print(f"  {v:+12.6f}", end="")
        marker = " <-- BEST" if cv == ratio_stats[0][1] else ""
        print(f"  {cv:8.4f}  {mean:+12.6f}{marker}")

    # Sign consistency check
    for r in ok:
        sign = "+" if r['da1_kernel'] > 0 else "-"
        print(f"  L={r['L']}: da1={r['da1_kernel']:+.6f} ({sign})")

    # Check if all same sign
    signs = [r['da1_kernel'] > 0 for r in ok]
    if all(signs) or not any(signs):
        print(f"  SIGN CONSISTENT across all L values")
    else:
        print(f"  WARNING: Sign flip detected!")

    # Power law
    if len(ok) >= 2:
        ms = np.array([r['m_eff'] for r in ok])
        da1s = np.array([r['da1_kernel'] for r in ok])
        valid = (np.abs(da1s) > 1e-12) & (ms > 1e-12)
        if np.sum(valid) >= 2:
            log_da1 = np.log(np.abs(da1s[valid]))
            log_m = np.log(ms[valid])
            p, A = np.polyfit(log_m, log_da1, 1)
            C = np.exp(A)
            print(f"\n  Power law: |da1| ~ {C:.4f} * m^{p:.4f}")
            for r in ok:
                pred = C * r['m_eff'] ** p
                print(f"    L={r['L']}: da1={r['da1_kernel']:+.6f}, "
                      f"|da1|_pred={pred:.6f}")

elif len(ok) == 1:
    r = ok[0]
    print(f"\n  Only one successful L={r['L']}:")
    print(f"    m_eff={r['m_eff']:.6f}, K_bulk={r['K_bulk']:.4f}, "
          f"da1_kernel={r['da1_kernel']:+.8f}")
else:
    print("  No successful L values.")

# =====================================================================
# Save
# =====================================================================
def sanitize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj

json_path = os.path.join(OUTPUT_DIR, "da1_spontaneous_vortex.json")
with open(json_path, "w") as f:
    json.dump(sanitize(all_results), f, indent=2)
print(f"\nResults saved to {json_path}")

# =====================================================================
# Figure
# =====================================================================
if not args.no_plot and len(ok) >= 1:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("da1 Co-evolutionary Spontaneous Vortex", fontsize=14)

    Ls = [r['L'] for r in ok]
    da1s = [r['da1_kernel'] for r in ok]
    ms = [r['m_eff'] for r in ok]
    Ks = [r['K_bulk'] for r in ok]

    ax = axes[0, 0]
    ax.plot(Ls, da1s, 'C0o-', ms=8)
    ax.set_xlabel("L"); ax.set_ylabel("da1_kernel")
    ax.set_title("(a) da1_kernel vs L")
    ax.axhline(0, color='gray', ls='--', lw=0.5)

    ax = axes[0, 1]
    ax.plot(Ls, ms, 'C3o-', ms=8, label='m_eff')
    ax2 = ax.twinx()
    ax2.plot(Ls, Ks, 'C2s--', ms=6, label='K_bulk')
    ax.set_xlabel("L"); ax.set_ylabel("m_eff", color='C3')
    ax2.set_ylabel("K_bulk", color='C2')
    ax.set_title("(b) m_eff & K_bulk vs L")

    ax = axes[1, 0]
    if len(ok) >= 2 and ratio_stats:
        for name, cv, mean, vals in ratio_stats[:3]:
            ax.plot(Ls, vals, 'o-', ms=6, label=f'{name} (CV={cv:.3f})')
        ax.legend(fontsize=7)
    ax.set_xlabel("L"); ax.set_ylabel("ratio")
    ax.set_title("(c) Most stable ratios")

    ax = axes[1, 1]
    ax.plot(Ls, [r['da1_kernel'] * r['m_eff'] ** 2 for r in ok], 'C2o-', ms=8)
    ax.set_xlabel("L"); ax.set_ylabel("da1*m^2")
    ax.set_title("(d) da1*m^2 vs L")
    ax.axhline(0, color='gray', ls='--', lw=0.5)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "da1_spontaneous_vortex.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")

print("\nDone.")
