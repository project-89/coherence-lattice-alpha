#!/usr/bin/env python3
"""
D9 Frame-Sector g-Factor Measurement (v2)

Creates a D9 skyrmion via Kuramoto spontaneous nucleation protocol,
then measures the g-factor using the 3N gauge-dressed Hamiltonian
with SU(2) Peierls phases.

v2 improvements over v1:
  - Identity frame init (stronger skyrmion, M_sky ~3.1)
  - 9000 growth sweeps (was 6000), 4000 independence (was 3000)
  - Adiabatic B-ramp for K_gauge channel: warm-starts from previous
    B point with gentle CLR+MC steps, replacing joint_reequilibrate
    which started from scratch and caused large V jumps.

Run:
    nohup .venv/bin/python -u scripts/d9_frame_gfactor.py --L 8 \
        --frame_init identity --n_anneal 9000 --n_indep 4000 \
        --living_mode adiabatic --n_ramp_steps 10 \
        > out/d9_frame_gfactor_v2_L8.log 2>&1 &
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import i0 as bessel_i0
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# Module imports
# =====================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
import d9_vortex_mc_gauge as d9
import gauge_dressed_gfactor_v2 as v2


def load_184b():
    """Load 184b module for Peierls phases, wrap, R0_vec, CLR."""
    return v2.load_lt183b_module()


# =====================================================================
# Phase 0: Lattice Construction
# =====================================================================

def build_lattice(L, verbose=True):
    """Build diamond lattice and enumerate plaquettes."""
    positions, ei, ej, site_nbr, site_bond, N, n_bonds, sublat = \
        d9.build_diamond_lattice(L)
    center = np.mean(positions, axis=0)

    if verbose:
        print(f"  N={N}, n_bonds={n_bonds}")
        print(f"  Enumerating plaquettes...")

    plaq_bonds, plaq_signs, btp_arr, btp_cnt = \
        d9.enumerate_hexagonal_plaquettes(ei, ej, site_nbr, site_bond, N, n_bonds)
    n_plaq = plaq_bonds.shape[0]
    if verbose:
        print(f"  {n_plaq} hexagonal plaquettes found")

    return {
        'positions': positions, 'ei': ei, 'ej': ej,
        'site_nbr': site_nbr, 'site_bond': site_bond,
        'N': N, 'n_bonds': n_bonds, 'sublat': sublat,
        'center': center,
        'plaq_bonds': plaq_bonds, 'plaq_signs': plaq_signs,
        'btp_arr': btp_arr, 'btp_cnt': btp_cnt,
    }


# =====================================================================
# Phase 1: D9 Kuramoto Spontaneous Protocol
# =====================================================================

def run_d9_protocol(lat, seed=42, K0=8.0, K_phi=4.0,
                    frame_init='identity',
                    beta_grow=4.0, beta_wilson=2.0,
                    eta_g=0.1, beta_g=2.0,
                    sigma_mc=0.5, sigma_g=0.1, sigma_theta=0.3,
                    n_anneal_sweeps=6000, n_independence_sweeps=3000,
                    reortho_every=50, measure_every=100,
                    n_clr_per_mc=5, fiedler_every=50,
                    verbose=True):
    """Run D9 Kuramoto spontaneous protocol.

    Phase 1a: Kuramoto bootstrap
    Phase 1b: Growth (n_anneal_sweeps at beta_grow)
    Phase 1c: Independence (n_independence_sweeps, vortex removed)

    Returns state dict with R, V, K, theta, cos_dphi, dtheta, history.
    """
    rng = np.random.default_rng(seed)

    N = lat['N']
    n_bonds = lat['n_bonds']
    positions = lat['positions']
    ei, ej = lat['ei'], lat['ej']
    site_nbr, site_bond = lat['site_nbr'], lat['site_bond']
    sublat = lat['sublat']
    center = lat['center']
    plaq_bonds = lat['plaq_bonds']
    plaq_signs = lat['plaq_signs']
    btp_arr, btp_cnt = lat['btp_arr'], lat['btp_cnt']

    # All sites active (no downfold)
    active_sites = np.arange(N)
    K_phi_arr = np.full(n_bonds, K_phi)

    history = []
    v2_ema = None
    fiedler_counter = 0

    # ---- Measurement helper ----
    def measure(phase_name, sweep_num):
        nonlocal theta, R, V, K
        cos_dr_now = d9.cos_dressed_all(R, V, ei, ej)
        wl = d9.wilson_mean(V, plaq_bonds, plaq_signs)
        snbr_alive, _ = d9.build_alive_site_nbr(
            site_nbr, site_bond, K, K_thresh=1e-4)
        topo = d9.spatial_topology_pruned(R, snbr_alive, positions, N)
        topo_dressed = d9.gauge_dressed_topo_charge(
            R, V, ei, ej, site_nbr, positions, N, n_bonds)
        w = d9.measure_winding_number(theta, positions, center, axis=2)

        obs = {
            'phase': phase_name, 'sweep': sweep_num,
            'M_sky': topo['M_sky_unsigned'],
            'M_dressed': topo_dressed['M_dressed'],
            'cos_dressed_mean': float(np.mean(cos_dr_now)),
            'wilson_mean': wl,
            'K_mean': float(np.mean(K)),
            'winding': w,
            'n_dead_bonds': int(np.sum(K < 1e-4)),
        }
        history.append(obs)
        return obs

    # ---- Phase 1a: Kuramoto Bootstrap ----
    if verbose:
        print(f"\n  PHASE 1a: KURAMOTO BOOTSTRAP")

    theta = np.zeros(N)
    if frame_init == 'identity':
        R = np.tile(np.eye(3), (N, 1, 1)).astype(np.float64)
    else:
        R = d9.random_rotation_small_batch(N, np.pi, rng)
    V = np.tile(np.eye(3), (n_bonds, 1, 1)).astype(np.float64)

    omega = d9.kuramoto_frequency_profile(
        positions, center, 'circulation', omega_gradient=1.0, axis=2, rng=rng)

    theta, winding, step_nuc = d9.kuramoto_bootstrap(
        theta, omega, positions, center, ei, ej, site_nbr,
        coupling=2.0, dt=0.1, n_steps=200, axis=2, verbose=verbose)

    cos_dphi = d9.compute_cos_dphi(theta, ei, ej)
    cos_dr = d9.cos_dressed_all(R, V, ei, ej)
    K = d9.compute_K_product(cos_dphi, cos_dr, K0)

    if verbose:
        print(f"    Winding: {winding}, cos_dphi_mean={np.mean(cos_dphi):.4f}")

    obs0 = measure('bootstrap', 0)
    if verbose:
        print(f"    Post-bootstrap: M_sky={obs0['M_sky']:.2f}, "
              f"M_dressed={obs0['M_dressed']:.2f}")

    # ---- Phase 1b: Growth ----
    if verbose:
        print(f"\n  PHASE 1b: GROWTH ({n_anneal_sweeps} sweeps, beta={beta_grow})")

    t_grow_start = time.time()

    for sweep in range(1, n_anneal_sweeps + 1):
        # Phase MC
        theta, _, _ = d9.phase_mc_sweep(
            theta, K_phi_arr, ei, ej, site_nbr, site_bond,
            beta_grow, sigma_theta, N, rng)
        cos_dphi = d9.compute_cos_dphi(theta, ei, ej)

        # Frame MC (half step size for growth)
        R, _, _ = d9.frame_mc_sweep(
            R, K, V, ei, ej, site_nbr, site_bond,
            beta_grow, sigma_mc * 0.5, N, rng, active_sites=active_sites)

        # Gauge MC
        V, _ = d9.gauge_mc_sweep(
            V, R, K, ei, ej,
            plaq_bonds, plaq_signs, btp_arr, btp_cnt,
            beta_grow, beta_wilson, sigma_g, n_bonds, rng)

        # Gauge CLR every n_clr_per_mc sweeps
        if sweep % n_clr_per_mc == 0:
            fiedler_counter += 1
            do_fiedler = (fiedler_counter %
                          max(fiedler_every // n_clr_per_mc, 1) == 0)
            V, v2_ema, _ = d9.gauge_clr_step(
                R, V, K, ei, ej,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                v2_ema, N, n_bonds,
                eta_g, beta_g, dt=0.01,
                fiedler_recompute=do_fiedler or v2_ema is None)

        # Update K
        cos_dr = d9.cos_dressed_all(R, V, ei, ej)
        K = d9.compute_K_product(cos_dphi, cos_dr, K0)

        # Reorthogonalize
        if sweep % reortho_every == 0:
            R = d9.reorthogonalize(R)
            for b in range(n_bonds):
                V[b] = d9.reorthogonalize_single(V[b])

        # Measure
        if sweep % measure_every == 0:
            obs = measure('grow', sweep)
            if verbose:
                print(f"    sweep {sweep:5d}: M_sky={obs['M_sky']:.2f}, "
                      f"M_d={obs['M_dressed']:.2f}, "
                      f"cos_dr={obs['cos_dressed_mean']:.4f}, "
                      f"wl={obs['wilson_mean']:.4f}, "
                      f"dead={obs['n_dead_bonds']}, "
                      f"w={obs['winding']}")

    t_grow = time.time() - t_grow_start
    obs_grow_end = measure('grow_end', n_anneal_sweeps)
    if verbose:
        print(f"\n    Growth done ({t_grow:.1f}s): M_sky={obs_grow_end['M_sky']:.2f}, "
              f"M_dressed={obs_grow_end['M_dressed']:.2f}")

    # ---- Phase 1c: Independence ----
    if verbose:
        print(f"\n  PHASE 1c: INDEPENDENCE ({n_independence_sweeps} sweeps)")
        print(f"    Removing vortex, all sites active")

    # Remove vortex: set cos_dphi = 1 everywhere
    cos_dphi_indep = np.ones(n_bonds)
    theta = np.zeros(N)
    cos_dphi = cos_dphi_indep

    # Recompute K without vortex
    cos_dr = d9.cos_dressed_all(R, V, ei, ej)
    K = d9.compute_K_product(cos_dphi_indep, cos_dr, K0)

    if verbose:
        print(f"    K_mean (no vortex)={np.mean(K):.4f}")

    t_indep_start = time.time()

    for sweep in range(1, n_independence_sweeps + 1):
        # Frame MC (all sites)
        R, _, _ = d9.frame_mc_sweep(
            R, K, V, ei, ej, site_nbr, site_bond,
            beta_grow, sigma_mc * 0.5, N, rng, active_sites=active_sites)

        # Gauge MC
        V, _ = d9.gauge_mc_sweep(
            V, R, K, ei, ej,
            plaq_bonds, plaq_signs, btp_arr, btp_cnt,
            beta_grow, beta_wilson, sigma_g, n_bonds, rng)

        # Gauge CLR
        if sweep % n_clr_per_mc == 0:
            fiedler_counter += 1
            do_fiedler = (fiedler_counter %
                          max(fiedler_every // n_clr_per_mc, 1) == 0)
            V, v2_ema, _ = d9.gauge_clr_step(
                R, V, K, ei, ej,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                v2_ema, N, n_bonds,
                eta_g, beta_g, dt=0.01,
                fiedler_recompute=do_fiedler or v2_ema is None)

        # Update K (no vortex)
        cos_dr = d9.cos_dressed_all(R, V, ei, ej)
        K = d9.compute_K_product(cos_dphi_indep, cos_dr, K0)

        # Reorthogonalize
        if sweep % reortho_every == 0:
            R = d9.reorthogonalize(R)
            for b in range(n_bonds):
                V[b] = d9.reorthogonalize_single(V[b])

        # Measure
        if sweep % measure_every == 0:
            obs = measure('independence', sweep)
            if verbose:
                print(f"    sweep {sweep:5d}: M_sky={obs['M_sky']:.2f}, "
                      f"M_d={obs['M_dressed']:.2f}, "
                      f"cos_dr={obs['cos_dressed_mean']:.4f}, "
                      f"wl={obs['wilson_mean']:.4f}")

    t_indep = time.time() - t_indep_start
    obs_final = measure('final', n_anneal_sweeps + n_independence_sweeps)

    if verbose:
        print(f"\n    Independence done ({t_indep:.1f}s): "
              f"M_sky={obs_final['M_sky']:.2f}, "
              f"M_dressed={obs_final['M_dressed']:.2f}")

    # dtheta = 0 after independence (all phases zeroed)
    dtheta = np.zeros(n_bonds)

    return {
        'R': R, 'V': V, 'K': K, 'theta': theta,
        'cos_dphi': cos_dphi, 'dtheta': dtheta,
        'history': history,
        'M_sky_final': obs_final['M_sky'],
        'M_dressed_final': obs_final['M_dressed'],
    }


# =====================================================================
# Zeeman term: on-site spin coupling to external B field
# =====================================================================
# LZ eigenvalues for spin-1 triplet: m = +1, 0, -1
_LZ_DIAG = np.array([1.0, 0.0, -1.0])

def add_zeeman_term(H, N, B_z, mu_Z):
    """Add on-site Zeeman term: H[3i+a, 3i+a] += -mu_Z * B_z * Lz[a].

    This breaks time-reversal symmetry, enabling linear Zeeman splitting.
    The Pauli spin coupling (automatic in Dirac equation) must be added
    explicitly in tight-binding models.

    mu_Z: Zeeman coupling strength (lattice Bohr magneton × g_tree).
          Set to bandwidth for natural normalization with extract_gfactor.
    """
    for a in range(3):
        idx = 3 * np.arange(N) + a
        H[idx, idx] += -mu_Z * B_z * _LZ_DIAG[a]
    return H


# =====================================================================
# Adiabatic gauge ramp helper
# =====================================================================

def _adiabatic_gauge_ramp(
        lat, state, mod,
        B_values, E_living_Kgauge, ov_Kgauge,
        E_triplet_Kgauge,
        E_K_field, E_Wilson,
        K0, dtheta, K_eq, V_eq, R_eq,
        ei, ej, N, n_bonds, positions,
        site_nbr, site_bond,
        plaq_bonds, plaq_signs, btp_arr, btp_cnt,
        A_sites, B_sites,
        e1, e2, e3, R_center, R_ring,
        psi_ref, peierls_mode,
        n_ramp_steps, dt_ramp_gauge, sigma_ramp, beta_ramp,
        eta_g, beta_g,
        pruning_mode='hard', T_death=3, eps_death=0.01,
        K_seed=1e-4, r_eff=6.0, T_soft=0.01,
        flux_mode='peierls',
        n_pre_equil=0,
        freeze_R_preequil=False,
        freeze_R_ramp=False,
        mu_Z=0.0,
        rng=None, verbose=True):
    """Adiabatic B-ramp for K_gauge channel.

    Ramps B from 0 outward in both directions, warm-starting each
    B point from the previous one. This avoids the large V jumps
    of joint_reequilibrate and maintains state tracking continuity.

    pruning_mode:
        'hard'    — v2 behavior: K = K0 * max(0, elig). Binary clip.
        'dynamic' — CLR-grounded: death timers + revival at B transitions.

    flux_mode:
        'peierls' — B enters Hamiltonian via Peierls phases (external).
        'native'  — B enters through phase field; Hamiltonian uses
                    dtheta_B = dtheta + A_B directly (LT-34).
    """
    N_B = len(B_values)
    i_zero = N_B // 2  # B=0 index (symmetric grid)
    use_dynamic = (pruning_mode == 'dynamic')
    use_soft = (pruning_mode == 'soft')
    death_thresh = 4.0 / r_eff  # CLR death threshold: elig < 4/r

    def _ramp_half(index_range, label):
        """Ramp from B=0 outward along index_range."""
        K_live = K_eq_pruned.copy() if use_soft else K_eq.copy()
        V_live = V_eq.copy()
        R_live = R_eq.copy()
        v2_ema = None
        psi_ref_kg = psi_ref_soft.copy() if use_soft else psi_ref.copy()

        # Copy triplet refs for this ramp half (each half starts from B=0)
        trip_refs_local = None
        if psi_trip_refs_kg is not None:
            trip_refs_local = [r.copy() for r in psi_trip_refs_kg]

        # Dynamic pruning state (persists across B points in this half)
        death_timer = np.zeros(n_bonds, dtype=np.int32)
        is_dead = np.zeros(n_bonds, dtype=bool)

        for ib in index_range:
            t_b = time.time()
            B_val = B_values[ib]
            B_vec = B_val * e3
            B_phases = mod.peierls_phases_general(ei, ej, positions, B_vec)

            # --- Revival at B transition (dynamic pruning) ---
            if use_dynamic:
                n_dead_before = int(np.sum(is_dead))
                # Seed dead bonds for re-evaluation
                if n_dead_before > 0:
                    K_live[is_dead] = K_seed
                # Reset all timers and death status
                death_timer[:] = 0
                is_dead[:] = False

            # Gentle re-equilibration: n_ramp_steps iterations
            for step in range(n_ramp_steps):
                # Compute eligibility
                cos_dphi_B = np.cos(mod.wrap(dtheta + B_phases))
                cos_dr = d9.cos_dressed_all(R_live, V_live, ei, ej)
                elig = cos_dphi_B * cos_dr

                if use_soft:
                    # Thermal softening: sigmoid at CLR death threshold
                    # K = K0 * max(0, elig) * sigmoid((elig - 4/r) / T_soft)
                    # Smooths the Heaviside step → dK/dB is finite everywhere
                    # Fixes the delta-function blindspot in g-2 extraction
                    elig_pos = np.maximum(0.0, elig)
                    sigmoid_arg = (elig - death_thresh) / T_soft
                    sigmoid_arg = np.clip(sigmoid_arg, -30, 30)  # prevent overflow
                    sigma = 1.0 / (1.0 + np.exp(-sigmoid_arg))
                    K_live = K0 * elig_pos * sigma
                elif use_dynamic:
                    # Product K with dynamic pruning
                    K_live = K0 * np.maximum(0.0, elig)

                    # Death velocity (K→0 limit of CLR)
                    v_death = elig / 2.0 - 2.0 / r_eff

                    # Update death timers
                    dying = v_death < -eps_death
                    growing = v_death > 0
                    death_timer[dying] += 1
                    death_timer[growing] = 0

                    # Prune bonds that have been dying long enough
                    newly_dead = (death_timer >= T_death) & (~is_dead)
                    is_dead |= newly_dead
                    K_live[is_dead] = 0.0
                else:
                    # Hard pruning (v2 behavior)
                    K_live = K0 * np.maximum(0.0, elig)

                # Gauge CLR (small dt)
                V_live, v2_ema, _ = d9.gauge_clr_step(
                    R_live, V_live, K_live, ei, ej,
                    plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                    v2_ema, N, n_bonds,
                    eta_g=eta_g, beta_g=beta_g, dt=dt_ramp_gauge,
                    fiedler_recompute=(step == 0))

                # Frame MC (conservative, every 2 steps, checkerboard)
                # Skip when freeze_R_ramp: isolates V response to B (Schwinger level)
                if not freeze_R_ramp and step % 2 == 0:
                    active = A_sites if (step // 2) % 2 == 0 else B_sites
                    R_live, _, _ = d9.frame_mc_sweep(
                        R_live, K_live, V_live, ei, ej,
                        site_nbr, site_bond,
                        beta_ramp, sigma_ramp, N, rng,
                        active_sites=active)

            # Final K update after equilibration
            cos_dphi_B = np.cos(mod.wrap(dtheta + B_phases))
            cos_dr = d9.cos_dressed_all(R_live, V_live, ei, ej)
            elig_final = cos_dphi_B * cos_dr
            if use_soft:
                elig_pos = np.maximum(0.0, elig_final)
                sigmoid_arg = np.clip((elig_final - death_thresh) / T_soft, -30, 30)
                sigma_f = 1.0 / (1.0 + np.exp(-sigmoid_arg))
                K_live = K0 * elig_pos * sigma_f
            else:
                K_live = K0 * np.maximum(0.0, elig_final)
                if use_dynamic:
                    K_live[is_dead] = 0.0

            # Diagonalize and track bound state
            use_native_flux = (flux_mode == 'native')
            h_dt = mod.wrap(dtheta + B_phases) if use_native_flux else dtheta
            h_bp = None if use_native_flux else B_phases
            h_pm = 'u1' if use_native_flux else peierls_mode
            H_kg = v2.build_gauge_dressed_hamiltonian_3N(
                N, ei, ej, K_live, h_dt, V_live,
                B_phases=h_bp, peierls_mode=h_pm)
            if mu_Z > 0:
                add_zeeman_term(H_kg, N, B_val, mu_Z)
            evals_kg, evecs_kg = np.linalg.eigh(H_kg)
            idx_kg, _, ov_kg = v2.select_bound_state_3N(
                evals_kg, evecs_kg, N, ei, ej, positions,
                e1, e2, e3, R_center, R_ring, psi_ref=psi_ref_kg)
            E_living_Kgauge[ib] = evals_kg[idx_kg]
            ov_Kgauge[ib] = ov_kg

            # Triplet extraction via continuous overlap tracking
            # (tracks 3 states by overlap with previous B point, not re-identifying)
            if E_triplet_Kgauge is not None and trip_refs_local is not None:
                for m in range(3):
                    ov_m = np.abs(evecs_kg.conj().T @ trip_refs_local[m]) ** 2
                    best_m = int(np.argmax(ov_m))
                    E_triplet_Kgauge[ib, m] = evals_kg[best_m]
                    # Update ref for next B point (adiabatic tracking)
                    trip_refs_local[m] = evecs_kg[:, best_m].copy()

            # Vacuum energy: K-field CLR potential over MARGINAL bonds only
            # Marginal bonds = those near the death boundary where vacuum
            # reorganization under B is physically meaningful.
            # Full lattice sum is extensive and drowns the intensive eigenvalue.
            K_pos = np.maximum(K_live, 1e-30)
            e_k_per_bond = K_pos**2 / (2.0 * r_eff) - np.log(bessel_i0(K_pos))
            marginal_mask = (elig_final > -0.1) & (elig_final < death_thresh + 0.1)
            e_k = np.sum(e_k_per_bond[marginal_mask])
            # Wilson: use full lattice (gauge links respond collectively)
            wl_mean = d9.wilson_mean(V_live, plaq_bonds, plaq_signs)
            n_plaq = plaq_bonds.shape[0]
            e_w = -beta_g * wl_mean * n_plaq
            E_K_field[ib] = e_k
            E_Wilson[ib] = e_w

            # Update psi_ref for next B point (adiabatic tracking)
            psi_ref_kg = evecs_kg[:, idx_kg].copy()

            # Diagnostic
            delta_V_norm = np.linalg.norm(V_live - V_eq) / max(np.linalg.norm(V_eq), 1e-12)
            n_dead_now = int(np.sum(is_dead)) if use_dynamic else int(np.sum(K_live < 1e-6))
            n_marginal = int(np.sum((elig_final > 0) & (elig_final < death_thresh)))
            dt_b = time.time() - t_b
            if verbose:
                extra = ""
                if use_dynamic:
                    extra = f" dead={n_dead_now} marg={n_marginal}"
                elif use_soft:
                    n_suppressed = int(np.sum((elig_final > 0) & (K_live < 0.1 * K0 * elig_final)))
                    extra = f" marg={n_marginal} supp={n_suppressed}"
                print(f"  {label} B[{ib}]={B_val:+.2e}: "
                      f"E_kg={E_living_Kgauge[ib]:.6f} "
                      f"ov_kg={ov_Kgauge[ib]:.3f} "
                      f"dV={delta_V_norm:.4e} "
                      f"K_mean={np.mean(K_live):.4f}"
                      f"{extra} ({dt_b:.1f}s)")

    # B=0 point: apply same pruning mode to K_eq for consistent reference
    use_native_flux = (flux_mode == 'native')
    eff_pm = 'u1' if use_native_flux else peierls_mode
    B_phases_zero = mod.peierls_phases_general(
        ei, ej, positions, B_values[i_zero] * e3)
    h_dt0 = mod.wrap(dtheta + B_phases_zero) if use_native_flux else dtheta
    h_bp0 = None if use_native_flux else B_phases_zero

    # Apply pruning-consistent K at B=0
    if use_soft:
        cos_dphi_0 = np.cos(mod.wrap(dtheta + B_phases_zero))
        cos_dr_0 = d9.cos_dressed_all(R_eq, V_eq, ei, ej)
        elig_0 = cos_dphi_0 * cos_dr_0
        elig_0_pos = np.maximum(0.0, elig_0)
        sig_0 = 1.0 / (1.0 + np.exp(-np.clip((elig_0 - death_thresh) / T_soft, -30, 30)))
        K_eq_pruned = K0 * elig_0_pos * sig_0
        if verbose:
            n_zombie = int(np.sum((elig_0 > 0) & (elig_0 < death_thresh)))
            print(f"  Soft pruning at B=0: {n_zombie} zombie bonds suppressed")
    else:
        K_eq_pruned = K_eq

    # --- Pre-equilibration: let V/R converge under soft-pruned K at B=0 ---
    if n_pre_equil > 0 and use_soft:
        freeze_R_label = " (R frozen)" if freeze_R_preequil else ""
        if verbose:
            print(f"  Pre-equilibrating V at B=0 ({n_pre_equil} steps){freeze_R_label}...")
        V_pre = V_eq.copy()
        R_pre = R_eq.copy()
        v2_pre = None
        t_pre = time.time()

        # With frozen R, K is fixed — compute once
        if freeze_R_preequil:
            cos_dr_frozen = d9.cos_dressed_all(R_pre, V_pre, ei, ej)
            cos_dphi_frozen = np.cos(dtheta)
            elig_frozen = cos_dphi_frozen * cos_dr_frozen
            elig_frozen_pos = np.maximum(0.0, elig_frozen)
            sig_frozen = 1.0 / (1.0 + np.exp(-np.clip(
                (elig_frozen - death_thresh) / T_soft, -30, 30)))
            K_frozen = K0 * elig_frozen_pos * sig_frozen

        for step in range(n_pre_equil):
            if freeze_R_preequil:
                K_pre = K_frozen  # fixed K throughout
            else:
                # Update K with current R (soft-pruned)
                cos_dr_pre = d9.cos_dressed_all(R_pre, V_pre, ei, ej)
                cos_dphi_pre = np.cos(dtheta)
                elig_pre = cos_dphi_pre * cos_dr_pre
                elig_pre_pos = np.maximum(0.0, elig_pre)
                sig_pre = 1.0 / (1.0 + np.exp(-np.clip(
                    (elig_pre - death_thresh) / T_soft, -30, 30)))
                K_pre = K0 * elig_pre_pos * sig_pre

            # Gauge CLR step
            V_pre, v2_pre, _ = d9.gauge_clr_step(
                R_pre, V_pre, K_pre, ei, ej,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                v2_pre, N, n_bonds,
                eta_g=eta_g, beta_g=beta_g, dt=dt_ramp_gauge,
                fiedler_recompute=(step % 10 == 0))

            # Frame MC (only when R is not frozen)
            if not freeze_R_preequil and step % 2 == 0:
                active = A_sites if (step // 2) % 2 == 0 else B_sites
                R_pre, _, _ = d9.frame_mc_sweep(
                    R_pre, K_pre, V_pre, ei, ej,
                    site_nbr, site_bond,
                    beta_ramp, sigma_ramp, N, rng,
                    active_sites=active)

            if verbose and (step + 1) % 100 == 0:
                dV_pre = np.linalg.norm(V_pre - V_eq) / max(np.linalg.norm(V_eq), 1e-12)
                print(f"    step {step+1}: dV={dV_pre:.6f}, K_mean={np.mean(K_pre):.4f}")

        dt_pre = time.time() - t_pre
        dV_final = np.linalg.norm(V_pre - V_eq) / max(np.linalg.norm(V_eq), 1e-12)
        if verbose:
            print(f"  Pre-equil done: dV={dV_final:.6f} ({dt_pre:.1f}s)")

        # Replace V_eq (and R_eq if not frozen) with converged state
        V_eq = V_pre
        if not freeze_R_preequil:
            R_eq = R_pre

        # Recompute K_eq_pruned with updated R/V
        cos_dr_new = d9.cos_dressed_all(R_eq, V_eq, ei, ej)
        cos_dphi_new = np.cos(dtheta)
        elig_new = cos_dphi_new * cos_dr_new
        elig_new_pos = np.maximum(0.0, elig_new)
        sig_new = 1.0 / (1.0 + np.exp(-np.clip(
            (elig_new - death_thresh) / T_soft, -30, 30)))
        K_eq_pruned = K0 * elig_new_pos * sig_new
        if verbose:
            n_zombie_new = int(np.sum((elig_new > 0) & (elig_new < death_thresh)))
            print(f"  After pre-equil: {n_zombie_new} zombie bonds, K_mean={np.mean(K_eq_pruned):.4f}")

    # Build scalar (NxN) reference for triplet tracking
    psi_scalar_ref_kg = None
    if E_triplet_Kgauge is not None:
        H_scalar_kg = np.zeros((N, N), dtype=complex)
        scalar_hop_kg = -K_eq_pruned * np.exp(1j * dtheta)
        np.add.at(H_scalar_kg, (ei, ej), scalar_hop_kg)
        np.add.at(H_scalar_kg, (ej, ei), np.conj(scalar_hop_kg))
        evals_s_kg, evecs_s_kg = np.linalg.eigh(H_scalar_kg)
        # Find scalar bound state near core
        dr = positions - R_center[np.newaxis, :]
        dist = np.linalg.norm(dr, axis=1)
        nn_d = np.linalg.norm(positions[ei[0]] - positions[ej[0]])
        core_cut = max(2.0 * nn_d, 1.0)
        core_mask = dist < core_cut
        rfrac_s_kg = np.array([
            np.sum(np.abs(evecs_s_kg[:, s]) ** 2 * core_mask)
            for s in range(len(evals_s_kg))])
        bw_s = max(np.max(np.abs(evals_s_kg)), 1e-12)
        central_s = np.abs(evals_s_kg) < 0.2 * bw_s
        if np.any(central_s):
            cand = np.where(central_s)[0]
            best_s = cand[np.argmax(rfrac_s_kg[cand])]
        else:
            best_s = int(np.argmax(rfrac_s_kg))
        psi_scalar_ref_kg = evecs_s_kg[:, best_s]
        if verbose:
            print(f"  Scalar ref (soft K): E={evals_s_kg[best_s]:.6f}, "
                  f"rfrac={rfrac_s_kg[best_s]:.4f}")

    H_kg0 = v2.build_gauge_dressed_hamiltonian_3N(
        N, ei, ej, K_eq_pruned, h_dt0, V_eq,
        B_phases=h_bp0, peierls_mode=eff_pm)
    evals_kg0, evecs_kg0 = np.linalg.eigh(H_kg0)
    idx_kg0, _, ov_kg0 = v2.select_bound_state_3N(
        evals_kg0, evecs_kg0, N, ei, ej, positions,
        e1, e2, e3, R_center, R_ring, psi_ref=psi_ref)
    E_living_Kgauge[i_zero] = evals_kg0[idx_kg0]
    ov_Kgauge[i_zero] = ov_kg0

    # Triplet extraction at B=0 — identify the 3 triplet states
    # and store their wavefunctions for continuous overlap tracking
    psi_trip_refs_kg = None  # will hold 3 wavefunctions for overlap tracking
    if E_triplet_Kgauge is not None and psi_scalar_ref_kg is not None:
        Et_kg0, ov_trip0 = v2.select_triplet_states_3N(
            evals_kg0, evecs_kg0, N, psi_scalar_ref_kg)
        E_triplet_Kgauge[i_zero] = Et_kg0
        # Store wavefunctions for continuous overlap tracking through B ramp
        psi_trip_refs_kg = []
        idx_site = np.arange(N)
        for m in range(3):
            psi_sector = np.zeros(3 * N, dtype=complex)
            psi_sector[3 * idx_site + m] = psi_scalar_ref_kg
            norm_sq = np.sum(np.abs(psi_sector) ** 2)
            if norm_sq > 1e-20:
                psi_sector /= np.sqrt(norm_sq)
            overlaps = np.abs(evecs_kg0.conj().T @ psi_sector) ** 2
            best = int(np.argmax(overlaps))
            psi_trip_refs_kg.append(evecs_kg0[:, best].copy())
        if verbose:
            print(f"  B=0 triplet (K_gauge): [{Et_kg0[0]:.6f}, {Et_kg0[1]:.6f}, {Et_kg0[2]:.6f}]"
                  f"  ov=[{ov_trip0[0]:.3f},{ov_trip0[1]:.3f},{ov_trip0[2]:.3f}]")

    # Update psi_ref to soft-pruned bound state (for adiabatic tracking)
    psi_ref_soft = evecs_kg0[:, idx_kg0].copy()

    # Vacuum energy at B=0 — marginal bonds only (consistent with ramp)
    K_ref = np.maximum(K_eq_pruned, 1e-30)
    death_thresh_0 = 4.0 / r_eff
    B_phases_z = mod.peierls_phases_general(ei, ej, positions, B_values[i_zero] * e3)
    cos_dphi_z = np.cos(mod.wrap(dtheta + B_phases_z))
    cos_dr_z = d9.cos_dressed_all(R_eq, V_eq, ei, ej)
    elig_z = cos_dphi_z * cos_dr_z
    marginal_z = (elig_z > -0.1) & (elig_z < death_thresh_0 + 0.1)
    e_k_z = K_ref**2 / (2.0 * r_eff) - np.log(bessel_i0(K_ref))
    E_K_field[i_zero] = np.sum(e_k_z[marginal_z])
    n_plaq = plaq_bonds.shape[0]
    wl0 = d9.wilson_mean(V_eq, plaq_bonds, plaq_signs)
    E_Wilson[i_zero] = -beta_g * wl0 * n_plaq

    if verbose:
        e_vac_0 = E_K_field[i_zero] + E_Wilson[i_zero]
        print(f"  B=0 [i={i_zero}]: E_kg={E_living_Kgauge[i_zero]:.6f} "
              f"ov_kg={ov_Kgauge[i_zero]:.3f} "
              f"E_K={E_K_field[i_zero]:.4f} E_W={E_Wilson[i_zero]:.4f} "
              f"E_vac={e_vac_0:.4f}")

    # Forward ramp: B=0 → +B_max (indices i_zero+1 → N_B-1)
    if i_zero + 1 < N_B:
        _ramp_half(range(i_zero + 1, N_B), "fwd")

    # Backward ramp: B=0 → -B_max (indices i_zero-1 → 0)
    if i_zero - 1 >= 0:
        _ramp_half(range(i_zero - 1, -1, -1), "bwd")


# =====================================================================
# Phase 3: B-scan with 3N gauge-dressed Hamiltonian
# =====================================================================

def run_b_scan_d9(lat, state, mod,
                  B_max=0.001, N_B=11, K0=8.0,
                  eta_clr=1.0, lam=0.1, dt_clr=0.01,
                  eta_g=0.1, beta_g=2.0, dt_gauge=0.01,
                  beta_mc=20.0, sigma_mc=0.05,
                  n_reequil_steps=200, n_reequil_mc=50,
                  peierls_mode='u1_su2',
                  flux_mode='peierls',
                  living_mode='adiabatic',
                  n_ramp_steps=10, dt_ramp_gauge=0.005,
                  sigma_ramp=0.02, beta_ramp=50.0,
                  pruning_mode='hard', T_death=3,
                  eps_death=0.01, K_seed=1e-4, r_eff=6.0,
                  T_soft=0.01, n_pre_equil=0,
                  freeze_R_preequil=False,
                  freeze_R_ramp=False,
                  mu_Z=0.0,
                  verbose=True):
    """Run frozen + living B-scan using D9 equilibrium state.

    Frozen channel: fixed K, V, R — only Peierls phases change.
    K_only channel: product-K update cos(dtheta+B)*cos_dr, V/R frozen.
    K_gauge channel: full K+V+R re-equilibration.
        living_mode='joint': v1 joint_reequilibrate (from scratch each B).
        living_mode='adiabatic': v2 warm-start ramp from B=0 outward.

    flux_mode:
        'peierls': B enters through Peierls phases in Hamiltonian builder
                   (borrowed condensed-matter construction).
        'native':  B enters through the phase field itself (dtheta += A_B).
                   The Hamiltonian uses the equilibrated state directly
                   with NO extra Peierls phases. Lattice-native: the
                   magnetic field IS the phase circulation (LT-34).
                   K/V/R still respond self-consistently in living channels.
    """
    N = lat['N']
    n_bonds = lat['n_bonds']
    positions = lat['positions']
    ei, ej = lat['ei'], lat['ej']
    site_nbr = lat['site_nbr']
    site_bond = lat['site_bond']
    sublat = lat['sublat']
    center = lat['center']
    plaq_bonds = lat['plaq_bonds']
    plaq_signs = lat['plaq_signs']
    btp_arr, btp_cnt = lat['btp_arr'], lat['btp_cnt']

    K_eq = state['K']
    V_eq = state['V']
    R_eq = state['R']
    dtheta = state['dtheta']

    # Lattice frame for B-field direction
    deltas = mod.make_simplex_deltas(3)
    e1, e2, e3, k0, _ = mod.get_3d_frame(deltas)
    R_center = center
    R_ring = 0.0  # vortex line, not ring

    B_values = np.linspace(-B_max, B_max, N_B)
    E_frozen = np.zeros(N_B)
    E_living_Konly = np.zeros(N_B)
    E_living_Kgauge = np.zeros(N_B)

    use_native = (flux_mode == 'native')
    # In native mode, peierls_mode is forced to 'u1' (flux in phase field)
    effective_peierls = 'u1' if use_native else peierls_mode
    is_su2 = effective_peierls in ('su2', 'u1_su2')
    E_triplet_frozen = np.zeros((N_B, 3)) if is_su2 else None
    E_triplet_Kgauge = np.zeros((N_B, 3)) if is_su2 else None

    ov_frozen = np.zeros(N_B)
    ov_Konly = np.zeros(N_B)
    ov_Kgauge = np.zeros(N_B)
    E_K_field = np.zeros(N_B)
    E_Wilson_arr = np.zeros(N_B)

    # ---- B=0 reference ----
    if verbose:
        print(f"\n  Building B=0 reference Hamiltonian (dim={3*N})...")

    H0 = v2.build_gauge_dressed_hamiltonian_3N(
        N, ei, ej, K_eq, dtheta, V_eq,
        peierls_mode=effective_peierls, debug=True)
    evals0, evecs0 = np.linalg.eigh(H0)
    bandwidth = evals0[-1] - evals0[0]

    # Resolve mu_Z = -1 → bandwidth
    if mu_Z < 0:
        mu_Z = bandwidth
    if mu_Z > 0 and verbose:
        print(f"  Zeeman coupling: mu_Z = {mu_Z:.4f}")

    idx0, rfrac0, _ = v2.select_bound_state_3N(
        evals0, evecs0, N, ei, ej, positions,
        e1, e2, e3, R_center, R_ring)
    psi_ref = evecs0[:, idx0]
    E_ref = evals0[idx0]

    # Scalar reference for triplet tracking (SU(2) modes)
    psi_scalar_ref = None
    if is_su2:
        H_scalar = np.zeros((N, N), dtype=complex)
        scalar_hop = -K_eq * np.exp(1j * dtheta)
        np.add.at(H_scalar, (ei, ej), scalar_hop)
        np.add.at(H_scalar, (ej, ei), np.conj(scalar_hop))
        evals_s, evecs_s = np.linalg.eigh(H_scalar)

        # Find scalar bound state near core (low-K region)
        dr = positions - R_center[np.newaxis, :]
        dist = np.linalg.norm(dr, axis=1)
        nn_d = np.linalg.norm(positions[ei[0]] - positions[ej[0]])
        core_cut = max(2.0 * nn_d, 1.0)
        core_mask = dist < core_cut

        rfrac_s = np.array([
            np.sum(np.abs(evecs_s[:, s]) ** 2 * core_mask)
            for s in range(len(evals_s))
        ])
        bw_s = max(np.max(np.abs(evals_s)), 1e-12)
        central_s = np.abs(evals_s) < 0.2 * bw_s
        if np.any(central_s):
            cand = np.where(central_s)[0]
            best_s = cand[np.argmax(rfrac_s[cand])]
        else:
            best_s = int(np.argmax(rfrac_s))
        psi_scalar_ref = evecs_s[:, best_s]

        if verbose:
            print(f"  Scalar bound state: E={evals_s[best_s]:.6f}, "
                  f"rfrac={rfrac_s[best_s]:.4f}")

    if verbose:
        print(f"  B=0: E_ref={E_ref:.6f}, rfrac={rfrac0:.4f}, "
              f"bandwidth={bandwidth:.4f}")

    # Pre-compute equilibrium cos_dr for K_only channel
    cos_dr_eq = d9.cos_dressed_all(R_eq, V_eq, ei, ej)

    # Compute soft-pruned reference eigenstate for K_only overlap tracking
    psi_ref_soft = psi_ref  # default: use hard reference
    if pruning_mode == 'soft':
        cos_dphi_eq = np.cos(dtheta)
        elig_eq = cos_dphi_eq * cos_dr_eq
        elig_eq_pos = np.maximum(0.0, elig_eq)
        dt_eq = 4.0 / r_eff
        sig_eq = 1.0 / (1.0 + np.exp(-np.clip((elig_eq - dt_eq) / T_soft, -30, 30)))
        K_soft_eq = K0 * elig_eq_pos * sig_eq
        H_soft0 = v2.build_gauge_dressed_hamiltonian_3N(
            N, ei, ej, K_soft_eq, dtheta, V_eq,
            peierls_mode=peierls_mode)
        evals_soft0, evecs_soft0 = np.linalg.eigh(H_soft0)
        idx_soft0, _, _ = v2.select_bound_state_3N(
            evals_soft0, evecs_soft0, N, ei, ej, positions,
            e1, e2, e3, R_center, R_ring, psi_ref=psi_ref)
        psi_ref_soft = evecs_soft0[:, idx_soft0].copy()
        if verbose:
            print(f"  Soft K_only ref: E={evals_soft0[idx_soft0]:.6f}")

    # Sublattice sets for checkerboard MC in adiabatic mode
    A_sites = np.where(lat['sublat'] == 0)[0]
    B_sites = np.where(lat['sublat'] == 1)[0]

    rng = np.random.default_rng(123)

    # ---- Frozen + K_only B-scan (all B points) ----
    for ib, B_val in enumerate(B_values):
        B_vec = B_val * e3
        B_phases = mod.peierls_phases_general(ei, ej, positions, B_vec)
        dtheta_B = mod.wrap(dtheta + B_phases)

        # --- Hamiltonian phase field: native vs peierls ---
        if use_native:
            # Lattice-native: flux IS the phase field (LT-34)
            h_dtheta_f = dtheta_B   # frozen: flux in phases, K/V/R unchanged
            h_bphases_f = None
            h_dtheta_ko = dtheta_B  # K_only: K responds, V/R frozen
            h_bphases_ko = None
        else:
            # Peierls: flux enters as external Hamiltonian modification
            h_dtheta_f = dtheta
            h_bphases_f = B_phases
            h_dtheta_ko = dtheta
            h_bphases_ko = B_phases

        # === Frozen channel ===
        H_f = v2.build_gauge_dressed_hamiltonian_3N(
            N, ei, ej, K_eq, h_dtheta_f, V_eq,
            B_phases=h_bphases_f, peierls_mode=effective_peierls)
        if mu_Z > 0:
            add_zeeman_term(H_f, N, B_val, mu_Z)
        evals_f, evecs_f = np.linalg.eigh(H_f)
        idx_f, rfrac_f, ov_f = v2.select_bound_state_3N(
            evals_f, evecs_f, N, ei, ej, positions,
            e1, e2, e3, R_center, R_ring, psi_ref=psi_ref)

        if is_su2 and psi_scalar_ref is not None:
            Et, Eov = v2.select_triplet_states_3N(
                evals_f, evecs_f, N, psi_scalar_ref)
            E_triplet_frozen[ib] = Et
            E_frozen[ib] = Et[0]  # m=+1 state for Zeeman
            ov_frozen[ib] = Eov[0]
        else:
            E_frozen[ib] = evals_f[idx_f]
            ov_frozen[ib] = ov_f

        # === Living channel: K_only (product K update, V/R frozen) ===
        cos_dphi_B = np.cos(dtheta_B)
        elig_ko = cos_dphi_B * cos_dr_eq
        if pruning_mode == 'soft':
            elig_ko_pos = np.maximum(0.0, elig_ko)
            dt_ko = 4.0 / r_eff
            sig_arg_ko = np.clip((elig_ko - dt_ko) / T_soft, -30, 30)
            sig_ko = 1.0 / (1.0 + np.exp(-sig_arg_ko))
            K_live_ko = K0 * elig_ko_pos * sig_ko
        else:
            K_live_ko = K0 * np.maximum(0.0, elig_ko)

        H_ko = v2.build_gauge_dressed_hamiltonian_3N(
            N, ei, ej, K_live_ko, h_dtheta_ko, V_eq,
            B_phases=h_bphases_ko, peierls_mode=effective_peierls)
        if mu_Z > 0:
            add_zeeman_term(H_ko, N, B_val, mu_Z)
        evals_ko, evecs_ko = np.linalg.eigh(H_ko)
        psi_ref_ko = psi_ref_soft if pruning_mode == 'soft' else psi_ref
        idx_ko, _, ov_ko = v2.select_bound_state_3N(
            evals_ko, evecs_ko, N, ei, ej, positions,
            e1, e2, e3, R_center, R_ring, psi_ref=psi_ref_ko)
        E_living_Konly[ib] = evals_ko[idx_ko]
        ov_Konly[ib] = ov_ko

        if verbose:
            trip_str = ""
            if is_su2:
                Et_now = E_triplet_frozen[ib]
                trip_str = f" trip=[{Et_now[0]:.6f},{Et_now[1]:.6f},{Et_now[2]:.6f}]"
            print(f"  B[{ib}]={B_val:+.2e}: "
                  f"E_f={E_frozen[ib]:.6f} "
                  f"E_ko={E_living_Konly[ib]:.6f} "
                  f"ov_f={ov_frozen[ib]:.3f} "
                  f"ov_ko={ov_Konly[ib]:.3f}"
                  f"{trip_str}")

    # ---- K_gauge channel ----
    if living_mode == 'adiabatic':
        if verbose:
            print(f"\n  K_gauge: ADIABATIC ramp (n_ramp_steps={n_ramp_steps}, "
                  f"dt={dt_ramp_gauge}, sigma={sigma_ramp}, beta={beta_ramp})")
            print(f"  pruning_mode={pruning_mode}", end="")
            if pruning_mode == 'dynamic':
                print(f" T_death={T_death} eps_death={eps_death} "
                      f"K_seed={K_seed} r_eff={r_eff} "
                      f"death_thresh={4.0/r_eff:.4f}")
            elif pruning_mode == 'soft':
                print(f" T_soft={T_soft} r_eff={r_eff} "
                      f"death_thresh={4.0/r_eff:.4f}")
            else:
                print()
        _adiabatic_gauge_ramp(
            lat, state, mod,
            B_values, E_living_Kgauge, ov_Kgauge,
            E_triplet_Kgauge,
            E_K_field, E_Wilson_arr,
            K0, dtheta, K_eq, V_eq, R_eq,
            ei, ej, N, n_bonds, positions,
            site_nbr, site_bond,
            plaq_bonds, plaq_signs, btp_arr, btp_cnt,
            A_sites, B_sites,
            e1, e2, e3, R_center, R_ring,
            psi_ref, peierls_mode,
            n_ramp_steps, dt_ramp_gauge, sigma_ramp, beta_ramp,
            eta_g, beta_g,
            pruning_mode=pruning_mode, T_death=T_death,
            eps_death=eps_death, K_seed=K_seed, r_eff=r_eff,
            T_soft=T_soft, flux_mode=flux_mode,
            n_pre_equil=n_pre_equil,
            freeze_R_preequil=freeze_R_preequil,
            freeze_R_ramp=freeze_R_ramp,
            mu_Z=mu_Z,
            rng=rng, verbose=verbose)
    else:
        # v1 joint_reequilibrate mode
        if verbose:
            print(f"\n  K_gauge: JOINT reequilibrate (n_steps={n_reequil_steps})")
        for ib, B_val in enumerate(B_values):
            t_b = time.time()
            B_vec = B_val * e3
            B_phases = mod.peierls_phases_general(ei, ej, positions, B_vec)

            K_live, V_live, R_live, jdiag = v2.joint_reequilibrate(
                mod, K_eq.copy(), V_eq.copy(), R_eq.copy(),
                dtheta, B_phases,
                ei, ej, N, n_bonds, sublat,
                site_nbr, site_bond,
                plaq_bonds, plaq_signs, btp_arr, btp_cnt,
                eta_clr=eta_clr, lam=lam, dt_clr=dt_clr,
                eta_g=eta_g, beta_g=beta_g, dt_gauge=dt_gauge,
                beta_mc=beta_mc, sigma_mc=sigma_mc,
                n_steps=n_reequil_steps, n_mc_equil=n_reequil_mc,
                rng=rng)

            H_kg = v2.build_gauge_dressed_hamiltonian_3N(
                N, ei, ej, K_live, dtheta, V_live,
                B_phases=B_phases, peierls_mode=peierls_mode)
            evals_kg, evecs_kg = np.linalg.eigh(H_kg)
            idx_kg, _, ov_kg = v2.select_bound_state_3N(
                evals_kg, evecs_kg, N, ei, ej, positions,
                e1, e2, e3, R_center, R_ring, psi_ref=psi_ref)
            E_living_Kgauge[ib] = evals_kg[idx_kg]
            ov_Kgauge[ib] = ov_kg

            dt_b = time.time() - t_b
            if verbose:
                dv_str = f" dV={jdiag['delta_V_norm']:.4e}" if 'delta_V_norm' in jdiag else ""
                print(f"  B[{ib}]={B_val:+.2e}: "
                      f"E_kg={E_living_Kgauge[ib]:.6f} "
                      f"ov_kg={ov_Kgauge[ib]:.3f}"
                      f"{dv_str} ({dt_b:.1f}s)")

    return {
        'B_values': B_values,
        'E_frozen': E_frozen,
        'E_living_Konly': E_living_Konly,
        'E_living_Kgauge': E_living_Kgauge,
        'E_triplet_frozen': E_triplet_frozen,
        'E_triplet_Kgauge': E_triplet_Kgauge,
        'bandwidth': bandwidth,
        'psi_ref': psi_ref,
        'E_ref': E_ref,
        'ov_frozen': ov_frozen,
        'ov_Konly': ov_Konly,
        'ov_Kgauge': ov_Kgauge,
        'E_K_field': E_K_field,
        'E_Wilson': E_Wilson_arr,
        'e1': e1, 'e2': e2, 'e3': e3,
        'peierls_mode': peierls_mode,
    }


# =====================================================================
# Phase 4: Analysis & Output
# =====================================================================

def analyze_and_report(scan, state, verbose=True):
    """Extract g-factors from both living channels and report."""
    B_values = scan['B_values']
    bandwidth = scan['bandwidth']
    peierls_mode = scan['peierls_mode']
    E_trip = scan['E_triplet_frozen']

    # g-factor from K_only channel
    gf_Konly = v2.extract_gfactor(
        B_values, scan['E_frozen'], scan['E_living_Konly'],
        bandwidth, peierls_mode=peierls_mode,
        E_triplet_frozen=E_trip)

    # g-factor from K_gauge channel (eigenvalue only)
    gf_Kgauge = v2.extract_gfactor(
        B_values, scan['E_frozen'], scan['E_living_Kgauge'],
        bandwidth, peierls_mode=peierls_mode,
        E_triplet_frozen=E_trip)

    # g-factor from E_total = E_bound + E_vacuum (eigenvalue + vacuum polarization)
    E_vacuum = scan['E_K_field'] + scan['E_Wilson']
    E_total = scan['E_living_Kgauge'] + E_vacuum
    gf_Etotal = v2.extract_gfactor(
        B_values, scan['E_frozen'], E_total,
        bandwidth, peierls_mode=peierls_mode,
        E_triplet_frozen=E_trip)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  g-FACTOR RESULTS (peierls_mode={peierls_mode})")
        print(f"{'='*60}")
        print(f"  Bandwidth: {bandwidth:.4f}")
        print(f"  M_sky final: {state['M_sky_final']:.2f}")
        print(f"  M_dressed final: {state['M_dressed_final']:.2f}")

        print(f"\n  --- Frozen channel ---")
        print(f"    a1_frozen = {gf_Konly['a1_frozen']:.6e}")
        print(f"    a2_frozen = {gf_Konly['a2_frozen']:.6e}")

        if 'triplet_slope' in gf_Konly:
            print(f"    triplet_slope = {gf_Konly['triplet_slope']:.6e}")
            print(f"    g_triplet_frozen = "
                  f"{gf_Konly.get('g_triplet_frozen', 0):.6f}")

        print(f"\n  --- K_only channel ---")
        print(f"    a1_living  = {gf_Konly['a1_living']:.6e}")
        print(f"    da1        = {gf_Konly['da1']:.6e}")
        print(f"    a_e        = {gf_Konly['a_e']:.6e}")
        print(f"    g          = {gf_Konly['g']:.8f}")
        print(f"    ratio_to_schwinger = {gf_Konly['ratio_to_schwinger']:.4f}")

        print(f"\n  --- K_gauge channel (eigenvalue only) ---")
        print(f"    a1_living  = {gf_Kgauge['a1_living']:.6e}")
        print(f"    da1        = {gf_Kgauge['da1']:.6e}")
        print(f"    a_e        = {gf_Kgauge['a_e']:.6e}")
        print(f"    g          = {gf_Kgauge['g']:.8f}")
        print(f"    ratio_to_schwinger = {gf_Kgauge['ratio_to_schwinger']:.4f}")

    # K_gauge triplet splitting analysis
    E_trip_kg = scan.get('E_triplet_Kgauge')
    gf_Kgauge_triplet = None
    if E_trip_kg is not None and np.any(E_trip_kg != 0):
        is_su2 = peierls_mode in ('su2', 'u1_su2')
        if is_su2:
            i_zero = len(B_values) // 2
            dE_trip_kg = E_trip_kg[:, 0] - E_trip_kg[:, 2]  # m=+1 minus m=-1
            dE_B0 = dE_trip_kg[i_zero]

            # Raw fit (includes B=0 splitting)
            pf_raw = np.polyfit(B_values, dE_trip_kg, 1)
            trip_slope_raw = pf_raw[0]

            # Subtracted fit: remove B=0 splitting to isolate B-dependent part
            dE_sub = dE_trip_kg - dE_B0
            pf_sub = np.polyfit(B_values, dE_sub, 1)
            trip_slope_sub = pf_sub[0]

            # Individual component linear slopes (quadratic fit, extract a1)
            comp_slopes = []
            comp_labels = ['m=+1', 'm=0', 'm=-1']
            for m in range(3):
                E_m = E_trip_kg[:, m] - E_trip_kg[i_zero, m]
                pf_m = np.polyfit(B_values, E_m, 2)
                comp_slopes.append(pf_m[1])  # linear coefficient

            # g from subtracted splitting slope
            g_trip_kg = abs(trip_slope_sub) / bandwidth if bandwidth > 1e-12 else 0.0
            # g from individual component difference (more robust)
            g_comp = abs(comp_slopes[0] - comp_slopes[2]) / bandwidth if bandwidth > 1e-12 else 0.0

            # Schwinger comparison
            K_BKT = 2.0 / np.pi
            from scipy.special import iv as I_bessel
            R0_BKT = float(I_bessel(1, K_BKT) / I_bessel(0, K_BKT))
            alpha_D = R0_BKT * (2.0 / np.pi) ** 4
            schwinger_ae = alpha_D / (2.0 * np.pi)
            a_e_trip_kg = (g_trip_kg - 2.0) / 2.0 if g_trip_kg > 2.0 else 0.0
            ratio_trip_kg = a_e_trip_kg / schwinger_ae if schwinger_ae > 1e-15 else 0.0
            a_e_comp = (g_comp - 2.0) / 2.0 if g_comp > 2.0 else 0.0
            ratio_comp = a_e_comp / schwinger_ae if schwinger_ae > 1e-15 else 0.0
            gf_Kgauge_triplet = {
                'triplet_slope_raw': trip_slope_raw,
                'triplet_slope_sub': trip_slope_sub,
                'g_triplet_Kgauge': g_trip_kg,
                'g_comp': g_comp,
                'a_e_triplet': a_e_trip_kg,
                'a_e_comp': a_e_comp,
                'ratio_to_schwinger': ratio_trip_kg,
                'ratio_comp': ratio_comp,
                'dE_B0': dE_B0,
                'comp_slopes': comp_slopes,
                'dE_triplet': dE_trip_kg,
            }
            if verbose:
                print(f"\n  --- K_gauge triplet splitting ---")
                print(f"    dE(m=+1 - m=-1) at B=0: {dE_B0:.6f}")
                print(f"    slope_raw (with B=0 offset) = {trip_slope_raw:.6e}")
                print(f"    slope_sub (B=0 subtracted)  = {trip_slope_sub:.6e}")
                print(f"    g_triplet (subtracted) = {g_trip_kg:.8f}")
                print(f"    ratio_to_schwinger     = {ratio_trip_kg:.4f}")
                print(f"    --- per-component linear slopes ---")
                for m in range(3):
                    print(f"      {comp_labels[m]}: a1 = {comp_slopes[m]:.6e}")
                print(f"    g_comp (|a1(+1)-a1(-1)|/bw) = {g_comp:.8f}")
                print(f"    ratio_comp = {ratio_comp:.4f}")

    if verbose:
        print(f"\n  --- E_total channel (eigenvalue + vacuum) ---")
        print(f"    E_vac(B=0) = {E_vacuum[len(B_values)//2]:.4f}")
        print(f"    dE_vac range = {np.max(E_vacuum)-np.min(E_vacuum):.6e}")
        print(f"    a1_living  = {gf_Etotal['a1_living']:.6e}")
        print(f"    da1        = {gf_Etotal['da1']:.6e}")
        print(f"    a_e        = {gf_Etotal['a_e']:.6e}")
        print(f"    g          = {gf_Etotal['g']:.8f}")
        print(f"    ratio_to_schwinger = {gf_Etotal['ratio_to_schwinger']:.4f}")

        print(f"\n  --- Reference ---")
        print(f"    alpha_D       = {gf_Konly['alpha_D']:.6e}")
        print(f"    Schwinger a_e = {gf_Konly['schwinger_ae']:.6e}")
        print(f"{'='*60}")

    return {'gf_Konly': gf_Konly, 'gf_Kgauge': gf_Kgauge, 'gf_Etotal': gf_Etotal,
            'gf_Kgauge_triplet': gf_Kgauge_triplet}


def make_figure(scan, state, results, L, outpath):
    """Generate diagnostic figure: E(B), triplet splitting, history."""
    B = scan['B_values']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0): E(B) for all channels
    ax = axes[0, 0]
    ax.plot(B * 1e3, scan['E_frozen'], 'ko-', ms=4, label='Frozen')
    ax.plot(B * 1e3, scan['E_living_Konly'], 'b^-', ms=4, label='K_only')
    ax.plot(B * 1e3, scan['E_living_Kgauge'], 'rs-', ms=4, label='K_gauge')
    ax.set_xlabel('B (x1e-3)')
    ax.set_ylabel('E_bound')
    ax.set_title('Bound State Energy vs B')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1): Triplet splitting (if SU(2))
    ax = axes[0, 1]
    is_su2 = scan['peierls_mode'] in ('su2', 'u1_su2')
    if is_su2 and scan['E_triplet_frozen'] is not None:
        Et = scan['E_triplet_frozen']
        dE = Et[:, 0] - Et[:, 2]  # m=+1 minus m=-1
        ax.plot(B * 1e3, dE * 1e6, 'go-', ms=4)
        ax.set_xlabel('B (x1e-3)')
        ax.set_ylabel('E(m=+1) - E(m=-1) (x1e-6)')
        ax.set_title('Triplet Splitting (frozen)')

        # Linear fit
        pf = np.polyfit(B, dE, 1)
        B_fit = np.linspace(B[0], B[-1], 100)
        ax.plot(B_fit * 1e3, np.polyval(pf, B_fit) * 1e6, 'g--',
                alpha=0.5, label=f'slope={pf[0]:.4e}')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'N/A (u1 mode)', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('Triplet Splitting')
    ax.grid(True, alpha=0.3)

    # (1,0): M_sky and M_dressed history
    ax = axes[1, 0]
    hist = state['history']
    sweeps = [h['sweep'] for h in hist]
    m_sky = [h['M_sky'] for h in hist]
    m_dressed = [h['M_dressed'] for h in hist]
    ax.plot(sweeps, m_sky, 'b-', alpha=0.7, label='M_sky')
    ax.plot(sweeps, m_dressed, 'r-', alpha=0.7, label='M_dressed')
    ax.axhline(3.11, color='k', ls='--', alpha=0.3, label='target')
    ax.set_xlabel('Sweep')
    ax.set_ylabel('Topological Charge')
    ax.set_title('Skyrmion History')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): Overlaps
    ax = axes[1, 1]
    ax.plot(B * 1e3, scan['ov_frozen'], 'ko-', ms=4, label='Frozen')
    ax.plot(B * 1e3, scan['ov_Konly'], 'b^-', ms=4, label='K_only')
    ax.plot(B * 1e3, scan['ov_Kgauge'], 'rs-', ms=4, label='K_gauge')
    ax.set_xlabel('B (x1e-3)')
    ax.set_ylabel('Overlap with B=0 ref')
    ax.set_title('State Tracking Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary text
    gf_ko = results['gf_Konly']
    gf_kg = results['gf_Kgauge']
    gf_et = results['gf_Etotal']
    summary = (
        f"L={L}, mode={scan['peierls_mode']}\n"
        f"M_dressed={state['M_dressed_final']:.2f}, BW={scan['bandwidth']:.2f}\n"
        f"K_only:  g={gf_ko['g']:.6f}, ratio={gf_ko['ratio_to_schwinger']:.3f}\n"
        f"K_gauge: g={gf_kg['g']:.6f}, ratio={gf_kg['ratio_to_schwinger']:.3f}\n"
        f"E_total: g={gf_et['g']:.6f}, ratio={gf_et['ratio_to_schwinger']:.3f}\n"
        f"Schwinger a_e={gf_ko['schwinger_ae']:.4e}"
    )
    fig.text(0.02, 0.02, summary, fontsize=8, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'D9 Frame-Sector g-Factor (L={L})', fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {outpath}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='D9 Frame-Sector g-Factor Measurement')
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--K0', type=float, default=8.0)
    parser.add_argument('--K_phi', type=float, default=4.0)
    parser.add_argument('--beta_grow', type=float, default=4.0)
    parser.add_argument('--beta_wilson', type=float, default=2.0)
    parser.add_argument('--frame_init', default='identity',
                        choices=['identity', 'random'])
    parser.add_argument('--n_anneal', type=int, default=9000)
    parser.add_argument('--n_indep', type=int, default=4000)
    parser.add_argument('--B_max', type=float, default=0.001)
    parser.add_argument('--N_B', type=int, default=11)
    parser.add_argument('--peierls_mode', default='u1_su2',
                        choices=['u1', 'su2', 'u1_su2'])
    parser.add_argument('--flux_mode', default='peierls',
                        choices=['peierls', 'native'],
                        help='peierls=external Peierls phases, '
                             'native=lattice-native flux in phase field (LT-34)')
    parser.add_argument('--eta_clr', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--n_reequil', type=int, default=200)
    parser.add_argument('--n_reequil_mc', type=int, default=50)
    parser.add_argument('--living_mode', default='adiabatic',
                        choices=['adiabatic', 'joint'])
    parser.add_argument('--load_state', type=str, default=None,
                        help='Path to saved .npz state file (skip growth)')
    parser.add_argument('--n_ramp_steps', type=int, default=10)
    parser.add_argument('--dt_ramp_gauge', type=float, default=0.005)
    parser.add_argument('--sigma_ramp', type=float, default=0.02)
    parser.add_argument('--beta_ramp', type=float, default=50.0)
    parser.add_argument('--pruning_mode', default='hard',
                        choices=['hard', 'dynamic', 'soft'])
    parser.add_argument('--T_death', type=int, default=3,
                        help='Death timer patience (consecutive dying steps)')
    parser.add_argument('--eps_death', type=float, default=0.01,
                        help='Death velocity threshold')
    parser.add_argument('--K_seed', type=float, default=1e-4,
                        help='Revival seed K value for dead bonds')
    parser.add_argument('--r_eff', type=float, default=6.0,
                        help='Effective SNR (eta/lambda) for death threshold')
    parser.add_argument('--T_soft', type=float, default=0.00126,
                        help='Thermal softening width for soft pruning sigmoid '
                             '(default: T_eff from CLR Langevin, LT-149)')
    parser.add_argument('--n_pre_equil', type=int, default=0,
                        help='Pre-equilibration steps at B=0 under soft-pruned K '
                             'before starting B ramp (0=skip)')
    parser.add_argument('--freeze_R_preequil', action='store_true',
                        help='Freeze R during pre-equilibration (only update V). '
                             'Allows V to converge to fixed K target.')
    parser.add_argument('--freeze_R_ramp', action='store_true',
                        help='Freeze R during B ramp (only update V via CLR). '
                             'Isolates V response to B; correct for Schwinger.')
    parser.add_argument('--mu_Z', type=float, default=0.0,
                        help='Zeeman coupling strength. >0 adds on-site '
                             'H_Z = -mu_Z*B*Lz to break time-reversal symmetry. '
                             'Use -1 for auto (=bandwidth). (default: 0=off)')
    args = parser.parse_args()

    t_total = time.time()

    print(f"\n{'='*70}")
    print(f"  D9 Frame-Sector g-Factor Measurement (v2)")
    print(f"  L={args.L}, seed={args.seed}, K0={args.K0}")
    print(f"  frame_init={args.frame_init}")
    print(f"  peierls_mode={args.peierls_mode}, flux_mode={args.flux_mode}")
    print(f"  B_max={args.B_max}, N_B={args.N_B}")
    print(f"  n_anneal={args.n_anneal}, n_indep={args.n_indep}")
    print(f"  living_mode={args.living_mode}")
    print(f"  eta_clr={args.eta_clr}, lam={args.lam}")
    print(f"  n_reequil={args.n_reequil}, n_reequil_mc={args.n_reequil_mc}")
    if args.living_mode == 'adiabatic':
        print(f"  n_ramp_steps={args.n_ramp_steps}, dt_ramp={args.dt_ramp_gauge}")
        print(f"  sigma_ramp={args.sigma_ramp}, beta_ramp={args.beta_ramp}")
        print(f"  pruning_mode={args.pruning_mode}")
        if args.pruning_mode == 'dynamic':
            print(f"  T_death={args.T_death}, eps_death={args.eps_death}, "
                  f"K_seed={args.K_seed}, r_eff={args.r_eff}")
        elif args.pruning_mode == 'soft':
            print(f"  T_soft={args.T_soft}, r_eff={args.r_eff}")
        if args.n_pre_equil > 0:
            print(f"  n_pre_equil={args.n_pre_equil}")
    print(f"{'='*70}")

    # Load 184b module
    print(f"\n  Loading 184b module...")
    mod = load_184b()

    # Phase 0: Build lattice
    print(f"\n  PHASE 0: LATTICE CONSTRUCTION (L={args.L})")
    lat = build_lattice(args.L)

    # Phase 1: D9 protocol
    # Phase 1: Create or load state
    state_path = os.path.join(OUTPUT_DIR, f"d9_state_L{args.L}_s{args.seed}.npz")
    if args.load_state and os.path.exists(args.load_state):
        print(f"\n  LOADING STATE from {args.load_state}")
        loaded = np.load(args.load_state, allow_pickle=True)
        state = {
            'R': loaded['R'], 'V': loaded['V'], 'K': loaded['K'],
            'theta': loaded['theta'],
            'cos_dphi': loaded['cos_dphi'], 'dtheta': loaded['dtheta'],
            'history': loaded['history'].tolist() if 'history' in loaded else [],
            'M_sky_final': float(loaded['M_sky_final']),
            'M_dressed_final': float(loaded['M_dressed_final']),
        }
        print(f"  M_sky={state['M_sky_final']:.2f}, "
              f"M_dressed={state['M_dressed_final']:.2f}")
    else:
        print(f"\n  PHASE 1: D9 KURAMOTO SPONTANEOUS PROTOCOL")
        state = run_d9_protocol(
            lat, seed=args.seed,
            K0=args.K0, K_phi=args.K_phi,
            frame_init=args.frame_init,
            beta_grow=args.beta_grow, beta_wilson=args.beta_wilson,
            n_anneal_sweeps=args.n_anneal,
            n_independence_sweeps=args.n_indep)

        # Save state for reuse
        print(f"  Saving state to {state_path}")
        np.savez(state_path,
                 R=state['R'], V=state['V'], K=state['K'],
                 theta=state['theta'], cos_dphi=state['cos_dphi'],
                 dtheta=state['dtheta'],
                 history=np.array(state['history'], dtype=object),
                 M_sky_final=state['M_sky_final'],
                 M_dressed_final=state['M_dressed_final'])

    # Check skyrmion survival
    if state['M_dressed_final'] < 1.0:
        print(f"\n  WARNING: M_dressed={state['M_dressed_final']:.2f} < 1.0")
        print(f"  Skyrmion may not have survived independence test!")
        print(f"  Proceeding with B-scan anyway...")

    # Phase 3: B-scan
    print(f"\n  PHASE 3: B-SCAN ({args.N_B} points, B_max={args.B_max})")
    scan = run_b_scan_d9(
        lat, state, mod,
        B_max=args.B_max, N_B=args.N_B,
        K0=args.K0,
        eta_clr=args.eta_clr, lam=args.lam, dt_clr=0.01,
        eta_g=0.1, beta_g=2.0, dt_gauge=0.01,
        beta_mc=20.0, sigma_mc=0.05,
        n_reequil_steps=args.n_reequil,
        n_reequil_mc=args.n_reequil_mc,
        peierls_mode=args.peierls_mode,
        flux_mode=args.flux_mode,
        living_mode=args.living_mode,
        n_ramp_steps=args.n_ramp_steps,
        dt_ramp_gauge=args.dt_ramp_gauge,
        sigma_ramp=args.sigma_ramp,
        beta_ramp=args.beta_ramp,
        pruning_mode=args.pruning_mode,
        T_death=args.T_death,
        eps_death=args.eps_death,
        K_seed=args.K_seed,
        r_eff=args.r_eff,
        T_soft=args.T_soft,
        n_pre_equil=args.n_pre_equil,
        freeze_R_preequil=args.freeze_R_preequil,
        freeze_R_ramp=args.freeze_R_ramp,
        mu_Z=args.mu_Z)

    # Phase 4: Analysis
    results = analyze_and_report(scan, state)

    # Figure
    fig_path = os.path.join(OUTPUT_DIR, f"d9_frame_gfactor_L{args.L}.png")
    make_figure(scan, state, results, args.L, fig_path)

    # Save results as JSON
    save_data = {
        'L': args.L, 'seed': args.seed,
        'K0': args.K0, 'K_phi': args.K_phi,
        'frame_init': args.frame_init,
        'beta_grow': args.beta_grow,
        'beta_wilson': args.beta_wilson,
        'B_max': args.B_max, 'N_B': args.N_B,
        'peierls_mode': args.peierls_mode,
        'flux_mode': args.flux_mode,
        'living_mode': args.living_mode,
        'eta_clr': args.eta_clr, 'lam': args.lam,
        'n_anneal': args.n_anneal, 'n_indep': args.n_indep,
        'n_reequil': args.n_reequil,
        'n_reequil_mc': args.n_reequil_mc,
        'n_ramp_steps': args.n_ramp_steps,
        'dt_ramp_gauge': args.dt_ramp_gauge,
        'sigma_ramp': args.sigma_ramp,
        'beta_ramp': args.beta_ramp,
        'pruning_mode': args.pruning_mode,
        'T_death': args.T_death,
        'eps_death': args.eps_death,
        'K_seed': args.K_seed,
        'r_eff': args.r_eff,
        'T_soft': args.T_soft,
        'M_sky_final': state['M_sky_final'],
        'M_dressed_final': state['M_dressed_final'],
        'bandwidth': float(scan['bandwidth']),
        'B_values': scan['B_values'].tolist(),
        'E_frozen': scan['E_frozen'].tolist(),
        'E_living_Konly': scan['E_living_Konly'].tolist(),
        'E_living_Kgauge': scan['E_living_Kgauge'].tolist(),
        'E_K_field': scan['E_K_field'].tolist(),
        'E_Wilson': scan['E_Wilson'].tolist(),
    }

    # Serialize g-factor results
    for key in ('gf_Konly', 'gf_Kgauge', 'gf_Etotal'):
        gf = results[key]
        save_data[key] = {}
        for k, v in gf.items():
            if isinstance(v, np.ndarray):
                save_data[key][k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                save_data[key][k] = float(v)
            else:
                save_data[key][k] = v

    if scan['E_triplet_frozen'] is not None:
        save_data['E_triplet_frozen'] = scan['E_triplet_frozen'].tolist()
    if scan.get('E_triplet_Kgauge') is not None:
        save_data['E_triplet_Kgauge'] = scan['E_triplet_Kgauge'].tolist()

    # Serialize history (keep only floats/ints)
    save_data['history'] = state['history']

    json_path = os.path.join(OUTPUT_DIR, f"d9_frame_gfactor_L{args.L}.json")
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {json_path}")

    t_elapsed = time.time() - t_total
    print(f"\n  Total runtime: {t_elapsed:.1f}s ({t_elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
