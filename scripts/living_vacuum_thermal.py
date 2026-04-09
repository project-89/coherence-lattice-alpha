"""
THE LIVING VACUUM: Thermal Regulation Edition

The key insight: bootstrap needs LOW noise (Kuramoto cooperates with CLR),
but vortex persistence needs HIGH noise (thermal fluctuations).

Solution: K-dependent noise that interpolates:
  - Low K (bootstrap):  noise ~ noise_quantum (small, fixed)
  - High K (regulated): noise ~ sqrt(2*dt/K_avg) (XY Langevin thermal)

The Langevin dynamics of the XY model at temperature T=1:
  dθ/dt = K Σ sin(θ_j - θ_i) + sqrt(2T) · ξ

With T=1: noise_std = sqrt(2*dt) ≈ 0.45 — TOO LARGE for bootstrap.
With T=0: noise_std ≈ 0 — bootstraps but over-orders.

Compromise: effective temperature T_eff(K) that's small at low K, unity at high K:
  T_eff(K) = T_target * (1 - exp(-K/K_thermal))
  noise_std = sqrt(2 * T_eff * dt)

K_thermal controls the transition scale. When K >> K_thermal, T_eff → T_target.
When K << K_thermal, T_eff → 0 (bootstrap regime).

Combined with Shannon CLR for vortex bond protection.
"""
import numpy as np
from scipy.special import i0, i1
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
import time
import json

# ============================================================
# Diamond lattice construction
# ============================================================
def build_diamond(L):
    N = 2 * L**3
    def a_idx(ix, iy, iz):
        return 2 * ((ix % L) * L**2 + (iy % L) * L + (iz % L))
    def b_idx(ix, iy, iz):
        return 2 * ((ix % L) * L**2 + (iy % L) * L + (iz % L)) + 1

    src_list, tgt_list = [], []
    for ix in range(L):
        for iy in range(L):
            for iz in range(L):
                a = a_idx(ix, iy, iz)
                for b in [b_idx(ix,iy,iz), b_idx((ix-1)%L,iy,iz),
                          b_idx(ix,(iy-1)%L,iz), b_idx(ix,iy,(iz-1)%L)]:
                    src_list.append(a)
                    tgt_list.append(b)

    src = np.array(src_list, dtype=np.int32)
    tgt = np.array(tgt_list, dtype=np.int32)
    n_edges = len(src)

    max_nbr = 4
    nbr_idx = np.full((N, max_nbr), -1, dtype=np.int32)
    nbr_eidx = np.full((N, max_nbr), -1, dtype=np.int32)
    nbr_count = np.zeros(N, dtype=np.int32)

    for e in range(n_edges):
        i, j = src[e], tgt[e]
        k = nbr_count[i]
        if k < max_nbr:
            nbr_idx[i, k] = j
            nbr_eidx[i, k] = e
            nbr_count[i] += 1
        k = nbr_count[j]
        if k < max_nbr:
            nbr_idx[j, k] = i
            nbr_eidx[j, k] = e
            nbr_count[j] += 1

    plaq_sites = []
    for ix in range(L):
        for iy in range(L):
            for iz in range(L):
                plaq_sites.append([
                    a_idx(ix,iy,iz), b_idx(ix,iy,iz),
                    a_idx((ix+1)%L,iy,iz), b_idx((ix+1)%L,(iy-1)%L,iz)])
                plaq_sites.append([
                    a_idx(ix,iy,iz), b_idx(ix,iy,iz),
                    a_idx((ix+1)%L,iy,iz), b_idx((ix+1)%L,iy,(iz-1)%L)])

    plaq_arr = np.array(plaq_sites, dtype=np.int32)
    return {
        'N': N, 'L': L, 'n_edges': n_edges,
        'src': src, 'tgt': tgt,
        'nbr_idx': nbr_idx, 'nbr_eidx': nbr_eidx, 'nbr_count': nbr_count,
        'plaq': plaq_arr,
    }

# ============================================================
# Phase dynamics with adaptive thermal noise
# ============================================================
def kuramoto_step_thermal(phases, K_edge, lat, dt, T_eff):
    """
    Kuramoto + thermal noise.
    noise_std = sqrt(2 * T_eff * dt)
    """
    nbr_idx = lat['nbr_idx']
    nbr_eidx = lat['nbr_eidx']
    valid = nbr_idx >= 0
    nbr_phases = np.where(valid, phases[np.maximum(nbr_idx, 0)], phases[:, None])
    nbr_K = np.where(valid, K_edge[np.maximum(nbr_eidx, 0)], 0.0)
    sin_diff = np.sin(nbr_phases - phases[:, None])
    drive = np.sum(nbr_K * sin_diff * valid, axis=1)

    noise_std = np.sqrt(2.0 * T_eff * dt) if T_eff > 0 else 0.0
    phases += dt * drive + noise_std * np.random.randn(lat['N'])
    return phases

# ============================================================
# CLR dynamics
# ============================================================
def clr_step(K_edge, phases, lat, eta_K, r_clr, cos_eff, birth_K=0.03):
    src, tgt = lat['src'], lat['tgt']
    cos_dtheta = np.cos(phases[src] - phases[tgt])
    safe_kappa = np.maximum(2 * K_edge, 1e-10)
    R0c = i1(safe_kappa) / i0(safe_kappa)
    signal = R0c * cos_eff * cos_dtheta
    cost = 2 * K_edge / r_clr
    K_edge += eta_K * (signal - cost)
    K_edge = np.maximum(K_edge, 0.0)
    dead = K_edge < 0.005
    if np.any(dead):
        born = dead & (np.random.random(len(K_edge)) < 0.1)
        K_edge[born] = birth_K
    return K_edge

def compute_fiedler_push(K_edge, lat, I_phase):
    N = lat['N']
    src, tgt = lat['src'], lat['tgt']
    n_edges = lat['n_edges']
    K_eff = np.maximum(K_edge, 1e-6)
    data = np.concatenate([-K_eff, -K_eff])
    row = np.concatenate([src, tgt])
    col = np.concatenate([tgt, src])
    L_off = csr_matrix((data, (row, col)), shape=(N, N))
    degree = np.zeros(N)
    np.add.at(degree, src, K_eff)
    np.add.at(degree, tgt, K_eff)
    L_mat = L_off + diags(degree)
    try:
        eigenvalues, eigenvectors = eigsh(L_mat, k=2, which='SM', tol=1e-6)
        idx = np.argsort(eigenvalues)
        lambda2 = float(eigenvalues[idx[1]])
        v2 = eigenvectors[:, idx[1]]
    except Exception:
        return np.zeros(n_edges), 0.0
    sensitivity = (v2[src] - v2[tgt])**2
    S = sensitivity - np.mean(sensitivity)
    S *= I_phase
    return S, lambda2

def shannon_clr_step(K_edge, phases, lat, eta_K, r_clr, cos_eff,
                     fiedler_push, eta_struct=1.0, birth_K=0.03):
    src, tgt = lat['src'], lat['tgt']
    cos_dtheta = np.cos(phases[src] - phases[tgt])
    safe_kappa = np.maximum(2 * K_edge, 1e-10)
    R0c = i1(safe_kappa) / i0(safe_kappa)
    signal = R0c * cos_eff * cos_dtheta
    cost = 2 * K_edge / r_clr
    K_edge += eta_K * (signal - cost) + eta_K * eta_struct * fiedler_push
    K_edge = np.maximum(K_edge, 0.0)
    dead = K_edge < 0.005
    if np.any(dead):
        born = dead & (np.random.random(len(K_edge)) < 0.1)
        K_edge[born] = birth_K
    return K_edge

# ============================================================
# Measurements
# ============================================================
def detect_vortices(phases, lat):
    plaq = lat['plaq']
    n_plaq = len(plaq)
    p = phases[plaq]
    dp = np.diff(p, axis=1, append=p[:, :1])
    dp = (dp + np.pi) % (2 * np.pi) - np.pi
    winding = np.sum(dp, axis=1)
    n_vortex = np.sum(np.abs(winding) > np.pi)
    return n_vortex / n_plaq, int(n_vortex)

def coherence_capital_at_K(K_bar, z=4):
    if K_bar < 1e-10:
        return 0, float('inf'), 0, 0
    R0 = float(i1(K_bar) / i0(K_bar))
    I_phase = R0**z
    alpha = I_phase
    for _ in range(50):
        n_eff = np.exp(-0.5) + alpha / (2 * np.pi)
        rho = (np.pi / z)**n_eff
        alpha = I_phase * rho
    return alpha, 1.0 / alpha if alpha > 0 else float('inf'), I_phase, rho

def measure_state(K_edge, phases, lat, lambda2=0.0, T_eff=0.0):
    src, tgt = lat['src'], lat['tgt']
    cos_dtheta = np.cos(phases[src] - phases[tgt])
    K_alive = K_edge[K_edge > 0.01]
    alive_frac = len(K_alive) / len(K_edge)
    vort_frac, n_vort = detect_vortices(phases, lat)
    K_bar = float(np.mean(K_alive)) if len(K_alive) > 0 else 0.0
    alpha, inv_alpha, I_phase, rho = coherence_capital_at_K(K_bar)

    dtheta = np.abs(phases[src] - phases[tgt])
    dtheta = np.minimum(dtheta, 2*np.pi - dtheta)
    core_mask = dtheta > np.pi/4
    n_core = int(np.sum(core_mask))
    K_core = float(np.mean(K_edge[core_mask])) if n_core > 0 else 0.0

    return {
        'K_mean': float(np.mean(K_edge)),
        'K_alive': K_bar,
        'K_std': float(np.std(K_edge)),
        'K_core': K_core,
        'n_core': n_core,
        'cos_mean': float(np.mean(cos_dtheta)),
        'alive_frac': float(alive_frac),
        'vortex_frac': float(vort_frac),
        'n_vortex': int(n_vort),
        'lambda2': lambda2,
        'alpha': alpha,
        'inv_alpha': inv_alpha,
        'T_eff': T_eff,
    }

# ============================================================
# Main simulation
# ============================================================
def run_thermal(L=6, n_steps=8000, seed=42, T_target=1.0, K_thermal=0.3,
                use_shannon=True, fiedler_interval=10, eta_struct=1.0):
    """
    T_eff(K) = T_target * (1 - exp(-K_avg / K_thermal))

    K_thermal: scale where thermal noise turns on.
      Small K_thermal → noise activates early (harder bootstrap)
      Large K_thermal → noise activates late (easier bootstrap, less regulation)
    """
    np.random.seed(seed)
    lat = build_diamond(L)
    N, n_edges = lat['N'], lat['n_edges']

    K_BKT = 2 / np.pi
    K_bulk = 16 / np.pi**2
    cos_eff = 2 / 3
    r_clr = 7.0
    eta_K = 0.02
    dt = 0.1
    kura_per_clr = 3
    meas_interval = 50

    phases = np.random.uniform(0, 2 * np.pi, N)
    K_edge = np.full(n_edges, 0.05)

    history = []
    bkt_crossed = False
    bkt_step = None
    bkt_data = None
    fiedler_push = np.zeros(n_edges)
    lambda2 = 0.0

    for step in range(n_steps):
        # Compute effective temperature from current K
        K_alive_arr = K_edge[K_edge > 0.01]
        K_avg = np.mean(K_alive_arr) if len(K_alive_arr) > 0 else 0.05
        T_eff = T_target * (1.0 - np.exp(-K_avg / K_thermal))

        # Kuramoto with adaptive thermal noise
        for _ in range(kura_per_clr):
            phases = kuramoto_step_thermal(phases, K_edge, lat, dt, T_eff)

        # CLR step
        if use_shannon and step % fiedler_interval == 0:
            I_ph = float((i1(K_avg) / i0(K_avg))**4)
            fiedler_push, lambda2 = compute_fiedler_push(K_edge, lat, I_ph)

        if use_shannon:
            K_edge = shannon_clr_step(K_edge, phases, lat, eta_K, r_clr,
                                      cos_eff, fiedler_push, eta_struct)
        else:
            K_edge = clr_step(K_edge, phases, lat, eta_K, r_clr, cos_eff)

        # Measure
        if step % meas_interval == 0:
            m = measure_state(K_edge, phases, lat, lambda2, T_eff)
            m['step'] = step

            if not bkt_crossed and m['K_alive'] > K_BKT:
                bkt_crossed = True
                bkt_step = step
                bkt_data = dict(m)
                m['phase'] = 'BKT!'
            elif m['K_alive'] > K_bulk * 0.9:
                m['phase'] = 'EQ'
            elif m['K_alive'] > K_BKT:
                m['phase'] = 'ORD'
            else:
                m['phase'] = 'BOOT'

            history.append(m)

    return history, bkt_step, bkt_data


if __name__ == '__main__':
    K_BKT = 2 / np.pi
    K_bulk = 16 / np.pi**2

    print("=" * 85)
    print("LIVING VACUUM: Thermal Regulation")
    print("Kuramoto + CLR with K-dependent thermal noise")
    print("=" * 85)

    # Sweep K_thermal to find the sweet spot
    configs = [
        # (T_target, K_thermal, use_shannon, label)
        (1.0, 0.2,  False, "Std CLR, K_th=0.2"),
        (1.0, 0.3,  False, "Std CLR, K_th=0.3"),
        (1.0, 0.5,  False, "Std CLR, K_th=0.5"),
        (1.0, 1.0,  False, "Std CLR, K_th=1.0"),
        (1.0, 0.3,  True,  "Shannon, K_th=0.3"),
        (1.0, 0.5,  True,  "Shannon, K_th=0.5"),
        (1.0, 1.0,  True,  "Shannon, K_th=1.0"),
    ]

    print(f"\n{'Config':<24s} {'BKTstep':>8s} {'K@BKT':>7s} {'K_final':>8s} "
          f"{'vort_fin':>8s} {'cos_fin':>8s} {'1/α_fin':>8s} {'T_eff':>6s}")
    print("-" * 88)

    all_results = {}

    for T_target, K_thermal, use_shannon, label in configs:
        h, bs, bd = run_thermal(
            L=6, n_steps=8000, seed=42,
            T_target=T_target, K_thermal=K_thermal,
            use_shannon=use_shannon, fiedler_interval=10)

        final = h[-1]
        K_bkt = bd['K_alive'] if bd else 0
        bkt_s = bs if bs else 'N/A'

        print(f"{label:<24s} {str(bkt_s):>8s} {K_bkt:7.4f} "
              f"{final['K_alive']:8.4f} {final['vortex_frac']*100:7.1f}% "
              f"{final['cos_mean']:8.4f} {final['inv_alpha']:8.2f} "
              f"{final['T_eff']:6.3f}")

        all_results[label] = {'history': h, 'bkt_step': bs, 'bkt_data': bd}

    # Detailed timeline for best configuration
    # Find the one with highest final vortex fraction that also bootstrapped
    best_label = None
    best_vort = -1
    for label, r in all_results.items():
        if r['bkt_step'] is not None:
            fv = r['history'][-1]['vortex_frac']
            if fv > best_vort:
                best_vort = fv
                best_label = label

    if best_label:
        print(f"\n\n{'='*85}")
        print(f"DETAILED TIMELINE: {best_label}")
        print(f"{'='*85}")

        h = all_results[best_label]['history']
        bs = all_results[best_label]['bkt_step']

        fmt = (f"{'step':>6s} {'K_alive':>7s} {'K_core':>7s} {'⟨cos⟩':>7s} "
               f"{'vort%':>6s} {'alive':>6s} {'n_core':>6s} {'T_eff':>6s} "
               f"{'1/α':>8s} {'phase':>5s}")
        print(f"\n{fmt}")
        print("-" * 80)
        for m in h:
            if m['step'] % 200 == 0 or m.get('phase') == 'BKT!':
                print(f"{m['step']:6d} {m['K_alive']:7.4f} {m['K_core']:7.4f} "
                      f"{m['cos_mean']:7.4f} {m['vortex_frac']*100:5.1f}% "
                      f"{m['alive_frac']*100:5.1f}% {m['n_core']:6d} "
                      f"{m['T_eff']:6.3f} {m['inv_alpha']:8.2f} "
                      f"{m.get('phase',''):>5s}")

        # Late-time statistics
        late = [m for m in h if m['step'] > 6000]
        if late:
            print(f"\nLate-time averages (step > 6000):")
            print(f"  K_alive  = {np.mean([m['K_alive'] for m in late]):.4f} "
                  f"± {np.std([m['K_alive'] for m in late]):.4f}")
            print(f"  ⟨cos⟩    = {np.mean([m['cos_mean'] for m in late]):.4f}")
            print(f"  Vortex   = {np.mean([m['vortex_frac'] for m in late])*100:.2f}%")
            print(f"  1/α      = {np.mean([m['inv_alpha'] for m in late]):.3f}")
            print(f"  T_eff    = {np.mean([m['T_eff'] for m in late]):.4f}")
    else:
        print("\nNo configuration bootstrapped successfully!")

    # Also print timeline for all configs that bootstrapped but have 0% vortex
    # to show the contrast
    for label, r in all_results.items():
        if label == best_label:
            continue
        h = r['history']
        final = h[-1]
        if r['bkt_step'] is not None:
            late = [m for m in h if m['step'] > 6000]
            if late:
                late_K = np.mean([m['K_alive'] for m in late])
                late_v = np.mean([m['vortex_frac'] for m in late])
                late_cos = np.mean([m['cos_mean'] for m in late])
                print(f"\n{label}: late K={late_K:.4f}, cos={late_cos:.4f}, "
                      f"vort={late_v*100:.2f}%")

    # Save
    try:
        save = {}
        for label, r in all_results.items():
            final = r['history'][-1]
            save[label] = {
                'bkt_step': r['bkt_step'],
                'final_K': final['K_alive'],
                'final_vort': final['vortex_frac'],
                'final_cos': final['cos_mean'],
                'final_inv_alpha': final['inv_alpha'],
            }
        with open('out/living_vacuum_thermal.json', 'w') as f:
            json.dump(save, f, indent=2)
        print(f"\nSaved to out/living_vacuum_thermal.json")
    except Exception as e:
        print(f"\nSave error: {e}")
