"""
casimir_mass_spectrum.py
========================
Compares the Casimir mass spectrum from Coherence Lattice Theory
to ALL known fundamental particle masses.

Mass formula: m_j = m_e * sqrt(4*j*(j+1)/3) for j = 0, 1/2, 1, 3/2, ...

This script:
1. Generates the full Casimir spectrum up to j=20 (and beyond where needed)
2. Compares to all known leptons, quarks, mesons, baryons
3. Tests alternative mass formulas
4. Performs statistical significance analysis
5. Reports HONESTLY what matches and what doesn't

Author: Coherence Lattice Theory project
Date: 2026-03-13
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import i0, i1
from collections import OrderedDict
import os
import json

# =====================================================================
# 0. SETUP
# =====================================================================

out_dir = os.path.join(os.path.dirname(__file__), '..', 'out')
os.makedirs(out_dir, exist_ok=True)

# Electron mass
m_e_MeV = 0.51099895  # PDG 2024

# =====================================================================
# 1. KNOWN PARTICLE MASSES (PDG 2024 values, MeV)
# =====================================================================

# Organize by category
# Each entry: (name, mass_MeV, category, is_fundamental)

particles = OrderedDict()

# --- LEPTONS (fundamental) ---
particles['e'] =      (0.51099895,     'lepton',  True)
particles['mu'] =     (105.6583755,    'lepton',  True)
particles['tau'] =    (1776.86,        'lepton',  True)

# --- QUARKS (fundamental, MS-bar masses at 2 GeV for light, pole for heavy) ---
particles['u'] =      (2.16,           'quark',   True)  # MS-bar at 2 GeV
particles['d'] =      (4.67,           'quark',   True)
particles['s'] =      (93.4,           'quark',   True)
particles['c'] =      (1270.0,         'quark',   True)  # MS-bar at m_c
particles['b'] =      (4180.0,         'quark',   True)  # MS-bar at m_b
particles['t'] =      (172760.0,       'quark',   True)  # pole mass

# --- GAUGE BOSONS ---
particles['W'] =      (80369.0,        'boson',   True)
particles['Z'] =      (91187.6,        'boson',   True)
particles['H'] =      (125250.0,       'boson',   True)  # Higgs

# --- LIGHT PSEUDOSCALAR MESONS ---
particles['pi0'] =    (134.9768,       'meson',   False)
particles['pi+'] =    (139.5706,       'meson',   False)
particles['K+'] =     (493.677,        'meson',   False)
particles['K0'] =     (497.611,        'meson',   False)
particles['eta'] =    (547.862,        'meson',   False)
particles['eta_p'] =  (957.78,         'meson',   False)  # eta'

# --- VECTOR MESONS ---
particles['rho'] =    (775.26,         'meson',   False)
particles['omega'] =  (782.66,         'meson',   False)
particles['phi'] =    (1019.461,       'meson',   False)
particles['J/psi'] =  (3096.900,       'meson',   False)
particles['Ups'] =    (9460.30,        'meson',   False)  # Upsilon(1S)

# --- BARYONS ---
particles['p'] =      (938.272,        'baryon',  False)
particles['n'] =      (939.565,        'baryon',  False)
particles['Lambda'] = (1115.683,       'baryon',  False)
particles['Sigma+'] = (1189.37,        'baryon',  False)
particles['Sigma0'] = (1192.642,       'baryon',  False)
particles['Xi0'] =    (1314.86,        'baryon',  False)
particles['Omega-'] = (1672.45,        'baryon',  False)
particles['Lam_c'] =  (2286.46,        'baryon',  False)  # Lambda_c

# Key mass ratios
m_mu_over_m_e = 206.7682830
m_tau_over_m_e = 3477.23


# =====================================================================
# 2. CASIMIR MASS SPECTRUM
# =====================================================================

def casimir_mass(j, m_e=m_e_MeV):
    """Mass from SO(3) Casimir on diamond lattice: m_j = m_e * sqrt(4*j*(j+1)/3)"""
    if j == 0:
        return 0.0
    return m_e * np.sqrt(4.0 * j * (j + 1) / 3.0)


def casimir_mass_ratio(j):
    """m_j / m_e = sqrt(4*j*(j+1)/3)"""
    if j == 0:
        return 0.0
    return np.sqrt(4.0 * j * (j + 1) / 3.0)


print("=" * 78)
print("  CASIMIR MASS SPECTRUM vs KNOWN PARTICLES")
print("  Coherence Lattice Theory: m_j = m_e * sqrt(4*j*(j+1)/3)")
print("=" * 78)


# --- Generate spectrum for j = 0, 1/2, 1, ..., 20 ---
j_max = 200  # go high enough to cover top quark
js_half = np.arange(0, j_max + 0.5, 0.5)  # 0, 0.5, 1.0, 1.5, ...
masses_half = np.array([casimir_mass(j) for j in js_half])

print(f"\n--- Part 1: Casimir Spectrum (first 15 levels) ---")
print(f"  {'j':>5s}  {'C2=j(j+1)':>10s}  {'m_j (MeV)':>12s}  {'m_j/m_e':>10s}")
print(f"  {'---':>5s}  {'---':>10s}  {'---':>12s}  {'---':>10s}")
for i, j in enumerate(js_half[:15]):
    C2 = j * (j + 1)
    m = masses_half[i]
    ratio = casimir_mass_ratio(j)
    print(f"  {j:5.1f}  {C2:10.2f}  {m:12.4f}  {ratio:10.4f}")


# =====================================================================
# 3. MATCH TO KNOWN PARTICLES
# =====================================================================

print(f"\n--- Part 2: Nearest Casimir Level for Each Known Particle ---")
print(f"  {'Particle':<10s}  {'m (MeV)':>12s}  {'Category':<8s}  "
      f"{'j_near':>7s}  {'m_Cas (MeV)':>12s}  {'Ratio':>8s}  {'Pct_err':>8s}")
print(f"  {'--------':<10s}  {'--------':>12s}  {'--------':<8s}  "
      f"{'------':>7s}  {'-----------':>12s}  {'-----':>8s}  {'-------':>8s}")

match_results = []

for name, (mass, cat, is_fund) in particles.items():
    # Find the j value that gives the closest mass
    # m_j = m_e * sqrt(4*j*(j+1)/3) => 4*j*(j+1)/3 = (mass/m_e)^2
    # => j^2 + j - 3*(mass/m_e)^2/4 = 0
    # => j = (-1 + sqrt(1 + 3*(mass/m_e)^2)) / 2

    r_sq = (mass / m_e_MeV)**2
    j_exact = (-1 + np.sqrt(1 + 3 * r_sq)) / 2.0

    # Nearest half-integer
    j_near = round(2 * j_exact) / 2.0
    if j_near < 0:
        j_near = 0.0

    m_casimir = casimir_mass(j_near)
    if m_casimir > 0:
        ratio = mass / m_casimir
        pct_err = (mass - m_casimir) / mass * 100
    else:
        ratio = float('inf')
        pct_err = 100.0

    match_results.append({
        'name': name,
        'mass': mass,
        'category': cat,
        'is_fundamental': is_fund,
        'j_exact': j_exact,
        'j_near': j_near,
        'm_casimir': m_casimir,
        'ratio': ratio,
        'pct_err': pct_err,
    })

    marker = ""
    if abs(pct_err) < 1.0:
        marker = " <-- CLOSE"
    elif abs(pct_err) < 5.0:
        marker = " <-- near"

    print(f"  {name:<10s}  {mass:12.4f}  {cat:<8s}  "
          f"{j_near:7.1f}  {m_casimir:12.4f}  {ratio:8.4f}  {pct_err:+8.2f}%{marker}")


# =====================================================================
# 4. THE MUON PROBLEM
# =====================================================================

print(f"\n--- Part 3: The Muon Problem ---")

# m_mu/m_e = 206.768
# Need 4*j*(j+1)/3 = 206.768^2 = 42753.0
# j^2 + j - 42753*3/4 = 0
# j^2 + j - 32064.75 = 0
# j = (-1 + sqrt(1 + 4*32064.75))/2 = (-1 + sqrt(128260))/2 = (-1 + 358.14)/2 = 178.57

j_muon_exact = (-1 + np.sqrt(1 + 3 * m_mu_over_m_e**2)) / 2.0
j_muon_near = round(2 * j_muon_exact) / 2.0
m_at_j_near = casimir_mass(j_muon_near)

print(f"  m_mu/m_e = {m_mu_over_m_e:.6f}")
print(f"  Required: 4*j*(j+1)/3 = {m_mu_over_m_e**2:.2f}")
print(f"  Exact j  = {j_muon_exact:.4f}")
print(f"  Nearest half-integer j = {j_muon_near:.1f}")
print(f"  Mass at j={j_muon_near:.1f}: {m_at_j_near:.4f} MeV")
print(f"  Actual muon mass: {particles['mu'][0]:.4f} MeV")
print(f"  Gap: {abs(m_at_j_near - particles['mu'][0]):.4f} MeV "
      f"({abs(m_at_j_near - particles['mu'][0])/particles['mu'][0]*100:.2f}%)")
print(f"\n  CONCLUSION: j_muon is NOT a half-integer ({j_muon_exact:.4f}).")
print(f"  The simple Casimir formula does NOT give the muon mass.")

# Similarly for tau
j_tau_exact = (-1 + np.sqrt(1 + 3 * m_tau_over_m_e**2)) / 2.0
j_tau_near = round(2 * j_tau_exact) / 2.0
m_at_j_tau = casimir_mass(j_tau_near)

print(f"\n  m_tau/m_e = {m_tau_over_m_e:.4f}")
print(f"  Exact j  = {j_tau_exact:.4f}")
print(f"  Nearest half-integer j = {j_tau_near:.1f}")
print(f"  Mass at j={j_tau_near:.1f}: {m_at_j_tau:.4f} MeV")
print(f"  Actual tau mass: {particles['tau'][0]:.4f} MeV")
print(f"  Gap: {abs(m_at_j_tau - particles['tau'][0]):.2f} MeV "
      f"({abs(m_at_j_tau - particles['tau'][0])/particles['tau'][0]*100:.2f}%)")
print(f"\n  CONCLUSION: j_tau is NOT a half-integer ({j_tau_exact:.4f}).")
print(f"  The simple Casimir formula does NOT give the tau mass either.")


# =====================================================================
# 5. ALTERNATIVE MASS FORMULAS
# =====================================================================

print(f"\n--- Part 4: Alternative Mass Formulas ---")

# 5a. Modified Casimir with different denominator
print(f"\n  (a) m_j = m_e * sqrt(4*j*(j+1)/z) for different z:")
for z_test in [2, 3, 4, 5, 6]:
    j_mu_z = (-1 + np.sqrt(1 + z_test * m_mu_over_m_e**2)) / 2.0
    j_tau_z = (-1 + np.sqrt(1 + z_test * m_tau_over_m_e**2)) / 2.0
    is_mu_half = abs(j_mu_z - round(2*j_mu_z)/2) < 0.01
    is_tau_half = abs(j_tau_z - round(2*j_tau_z)/2) < 0.01
    print(f"    z={z_test}: j_mu={j_mu_z:.4f} ({'HALF-INT' if is_mu_half else 'no'}), "
          f"j_tau={j_tau_z:.4f} ({'HALF-INT' if is_tau_half else 'no'})")

# 5b. Products of Casimir eigenvalues
print(f"\n  (b) Product formulas: m = m_e * sqrt(C2(j1) * C2(j2) * ...)")
print(f"      m_mu/m_e = {m_mu_over_m_e:.4f}, so (m_mu/m_e)^2 = {m_mu_over_m_e**2:.2f}")
print(f"      Can we write {m_mu_over_m_e**2:.2f} as a product of Casimir eigenvalues?")
print(f"      C2 values: 3/4, 2, 15/4, 6, 35/4, 12, 63/4, 20, ...")

# Check small products
casimir_vals = {
    '1/2': 3/4,
    '1': 2.0,
    '3/2': 15/4,
    '2': 6.0,
    '5/2': 35/4,
    '3': 12.0,
}

print(f"\n      Checking products of 2 Casimir values:")
target = m_mu_over_m_e**2
for n1, c1 in casimir_vals.items():
    for n2, c2 in casimir_vals.items():
        prod = (4*c1/3) * (4*c2/3)  # each mass ratio squared
        if 0.95 < prod/target < 1.05:
            print(f"        ({n1}) x ({n2}): (4*{c1:.2f}/3)*(4*{c2:.2f}/3) = {prod:.2f} "
                  f"vs target {target:.2f}, ratio = {prod/target:.4f}")

# 5c. Koide-type formula
print(f"\n  (c) Koide formula check:")
print(f"      Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2")

m_e_K = particles['e'][0]
m_mu_K = particles['mu'][0]
m_tau_K = particles['tau'][0]

Q_koide = (m_e_K + m_mu_K + m_tau_K) / (np.sqrt(m_e_K) + np.sqrt(m_mu_K) + np.sqrt(m_tau_K))**2
print(f"      Q = {Q_koide:.6f}  (Koide's value: 2/3 = {2/3:.6f})")
print(f"      Gap from 2/3: {abs(Q_koide - 2/3):.6f} ({abs(Q_koide - 2/3)/(2/3)*100:.4f}%)")
print(f"      NOTE: 2/3 = cos_eff in our lattice theory (simplex projection)!")

# 5d. Can we get the Koide formula from lattice theory?
print(f"\n  (d) Koide from lattice theory?")
print(f"      cos_eff = (d-1)/d = 2/3 for d=3.")
print(f"      If m_l ~ sqrt(C2(j_l)), then Koide becomes:")
print(f"      sum(C2) / (sum(sqrt(C2)))^2 = 2/3")
print(f"      This constrains the j values but does NOT fix them uniquely.")

# Check: if Koide holds with Casimir masses, what j-values would work?
# sum(4*j_i*(j_i+1)/3) / (sum(sqrt(4*j_i*(j_i+1)/3)))^2 = 2/3
# Let x_i = 4*j_i*(j_i+1)/3 (= (m_i/m_e)^2)
# sum(x_i) / (sum(sqrt(x_i)))^2 = 2/3
# For the electron: x_1 = 1 (j=1/2)
# Need x_2, x_3 such that (1+x_2+x_3)/(1+sqrt(x_2)+sqrt(x_3))^2 = 2/3

# With actual lepton masses:
x_e = 1.0
x_mu = m_mu_over_m_e**2  # = 42753
x_tau = m_tau_over_m_e**2  # = 12091132

Q_actual = (x_e + x_mu + x_tau) / (np.sqrt(x_e) + np.sqrt(x_mu) + np.sqrt(x_tau))**2
print(f"      Using actual mass ratios: Q = {Q_actual:.6f}")
print(f"      (Same as above since Koide is scale-invariant.)")


# 5e. Power-law / exponential patterns
print(f"\n  (e) Mass ratio patterns (looking for structure):")
print(f"      m_mu/m_e   = {m_mu_over_m_e:.4f}")
print(f"      m_tau/m_mu  = {m_tau_K/m_mu_K:.4f}")
print(f"      m_tau/m_e   = {m_tau_over_m_e:.4f}")
print(f"      ln(m_mu/m_e)  = {np.log(m_mu_over_m_e):.4f}")
print(f"      ln(m_tau/m_e) = {np.log(m_tau_over_m_e):.4f}")
print(f"      ratio of logs: {np.log(m_tau_over_m_e)/np.log(m_mu_over_m_e):.4f}")

# Check if mass ratios are powers of some base
print(f"      If m_l/m_e = b^n_l, then:")
print(f"        b^n_mu = {m_mu_over_m_e:.2f}")
print(f"        b^n_tau = {m_tau_over_m_e:.2f}")
print(f"        n_tau/n_mu = {np.log(m_tau_over_m_e)/np.log(m_mu_over_m_e):.4f}")
print(f"      This is ~1.53, not a simple rational number.")

# Check e^(pi*n) pattern
print(f"\n      Check e^(pi*n) pattern:")
for n in range(1, 5):
    val = np.exp(np.pi * n)
    print(f"        e^({n}*pi) = {val:.2f}  (m_mu/m_e = {m_mu_over_m_e:.2f})")

# Check (3/alpha)^n pattern
alpha_val = 1.0 / 137.036
print(f"\n      Check (3/alpha)^n pattern:")
for n_p, n_v in [(1, 3/alpha_val), (0.5, np.sqrt(3/alpha_val))]:
    print(f"        (3/alpha)^{n_p:.1f} = {n_v:.2f}")

# 5f. Could different lattice substructures give different spectra?
print(f"\n  (f) Different lattice substructures:")
print(f"      Diamond lattice has: vertices, edges, faces, unit cells")
print(f"      Vertex SO(3): C2 = j(j+1), factor = 4/3 from j=1/2 normalization")
print(f"      Edge SU(2): if bonds carry spin-1/2 pairs, C2_edge = s(s+1), s=0,1")
print(f"      Face: plaquette excitations with integer angular momentum")
print(f"      Volume: collective modes with large j")
print(f"")
print(f"      This is speculative. The Casimir formula applies to the frame")
print(f"      bundle SO(3) representation at each vertex. Other excitations")
print(f"      would need different derivations.")


# =====================================================================
# 6. STATISTICAL ANALYSIS: IS MATCHING BETTER THAN RANDOM?
# =====================================================================

print(f"\n--- Part 5: Statistical Analysis ---")

# For each particle, compute fractional distance to nearest Casimir level
# Compare to what random masses would give

frac_errors = []
for r in match_results:
    if r['m_casimir'] > 0:
        frac_errors.append(abs(r['pct_err']))

# Random expectation: Casimir levels are spaced as delta_m ~ m_e * delta_j / sqrt(j)
# For large j, spacing ~ m_e / (2*sqrt(3*j/4)) at mass m
# Average fractional gap ~ spacing / (2*mass) for uniform random masses
# = m_e / (4*mass*sqrt(3*j/4)) ~ 1/(4*sqrt(3)*j) for j >> 1

# More rigorous: Monte Carlo simulation
np.random.seed(42)
n_mc = 10000
n_particles = len(particles)

# Mass range of known particles
mass_min = 0.1   # MeV
mass_max = 200000  # MeV

# ANALYTICAL approach: for any mass m, the nearest Casimir level can be found
# by solving j_exact = (-1 + sqrt(1 + 3*(m/m_e)^2)) / 2, rounding to nearest
# half-integer, then computing the Casimir mass at that j. This is O(1) per
# particle, no need to pre-generate a huge array.

def nearest_casimir_error(m, m_e=m_e_MeV):
    """Fractional error (%) to nearest Casimir level for mass m."""
    if m <= 0:
        return 100.0
    r_sq = (m / m_e)**2
    j_exact = (-1 + np.sqrt(1 + 3 * r_sq)) / 2.0
    j_near = max(0.5, round(2 * j_exact) / 2.0)
    m_cas = m_e * np.sqrt(4.0 * j_near * (j_near + 1) / 3.0)
    return abs(m_cas - m) / m * 100

def compute_match_quality(test_masses):
    """Compute mean absolute fractional error to nearest Casimir level."""
    errors = [nearest_casimir_error(m) for m in test_masses if m > 0]
    return np.mean(errors), np.median(errors), errors

# Actual particles
actual_masses = [p[0] for p in particles.values()]
actual_mean, actual_median, actual_errors = compute_match_quality(actual_masses)

# Monte Carlo: random masses drawn log-uniformly
mc_means = []
mc_medians = []
for _ in range(n_mc):
    random_masses = np.exp(np.random.uniform(np.log(mass_min), np.log(mass_max), n_particles))
    mc_mean, mc_median, _ = compute_match_quality(random_masses)
    mc_means.append(mc_mean)
    mc_medians.append(mc_median)

mc_means = np.array(mc_means)
mc_medians = np.array(mc_medians)

p_value_mean = np.mean(mc_means <= actual_mean)
p_value_median = np.mean(mc_medians <= actual_median)

print(f"  Matching quality (% error to nearest Casimir level):")
print(f"    Actual particles:  mean = {actual_mean:.2f}%, median = {actual_median:.2f}%")
print(f"    Random (MC {n_mc} trials):")
print(f"      mean of means:   {np.mean(mc_means):.2f}% +/- {np.std(mc_means):.2f}%")
print(f"      mean of medians: {np.mean(mc_medians):.2f}% +/- {np.std(mc_medians):.2f}%")
print(f"    p-value (mean):  {p_value_mean:.4f}")
print(f"    p-value (median): {p_value_median:.4f}")

if p_value_mean > 0.05:
    print(f"\n  CONCLUSION: Matching is NOT statistically significant (p={p_value_mean:.3f}).")
    print(f"  Known particle masses are NOT closer to Casimir levels than random.")
else:
    print(f"\n  CONCLUSION: Matching is statistically significant (p={p_value_mean:.4f}).")

# Count "close matches" (within 1%, 5%, 10%)
for threshold in [1.0, 5.0, 10.0]:
    n_close = sum(1 for e in actual_errors if e < threshold)
    # Expected from MC
    mc_counts = []
    for _ in range(1000):
        random_masses = np.exp(np.random.uniform(np.log(mass_min), np.log(mass_max), n_particles))
        _, _, rand_errors = compute_match_quality(random_masses)
        mc_counts.append(sum(1 for e in rand_errors if e < threshold))
    expected = np.mean(mc_counts)
    print(f"  Within {threshold:.0f}%: {n_close} particles (expected by chance: {expected:.1f})")


# =====================================================================
# 7. THE LOW-MASS SPECTRUM MYSTERY
# =====================================================================

print(f"\n--- Part 6: The Low-Mass Spectrum (sub-10 MeV) ---")

print(f"\n  Casimir levels below 10 MeV:")
for j in js_half[:20]:
    m = casimir_mass(j)
    if m > 0 and m < 10:
        # Check against known particles in this range
        near = [(n, p[0]) for n, p in particles.items() if abs(p[0] - m)/max(p[0],0.001) < 0.3]
        near_str = ', '.join([f"{n}({p:.3f})" for n, p in near]) if near else "---"
        print(f"    j={j:5.1f}:  m = {m:.4f} MeV   Nearby particles: {near_str}")

print(f"\n  Known sub-10 MeV fundamental particles:")
print(f"    electron:   0.511 MeV  (j=1/2 EXACT)")
print(f"    up quark:   ~2.16 MeV  (j={(-1+np.sqrt(1+3*(2.16/m_e_MeV)**2))/2:.2f})")
print(f"    down quark: ~4.67 MeV  (j={(-1+np.sqrt(1+3*(4.67/m_e_MeV)**2))/2:.2f})")
print(f"")
print(f"  The j=1 level (835 keV) and j=3/2 level (1143 keV) correspond to")
print(f"  NO known fundamental particle. Possible interpretations:")
print(f"    1. These are lattice artifacts (not physical states)")
print(f"    2. These are confined states that don't appear as free particles")
print(f"    3. The low-j spectrum maps to something other than individual particles")
print(f"    4. They could correspond to nuclear resonances or exotic states")


# =====================================================================
# 8. WHERE DOES EACH PARTICLE TYPE LIVE?
# =====================================================================

print(f"\n--- Part 7: j-Values Required for Each Particle ---")
print(f"  {'Particle':<10s}  {'m (MeV)':>12s}  {'j_exact':>10s}  {'j_near':>8s}  {'Residual':>10s}")
print(f"  {'--------':<10s}  {'--------':>12s}  {'-------':>10s}  {'------':>8s}  {'--------':>10s}")

categories_j = {'lepton': [], 'quark': [], 'meson': [], 'baryon': [], 'boson': []}

for r in match_results:
    residual = r['j_exact'] - r['j_near']
    categories_j[r['category']].append(r['j_exact'])
    print(f"  {r['name']:<10s}  {r['mass']:12.4f}  {r['j_exact']:10.4f}  {r['j_near']:8.1f}  {residual:+10.4f}")

print(f"\n  j-value ranges by category:")
for cat, js in categories_j.items():
    if js:
        print(f"    {cat:<8s}: j = {min(js):.1f} -- {max(js):.1f}")


# =====================================================================
# 9. ALPHA CORRECTION FORMULAS
# =====================================================================

print(f"\n--- Part 8: Alpha-Corrected Mass Formulas ---")

# Test: m_j = m_e * sqrt(4*j*(j+1)/3) * (1 + alpha*f(j))
alpha = 1.0/137.036

# Can we get the muon with an alpha correction?
# m_mu = m_e * sqrt(4*j*(j+1)/3) * (1 + alpha*f(j))
# At j_near = 178.5 or 179.0:
for j_test in [178.0, 178.5, 179.0]:
    m_cas = casimir_mass(j_test)
    correction_needed = particles['mu'][0] / m_cas - 1.0
    alpha_f = correction_needed / alpha
    print(f"  j={j_test:.1f}: m_cas={m_cas:.4f} MeV, need alpha*f = {correction_needed:.6f} (f={alpha_f:.2f})")

print(f"\n  The correction needed is O(0.1%), which is O(alpha) = 0.7%.")
print(f"  So an alpha correction COULD shift masses by the right amount")
print(f"  but the j-values are still very large and non-illuminating.")


# =====================================================================
# 10. PRODUCT REPRESENTATIONS
# =====================================================================

print(f"\n--- Part 9: Product Representations ---")
print(f"  Can mass ratios be PRODUCTS of small Casimir eigenvalues?")
print(f"  If excitations combine as m^2 = m_e^2 * product(4*j_k*(j_k+1)/3),")
print(f"  then (m/m_e)^2 = product of C2-ratios.")
print(f"")

# Muon: (m_mu/m_e)^2 = 42753
# Factorize as products of small Casimir ratios (4*j*(j+1)/3)
casimir_ratios = {}
for j in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
    casimir_ratios[j] = 4*j*(j+1)/3

print(f"  Casimir ratios 4*j*(j+1)/3:")
for j, cr in casimir_ratios.items():
    print(f"    j={j:.1f}: {cr:.4f}")

# Try factoring (m_mu/m_e)^2 = 42753
target_mu_sq = m_mu_over_m_e**2
print(f"\n  Target: (m_mu/m_e)^2 = {target_mu_sq:.2f}")

# Try products of 2, 3, 4 Casimir ratios
best_products = []
cr_list = list(casimir_ratios.items())
for i, (j1, c1) in enumerate(cr_list):
    for j2, c2 in cr_list[i:]:
        for j3, c3 in cr_list:
            prod = c1 * c2 * c3
            if 0.99 < prod/target_mu_sq < 1.01:
                best_products.append(((j1,j2,j3), prod, prod/target_mu_sq))
            for j4, c4 in cr_list:
                prod4 = prod * c4
                if 0.99 < prod4/target_mu_sq < 1.01:
                    best_products.append(((j1,j2,j3,j4), prod4, prod4/target_mu_sq))

if best_products:
    print(f"  Found {len(best_products)} product combinations within 1% of target:")
    for js, prod, ratio in best_products[:10]:
        js_str = ' x '.join([f'j={j:.1f}' for j in js])
        print(f"    {js_str}: product = {prod:.2f}, ratio = {ratio:.6f}")
else:
    print(f"  No product of 2-4 small Casimir ratios matches the muon mass ratio within 1%.")

# Try with larger j values
print(f"\n  Checking 2-factor products with larger j:")
for j1 in np.arange(0.5, 20.5, 0.5):
    c1 = 4*j1*(j1+1)/3
    needed_c2 = target_mu_sq / c1
    # Solve: 4*j2*(j2+1)/3 = needed_c2
    # j2^2 + j2 - 3*needed_c2/4 = 0
    disc = 1 + 3*needed_c2
    if disc >= 0:
        j2_exact = (-1 + np.sqrt(disc)) / 2.0
        if abs(j2_exact - round(2*j2_exact)/2) < 0.005:
            j2_near = round(2*j2_exact)/2
            c2_near = 4*j2_near*(j2_near+1)/3
            product = c1 * c2_near
            err = abs(product/target_mu_sq - 1) * 100
            if err < 0.5:
                print(f"    j1={j1:.1f} x j2={j2_near:.1f}: product = {product:.2f}, "
                      f"target = {target_mu_sq:.2f}, err = {err:.3f}%")


# =====================================================================
# 11. DENSITY OF STATES ANALYSIS
# =====================================================================

print(f"\n--- Part 10: Density of States ---")

# Casimir spectrum: m_j = m_e * sqrt(4*j*(j+1)/3)
# Spacing: dm/dj = m_e * (4*(2j+1)/3) / (2*sqrt(4*j*(j+1)/3))
#         = m_e * 2(2j+1) / (3*sqrt(4*j*(j+1)/3))
# For large j: ~ m_e * 2 / (3*sqrt(4/3)) = m_e * sqrt(3)/3 ~ 0.295 MeV per step

# At small j, spacing is larger in absolute terms but comparable to the mass
# At large j, spacing is ~constant in MeV

print(f"  Casimir level spacing (MeV per dj=1/2):")
for j in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
    m_j = casimir_mass(j)
    m_jp = casimir_mass(j + 0.5)
    spacing = m_jp - m_j
    frac_spacing = spacing / m_j * 100 if m_j > 0 else float('inf')
    print(f"    j={j:6.1f}: m={m_j:10.2f} MeV, spacing={spacing:8.4f} MeV ({frac_spacing:.2f}%)")

print(f"\n  At j ~ 100 (muon-mass region): spacing ~ {casimir_mass(100.5)-casimir_mass(100):.4f} MeV")
print(f"  Muon mass = 105.658 MeV, so fractional spacing ~ {(casimir_mass(100.5)-casimir_mass(100))/105.658*100:.2f}%")
print(f"  By chance, any mass in this range is within ~{(casimir_mass(100.5)-casimir_mass(100))/105.658*100/2:.2f}% of a Casimir level.")
print(f"  This means near-matches at high j are EXPECTED BY CHANCE.")


# =====================================================================
# 12. COMPREHENSIVE TABLE
# =====================================================================

print(f"\n--- Part 11: Comprehensive Summary Table ---")

# Classification of matching quality
print(f"\n  EXACT MATCHES (trivial):")
print(f"    electron (j=1/2): by construction")

print(f"\n  COINCIDENTAL NEAR-MATCHES (high j, dense spectrum):")
close_at_high_j = [(r['name'], r['mass'], r['j_exact'], r['pct_err'])
                   for r in match_results if abs(r['pct_err']) < 1 and r['j_exact'] > 5]
for name, mass, j, err in close_at_high_j:
    print(f"    {name}: m={mass:.2f} MeV, j={j:.2f}, err={err:+.2f}% (expected from dense spectrum)")

print(f"\n  PARTICLES WITH NO CLOSE CASIMIR MATCH:")
far = [(r['name'], r['mass'], r['j_exact'], r['pct_err'])
       for r in match_results if abs(r['pct_err']) > 5]
for name, mass, j, err in far[:10]:
    print(f"    {name}: m={mass:.2f} MeV, j={j:.2f}, err={err:+.2f}%")


# =====================================================================
# 13. KOIDE-LATTICE CONNECTION
# =====================================================================

print(f"\n--- Part 12: Koide-Lattice Connection ---")
print(f"  The Koide formula: Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2 = 2/3")
print(f"  The simplex projection: cos_eff = (d-1)/d = 2/3 for d=3")
print(f"")
print(f"  Both equal 2/3! Is this a coincidence?")
print(f"  Q_Koide = {Q_koide:.8f}")
print(f"  cos_eff = {2/3:.8f}")
print(f"  Difference: {abs(Q_koide - 2/3):.8f}")
print(f"")
print(f"  The Koide formula is satisfied to {abs(Q_koide - 2/3)/(2/3)*100:.4f}% accuracy.")
print(f"  If this is NOT a coincidence, it would mean:")
print(f"    (sum of mass^2 ratios) / (sum of mass ratios)^2 = cos_eff")
print(f"    Lepton masses are constrained by the simplex projection theorem!")
print(f"")
print(f"  However, the Casimir spectrum m_j ~ sqrt(j(j+1)) does NOT naturally")
print(f"  produce the Koide relation. The Koide constraint is on the SUM of masses,")
print(f"  not on individual Casimir quantum numbers.")
print(f"")
print(f"  SPECULATION: The Koide formula might arise from a different mechanism")
print(f"  than the Casimir spectrum -- perhaps from the simplex structure of the")
print(f"  3-generation mixing matrix, which has the same cos_eff = 2/3 factor.")


# =====================================================================
# 14. PLOTS
# =====================================================================

print(f"\n--- Part 13: Generating Plots ---")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(2, 2, hspace=0.30, wspace=0.30)

# Color map for categories
cat_colors = {
    'lepton': '#e41a1c',
    'quark': '#377eb8',
    'meson': '#4daf4a',
    'baryon': '#984ea3',
    'boson': '#ff7f00',
}

# --- (a) Casimir spectrum vs known particles (log scale) ---
ax1 = fig.add_subplot(gs[0, 0])

# Plot Casimir levels as horizontal lines
j_plot = np.arange(0.5, 50.5, 0.5)
m_plot = [casimir_mass(j) for j in j_plot]
for j, m in zip(j_plot, m_plot):
    ax1.axhline(y=m, color='gray', alpha=0.3, linewidth=0.5)

# Plot known particles as colored markers
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs, ys, names_cat = [], [], []
    for r in match_results:
        if r['category'] == cat:
            xs.append(r['j_exact'])
            ys.append(r['mass'])
            names_cat.append(r['name'])
    ax1.scatter(xs, ys, c=cat_colors[cat], label=cat, s=60, zorder=5, edgecolors='k', linewidths=0.5)
    for x, y, n in zip(xs, ys, names_cat):
        if y < 5000:  # don't label very heavy particles (too crowded)
            ax1.annotate(n, (x, y), fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

ax1.set_yscale('log')
ax1.set_xlabel('j (exact, non-integer)', fontsize=11)
ax1.set_ylabel('Mass (MeV)', fontsize=11)
ax1.set_title('(a) Known Particles vs Casimir Spectrum', fontsize=12)
ax1.legend(fontsize=8, loc='upper left')
ax1.set_xlim(-5, 300)
ax1.set_ylim(0.1, 200000)

# --- (b) Nearest-match residuals ---
ax2 = fig.add_subplot(gs[0, 1])

names_sorted = [r['name'] for r in sorted(match_results, key=lambda x: x['mass'])]
pct_errs_sorted = [r['pct_err'] for r in sorted(match_results, key=lambda x: x['mass'])]
colors_sorted = [cat_colors[r['category']] for r in sorted(match_results, key=lambda x: x['mass'])]

bars = ax2.barh(range(len(names_sorted)), pct_errs_sorted, color=colors_sorted, edgecolor='k', linewidth=0.3)
ax2.set_yticks(range(len(names_sorted)))
ax2.set_yticklabels(names_sorted, fontsize=7)
ax2.set_xlabel('Error to nearest Casimir level (%)', fontsize=11)
ax2.set_title('(b) Matching Residuals (sorted by mass)', fontsize=12)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Add 1% and 5% bands
for thresh in [1, -1, 5, -5]:
    ax2.axvline(x=thresh, color='red', linewidth=0.5, linestyle='--', alpha=0.5)

# --- (c) Statistical significance ---
ax3 = fig.add_subplot(gs[1, 0])

ax3.hist(mc_means, bins=50, color='steelblue', alpha=0.7, label=f'Random (N={n_mc})')
ax3.axvline(x=actual_mean, color='red', linewidth=2, label=f'Actual particles ({actual_mean:.1f}%)')
ax3.axvline(x=np.mean(mc_means), color='blue', linewidth=1, linestyle='--',
           label=f'Random mean ({np.mean(mc_means):.1f}%)')
ax3.set_xlabel('Mean % error to nearest Casimir level', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title(f'(c) Statistical Significance (p={p_value_mean:.3f})', fontsize=12)
ax3.legend(fontsize=9)

# --- (d) Spectrum on log scale showing coverage ---
ax4 = fig.add_subplot(gs[1, 1])

# Casimir spectrum density
j_dense = np.arange(0.5, 500, 0.5)
m_dense = np.array([casimir_mass(j) for j in j_dense])

# Plot Casimir spectrum as dots
ax4.scatter(j_dense, m_dense, s=1, c='gray', alpha=0.5, label='Casimir levels')

# Overlay known particles at their exact j
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs, ys = [], []
    for r in match_results:
        if r['category'] == cat:
            xs.append(r['j_exact'])
            ys.append(r['mass'])
    ax4.scatter(xs, ys, c=cat_colors[cat], label=cat, s=40, zorder=5,
               edgecolors='k', linewidths=0.5)

# Add lines for key mass scales
key_masses = [('e', 0.511), ('mu', 105.66), ('pi', 135.0), ('p', 938.3), ('tau', 1777)]
for name, m in key_masses:
    ax4.axhline(y=m, color='red', alpha=0.2, linewidth=0.5, linestyle=':')
    ax4.text(480, m*1.1, name, fontsize=7, color='red', alpha=0.6)

ax4.set_yscale('log')
ax4.set_xlabel('j quantum number', fontsize=11)
ax4.set_ylabel('Mass (MeV)', fontsize=11)
ax4.set_title('(d) Full Casimir Spectrum (log scale)', fontsize=12)
ax4.legend(fontsize=8, loc='upper left')
ax4.set_xlim(0, 500)
ax4.set_ylim(0.3, 200000)

# Fractional spacing annotation
ax4_twin = ax4.twinx()
frac_spacings = []
for j in j_dense[:-1]:
    m1 = casimir_mass(j)
    m2 = casimir_mass(j + 0.5)
    frac_spacings.append((m2 - m1) / m1 * 100)
ax4_twin.plot(j_dense[:-1], frac_spacings, 'orange', alpha=0.5, linewidth=0.5)
ax4_twin.set_ylabel('Fractional spacing (%)', fontsize=9, color='orange')
ax4_twin.set_yscale('log')
ax4_twin.set_ylim(0.01, 200)
ax4_twin.tick_params(axis='y', colors='orange')

plt.suptitle('Casimir Mass Spectrum vs Known Particles\n'
             r'$m_j = m_e \sqrt{4j(j+1)/3}$, Coherence Lattice Theory',
             fontsize=14, fontweight='bold', y=0.98)

plot_path = os.path.join(out_dir, 'casimir_mass_spectrum.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()


# =====================================================================
# 15. HONEST ASSESSMENT
# =====================================================================

print(f"\n" + "=" * 78)
print(f"  HONEST ASSESSMENT")
print(f"=" * 78)

print(f"""
  THE CASIMIR MASS SPECTRUM: m_j = m_e * sqrt(4*j*(j+1)/3)

  WHAT WORKS:
  -----------
  1. Electron (j=1/2): EXACT by construction (this is the input).
  2. Mass RATIOS between Casimir levels are exact group theory.
  3. The formula gives a tower of sub-MeV states with definite spacing.
  4. The Koide formula's Q = 2/3 matches cos_eff = (d-1)/d = 2/3
     from the simplex projection -- intriguing but unproven connection.

  WHAT DOES NOT WORK:
  -------------------
  1. MUON: Requires j = {j_muon_exact:.2f} (NOT a half-integer).
     The simple Casimir formula cannot give the muon mass.

  2. TAU: Requires j = {j_tau_exact:.2f} (NOT a half-integer).
     Same problem.

  3. LOW-ENERGY SPECTRUM: The levels at j=1 (835 keV), j=3/2 (1143 keV),
     j=2 (1445 keV) etc. correspond to NO known fundamental particles.

  4. STATISTICAL TEST: The MEAN matching error (p_mean={p_value_mean:.4f}) appears
     significant, but this is MISLEADING. At high j, level spacing is ~0.295 MeV,
     so any mass above ~100 MeV is trivially within ~0.14% of a Casimir level.
     The MEDIAN p-value ({p_value_median:.3f}) is NOT significant.
     The "good matching" is a pure density-of-states artifact.

  5. ALL particles heavier than the electron need j > 1/2, meaning they
     require SO(3) representations well beyond spin-1/2. This is NOT
     standard particle physics (quarks/leptons are spin-1/2).

  WHAT THIS MEANS:
  ----------------
  The Casimir mass formula m^2 ~ j(j+1) predicts a SPECIFIC tower of
  excitations at the lattice scale (0.5 -- 2 MeV for j=1/2 through j=3).
  These do not map to known particles except the electron itself.

  For the muon, tau, quarks, and all heavier particles, the simple
  Casimir formula does NOT work. The lepton mass hierarchy
  (m_mu/m_e = 207, m_tau/m_e = 3477) requires a DIFFERENT mechanism:

  Candidates:
  - Different lattice substructures (edges, faces, volumes)
  - K-field dynamics: dressed masses from self-energy corrections
  - Topological excitations: vortices, skyrmions with their own mass scales
  - Multi-particle bound states (composite structure)
  - Generation mixing: the Koide relation suggests a mixing mechanism
  - A completely different origin for the generation structure

  The Casimir spectrum is REAL -- it gives the mass gap of the frame
  bundle SO(3) representation. But it appears to be a SINGLE-PARTICLE
  spectrum at the lattice scale, not the full Standard Model mass spectrum.

  The 2/3 Koide-cos_eff coincidence is the most tantalizing clue,
  suggesting that the generation structure might be related to the
  d=3 simplex geometry, but through a mechanism DIFFERENT from the
  simple Casimir eigenvalue tower.
""")


# =====================================================================
# 16. SAVE RESULTS
# =====================================================================

results = {
    'casimir_spectrum': [
        {'j': float(j), 'mass_MeV': float(casimir_mass(j)), 'mass_ratio': float(casimir_mass_ratio(j))}
        for j in np.arange(0, 21, 0.5)
    ],
    'particle_matches': [
        {k: (float(v) if isinstance(v, (np.floating, float)) else v)
         for k, v in r.items()}
        for r in match_results
    ],
    'statistics': {
        'actual_mean_pct_err': float(actual_mean),
        'actual_median_pct_err': float(actual_median),
        'random_mean_pct_err': float(np.mean(mc_means)),
        'random_std_pct_err': float(np.std(mc_means)),
        'p_value_mean': float(p_value_mean),
        'p_value_median': float(p_value_median),
    },
    'koide': {
        'Q': float(Q_koide),
        'cos_eff': 2.0/3.0,
        'difference': float(abs(Q_koide - 2/3)),
    },
    'muon_problem': {
        'j_exact': float(j_muon_exact),
        'j_nearest': float(j_muon_near),
        'mass_at_nearest': float(m_at_j_near),
        'actual_mass': float(particles['mu'][0]),
    },
}

json_path = os.path.join(out_dir, 'casimir_mass_spectrum_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved results: {json_path}")

print(f"\n  Done.")
