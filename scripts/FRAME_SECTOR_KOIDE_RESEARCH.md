# Frame Sector and the Koide Formula: Research Report

**Researched:** 2026-04-04
**Domain:** SO(3) frame quantization, gauge-dressed vortex mass, lepton generation problem
**Confidence:** MEDIUM (framework is solid; the specific Koide mechanism is speculative)

---

## Summary

The phase sector (U(1)) cannot generate the lepton mass hierarchy: the simplex isotropy theorem guarantees that all three tetrahedral cross-section planes see identical K_eff (LT-215, FALSIFIED). Mass must come from the **frame sector** (SO(3)), where the Casimir spectrum m_l^2 ~ l(l+1) provides a tower of mass levels (LT-43, Thm 10.1). The key question is whether the gauge-dressed vortex mass -- the mass of a U(1) vortex dressed by SO(3) frame fluctuations through SU(2) gauge links -- depends on the frame angular momentum quantum number j in a way that selects three specific values satisfying the Koide relation Q = 2/3.

**Primary recommendation:** Measure the gauge-dressed vortex mass M_dressed(j) as a function of the frame angular momentum channel j using the D9 infrastructure with angular-momentum-projected frame configurations. If M_dressed(j) ~ M_cl + j(j+1)/(2*Lambda_vortex), then three j values satisfying the Brannen parametrization would constitute a derivation of Koide from the lattice.

---

## 1. What Was Falsified (LT-215)

The hypothesis that the 3 tetrahedral cross-sections provide different K_eff values was tested with the full D9 protocol (Kuramoto bootstrap + Shannon+Fiedler CLR). Result: Q ~ 1/3 (equal-mass limit), theta_K ~ 90 degrees, mass ratios 1:1:1 for all three vortex axes.

**Root cause:** The simplex isotropy theorem guarantees every bond projects 2/3 of its norm-squared onto ALL 3 cross-section planes. Bond death removes equal coupling from all planes. The isotropy is too perfect -- the same property that makes cos_eff = (d-1)/d = 2/3 exact (enabling the alpha derivation) prevents the cross-section mechanism from splitting masses.

**What this rules out definitively:**
- Any mass-splitting mechanism in the PHASE sector alone
- Linear K_eff-to-mass mapping through cross-section projections
- Any mechanism relying solely on the U(1) vortex without frame dressing

---

## 2. The Frame Sector Mass Theory

### 2.1 Casimir Spectrum (LT-43, Thm 10.1)

Frame excitations in angular momentum channel l have mass-squared:

    m_l^2 = (nu_R / c^2) * l(l+1)

where nu_R = K_frame * h^(2-d) is the continuum frame coupling stiffness. With the SU(2) cover (LT-45), half-integer j is allowed:

| Channel j | m^2 proportional to | Statistics |
|-----------|---------------------|------------|
| j = 0     | 0                   | Boson      |
| j = 1/2   | 3/4                 | Fermion    |
| j = 1     | 2                   | Boson      |
| j = 3/2   | 15/4                | Fermion    |
| j = 2     | 6                   | Boson      |

The j = 1/2 state is the lightest massive fermion (the electron analog). But LT-43 gives mass RATIOS between channels, not the absolute mass scale, and not why exactly three fermionic channels are populated.

### 2.2 The Naive Casimir Tower Problem

The naive m ~ sqrt(j(j+1)) with j = 1/2 as the electron gives:
- Muon would need j ~ 178.6 (not half-integer)
- Tau would need j ~ 3011 (not half-integer)

These are not small quantum numbers. The naive Casimir tower does NOT reproduce the SM mass spectrum (previously documented in casimir_mass_spectrum.py as an HONEST NEGATIVE).

**Critical insight:** The mass hierarchy is not from different j values of BULK frame excitations. It must come from the dressing of a TOPOLOGICAL object (the vortex) by the frame sector.

### 2.3 The Gauge-Dressed Vortex

The electron is a U(1) phase vortex (pi_1). Its mass receives contributions from:

1. **Phase sector:** Vortex core energy from the K-depletion zone (bare vortex mass M_vortex ~ K * ln(L/xi))
2. **Frame sector dressing:** The gauge CLR dresses the vortex core with SO(3) frame topology. The effective coupling at the core is:

       cos_eff_ij = cos(Delta_phi_ij) * cos_dressed_ij

   where cos_dressed_ij = (1/3) tr(R_i^T V_ij R_j)

3. **Gauge topology reservoir:** D9 proved M_dressed ~ 4x M_bare during nucleation (LT-198). Most of the mass is in the gauge sector.

**The key physics:** The gauge-dressed vortex mass depends on which angular momentum channel j the frame sector occupies near the vortex core. Different j values give different cos_dressed profiles, different M_dressed, and thus different physical masses.

---

## 3. The Koide Formula: Mathematical Structure

### 3.1 The Formula

    Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2 = 2/3

Experimental: Q = 0.666660 +/- 0.000005 (9 ppm agreement with 2/3).

### 3.2 Brannen Parametrization

The square roots of the charged lepton masses follow:

    sqrt(m_n) = mu * (1 + sqrt(2) * cos(delta + 2*pi*n/3))

where:
- mu is an overall mass scale
- delta = 2/9 radians (the "Koide angle")
- n = 0, 1, 2 labels the three generations

This gives Q = 2/3 EXACTLY for ANY value of delta and mu. The value delta = 2/9 fixes the specific mass ratios m_e : m_mu : m_tau.

### 3.3 Why Q = 2/3

The formula Q = 2/3 is a geometric identity: it says the vector (sqrt(m_e), sqrt(m_mu), sqrt(m_tau)) makes a 45-degree angle with the vector (1, 1, 1) in R^3. Equivalently, the S_3-symmetric part of the squared mass vector is exactly 1/3 of the total.

### 3.4 The 2/3 Coincidence

In the coherence lattice:
- cos_eff = (d-1)/d = 2/3 from the simplex projection (proven, used in the alpha derivation)
- Koide Q = 2/3 experimentally

This is the SAME number 2/3. If this is not a coincidence, there must be a mechanism connecting simplex geometry to the lepton mass relation. The simplex isotropy that gives cos_eff = 2/3 acts on the PHASE sector. The Koide formula acts on the MASS sector. The bridge between them is the gauge-dressed product coupling.

---

## 4. Literature on Koide Mechanisms

### 4.1 Koide's Original Mechanism (1981)

Mass matrix from preon model: m ~ (z_0 + z_i)^2 with constraints z_1 + z_2 + z_3 = 0 and (z_1^2 + z_2^2 + z_3^2)/3 = z_0^2. The S_3 permutation symmetry plus the constraint gives Q = 2/3. Preon model predictions for quarks are now falsified, but the mass matrix structure survives.

### 4.2 Sumino's Family Gauge Symmetry (2009)

U(3) family gauge symmetry cancels QED radiative corrections to Koide's formula, explaining why it holds at low energies despite running. Key paper: arXiv:0812.2103.

### 4.3 Foot's Geometric Interpretation

The Koide formula as a dot product: (sqrt(m) . (1,1,1)) / (|sqrt(m)| * |(1,1,1)|) = cos(45 degrees). This is a constraint on the direction of the mass vector in sqrt-mass space.

### 4.4 Brannen's Trigonometric Form

sqrt(m_n) = mu(1 + sqrt(2) cos(2*pi*n/3 + delta)) with delta = 2/9. Extended to neutrinos with delta_nu = 2/9 + pi/12 (assuming negative sqrt mass for lightest neutrino). Extended to hadron triplets in some cases.

### 4.5 The 2/9 from G2 Casimirs (Recent)

Claim: delta = 2/9 = C_2(3)/C_2(Sym^3(3)) within the embedding SU(3) subset G_2 = Aut(O), where G_2 is the automorphism group of the octonions. If validated, this connects the Koide angle to exceptional Lie group structure. Status: UNVERIFIED CONJECTURE.

### 4.6 Solitonic Coherence Approach (2025)

A recent preprint (preprints.org 202505.2156) derives the Koide formula from phase coherence of topological solitons, assigning topological weight 2 to leptons from pi_1(M) = Z_2 (the double cover / spinor structure). Claims Q = 2/3 as a topological identity. Status: NOT PEER REVIEWED.

### 4.7 What Nobody Has Done

**Nobody has connected the Koide formula to lattice gauge theory with explicit SO(3) frame dressing of vortices.** The closest approach is the solitonic coherence paper, which uses topological soliton language but not lattice gauge dynamics. The coherence lattice framework is in genuinely novel territory here.

---

## 5. The Proposed Mechanism: Gauge-Dressed Vortex Mass with Frame Quantum Numbers

### 5.1 Physical Picture

The electron is a U(1) vortex (pi_1(S^1) = Z) dressed by the SO(3) frame sector through SU(2) gauge links. The dressing creates an effective mass that depends on the angular momentum channel j of the frame fluctuations at the vortex core.

In the Skyrme model (the closest established analogy), the nucleon and Delta masses are:

    M = M_cl + j(j+1) / (2 * Lambda)

where M_cl is the classical Skyrmion mass and Lambda is the moment of inertia. The nucleon (j = 1/2) and Delta (j = 3/2) are the two lowest-lying states. This is rigid-body quantization of the zero modes.

### 5.2 Adaptation to the Coherence Lattice

For the gauge-dressed vortex:

    M_vortex(j) = M_core(K) + j(j+1) * f(K, beta_g)

where:
- M_core(K) is the bare vortex core energy (from the phase sector K-depletion)
- j is the angular momentum quantum number of the frame configuration at the core
- f(K, beta_g) is a function of the coupling and gauge coupling

The three charged lepton masses would correspond to three specific j values. Using the Brannen parametrization, we need:

    sqrt(M_vortex(j_n)) = mu * (1 + sqrt(2) * cos(2*pi*n/3 + delta))

for n = 0, 1, 2 with some specific j_0, j_1, j_2.

### 5.3 Why This Might Work

1. **The 2/3 connection:** cos_eff = 2/3 from simplex geometry appears in the gauge-dressed product coupling. The Koide Q = 2/3 involves the SAME number. If M_vortex depends on cos_eff through the gauge dressing, the 2/3 propagates to the mass relation.

2. **The Skyrme analogy:** In the Skyrme model, rigid-body quantization of zero modes gives EXACTLY the j(j+1) mass splitting. The coherence lattice frame sector has the same SO(3) structure.

3. **Three generations from selection rules:** Only three j values may be stable against decay to lower j states. Or only three j values may be compatible with the vortex topology (e.g., j = 1/2, 3/2, 5/2 if the vortex winding constrains the allowed representations).

4. **The gauge reservoir:** D9 showed M_dressed/M_bare ~ 4 during nucleation. The gauge sector carries most of the mass. Different j values at the core change the gauge-frame coupling (cos_dressed) and thus the gauge contribution to the mass.

### 5.4 Why This Might Fail

1. **Naive j(j+1) doesn't work:** The simple m ~ sqrt(j(j+1)) with j = 1/2, 3/2, 5/2 gives mass ratios sqrt(3/4) : sqrt(15/4) : sqrt(35/4) = 1 : 2.24 : 3.42. The actual ratios are 1 : 206.8 : 3477. The required hierarchy is HUGE. Linear j(j+1) is far too gentle.

2. **BKT amplification needed:** The vortex mass depends on K through xi ~ exp(b/sqrt(K - K_BKT)). If the frame dressing modifies K_eff slightly differently for each j, the BKT essential singularity could amplify tiny K_eff differences into the 1:207:3477 hierarchy. This is the most promising (and most speculative) piece.

3. **Selection of exactly 3 j values:** The frame spectrum is infinite (j = 1/2, 3/2, 5/2, ...). Why exactly 3 contribute? Possible answers:
   - Stability: only the 3 lowest j states are topologically stable against vortex pair annihilation
   - Energetics: higher j vortices decay too fast to be observed as particles
   - Symmetry: the tetrahedral symmetry of the diamond lattice (T_d has dimension... but the point group has specific representations)

4. **The angle delta = 2/9:** Even if we get Q = 2/3, we need delta = 2/9 to get the right mass ratios. Where does 2/9 come from in the lattice? The G2 Casimir claim is interesting but unverified, and G2 structure is not obviously present in the diamond lattice.

---

## 6. Computable Predictions Using D9 Infrastructure

### 6.1 Experiment 1: Angular-Momentum-Projected Frame Energy

**Measure the vortex core energy as a function of frame j.**

Protocol:
1. Run standard D9 (Kuramoto bootstrap + gauge CLR) to establish vortex + gauge configuration
2. At equilibrium, project the frame configuration at core bonds onto angular momentum channels j = 1/2, 1, 3/2, 2, 5/2, 3
3. For each j channel, compute:
   - M_dressed(j) = total energy of vortex with frame sector restricted to channel j
   - cos_dressed_core(j) = average gauge-dressed cosine at core bonds in channel j
   - The K_eff(j) = K * cos_eff * cos_dressed_core(j)

**Implementation:** Use the Peter-Weyl decomposition. The D^j matrices are known. Project R_i onto the j-th irreducible representation and compute the restricted energy.

**Observable:** M_dressed(j) vs j. Plot. Check for j(j+1) dependence and for BKT-amplified hierarchy.

### 6.2 Experiment 2: Effective K Modulation by Frame Channel

**Measure how the frame angular momentum modifies K_eff at the vortex core.**

Protocol:
1. Freeze a D9 vortex+gauge configuration
2. Impose a specific frame j at core bonds:
   - j = 1/2: construct a hedgehog-like frame with minimal angular momentum
   - j = 3/2: construct a frame with the next-higher representation
   - Measure K_eff in each case
3. Feed K_eff(j) through the BKT correlation length:
   - xi(j) ~ exp(b / sqrt(K_eff(j) - K_BKT))
   - M_vortex(j) ~ K_eff(j) * ln(L / xi(j))

**Observable:** M_vortex(j) ratios. Compare to lepton mass ratios.

### 6.3 Experiment 3: Gauge-Dressed Moment of Inertia

**Measure the rotational moment of inertia of the gauge-dressed vortex.**

Protocol:
1. At D9 equilibrium, perturb the frame configuration by a global SO(3) rotation of angle epsilon
2. Measure the energy cost: Delta_E = epsilon^2 / (2 * Lambda_vortex)
3. Lambda_vortex is the moment of inertia

If Lambda_vortex is small, the j(j+1)/(2*Lambda) term is large and mass splitting is significant. If Lambda_vortex is large, the rotational contribution is negligible.

**Observable:** Lambda_vortex and the mass splitting j(j+1)/(2*Lambda_vortex) for j = 1/2, 3/2, 5/2.

### 6.4 Experiment 4: Cross-Check with Koide Angle

**Test whether any set of 3 frame quantum numbers gives Q = 2/3 with delta = 2/9.**

Protocol:
1. From Experiments 1-3, extract M_vortex(j) for j = 1/2, 3/2, 5/2, 7/2, ...
2. For all triplets (j_a, j_b, j_c), compute:
   - Q(j_a, j_b, j_c) = (M_a + M_b + M_c) / (sqrt(M_a) + sqrt(M_b) + sqrt(M_c))^2
   - theta_K = the Koide angle from these masses
3. Search for triplets with Q ~ 2/3 and theta_K ~ 2/9

**Observable:** The (j_a, j_b, j_c) triplet that best matches Koide.

---

## 7. The BKT Amplification Mechanism (Key Speculative Element)

### 7.1 The Problem

The mass ratios m_e : m_mu : m_tau = 1 : 207 : 3477 span 3.5 orders of magnitude. No simple polynomial function of quantum numbers can produce this. But the BKT correlation length:

    xi(K) ~ exp(b / sqrt(K - K_BKT))

is an essential singularity. Near K_BKT, tiny changes in K produce huge changes in xi (and thus in the vortex energy).

### 7.2 The Mechanism

If the frame j value shifts K_eff slightly:

    K_eff(j) = K_BKT + delta_K(j)

where delta_K(j) is a small correction from the frame dressing, then:

    M_vortex(j) ~ K_eff * ln(L/xi) ~ K_eff * (b / sqrt(delta_K(j)))

and the mass RATIO between two generations is:

    M_vortex(j_1) / M_vortex(j_2) ~ sqrt(delta_K(j_2) / delta_K(j_1))

This is algebraic (not exponential) in the delta_K ratio, which may not be enough. The full BKT formula is:

    xi ~ a * exp(b / sqrt(K/K_BKT - 1))

and the vortex energy involves an integral over the K profile. The amplification could be much stronger if the frame dressing affects K right at the BKT threshold.

### 7.3 Required Numerical Test

Compute delta_K(j) for j = 1/2, 3/2, 5/2 from Experiment 2. Feed into the BKT formula. Compare mass ratios to 1 : 207 : 3477.

**If the BKT amplification is insufficient, the mechanism fails.** This is the make-or-break test.

---

## 8. Conventions

| Choice | Convention | Alternatives | Source |
|--------|-----------|--------------|--------|
| Frame group | SO(3) with SU(2) cover | O(3) | LT-43, LT-45 |
| Gauge group | SU(2) on bonds | SO(3) | D9, LT-195 |
| Mass formula | m^2 ~ j(j+1) for bulk | m ~ j(j+1) | LT-43 Thm 10.1 |
| Koide parametrization | Brannen trigonometric | Koide preon | PDG + Brannen 2006 |
| BKT critical coupling | K_BKT = 2/pi (bare, z=4) | K_BKT = pi/(2z) | LT-214 |
| Lattice | 3-diamond, d=3, z=4 | FCC, simple cubic | Throughout |

---

## 9. Existing Results to Leverage (DO NOT RE-DERIVE)

| Result | Source | Role |
|--------|--------|------|
| cos_eff = (d-1)/d = 2/3 | Simplex isotropy theorem | The 2/3 that appears in both alpha and Koide |
| m_l^2 ~ l(l+1) | LT-43 Thm 10.1 | Casimir mass hierarchy from frame sector |
| Gauge-dressed cos = (1/3) tr(R^T V R') | Gauge-dressed coherence capital Def 2.1 | How the frame dresses the vortex |
| M_dressed ~ 4x M_bare | LT-198 | Gauge reservoir: mass is mostly in gauge sector |
| Spontaneous Skyrmion genesis | LT-197 | Topology is an attractor |
| Wilson confinement stabilizes Skyrmions | LT-195 | Gauge confinement provides permanent stability |
| Brannen parametrization | Brannen 2006 | sqrt(m_n) = mu(1 + sqrt(2) cos(2pi*n/3 + 2/9)) |
| Koide Q = 2/3 exact for any delta, mu | Mathematical identity | Q = 2/3 follows from the S_3-symmetric parametrization |

---

## 10. Common Pitfalls

### Pitfall 1: Confusing Bulk Casimir with Vortex Dressing

**What goes wrong:** Assuming m ~ sqrt(j(j+1)) for j = 1/2 gives the electron mass, and searching for j ~ 179 for the muon.
**Why it fails:** The naive Casimir tower gives mass ratios of order 1, not order 1000. Lepton masses are not rigid-rotor levels of the bulk frame field.
**How to avoid:** Always distinguish BULK frame excitations (LT-43) from VORTEX-DRESSED frame states. The mass hierarchy comes from the vortex dressing, not the bulk spectrum.

### Pitfall 2: Expecting the Phase Sector to Split Masses

**What goes wrong:** Trying to get different K_eff per cross-section plane.
**Why it fails:** Simplex isotropy (LT-215, FALSIFIED).
**How to avoid:** Accept that mass splitting MUST come from the frame sector.

### Pitfall 3: Forgetting BKT Essential Singularity

**What goes wrong:** Using linear or polynomial mass-vs-quantum-number relations.
**Why it fails:** Cannot produce 3+ orders of magnitude hierarchy from small quantum number differences.
**How to avoid:** Always include the BKT correlation length xi ~ exp(b/sqrt(K - K_BKT)) in the vortex mass formula.

### Pitfall 4: Ignoring the Gauge Reservoir

**What goes wrong:** Computing vortex mass from phase + frame sectors only, ignoring gauge links.
**Why it fails:** D9 proved that M_dressed ~ 4x M_bare. The gauge sector carries most of the energy.
**How to avoid:** Always include gauge link energy in the dressed vortex mass.

### Pitfall 5: Numerology vs Derivation

**What goes wrong:** Finding 3 j values that fit the Koide formula by scanning, without a selection mechanism.
**Why it fails:** With infinitely many j values, finding a matching triplet is not surprising -- it's parameter fitting.
**How to avoid:** Need a MECHANISM that selects the three j values (stability, topology, symmetry) independently of the mass measurement.

---

## 11. Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
|-------|------------------|----------------|-----------------|
| cos_eff recovery | Gauge dressing reduces to bare in V=I limit | Set V=I, check cos_dressed = tr(R^T R')/3 = cos_bare | Exact agreement |
| Budget conservation | CLR dynamics conserve total K | Sum gauge CLR contributions | Sum < 1e-10 |
| Symmetry of M_dressed(j) | Degenerate j substates give same energy | Check m_j = m_{-j} | Exact |
| BKT scaling | xi diverges at K_BKT | Measure xi vs K near K_BKT | Exponential divergence |

### Known Limits

| Limit | Parameter Regime | Known Result | Source |
|-------|-----------------|--------------|--------|
| No gauge (V=I) | beta_g = 0 | Bare vortex mass, no dressing | LT-02 |
| Strong gauge | beta_g >> 1 | Wilson confinement, cos_dressed -> 1 | LT-195 |
| j = 0 | Trivial frame | M_vortex = M_core (no frame contribution) | Def. |
| j -> infinity | Dense rotor spectrum | M_vortex -> infinity | Casimir unbounded |

### Red Flags

- If M_dressed(j) is independent of j: frame dressing is negligible, mechanism fails
- If M_dressed(j) depends on j but not through j(j+1): the SO(3) Casimir is not the relevant quantum number
- If no set of 3 j values gives Q near 2/3: the Koide connection to the lattice is coincidental
- If the mass hierarchy requires non-half-integer j: the fermionic interpretation is lost

---

## 12. Computational Tools

### Core (Already Available)

| Tool | Location | Purpose |
|------|----------|---------|
| D9 vortex protocol | d9_vortex_mc_gauge.py | Vortex + gauge CLR infrastructure |
| Koide measurement | koide_d9_protocol.py | Cross-section K_eff measurement (modify for frame) |
| Diamond lattice builder | d9_vortex_mc_gauge.py::build_diamond_lattice | Lattice geometry |
| Fiedler eigenvector | koide_d9_protocol.py::deterministic_fiedler | Structural sensitivity |
| SU(2) gauge links | d9_vortex_mc_gauge.py | V_ij matrices, Wilson action |

### Needed (Build or Import)

| Tool | Purpose | Complexity |
|------|---------|------------|
| Peter-Weyl projector | Project R_i onto D^j representation | Medium (use Wigner D-matrices) |
| Frame channel energy | Compute energy restricted to j channel | Medium |
| Gauge-dressed moment of inertia | Lambda_vortex from energy vs rotation angle | Low |
| BKT correlation length | xi(K) from the vortex mass formula | Low (known formula) |

### Python Packages

```python
# Already available in the project
import numpy as np
from scipy.special import i0, i1  # Bessel functions for R_0(K)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# May need for Wigner D-matrices
# pip install sympy  (for Wigner d-function computation)
# Or implement directly: d^j_{m'm}(beta) from recursion
```

---

## 13. The Skyrme Model Analogy (Closest Established Physics)

In the standard Skyrme model:
- The Skyrmion is a topological soliton in pi_3(SU(2)) = Z
- Quantization of the rotational zero modes gives M = M_cl + j(j+1)/(2*Lambda)
- Lambda is the moment of inertia (computable from the classical profile)
- j = 1/2 is the nucleon (938 MeV), j = 3/2 is the Delta (1232 MeV)
- Mass ratio Delta/nucleon = 1.31

The coherence lattice analog:
- The electron is a topological soliton in pi_1(S^1) = Z
- The frame sector dresses it with SO(3) quantum numbers
- Quantization should give M_vortex(j) = M_core + j(j+1)/(2*Lambda_vortex)
- Lambda_vortex depends on the gauge coupling beta_g and the vortex profile

**Key difference from Skyrme:** In the Skyrme model, Lambda is fixed by the classical Skyrmion profile -- there is ONE moment of inertia for ONE soliton. Different j values give different states of the SAME soliton.

For leptons, we need different particles (e, mu, tau) with masses spanning 3.5 decades. The simple j(j+1)/(2*Lambda) with fixed Lambda gives mass differences of order 1/Lambda, not ratios of order 1000. This is why the BKT amplification is essential.

---

## 14. Proposed Strategy (Priority Order)

### Phase A: Reconnaissance (1-2 days)

1. **Measure Lambda_vortex:** Using D9 at equilibrium, perturb frame by global SO(3) rotation, extract the rotational energy cost. This determines whether the frame dressing contributes significantly to the vortex mass.

2. **Measure K_eff(j) at core:** For j = 1/2, 3/2, 5/2, compute the gauge-dressed K_eff at core bonds when the frame is restricted to channel j. This gives delta_K(j).

3. **Quick BKT check:** Feed delta_K(j) into the BKT formula. Can we get mass ratios of order 100-1000? If yes, proceed. If delta_K differences are too small or BKT amplification is insufficient, STOP.

### Phase B: Detailed Measurement (3-5 days, only if Phase A shows promise)

4. **Full M_dressed(j) curve:** Sweep j from 1/2 to 5/2 in steps of 1/2, measure M_dressed at each j. Multiple seeds, L = 8 and L = 10.

5. **Koide triplet search:** From M_dressed(j) data, compute Q for all triplets. Look for Q ~ 2/3.

6. **Stability analysis:** Which j values are topologically stable? Can higher-j vortices decay?

### Phase C: Derivation (1-2 weeks, only if Phase B gives Koide)

7. **Derive the Koide angle:** If a specific (j_a, j_b, j_c) gives Q = 2/3, derive the Koide angle delta = 2/9 from the lattice parameters.

8. **Running of the Koide formula:** Does the gauge-dressed mechanism provide the Sumino-like protection against QED running?

---

## 15. Open Questions

1. **How does the frame j value couple to the vortex core?** The cos_dressed = (1/3) tr(R^T V R') depends on the specific frame configuration, which decomposes into j channels. But the product coupling cos(Delta_phi) * cos_dressed means that at the vortex core (cos(Delta_phi) ~ 0), the frame contribution is SUPPRESSED. Is the frame dressing important where the vortex is, or only in the bulk?

   **This is potentially a show-stopper.** If the product rule kills the frame contribution at exactly the location (the vortex core) where it matters, the mechanism fails. Counter-argument: the independence phase of D9 showed that cos_dressed RECOVERS to 0.88 even after the vortex is removed, suggesting the frame dressing is not killed by the vortex but rather reorganizes around it.

2. **Why 3 generations specifically?** The diamond lattice has T_d point group symmetry. T_d has irreps of dimension 1, 1, 2, 3. Could the 3-dimensional irrep (T_1 or T_2) select exactly 3 frame channels? This would connect the number of generations to the lattice symmetry.

3. **Where does delta = 2/9 come from?** The G_2 Casimir claim (C_2(3)/C_2(Sym^3(3)) = 2/9) is intriguing but G_2 is not obviously present in the diamond lattice. The diamond lattice has Fd-3m symmetry, which is a subgroup of O_h. No obvious G_2 connection.

4. **Renormalization protection:** The Koide formula holds at the pole mass (low energy), which is remarkable because QED radiative corrections would normally spoil it. Sumino's family gauge symmetry mechanism cancels these corrections. Does the gauge CLR provide an analogous protection?

---

## 16. Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost |
|---------------|-----------|-----------|------|
| Frame j dressing | Product rule kills frame at core | Gauge plaquette dressing (Wilson action j-dependence) | Moderate: new observable |
| BKT amplification | delta_K too small | Non-perturbative vortex mass from D9 energy measurement | Low: already have infrastructure |
| Selection of 3 j | No mechanism for exactly 3 | Inter-vortex interaction spectrum (bound states) | High: multi-vortex simulation |
| The whole approach | No frame j dependence in M_dressed | Abandon Casimir route; try RG-generated cross-section anisotropy (direction 3 in negative result notes) | High: requires new theoretical framework |

---

## 17. Sources

### Primary (HIGH confidence)
- LT-43: Frame Quantization Theorem (frame_quantization_theorem.md) -- Casimir spectrum, mass hierarchy
- LT-195, LT-197, LT-198: Skyrmion genesis papers (spontaneous_skyrmion_genesis.md) -- gauge confinement, topology reservoir
- LT-215: Koide cross-section negative result (journal/theory/2026-04-04-koide-cross-section-negative.md)
- Gauge-dressed coherence capital paper (gauge_dressed_coherence_capital.md) -- product rule, torque calculus
- [Koide formula - Wikipedia](https://en.wikipedia.org/wiki/Koide_formula) -- standard reference for the formula
- [Brannen, "The Lepton Masses"](https://brannenworks.com/MASSES2.pdf) -- trigonometric parametrization

### Secondary (MEDIUM confidence)
- [Sumino, "Family gauge symmetry and Koide's mass formula"](https://arxiv.org/abs/0812.2103) -- family gauge protection
- [Rivero & Gsponer, "The strange formula of Dr. Koide"](https://arxiv.org/abs/hep-ph/0505220) -- historical review
- [Gudnason & Halcrow, "Nonlinear rigid-body quantization of Skyrmions"](https://arxiv.org/abs/2311.11667) -- Skyrme model quantization
- [Baez, "The Koide Formula"](https://johncarlosbaez.wordpress.com/2021/04/04/the-koide-formula/) -- accessible overview
- [Dynamical mass generation for ferromagnetic skyrmions](https://ui.adsabs.harvard.edu/abs/2022JMMM..56470062W/abstract) -- skyrmion mass generation

### Tertiary (LOW confidence)
- [G2 Casimir derivation of theta = 2/9](https://www.physicsforums.com/threads/koide-angle-th-2-9-from-g-casimirs-neutrino-mass-prediction.1084336/) -- Physics Forums, unverified
- [Koide formula from phase coherence of solitons](https://www.preprints.org/manuscript/202505.2156) -- preprint, not peer reviewed

---

## 18. Caveats and Self-Critique

### What assumption might be wrong?

The central assumption is that the frame angular momentum j is a good quantum number for the gauge-dressed vortex. In reality, the vortex core is a strongly interacting region where the frame, gauge, and phase sectors are all coupled. The j decomposition may not be meaningful in this non-perturbative regime.

### What alternative was dismissed too quickly?

The RG-generated anisotropy (option 3 in the negative result notes): different cross-sections see different RENORMALIZED K at different scales. This breaks the bare isotropy that killed LT-215 but requires understanding the lattice RG flow, which is a separate research program.

### What limitation am I understating?

The BKT amplification is the weakest link. The mass hierarchy of 3.5 decades requires the three j values to produce K_eff values that differ by a PRECISE amount near K_BKT. This is fine-tuning unless there is a mechanism that pins the K_eff values relative to K_BKT. The CLR self-organized criticality (LT-214) pins the BULK K to K_BKT -- but whether it also pins the core K_eff(j) differences is unknown.

### Is there a simpler method I overlooked?

The simplest explanation for Q = 2/3 would be if it follows directly from cos_eff = 2/3 by some identity, without involving the frame sector at all. But the negative result (LT-215) rules this out for the phase sector. The frame sector IS the simpler remaining option.

### Would a specialist disagree?

A lattice gauge theorist would likely question whether the SO(3) frame field on the coherence lattice is sufficiently analogous to QCD to use Skyrme model intuition. The coherence lattice is a classical dynamical system with CLR dynamics, not a quantum field theory. The "quantization" (LT-43) is formal -- it constructs a Hilbert space on SO(3) but does not demonstrate that the lattice dynamics populates it quantum mechanically. This is a real gap.

---

## Metadata

**Confidence breakdown:**
- Mathematical framework (Casimir spectrum, Brannen parametrization): HIGH -- textbook results applied correctly
- Standard approaches (Skyrme quantization analogy): MEDIUM -- the analogy is qualitative, not exact
- Computational tools (D9 infrastructure): HIGH -- tested and validated
- Proposed mechanism (gauge-dressed vortex mass with frame j): LOW -- genuinely novel, untested
- BKT amplification of mass hierarchy: LOW -- speculative, requires numerical verification
- Koide angle delta = 2/9 from lattice: LOW -- no mechanism identified yet

**Research date:** 2026-04-04
**Valid until:** Results are stable unless D9 infrastructure changes or new lattice theorems are proved
