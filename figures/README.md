# Figures

## Main Figures

| Figure # | Description | Source Script | Output File |
|----------|------------|---------------|-------------|
| Table 1 | Progressive LCE convergence (CENTRAL TABLE) | `two_vertex_lce.py` | `two_vertex_lce.png` |
| Table 2 | Proof status assessment | (text table in paper) | — |
| Fig. 1 | LCE convergence visualization | `two_vertex_lce.py` | `lce_alpha_visualization.html` |
| Fig. 2 | Running alpha curve $\alpha(Q)$ | `running_alpha_curve.py` | `running_alpha_curve.png` |
| Fig. 3 | Diamond lattice Green's function $G_\text{diff} = 1/z$ | `diamond_greens_function.py` | `diamond_greens_function_3d.png` |
| Fig. 4 | BKT RG flow and $1/\sqrt{e}$ connection | `bkt_rg_flow.py` | `bkt_rg_flow.png` |
| Fig. 5 | Plaquette MC vertex factor | `plaquette_correction_alpha.py` | `plaquette_correction_alpha.png` |
| Fig. 6 | Casimir mass spectrum | `casimir_mass_spectrum.py` | `casimir_mass_spectrum.png` |

## Files in This Directory

| File | Description |
|------|-------------|
| `two_vertex_lce.png` | LCE progressive convergence (Table 1 visual) |
| `running_alpha_curve.png` | $\alpha(Q)$ from UV to IR |
| `diamond_greens_function_3d.png` | 3D visualization of lattice Green's function |
| `bkt_rg_flow.png` | BKT RG trajectories and $1/\sqrt{e}$ |
| `plaquette_correction_alpha.png` | MC plaquette vertex measurement |
| `casimir_mass_spectrum.png` | Mass spectrum $m_j = m_e\sqrt{4j(j+1)/3}$ |
| `lce_alpha_visualization.html` | Interactive LCE convergence (HTML) |

## Scripts (in `scripts/` directory)

### Core α Derivation

| Script | Purpose | Lines |
|--------|---------|-------|
| `alpha_137_verification.py` | Full α formula verification | — |
| `g_factor_from_lattice.py` | α → QED series → g-factor | — |
| `two_vertex_lce.py` | Two-vertex LCE correction (c = 2.9858) | — |
| `double_plaquette_lce.py` | Single-vertex LCE (Layer 1) | — |
| `alpha_crossover_scale.py` | VP crossover scale matching | — |
| `diamond_greens_function.py` | $G_\text{diff} = 1/z$ verification | — |
| `bkt_rg_flow.py` | BKT RG flow, $1/\sqrt{e}$ connection | — |
| `running_alpha_curve.py` | Running α from UV to IR | — |
| `electron_mass_from_lattice.py` | Lattice spacing h = √3/2 λ_C | — |
| `casimir_mass_spectrum.py` | Casimir mass tower | — |

### Vortex and Skyrmion

| Script | Purpose | Lines |
|--------|---------|-------|
| `da1_spontaneous_vortex.py` | Spontaneous vortex from co-evolution | 825 |
| `d9_vortex_mc_gauge.py` | D9 spontaneous Skyrmion genesis | 1968 |
| `gauge_skyrmion.py` | Gauge confinement stabilization | 1825 |
| `skyrmion_spontaneous.py` | Kuramoto bootstrap nucleation | 2050 |
| `coupled_vortex_skyrmion.py` | Vortex-Skyrmion interaction | 2007 |
| `d9_frame_gfactor.py` | Adiabatic B-ramp g-factor measurement | 1575 |

### Verification and Measurement

| Script | Purpose | Lines |
|--------|---------|-------|
| `plaquette_correction_alpha.py` | MC lattice vertex factor measurement | — |
| `alpha_analytical_corrections.py` | Perturbative corrections (all wrong direction) | 665 |
| `vertex_rg_flow.py` | Vertex RG at BKT, α universality | 825 |
| `vD_eff_measurement.py` | Effective Dirac velocity, v_D² = 1/3 | 552 |
| `living_vacuum_thermal.py` | Thermal CLR dynamics simulation | 428 |

## Data Files (in `data/` directory)

| File | Description |
|------|-------------|
| `casimir_mass_spectrum_results.json` | Casimir mass spectrum computation |
| `plaquette_correction_alpha_results.json` | Plaquette MC measurement results |
| `d9_frame_gfactor_L8.json` | D9 g-factor scan at L=8 |

## Source Locations

All original scripts: `experiments/lattice_theory/scripts/`
All original outputs: `experiments/lattice_theory/out/`
