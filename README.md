# TRACEBIND: Hierarchical Spatial & Kinematic Descriptor Framework

[![Pipeline Status](https://img.shields.io/badge/Pipeline-PASSING-success.svg)](#execution--reproducibility)
[![Phase 6 Baseline](https://img.shields.io/badge/Phase%206-FROZEN%20v1.0-blue.svg)](https://github.com/bigwiginfohub-wq/TRACEBIND-Atmosphere/releases/tag/phase6-v1.0)
[![Phase 8 C2 Audit](https://img.shields.io/badge/Phase%208%20C2-FROZEN%20%26%20AUDITED-success.svg)](#phase-8-c2-blinded-empirical-validation)
[![Manuscript](https://img.shields.io/badge/Manuscript-v2.0-green.svg)](manuscript/)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

**TRACEBIND** is a computational framework and diagnostic taxonomy designed to evaluate continuous spatial field structures across local gradient, global phase, and geometric regimes. 

While classical second-order spatial statistics (e.g., Moran's $I$, Geary's $C$) quantify spatial autocorrelation via local covariance, they exhibit limited response under non-local Fourier phase randomization and compress complex, anisotropic spatial velocity fields into monolithic summaries. TRACEBIND establishes a **Two-Tier Hierarchical Descriptor Taxonomy** that explicitly disentangles macro-scale phase organization from specific domain geometries.

---

## Key Features & Descriptor Taxonomy

TRACEBIND partitions spatial characterization into two decoupled feature subspaces:

```text
                  ┌──────────────────────────────────────────┐
                  │    Input Field (Spatial Kinematics)      │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                    TIER 1: PHASE ORGANIZATION SUBSPACE                   │
  │  Descriptors: GE (Gradient Energy), LE (Laplacian Energy),               │
  │               C_orient (Phase Coherence)                                 │
  │  Variance Explained: ~84.8% (PC1)                                        │
  │  Physical Meaning: Macro coherent phase structure vs randomized noise    │
  └────────────────┬─────────────────────────────────────────┬───────────────┘
                   │ (Passes: Non-random phase)               │ (Fails: Phase noise)
                   ▼                                         ▼
  ┌──────────────────────────────────────────┐     ┌──────────────────┐
  │    TIER 2: CYCLONE GEOMETRY SUBSPACE     │     │ Reject as Random │
  │  Descriptors: A_radial (Radial Symmetry),│     └──────────────────┘
  │               S_orient (Shear/Orient)    │
  │  Variance Contribution: Independent PCs  │
  │  Physical Meaning: Kinematic shape,      │
  │  radial symmetry & anisotropic rotation  │
  └────────────────┬─────────────────────────┘
                   │
                   ▼
  ┌──────────────────────────────────────────┐
  │         PHYSICAL KINEMATIC PROFILE       │
  │  Disentangles macro phase organization   │
  │  from specific tropical vortex geometry  │
  └──────────────────────────────────────────┘

```

### 1. Tier 1: Phase Organization Subspace

Evaluates macro-scale spatial coherence and structural energy concentration relative to random phase noise:

* **Gradient Energy Density ($GE$):** Concentration of local spatial finite differences.
* **Laplacian Energy Density ($LE$):** Local structural curvature concentration via the trace of the spatial Hessian matrix.
* **Global Phase Alignment ($C_{\text{orient}}$ / $C_\phi$):** Directional alignment of 2D Fourier phase angles across spectral bands.

### 2. Tier 2: Cyclone & Field Geometry Subspace

Evaluates domain-specific structural kinematics and shape anisotropy:

* **Radial Anisotropy & Symmetry ($A_{\text{radial}}$):** Measures deviations from pure azimuthal radial symmetry about the field centroid.
* **Shear-Oriented Anisotropy ($S_{\text{orient}}$):** Structural stretch and rotational shear along principal kinematic axes derived from the spatial structure tensor.

---

## Phase 8 C2: Blinded Empirical Validation

In Phase 8 C2, TRACEBIND was subjected to a **prospective, blinded, access-controlled empirical validation experiment** evaluating scalar Kinematic Phase Coherence ($C_\phi$) on ERA5 atmospheric reanalysis fields ($N = 20$, $10$ Tropical Cyclones vs. $10$ Matched Controls).

### Protocol & Audit Trail Integrity

* **Cryptographic Provenance:** Pre-unblinding manifest snapshot (`pre_unblinding_audit_snapshot.json`) verified with SHA-256 integrity checks prior to keycard decryption.
* **Blind Unblinding Audit:** $100\%$ line-by-line verification passed ($0$ mapping discrepancies across $20/20$ UUID keys).
* **Experimental Lock:** Phase 8 C2 is permanently frozen to preserve protocol integrity without post-hoc modifications.

### Experimental Results Summary

| Metric / Test | Cyclone Cohort ($n=10$) | Control Cohort ($n=10$) | Test Statistic / Difference | $p$-value / Effect Size |
| --- | --- | --- | --- | --- |
| **Mean $C_\phi$** | $0.746451$ | $0.747183$ | $\Delta = -0.000732$ | — |
| **Std. Deviation** | $0.211402$ | $0.180214$ | — | — |
| **Median $C_\phi$** | $0.762025$ | $0.760811$ | — | — |
| **Welch's $t$-test** | — | — | $t = -0.0083$ | $p = 0.9934$ (two-sided) |
| **Mann-Whitney $U$** | — | — | $U = 46.0$ | $p = 0.7863$ (one-sided pre-registered) |
| **Hedges' $g$** | — | — | — | $g = -0.0037$ |
| **Cliff's $\delta$** | — | — | — | $\delta = -0.0800$ |

### Confirmatory Conclusion

Under the pre-registered Phase 8 C2 protocol, the hypothesis that cyclone cases exhibit systematically higher scalar phase coherence ($C_\phi$) than matched control cases was **not supported** ($p = 0.7863$, $g = -0.0037$). The observed distributions substantially overlap across both cohorts.

### Exploratory Observation (Hypothesis-Generating)

Several non-cyclonic control cases exhibiting high hydrodynamic organization (e.g., active monsoon troughs, shear zones, and surge flows) yielded $C_\phi$ values ($> 0.85\text{--}0.93$) comparable to intense tropical cyclones (e.g., `TC_TAUKTAE` $C_\phi = 0.9405$, `TC_KYARR` $C_\phi = 0.9246$). This indicates that **scalar $C_\phi$ quantifies broad synoptic-scale flow organization rather than cyclone-specific identity**. This finding serves as a baseline for future tensor-based metric formulations in prospective phases.

---

## Repository Structure

```text
TRACEBIND-Atmosphere/
├── tracebind/           # Core TRACEBIND package & descriptor extraction modules
├── phase4/              # Synthetic generation & cross-metric benchmarking
├── phase5/              # ERA5 empirical pipeline, stability & taxonomy scripts
├── phase6/              # Frozen baseline, verification logs, and validation artifacts
├── phase7/              # Atmospheric data processing & integration pipeline
├── phase8/              # Blinded cohort execution, keycard manifest, audit & stats
│   └── c2/
│       ├── manifest/                 # Access-controlled keycard & cohort manifest
│       ├── extraction/               # Extracted C_phi scalar results
│       ├── diagnostics/              # Pre-unblinding audit snapshots & logs
│       ├── unblinded_master_dataset.csv  # Frozen, verified master dataset
│       ├── verify_pre_unblind.py     # Cryptographic stage-gate verification script
│       ├── unblind.py                # Deterministic unblinding pipeline
│       ├── statistical_analysis.py   # Pre-registered hypothesis testing runner
│       ├── audit_unblinding.py       # Line-by-line keycard cross-reference auditor
│       └── inference_report.json     # Final statistical results export
├── manuscript/          # Publication-ready draft & LaTeX assets
├── experiments/         # Exploratory validation setups & notebooks
├── benchmarks/          # Synthetic frozen benchmark suite (v1.0)
├── data/                # Atmospheric reanalysis & synthetic data arrays
├── tests/               # Pytest test suite for regression & determinism
├── validate_release.py  # End-to-end automated validation runner
├── requirements.txt     # Python dependency manifest
└── README.md            # Project documentation

```

---

## Quickstart & Installation

### Prerequisites

* Python 3.11+
* Git

### Setup

1. **Clone the repository:**

```bash
git clone [https://github.com/bigwiginfohub-wq/TRACEBIND-Atmosphere.git](https://github.com/bigwiginfohub-wq/TRACEBIND-Atmosphere.git)
cd TRACEBIND-Atmosphere

```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

```

3. **Install dependencies:**

```bash
pip install -r requirements.txt

```

---

## Execution & Reproducibility

### 1. Run Phase 8 Pre-Unblinding Verification & Audit

To verify the cryptographic integrity and execute line-by-line audit checks on Phase 8 C2:

```bash
cd phase8/c2
python verify_pre_unblind.py
python audit_unblinding.py

```

### 2. Execute Statistical Inference Test

To re-run the pre-registered statistical analysis on the unblinded master dataset:

```bash
python phase8/c2/statistical_analysis.py

```

### 3. Run Descriptor Taxonomy & Hierarchical Analysis

To execute the non-parametric validation, Leave-One-Storm-Out rank stability, correlation, and PCA analysis:

```bash
python phase5/scripts/41_hierarchical_validation_and_stability.py

```

---

## Citation

If you utilize TRACEBIND in your research, please cite the manuscript located in [`manuscript/`](https://www.google.com/search?q=manuscript/):

```bibtex
@article{tracebind2026,
  title={TRACEBIND: A Hierarchical Descriptor Taxonomy for Spatial Phase Organization and Atmospheric Kinematics},
  author={Mohammed Ali},
  journal={Formal Methodological Research Manuscript},
  year={2026},
  version={2.0}
}

```

---

## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://www.google.com/search?q=LICENSE.txt) file for details.

```

```
