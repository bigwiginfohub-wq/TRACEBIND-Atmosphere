# TRACEBIND: Hierarchical Spatial & Kinematic Descriptor Framework

[![Pipeline Status](https://img.shields.io/badge/Pipeline-PASSING-success.svg)](#validation--reproducibility)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TRACEBIND** is a computational framework and diagnostic taxonomy designed to evaluate continuous spatial field structures across local gradient, global phase, and geometric regimes. 

While classical second-order spatial statistics (e.g., Moran's $I$, Geary's $C$) quantify spatial autocorrelation via local covariance, they exhibit limited response under non-local Fourier phase randomization and compress complex, anisotropic spatial velocity fields into monolithic summaries. TRACEBIND establishes a **Two-Tier Hierarchical Descriptor Taxonomy** that explicitly disentangles macro-scale phase organization from specific domain geometries.

---

## Key Features & Descriptor Taxonomy

TRACEBIND partitions spatial characterization into two decoupled feature subspaces:


```

```
                  ┌──────────────────────────────────────────┐
                  │    Input Field (Spatial Kinematics)      │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                   TIER 1: PHASE ORGANIZATION SUBSPACE                     │
  │  Descriptors: GE (Gradient Energy), LE (Laplacian Energy),               │
  │               C_orient (Phase Coherence)                                 │
  │  Variance Explained: ~84.8% (PC1)                                        │
  │  Physical Meaning: Macro coherent phase structure vs randomized noise    │
  └────────────────┬─────────────────────────────────────────┬───────────────┘
                   │ (Passes: Non-random phase)              │ (Fails: Phase noise)
                   ▼                                         ▼
  ┌──────────────────────────────────────────┐     ┌──────────────────┐
  │   TIER 2: CYCLONE GEOMETRY SUBSPACE       │     │ Reject as Random │
  │  Descriptors: A_radial (Radial Symmetry),│     └──────────────────┘
  │               S_orient (Shear/Orient)    │
  │  Variance Contribution: Independent PCs  │
  │  Physical Meaning: Kinematic shape,      │
  │  radial symmetry & anisotropic rotation  │
  └────────────────┬─────────────────────────┘
                   │
                   ▼
  ┌──────────────────────────────────────────┐
  │       PHYSICAL KINEMATIC PROFILE         │
  │  Disentangles macro phase organization   │
  │  from specific tropical vortex geometry  │
  └──────────────────────────────────────────┘

```

```

### 1. Tier 1: Phase Organization Subspace
Evaluates macro-scale spatial coherence and structural energy concentration relative to random phase noise:
* **Gradient Energy Density ($GE$):** Concentration of local spatial finite differences.
* **Laplacian Energy Density ($LE$):** Local structural curvature concentration via the trace of the spatial Hessian matrix.
* **Global Phase Alignment ($C_{\text{orient}}$):** Directional alignment of 2D Fourier phase angles across spectral bands.

### 2. Tier 2: Cyclone & Field Geometry Subspace
Evaluates domain-specific structural kinematics and shape anisotropy:
* **Radial Anisotropy & Symmetry ($A_{\text{radial}}$):** Measures deviations from pure azimuthal radial symmetry about the field centroid.
* **Shear-Oriented Anisotropy ($S_{\text{orient}}$):** Structural stretch and rotational shear along principal kinematic axes derived from the spatial structure tensor.

---

## Validation Highlights & Scientific Findings

TRACEBIND has been evaluated across synthetic benchmark suites and empirical ERA5 atmospheric reanalysis fields (Tropical Cyclones vs. Negative Controls):

* **Rank Stability (LOSO):** Achieves an average Spearman’s $\rho = 1.000$ under Leave-One-Storm-Out iterations, proving that descriptor ordering is invariant and not artificially driven by extreme events.
* **Feature Orthogonality:** Empirical correlation analysis demonstrates strong mutual correlation within Tier 1 metrics ($\lvert r\rvert > 0.79$), while Tier 2 geometric metrics are virtually independent of Tier 1 ($\lvert r\rvert < 0.14$).
* **Unsupervised PCA proof:** Principal Component Analysis confirms that Tier 1 captures macro phase organization ($\text{PC}_1 = 84.8\%$ of population variance), while Tier 2 geometric descriptors operate on independent, highly specific orthogonal axes ($\text{PC}_2$ and $\text{PC}_5$).
* **Asymptotic Computational Scaling:** Execution time and memory scale linearly $O(P)$ with total pixel count ($P = N^2$), making the implementation scalable to high-resolution grid domains.

---

## Repository Structure

```text
TRACEBIND-Atmosphere/
├── tracebind/              # Core TRACEBIND package & descriptor extraction modules
├── phase4/                 # Synthetic generation & cross-metric benchmarking
├── phase5/                 # ERA5 empirical pipeline, stability & taxonomy scripts
│   ├── scripts/            # Pipeline execution scripts (40_*, 41_*)
│   └── results/            # Outputs (bootstrapped CIs, PCA, LOSO stability CSVs)
├── manuscript/             # Publication-ready draft & latex assets
├── experiments/            # Exploratory validation setups & notebooks
├── benchmarks/             # Synthetic frozen benchmark suite (v1.0)
├── data/                   # Atmospheric reanalysis & synthetic data arrays
├── tests/                  # Pytest test suite for regression & determinism
├── validate_release.py     # End-to-end automated validation runner
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation

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

### 1. Run Descriptor Taxonomy & Hierarchical Analysis

To execute the refactored non-parametric validation, Leave-One-Storm-Out rank stability, correlation, and PCA analysis:

```bash
python phase5/scripts/41_hierarchical_validation_and_stability.py

```

Outputs are written directly to `phase5/results/`.

### 2. Automated Release Validation

To verify pipeline determinism, synthetic benchmarks, and environment integrity:

```bash
python validate_release.py

```

---

## Citation

If you utilize TRACEBIND in your research, please cite the manuscript:

```bibtex
@article{tracebind2026,
  title={TRACEBIND: A Hierarchical Descriptor Taxonomy for Spatial Phase Organization and Atmospheric Kinematics},
  author={Mohammed Ali, Independent Researcher},
  journal={Formal Methodological Research Manuscript},
  year={2026},
  version={2.0}
}

```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
