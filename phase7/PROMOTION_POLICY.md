# Phase 7 Metric Promotion Policy & Sandbox Governance

**Document Version:** 1.2  
**Effective Date:** July 28, 2026  
**Status:** Active Governance Protocol  

---

## Metric Promotion Gatekeeper Requirements

Before any experimental diagnostic in `sandbox/metrics/` is promoted to production (`phase7/src/`) or evaluated against ERA5 atmospheric datasets, it must pass all 7 automated Gatekeeper tests in `sandbox/tests/run_regression.py`:

| Requirement | Metric / Operator | Pass Criterion | Verification Module |
| :--- | :--- | :--- | :--- |
| **Analytical Accuracy** | Discretization Error | Relative $\text{RMSE} < 1.0\%$ | `verify_convergence.py` |
| **Grid Convergence** | Empirical Order (EOC) | Asymptotic $\text{EOC} \ge 1.80$ ($O(h^2)$) | `verify_convergence.py` |
| **Rotation Invariance** | Coordinate + Vector Shift | Relative Delta $\Delta < 0.1\%$ | `run_regression.py` |
| **Translation Invariance**| Coordinate Shift ($\pm 50\text{ km}$) | Relative Delta $\Delta < 0.1\%$ | `run_regression.py` |
| **Noise Monotonicity** | SNR Spectrum ($20\rightarrow 0\text{ dB}$) | Strictly monotonic decrease | `run_regression.py` |
| **Determinism** | Master Seed ($RNG=42$) | Identical bitwise output | `run_regression.py` |
| **Provenance Locking** | SHA-256 + Environment Snapshot | $100\%$ artifact audit match | `verification_manifest.json` |

---

## Decision Flowchart

```text
               Experimental Metric Idea
                          │
                          ▼
             Implement in sandbox/metrics/
                          │
                          ▼
              Synthetic Regression Suite
              (sandbox/tests/run_regression.py)
                          │
                 All Gatekeeper Tests PASS?
                    /          \
                  NO            YES
                  │              │
        Revise Metric Code       Promote to Production (phase7/src/)
                                 │
                                 ▼
                        Evaluate on ERA5 Data
                                 │
                                 ▼
                       Statistical Inference

### Audit Output Manifest Summary (`verification_manifest.json`)

Running `run_regression.py` populates `sandbox/reports/verification_manifest.json` with the following structure:

```json
{
  "sandbox_version": "1.0.0",
  "verification_date_utc": "2026-07-28T11:47:58Z",
  "environment": {
    "git_commit": "a3f892c4b71d",
    "python_version": "3.11.9",
    "numpy_version": "1.26.4",
    "scipy_version": "1.12.0",
    "master_rng_seed": 42
  },
  "overall_status": "PASS",
  "tests": [
    {
      "id": "TEST_001_ACCURACY_AND_CONVERGENCE",
      "metric": "vorticity_eoc",
      "observed_eoc": 1.9842,
      "observed_rmse_pct": 0.0381,
      "observed_linf_pct": 0.1124,
      "target": "EOC >= 1.80, RMSE < 1.0%",
      "status": "PASS",
      "runtime_ms": 142.3
    },
    {
      "id": "TEST_002_INVARIANCE_ROTATION",
      "metric": "radial_coherence",
      "relative_delta": 0.0,
      "target": "< 0.001",
      "status": "PASS",
      "runtime_ms": 4.1
    },
    {
      "id": "TEST_003_INVARIANCE_TRANSLATION",
      "metric": "radial_coherence",
      "relative_delta": 0.0,
      "target": "< 0.001",
      "status": "PASS",
      "runtime_ms": 5.8
    },
    {
      "id": "TEST_004_PERTURBATION_NOISE_DECAY",
      "metric": "phase_coherence_index",
      "snr_spectrum_db": [20.0, 15.0, 10.0, 5.0, 0.0],
      "coherence_spectrum": [0.9902, 0.9698, 0.9124, 0.7853, 0.6120],
      "strictly_monotonic": true,
      "status": "PASS",
      "runtime_ms": 18.5
    }
  ],
  "artifact_hashes": {
    "synthetic_vortex.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "coherence.py": "8f4e2a1b9c3d7e5f6a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f",
    "verify_convergence.py": "2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3",
    "PROMOTION_POLICY.md": "c4d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7",
    "P7_PREREG.md": "7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8"
  }
}                       

