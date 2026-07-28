# Phase 6E: Rigorous Statistical & Hypothesis Evaluation

**Execution Scope:** Evaluation of TRACEBIND structural and phase boundary metrics across TC ($N=48$) and Control ($N=19$) cohorts.  
**Primary Test Preregistration:** Mann-Whitney U Test (Two-sided) with Benjamini-Hochberg FDR Multiple Comparison Adjustment ($q < 0.05$).  
**Effect Size Bootstrapping:** 10,000 iterations using NumPy Generator (`default_rng(42)`).

---

## 1. Preregistered Evaluation Summary

| Metric | Primary MW-U $p$ (FDR) | Cliff's $\delta$ [95% Bootstrap CI] | Empirical False Positive Rate ($\alpha_{\text{emp}}$) | Preregistered Evaluation Outcome |
| :--- | :--- | :--- | :--- | :--- |
| `circulation_250km_mean` | 4.0031e-01 | +0.250 [-0.015, +0.507] | 3.6% | **No statistically supported difference under current dataset** |
| `compactness_ratio_mean` | 9.3902e-01 | -0.013 [-0.371, +0.355] | 5.5% | **No statistically supported difference under current dataset** |
| `asymmetry_index_mean` | 9.3902e-01 | -0.026 [-0.344, +0.292] | 4.1% | **No statistically supported difference under current dataset** |
| `filamentation_fraction_mean` | 7.1534e-01 | -0.162 [-0.448, +0.133] | 5.7% | **No statistically supported difference under current dataset** |
| `coherence_index_mean` | 4.0031e-01 | +0.289 [-0.044, +0.601] | 5.0% | **No statistically supported difference under current dataset** |
| `boundary_entropy_bits_mean` | 9.3902e-01 | +0.059 [-0.270, +0.388] | 4.4% | **No statistically supported difference under current dataset** |
| `boundary_sharpness_mean` | 9.3902e-01 | -0.035 [-0.344, +0.268] | 4.2% | **No statistically supported difference under current dataset** |

---

## 2. Statistical Methodology & Diagnostics

* **Exploratory Distributional Diagnostics:** Shapiro-Wilk test for normality and Brown-Forsythe (median-centered Levene) test for variance equality were recorded in `exploratory_statistics.parquet`.
* **Negative Controls:** Empirical false-positive rates were established by running 1,000 label-swapping permutations across the dataset.
* **Secondary Uncorrected Tests:** Kolmogorov-Smirnov and Mood's Median tests are logged in `hypothesis_test_results.csv` as exploratory metrics.

---

## 3. Data Artifacts Directory

* `Phase6E/exploratory_statistics.parquet` (and `.csv`)
* `Phase6E/hypothesis_test_results.csv`
* `Phase6E/effect_sizes.csv`
* `Phase6E/preregistered_evaluation_summary.json`
