\# Phase 7 Pre-Registration Manifest (P7\_PREREG.md)



\*\*Project:\*\* TRACEBIND Atmospheric Research Suite  

\*\*Phase:\*\* Phase 7 – Kinematic Phase Coherence \& Emergent Structure Diagnostics  

\*\*Status:\*\* Pre-Registered (Locked Prior to ERA5 Primary Evaluation)  

\*\*Date:\*\* July 28, 2026  

\*\*Primary Investigator / Author:\*\* Mohammed Ali  



\---



\## 1. Context \& Research Objectives



Phase 7 evaluates whether spatial phase alignment and kinematic coherence metrics derived from atmospheric velocity fields ($u, v$) systematically distinguish organized tropical cyclones (TCs) from non-developing synoptic disturbances in ERA5 reanalysis datasets.



To prevent p-hacking, metric tuning, or selective sample choices, all candidate metrics must clear the synthetic sandbox verification suite (`PROMOTION\_POLICY.md`) before running against primary ERA5 datasets under the fixed rules below.



\---



\## 2. Formal Research Questions \& Hypotheses



\### Research Question 1 (RQ1: Vortex Coherence Differentiation)

\* \*\*Question:\*\* Does the spatial phase coherence metric ($\\mathcal{C}\_{\\phi}$) display a statistically significant elevation in developing tropical cyclone cores relative to matched non-developing tropical waves?

\* \*\*Primary Hypothesis ($\\mathcal{H}\_{1,a}$):\*\* Mean core coherence ($\\mathcal{C}\_{\\phi}$) within $R \\le 200\\text{ km}$ of the storm center is significantly higher for developing tropical cyclones than for matched non-developing control perturbations ($p < 0.001$, Cohen’s $d \\ge 0.8$).

\* \*\*Null Hypothesis ($\\mathcal{H}\_{1,0}$):\*\* No statistically significant difference in mean core coherence exists between developing and non-developing systems ($p \\ge 0.05$).



\### Research Question 2 (RQ2: Lead-Time Predictive Signal)

\* \*\*Question:\*\* Does elevated kinematic phase coherence precede maximum intensity (peak $V\_{max}$) or tropical depression genesis ($V\_{max} \\ge 34\\text{ kt}$) by a measurable temporal lead time?

\* \*\*Secondary Hypothesis ($\\mathcal{H}\_{2,a}$):\*\* Phase coherence rate of increase ($\\frac{d\\mathcal{C}\_{\\phi}}{dt}$) exhibits a statistically significant positive lead time ($\\Delta t \\ge 12\\text{ hours}$) relative to pressure drop or wind speed intensification.

\* \*\*Null Hypothesis ($\\mathcal{H}\_{2,0}$):\*\* Kinematic phase coherence does not show consistent lead-time behavior prior to pressure or wind speed thresholds ($\\Delta t \\le 0\\text{ hours}$).



\### Research Question 3 (RQ3: Resilience Under Environmental Shear)

\* \*\*Question:\*\* Does phase coherence provide a robust signal under vertical shear conditions ($850\\text{--}200\\text{ hPa}$) where conventional vorticity metrics suffer from structural distortion?

\* \*\*Hypothesis ($\\mathcal{H}\_{3,a}$):\*\* The ratio of core coherence to total vorticity remains higher in sheared developing systems than in sheared non-developing systems ($p < 0.01$).



\---



\## 3. Data Protocols \& Hold-Out Rules



\### 3.1 Primary Evaluation Dataset

\* \*\*Source:\*\* ERA5 Reanalysis ($0.25^\\circ \\times 0.25^\\circ$ grid, 6-hourly temporal resolution).

\* \*\*Domain:\*\* North Atlantic \& Western North Pacific basins (2010–2024).

\* \*\*Sample Stratification:\*\*

&#x20; \* \*\*Exploratory Set (50%):\*\* Used for initial exploratory analysis and baseline calibration.

&#x20; \* \*\*Confirmatory Hold-Out Set (50%):\*\* Kept strictly locked until candidate metrics clear all sandbox promotion gates.



\### 3.2 Hold-Out Access Policy

1\. No execution of candidate metrics on the Confirmatory Hold-Out Set is permitted prior to logging a passing `verification\_manifest.json` from the sandbox suite.

2\. Single-pass evaluation on the Confirmatory Hold-Out Set; no re-calibration or parameter tuning is permitted after hold-out execution.



\---



\## 4. Methodological \& Metric Promotion Requirements



Every candidate metric evaluated in Phase 7 must pass the five gates specified in `C:\\TRACEBIND-Atmosphere\\phase7\\PROMOTION\_POLICY.md`:



1\. \*\*Numerical Accuracy Gate:\*\* RMSE $< 1.0\\%$ vs analytical vortex ground truth (Lamb–Oseen/Rankine).

2\. \*\*Operator Invariance Gate:\*\* Relative shift $\\Delta < 0.1\\%$ under $90^\\circ$ spatial rotation and coordinate translation.

3\. \*\*Response Dynamics Gate:\*\* Monotonic decrease under controlled environmental shear and noise injection.

4\. \*\*CI/CD Regression Pass Gate:\*\* 100% pass rate in `phase7/sandbox/tests/run\_regression.py`.

5\. \*\*Provenance Gate:\*\* Documented LaTeX mathematical spec, defined physical units, and recorded SHA-256 code hash.



\---



\## 5. Statistical Decision Criteria \& Falsification Thresholds



| Hypothesis | Target Metric | Test Statistic | Significance Threshold | Effect Size Target |

|---|---|---|---|---|

| \*\*$\\mathcal{H}\_{1,a}$ (Coherence Elevation)\*\* | Core Coherence $\\mathcal{C}\_{\\phi}$ | Two-sample Welch's $t$-test / Mann-Whitney $U$ | $p < 0.001$ (Two-tailed) | Cohen's $d \\ge 0.80$ |

| \*\*$\\mathcal{H}\_{2,a}$ (Precursor Lead Time)\*\* | Lead time $\\Delta t$ ($\\frac{d\\mathcal{C}\_{\\phi}}{dt}$ lead) | Cross-correlation / Lag Analysis | $p < 0.01$ | Median $\\Delta t \\ge 12\\text{ hrs}$ |

| \*\*$\\mathcal{H}\_{3,a}$ (Shear Resilience)\*\* | Coherence / Vorticity Ratio | Two-way ANOVA with shear interaction | $p < 0.01$ | Partial $\\eta^2 \\ge 0.15$ |



\*If any primary hypothesis fails to meet both the statistical significance threshold and effect size target on the Confirmatory Hold-Out Set, it will be declared \*\*falsified\*\*.\*



\---



\## 6. Pre-Registration Integrity Lock



\* \*\*Manifest File Location:\*\* `C:\\TRACEBIND-Atmosphere\\phase7\\preregistration\\P7\_PREREG.md`

\* \*\*Governing Ruleset:\*\* `C:\\TRACEBIND-Atmosphere\\phase7\\PROMOTION\_POLICY.md`

\* \*\*Verification Runner:\*\* `C:\\TRACEBIND-Atmosphere\\phase7\\sandbox\\tests\\run\_regression.py`



```





