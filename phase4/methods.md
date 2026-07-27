

\## 1. Metric Response Characteristics Across Controlled Perturbations



The continuous response curves in \*\*Experiment 6 (Continuous Linear Mixture)\*\* explicitly decouple large-scale value distributions from local spatial alignment:



```

&#x20;Alpha    TB-v1    TB-v2   Moran I  Geary C

&#x20;  0.0 0.985482 0.993276  0.958036 0.049835

&#x20;  0.1 0.977002 0.590833  0.946248 0.061513

&#x20;  0.2 0.967036 0.554766  0.901298 0.106044

&#x20;  ...

&#x20;  0.9 0.893288 0.503030  0.010534 0.988482

&#x20;  1.0 0.882891 0.502644 -0.000017 0.998934



```



\### Precise Mathematical Characterizations



1\. \*\*TRACEBIND v1 (Global Distribution Summary Statistic)\*\*

TB-v1 exhibits a smooth, approximately linear decay ($0.985 \\to 0.883$) as random noise is added. It behaves as a global summary statistic that is primarily influenced by the overall distribution of field values and large-scale variance structure, while displaying relatively limited sensitivity to local topological disruption.

2\. \*\*TRACEBIND v2 (Local Gradient-Alignment Coherence Metric)\*\*

TB-v2 drops rapidly from \*\*0.993 to 0.591\*\* at $\\alpha = 0.1$, subsequently saturating near $0.500$. It demonstrates a strong early non-linear response to initial disruptions in local directional structure, functioning as a high-sensitivity local gradient-alignment metric.

3\. \*\*Origin and Meaning of the $0.500$ Asymptote\*\*

The saturation near $0.500$ under pure noise ($\\alpha = 1.0$) is a direct mathematical consequence of the bounded cosine-similarity transformation. Uniformly distributed random gradient vectors yield an expected inner product of zero, mapping directly to a baseline of $0.5$. \*\*Consequently, the lower bound of approximately $0.5$ should not be interpreted as residual atmospheric organization, but as the expected baseline of the metric under isotropic random gradient orientations.\*\*

4\. \*\*Moran’s I (Second-Order Spatial Autocorrelation)\*\*

Exhibits a smooth, non-linear decay ($0.958 \\to 0.000$), directly measuring distance-dependent covariance decay across the domain.



\---



\## 2. Benchmark Response Comparison



| Metric | Dominant Response Characteristic | Behavior Under Noise / Scrambling | Asymptotic Baseline Value |

| --- | --- | --- | --- |

| \*\*TRACEBIND v1\*\* | Overall value distribution \& large-scale variance | Smooth, proportional linear decay | \*\*Elevated (\~0.88)\*\* under complete scrambling |

| \*\*TRACEBIND v2\*\* | Local micro-gradient directional alignment | Strong early non-linear drop | \*\*Random-orientation baseline (\~0.50)\*\* |

| \*\*Moran’s I\*\* | Distance-dependent spatial autocorrelation | Gradual non-linear covariance decay | \*\*Zero (\~0.00)\*\* under complete scrambling |



\---



\## 3. Reprojection \& Affine Transformation Robustness Benchmark



Before transitioning to observational data, an additional synthetic experiment will be executed to test metric behavior under coordinate remapping, spatial interpolation, and grid warping:



```

&#x20;                 ┌─────────────────────────────────────┐

&#x20;                 │      Affine Distortion Suite        │

&#x20;                 ├─────────────────────────────────────┤

&#x20;                 │ 1. Isotropic Scaling                │

&#x20;                 │ 2. Anisotropic Scaling              │

&#x20;                 │ 3. Shear Transformation             │

&#x20;                 │ 4. Mild Perspective Distortion      │

&#x20;                 └──────────────────┬──────────────────┘

&#x20;                                    │

&#x20;                                    ▼

&#x20;                     Quantify Metric Sensitivity



```



\* \*\*Target Validation Bounds:\*\* Verify that grid re-sampling and mild reprojection distortions preserve metric rankings and do not introduce artifactual phase shifts.



\---



\## 4. Formal Metric Formulation Freeze



Prior to analyzing observational fields, the mathematical definitions of \*\*TRACEBIND v1\*\* and \*\*TRACEBIND v2\*\* are \*\*frozen\*\*.



> \*\*Methodological Principle:\*\* Locking the formulations at the conclusion of the synthetic benchmark phase prevents post-hoc parameter tuning on real atmospheric data. This preserves a strict, defensible separation between metric development and empirical evaluation.



\---



\## 5. ERA5 Tropical Cyclone Pre-Registered Case Study



The empirical evaluation phase applies the frozen metric formulations to a high-resolution ERA5 reanalysis dataset during a tropical cyclone (TC) intensification event.



\### Case Study Design



\* \*\*Target Domain:\*\* ERA5 500 hPa Geopotential Height

\* \*\*Temporal Scope:\*\* Hourly temporal resolution over a 72-hour intensification window

\* \*\*Spatial Framing:\*\* Moving $20^\\circ \\times 20^\\circ$ bounding box centered on the cyclone core

\* \*\*Evaluated Metrics:\*\* `TB-v1`, `TB-v2`, `Moran's I`, `Geary's C`, `Gradient Energy`, `Spectral Slope`

\* \*\*Observational Anchors:\*\* Minimum Sea-Level Pressure (MSLP), Maximum Sustained Winds



\### Pre-Registered Hypotheses



\* \*\*Primary Hypothesis ($H\_1$):\*\* `TB-v2` exhibits statistically detectable inflection points prior to one or more conventional storm-intensity indicators during the selected case study.

\* \*\*Secondary Hypothesis ($H\_2$):\*\* `TB-v1` and `TB-v2` capture complementary aspects of spatial organization rather than reproducing classical second-order autocorrelation metrics like `Moran's I`.



\---



\## Methodological Summary Statement



> The initial synthetic property-testing phase establishes a reproducible benchmark for characterizing metric behavior under controlled perturbations. Future TRACEBIND variants can be evaluated against this benchmark using identical experimental protocols.

