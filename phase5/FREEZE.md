\# TRACEBIND Phase 5 Execution \& Protocol Freeze



\*\*Freeze Date:\*\* 2026-07-25  

\*\*Pipeline Identifier:\*\* `TRACEBIND-P5.0-FROZEN`  

\*\*Status:\*\* Protocol Frozen for Expanded Cohort Validation ($N \\ge 30$)



\---



\## 1. Frozen Analysis Choices

\- \*\*Dataset Source:\*\* Copernicus CDS ERA5 Reanalysis (`reanalysis-era5-single-levels`)

\- \*\*Target Variable:\*\* Mean Sea Level Pressure (`msl`, Units: Pa)

\- \*\*Spatial Resolution:\*\* $0.25^\\circ \\times 0.25^\\circ$

\- \*\*Temporal Resolution:\*\* Hourly (`valid\_time`)

\- \*\*Spatial Diagnostic (Phase 5A):\*\* Spatial Gradient Energy ($GE$)

\- \*\*Spatial Null Model:\*\* Array-wide random spatial permutation (`rng.permutation`)

\- \*\*Temporal Diagnostic (Phase 5B):\*\* Gradient Energy Trajectory Rate ($|dGE/dt|$)

\- \*\*Temporal Null Model:\*\* Fourier Phase Randomization Surrogates (`rfft` / `irfft`)

\- \*\*Permutation Count:\*\* $N\_{perm} = 1000$

\- \*\*Random Seed:\*\* $42$ (`np.random.default\_rng(42)`)

\- \*\*Significance Threshold:\*\* $p < 0.05$



\---



\## 2. Freeze Governance Rules

1\. \*\*No Algorithmic Modifications:\*\* The code generating spatial and temporal metrics must not be altered while expanding the cohort.

2\. \*\*Bug Fix Exception:\*\* Only strict execution errors (e.g., IO file errors, pathing, or library deprecations) may be patched. Statistical logic remains locked.

3\. \*\*Audit Verification:\*\* Every output directory must include automated SHA-256 script hashes and NetCDF MD5 checksum logs.

