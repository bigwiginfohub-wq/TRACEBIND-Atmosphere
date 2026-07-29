\# TRACEBIND Phase 8 (C2) - Cohort Selection \& Matching Protocol



\## 1. Objective

Establish a non-confounded, statistically paired atmospheric sample (10 Tropical Cyclones vs. 10 Matched Non-Cyclonic Controls) across the North Indian Ocean (NIO) basin to evaluate the discriminative power of phase coherence ($C\_\\phi$).



\## 2. Cyclone Cohort Selection Criteria (N = 10)

\- \*\*Geographic Basin:\*\* North Indian Ocean (Bay of Bengal / Arabian Sea: $0^\\circ - 30^\\circ\\text{N}$, $45^\\circ - 100^\\circ\\text{E}$).

\- \*\*Intensity Threshold:\*\* Severe Cyclonic Storm (SCS) or higher (10-min sustained winds $\\ge 48\\text{ knots}$).

\- \*\*Maturity Stage:\*\* Frozen at time of minimum central pressure ($p\_{\\min}$).

\- \*\*Independence:\*\* Distinct storm systems across $2008 - 2023$ (zero temporal overlap).

\- \*\*Domain Size:\*\* $10^\\circ \\times 10^\\circ$ bounding box centered on $p\_{\\min}$ coordinates.



\## 3. Control Cohort Matching Criteria (N = 10)

To isolate rotational organization from background atmospheric state, each Control case $k$ is paired 1-to-1 with Cyclone case $k$ via exact covariate matching:

1\. \*\*Latitude Band:\*\* $\\pm 2.0^\\circ$ of Cyclone $k$ center.

2\. \*\*Longitude Band:\*\* Same basin ($\\pm 5.0^\\circ$ shift away from active convection/cyclogenesis).

3\. \*\*Seasonal/Monthly Match:\*\* Same calendar month ($\\pm 7\\text{ days}$) in a non-cyclone year or non-active period.

4\. \*\*Diurnal Time Match:\*\* Exact same UTC analysis hour ($t\_{\\text{UTC}}$) as Cyclone $k$.

5\. \*\*Exclusion Criteria:\*\* Must not contain an IMD-classified depression, deep depression, or organized synoptic low.



\## 4. Blinding Integrity

The acquisition script ingests this protocol specification and writes directly to `phase8/c2/raw/c2\_uuid\_<token>.nc`. No case names or category tags exist in the execution directory.

