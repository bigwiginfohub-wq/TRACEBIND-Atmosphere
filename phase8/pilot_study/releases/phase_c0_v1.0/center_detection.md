# TRACEBIND Phase C0: Center Detection Specification (v1.0)

## Overview
Phase C0 provides spatial alignment and quality control auditing prior to metric calculation.

## Algorithm Sequence
1. **Coordinate Alignment**: Verify descending/ascending latitudes and enforce standard ordering.
2. **Geodesic Spacing Calculation**: Calculate localized grid spacing $(\Delta x, \Delta y)$ using WGS84 geodesics evaluated at domain midpoints.
3. **Primary Center**: Identify local $P_{min}$ (MSLP Minimum).
4. **Local Search Mask**: Construct geodesic search radius ($r \le \text{max\_center\_sep\_km}$) centered on $P_{min}$.
5. **Local Vorticity Center**: Find maximum relative vorticity $\zeta$ strictly within candidate cells ($N > 0$).
6. **Stability Audit**: Measure geodesic distance separation between $P_{min}$ and local $\zeta_{max}$. Reject systems with separation $> \text{max\_center\_sep\_km}$.
