\# TRACEBIND Boundary \& Limitations Register



\## 1. Current Technical Scope

The TRACEBIND core framework explicitly assumes:

\*   Static scalar fields per time step.

\*   Regular spatial coordinate sampling grids.

\*   Local stationarity within the neighborhood boundaries.

\*   Independent permutations during reference distribution generation.



\## 2. Explicit Out-of-Scope Configurations

The framework does not currently evaluate or account for:

\*   Anisotropic spatial neighborhoods.

\*   Directional flow fields or vector velocity vectors.

\*   Graph neural network topologies.

\*   Uncertainty metrics embedded inside raw observations.



These items remain open research avenues for future versioned expansions.

