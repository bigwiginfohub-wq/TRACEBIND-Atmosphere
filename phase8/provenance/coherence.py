"""TRACEBIND Phase 7 - Spatial Phase Coherence Diagnostic

======================================================
File: phase7/sandbox/metrics/coherence.py
Computes radial/tangential phase alignment coherence C_phi between velocity vectors
and grid center origin.
C_phi = 1.0 represents perfect rotational alignment.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def compute_relative_vorticity(
    u: np.ndarray, v: np.ndarray, dx: float = 2500.0, dy: float = 2500.0
) -> np.ndarray:
  """Computes vertical component of relative vorticity (dv/dx - du/dy)."""
  u = np.asarray(u, dtype=np.float64)
  v = np.asarray(v, dtype=np.float64)

  dudx, dudy = np.gradient(u, dx, axis=1), np.gradient(u, dy, axis=0)
  dvdx, dvdy = np.gradient(v, dx, axis=1), np.gradient(v, dy, axis=0)

  return dvdx - dudy


def compute_phase_coherence(
    u: np.ndarray,
    v: np.ndarray,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
  """Computes radial phase alignment coherence C_phi relative to center origin.

  Parameters:
  -----------
  u, v : np.ndarray (2D)
      Zonal and meridional wind components.
  X, Y : np.ndarray (2D), optional
      Coordinate grids. If None, auto-generated based on grid shape.
  center : Tuple[float, float], optional
      (xc, yc) center coordinates. If None, defaults to grid midpoint.
  mask : np.ndarray (2D boolean or float), optional
      Optional evaluation domain or eyewall mask.

  Returns:
  --------
  float : C_phi phase coherence metric in [0.0, 1.0].
  """
  u = np.asarray(u, dtype=np.float64)
  v = np.asarray(v, dtype=np.float64)
  ny, nx = u.shape

  # Auto-generate coordinate grid if not provided
  if X is None or Y is None:
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

  # Auto-set center to geometric midpoint if not provided
  if center is None:
    center = (float(X.mean()), float(Y.mean()))

  Xc = X - center[0]
  Yc = Y - center[1]
  R = np.sqrt(Xc**2 + Yc**2) + 1e-12

  # Tangential unit vectors e_theta = (-Yc/R, Xc/R)
  e_theta_x = -Yc / R
  e_theta_y = Xc / R

  # Velocity magnitude
  speed = np.sqrt(u**2 + v**2) + 1e-12

  # Normalized tangential dot product
  dot_tangential = (u * e_theta_x + v * e_theta_y) / speed

  # Apply mask if provided
  if mask is not None:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
      return 0.0
    dot_tangential = dot_tangential[mask_bool]

  # Mean tangential alignment magnitude C_phi
  c_phi = float(np.mean(np.abs(dot_tangential)))

  return c_phi