import numpy as np

def compute_numerical_vorticity(u, v, dx_km):
    """
    Finite-difference spatial derivatives: dv/dx - du/dy
    dx_km is grid spacing in kilometers (converted to meters for SI output).
    """
    dx_m = dx_km * 1000.0
    dv_dx = np.gradient(v, dx_m, axis=1)
    du_dy = np.gradient(u, dx_m, axis=0)
    return dv_dx - du_dy

def verify_gradient_accuracy(x, y, u, v, analytical_vorticity):
    dx_km = x[0, 1] - x[0, 0]
    num_vorticity = compute_numerical_vorticity(u, v, dx_km)
    
    # Ignore boundary ring (where finite difference order degrades)
    inner_mask = (np.abs(x) < 400.0) & (np.abs(y) < 400.0)
    
    err_map = np.abs(num_vorticity - analytical_vorticity)[inner_mask]
    analytical_norm = np.max(np.abs(analytical_vorticity[inner_mask]))
    
    rmse_pct = (np.sqrt(np.mean(err_map**2)) / analytical_norm) * 100.0
    linf_pct = (np.max(err_map) / analytical_norm) * 100.0
    
    return rmse_pct, linf_pct