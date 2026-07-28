import json
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

def compute_metrics_for_system(nc_path: Path):
    with xr.open_dataset(nc_path) as ds:
        # Load grid coordinates and core arrays
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        times = ds["time"].values
        
        vort = ds["vorticity"].values
        ow = ds["okubo_weiss"].values

        # Ensure lats are ascending for interpolation consistency
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            vort = np.flip(vort, axis=-2)
            ow = np.flip(ow, axis=-2)

        n_time = len(times)
        
        # Grid conversion (approx 1 deg lat = 111 km)
        lat_center = np.mean(lats)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat_center))

        # Grid spacing in meters for Stokes' area integration
        dy_m = (abs(lats[1] - lats[0]) * km_per_deg_lat * 1000.0) if len(lats) > 1 else 10000.0
        dx_m = (abs(lons[1] - lons[0]) * km_per_deg_lon * 1000.0) if len(lons) > 1 else 10000.0
        cell_area_m2 = dy_m * dx_m

        # Time-averaged spatial metrics initialization
        circ_list, compactness_list, asymmetry_list, filament_list = [], [], [], []
        entropy_list, sharpness_list = [], []

        # Coordinate grids relative to spatial center
        mid_lat, mid_lon = lats[len(lats)//2], lons[len(lons)//2]
        y_km = (lats - mid_lat) * km_per_deg_lat
        x_km = (lons - mid_lon) * km_per_deg_lon
        X_km, Y_km = np.meshgrid(x_km, y_km)
        R_km = np.sqrt(X_km**2 + Y_km**2)

        # Process each time slice
        core_masks = []
        for t in range(n_time):
            vort_t = vort[t]
            ow_t = ow[t]

            # --- 1. Okubo-Weiss Filamentation Fraction ---
            ow_std = np.nanstd(ow_t)
            threshold = -0.2 * ow_std
            core_mask = ow_t < threshold
            core_masks.append(core_mask)
            filament_fraction = np.sum(core_mask) / core_mask.size
            filament_list.append(filament_fraction)

            # --- 2. Compactness Ratio (100km vs 500km) ---
            core_vort = np.sum(vort_t[R_km <= 100])
            outer_vort = np.sum(vort_t[R_km <= 500])
            compactness = float(core_vort / outer_vort) if outer_vort != 0 else 0.0
            compactness_list.append(compactness)

            # --- 3. Vortex Asymmetry Index ---
            r_mask = R_km <= 200
            if np.any(r_mask):
                vort_core = vort_t[r_mask]
                mean_vort = np.mean(vort_core)
                asymmetry = np.sum((vort_core - mean_vort)**2) / (np.sum(mean_vort**2) + 1e-12)
                asymmetry_list.append(float(asymmetry))
            else:
                asymmetry_list.append(0.0)

            # --- 4. Circulation Score (at R=250km via Stokes' Theorem) ---
            circ_mask = R_km <= 250.0
            if np.any(circ_mask):
                circ = np.sum(vort_t[circ_mask]) * cell_area_m2
                circ_list.append(float(circ))

            # --- 5. Phase Boundary Entropy & Sharpness ---
            dy, dx = np.gradient(vort_t)
            grad_mag = np.sqrt(dx**2 + dy**2)
            grad_angle = np.arctan2(dy, dx)

            # Boundary zone (100km <= R <= 300km)
            bnd_mask = (R_km >= 100) & (R_km <= 300)
            if np.any(bnd_mask):
                angles = grad_angle[bnd_mask]
                hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi), density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                entropy_list.append(float(entropy))
                
                sharpness = np.mean(grad_mag[bnd_mask])
                sharpness_list.append(float(sharpness))

        # --- 6. Coherence Index ---
        coherence_scores = []
        if len(core_masks) > 1:
            # Multi-timestep Temporal Jaccard Persistence
            for t in range(len(core_masks) - 1):
                m1, m2 = core_masks[t], core_masks[t+1]
                intersection = np.sum(m1 & m2)
                union = np.sum(m1 | m2)
                jaccard = float(intersection / union) if union > 0 else 0.0
                coherence_scores.append(jaccard)
        else:
            # Single-snapshot Spatial Structural Coherence Proxy
            m = core_masks[0]
            m_shift = np.roll(m, shift=1, axis=0)
            intersection = np.sum(m & m_shift)
            union = np.sum(m | m_shift)
            jaccard = float(intersection / union) if union > 0 else 0.0
            coherence_scores.append(jaccard)

        return {
            "circulation_250km_mean": float(np.mean(circ_list)) if circ_list else None,
            "compactness_ratio_mean": float(np.mean(compactness_list)),
            "asymmetry_index_mean": float(np.mean(asymmetry_list)),
            "filamentation_fraction_mean": float(np.mean(filament_list)),
            "coherence_index_mean": float(np.mean(coherence_scores)) if coherence_scores else None,
            "boundary_entropy_bits_mean": float(np.mean(entropy_list)) if entropy_list else None,
            "boundary_sharpness_mean": float(np.mean(sharpness_list)) if sharpness_list else None
        }

def main():
    print("[INFO] Starting Phase 6D.2 & 6D.3 Metrics Calculation Engine...")
    base_dir = Path(__file__).resolve().parent
    manifest_path = base_dir / "feature_manifest.json"
    features_dir = base_dir / "data" / "features"
    out_dir = base_dir / "data" / "summary"

    if not manifest_path.exists():
        raise FileNotFoundError(f"[ERROR] Master manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows = []
    for sys_entry in manifest["systems"]:
        sys_id = sys_entry["system_id"]
        nc_path = features_dir / sys_entry["feature_file"]

        metrics = compute_metrics_for_system(nc_path)
        
        row = {
            "system_id": sys_id,
            "cohort_id": sys_entry["cohort_id"],
            "cohort_name": sys_entry["cohort_name"],
            "system_class": sys_entry["system_class"],
            "feature_sha256": sys_entry["feature_sha256"]
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_out = out_dir / "structural_tracebind_metrics.csv"
    parquet_out = out_dir / "structural_tracebind_metrics.parquet"

    df.to_csv(csv_out, index=False)
    df.to_parquet(parquet_out, index=False)

    print(f"[INFO] Phase 6D.2/3 Complete! Output written to:\n - {parquet_out}\n - {csv_out}")

if __name__ == "__main__":
    main()