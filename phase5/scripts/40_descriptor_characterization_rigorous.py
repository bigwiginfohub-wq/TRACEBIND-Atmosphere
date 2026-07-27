"""
40_descriptor_characterization_rigorous.py
-----------------------------------------------------
TRACEBIND Phase 5C (Refactored): Rigorous Descriptor Characterization Framework
Features:
- Parameterized Synthetic Ensembles (N=100 per class)
- Non-parametric Effect Sizes (Cliff's Delta)
- Surrogate Rigidity Metrics (Surrogate Std Dev & Coefficient of Variation)
- Exploratory Classifier Weighting (Explicitly Flagged)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tracebind_core import compute_row_wise_grid_spacing, compute_reduced_vector, generate_exact_fourier_surrogate

SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
RESULTS_DIR = PHASE5_DIR / "results"

DESCRIPTORS_IN = RESULTS_DIR / "storm_descriptors_5d.csv"
OUT_PROFILES = RESULTS_DIR / "zscore_profiles_rigorous.csv"
OUT_SYNTH_ENSEMBLE = RESULTS_DIR / "synthetic_ensemble_profiles.csv"
OUT_CHARACTERIZATION = RESULTS_DIR / "descriptor_rigidity_and_effects.csv"

FEATURE_NAMES = ["GE", "LE", "C_orient", "A_radial", "S_orient"]
N_SURROGATES = 100
N_SYNTH_SAMPLES = 100  # Parameterized ensemble size per synthetic class


def calculate_cliffs_delta(group1, group2):
    """Non-parametric effect size robust to unequal and small sample sizes."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    more = sum(x > y for x in group1 for y in group2)
    less = sum(x < y for x in group1 for y in group2)
    return (more - less) / (n1 * n2)


def generate_synthetic_ensemble_batch(ny=100, nx=100, count=100):
    """Generates stochastic parameter-varied families for synthetic geometric classes."""
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    
    ensembles = {
        "Synthetic_Radial": [],
        "Synthetic_Elliptical": [],
        "Synthetic_Frontal": [],
        "Synthetic_Noise": []
    }
    
    np.random.seed(42)
    
    for i in range(count):
        # 1. Radial Vortices: Randomize radius, depth, center offset, background noise
        r_scale = np.random.uniform(10.0, 25.0)
        depth = np.random.uniform(3000.0, 7000.0)
        off_x, off_y = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
        r2 = (x - (cx + off_x))**2 + (y - (cy + off_y))**2
        radial = 101325.0 - depth * np.exp(-r2 / (2 * r_scale**2)) + np.random.normal(0, 100, (ny, nx))
        ensembles["Synthetic_Radial"].append(radial)
        
        # 2. Elliptical Vortices: Randomize major/minor axes, orientation, shear
        a = np.random.uniform(10.0, 20.0)
        b = np.random.uniform(25.0, 45.0)
        angle = np.random.uniform(0, np.pi)
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        r2_e = (x_rot / a)**2 + (y_rot / b)**2
        elliptical = 101325.0 - depth * np.exp(-r2_e / 2.0) + np.random.normal(0, 100, (ny, nx))
        ensembles["Synthetic_Elliptical"].append(elliptical)
        
        # 3. Frontal Waves: Randomize wavelength, wave amplitude, gradient, orientation
        wavelen = np.random.uniform(20.0, 60.0)
        amp = np.random.uniform(1000.0, 4000.0)
        grad = np.random.uniform(2000.0, 5000.0)
        front = 101325.0 + amp * np.sin(2 * np.pi * x / wavelen) + grad * (y - cy) / ny + np.random.normal(0, 150, (ny, nx))
        ensembles["Synthetic_Frontal"].append(front)
        
        # 4. Pure Gaussian White Noise
        scale = np.random.uniform(500.0, 1500.0)
        noise = np.random.normal(loc=101325.0, scale=scale, size=(ny, nx))
        ensembles["Synthetic_Noise"].append(noise)
        
    return ensembles


def profile_single_field(field_2d, dx_rows, dy, storm_id, cohort_type):
    v_obs = compute_reduced_vector(field_2d, dx_rows, dy)
    
    surr_vecs = []
    for s_idx in range(N_SURROGATES):
        surr_field = generate_exact_fourier_surrogate(field_2d, seed=1000 + s_idx)
        v_surr = compute_reduced_vector(surr_field, dx_rows, dy)
        surr_vecs.append(v_surr)
        
    surr_vecs = np.array(surr_vecs)
    mu_surr = np.mean(surr_vecs, axis=0)
    std_surr = np.std(surr_vecs, axis=0)
    
    # Coefficient of Variation (CV) under null
    cv_surr = std_surr / (np.abs(mu_surr) + 1e-12)
    
    z_scores = (v_obs - mu_surr) / (std_surr + 1e-12)
    
    record = {
        "storm_id": storm_id,
        "cohort_type": cohort_type
    }
    for i, f in enumerate(FEATURE_NAMES):
        record[f"{f}_obs"] = float(v_obs[i])
        record[f"{f}_z"] = float(z_scores[i])
        record[f"{f}_abs_z"] = float(abs(z_scores[i]))
        record[f"{f}_surr_mu"] = float(mu_surr[i])
        record[f"{f}_surr_std"] = float(std_surr[i])
        record[f"{f}_surr_cv"] = float(cv_surr[i])
        
    return record


def run_rigorous_characterization():
    print("[*] Starting TRACEBIND Rigorous Descriptor Characterization Framework...\n")
    df_desc = pd.read_csv(DESCRIPTORS_IN)
    real_records = []
    
    # 1. Profile Real ERA5 Datasets
    print("--- 1. Processing ERA5 Cohorts ---")
    for _, row in df_desc.iterrows():
        storm_id = str(row['storm_id']).strip()
        cohort = str(row['cohort_type']).strip()
        
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"
        if not nc_file.exists():
            nc_file = DATA_DIR / f"era5_{storm_id}.nc"
            
        if not nc_file.exists():
            continue
            
        ds = xr.open_dataset(nc_file)
        dx_rows, dy = compute_row_wise_grid_spacing(ds)
        var_name = 'msl' if 'msl' in ds else 'mean_sea_level_pressure'
        field = ds[var_name].values
        field_2d = field[field.shape[0] // 2, :, :] if field.ndim >= 3 else field
        ds.close()
        
        rec = profile_single_field(field_2d, dx_rows, dy, storm_id, cohort)
        real_records.append(rec)
        print(f"  [✓] ERA5 Profiled: {storm_id:15s} ({cohort:15s})")

    df_real = pd.DataFrame(real_records)
    df_real.to_csv(OUT_PROFILES, index=False)

    # 2. Profile Synthetic Ensembles (100 per class)
    print("\n--- 2. Processing Parameterized Synthetic Ensembles (N=100 per class) ---")
    ny, nx = 100, 100
    dx_synth, dy_synth = np.full((ny,), 25000.0), 25000.0
    synth_batches = generate_synthetic_ensemble_batch(ny, nx, count=N_SYNTH_SAMPLES)
    
    synth_records = []
    for cls_name, field_list in synth_batches.items():
        print(f"  [...] Profiling Ensemble: {cls_name} (N={len(field_list)})")
        for idx, field_2d in enumerate(field_list):
            item_id = f"{cls_name}_{idx:03d}"
            rec = profile_single_field(field_2d, dx_synth, dy_synth, item_id, cls_name)
            synth_records.append(rec)
            
    df_synth = pd.DataFrame(synth_records)
    df_synth.to_csv(OUT_SYNTH_ENSEMBLE, index=False)
    print(f"  [✓] Synthetic Ensemble Matrix Saved: {OUT_SYNTH_ENSEMBLE}")

    # 3. Analyze Descriptor Rigidity vs. Physical Deviation
    print("\n==========================================================================================")
    print("                 DESCRIPTOR SURROGATE RIGIDITY & STABILITY METRICS                        ")
    print("==========================================================================================")
    rigidity_table = []
    for f in FEATURE_NAMES:
        mean_std = df_real[f"{f}_surr_std"].mean()
        mean_cv = df_real[f"{f}_surr_cv"].mean()
        mean_abs_z = df_real[f"{f}_abs_z"].mean()
        
        rigidity_table.append({
            "Descriptor": f,
            "Mean_Surr_Std": round(mean_std, 6),
            "Mean_Surr_CV": round(mean_cv, 6),
            "Mean_Abs_Z": round(mean_abs_z, 2)
        })
    df_rigidity = pd.DataFrame(rigidity_table)
    print(df_rigidity.to_string(index=False))

    # 4. Compute Non-Parametric Separation (Cliff's Delta) & Exploratory Classifier Weights
    # Normalize string matching to handle space variations
    df_real['cohort_clean'] = df_real['cohort_type'].str.replace(' ', '').str.lower()
    
    tc_data = df_real[df_real['cohort_clean'].isin(['coretc', 'expandedtc', 'core', 'expanded'])]
    ctrl_data = df_real[df_real['cohort_clean'].isin(['negativecontrol', 'control'])]

    if len(ctrl_data) == 0:
        print("\n[!] Warning: No negative control cases matched! Checking unique cohorts present:")
        print(df_real['cohort_type'].unique())
        return

    # Prepare Exploratory Linear Classifiers
    X_expl = np.vstack([
        tc_data[[f"{f}_z" for f in FEATURE_NAMES]].values,
        ctrl_data[[f"{f}_z" for f in FEATURE_NAMES]].values
    ])
    y_expl = np.hstack([np.ones(len(tc_data)), np.zeros(len(ctrl_data))])

    # Fix: Use penalty='l2' with C=1e6 to simulate unpenalized regression without deprecation warnings
    clf_log = LogisticRegression(C=1e6, solver='lbfgs')
    clf_log.fit(X_expl, y_expl)
    
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_expl, y_expl)

    print("\n==========================================================================================")
    print("          TC vs CONTROL SEPARATION (CLIFF'S DELTA & EXPLORATORY MODEL WEIGHTS)             ")
    print("==========================================================================================")
    
    char_records = []
    for idx, f in enumerate(FEATURE_NAMES):
        tc_z = tc_data[f"{f}_z"].values
        ctrl_z = ctrl_data[f"{f}_z"].values
        
        c_delta = calculate_cliffs_delta(tc_z, ctrl_z)
        log_w = clf_log.coef_[0][idx]
        lda_w = clf_lda.coef_[0][idx]
        
        # Pull synthetic median Z-scores for context
        rad_z = df_synth[df_synth['cohort_type'] == 'Synthetic_Radial'][f"{f}_z"].median()
        ell_z = df_synth[df_synth['cohort_type'] == 'Synthetic_Elliptical'][f"{f}_z"].median()
        front_z = df_synth[df_synth['cohort_type'] == 'Synthetic_Frontal'][f"{f}_z"].median()
        
        char_records.append({
            "Descriptor": f,
            "Cliffs_Delta": round(c_delta, 3),
            "Synth_Radial_Z": round(rad_z, 2),
            "Synth_Elliptical_Z": round(ell_z, 2),
            "Synth_Front_Z": round(front_z, 2),
            "Exploratory_LogReg_W": round(log_w, 4),
            "Exploratory_LDA_W": round(lda_w, 4)
        })

    df_char = pd.DataFrame(char_records).sort_values(by="Cliffs_Delta", ascending=False)
    df_char.to_csv(OUT_CHARACTERIZATION, index=False)
    print(df_char.to_string(index=False))
    print("==========================================================================================\n")
    print("[*] Rigorous characterization complete!\n")

if __name__ == "__main__":
    run_rigorous_characterization()