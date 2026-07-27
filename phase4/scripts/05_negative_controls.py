import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, gaussian_filter

from metrics import (
    compute_tracebind_v1,
    compute_tracebind_v2,
    compute_morans_i,
    compute_gearys_c,
    compute_gradient_energy,
    compute_texture_entropy
)

# Output Paths
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CSV_DIR = os.path.join(OUTPUT_DIR, "metrics")

for d in [FIG_DIR, CSV_DIR]:
    os.makedirs(d, exist_ok=True)

def generate_base_cyclone(shape=(128, 128)):
    """Generates synthetic cyclonic vortex field."""
    x = np.linspace(-3, 3, shape[1])
    y = np.linspace(-3, 3, shape[0])
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    return np.exp(-r**2) * np.cos(3 * np.arctan2(Y, X))

def run_all_property_tests():
    np.random.seed(42)
    base = generate_base_cyclone(shape=(128, 128))
    
    print("=========================================================================")
    print("      TRACEBIND PROPERTY & INVARIANCE EXPERIMENTAL BENCHMARK             ")
    print("=========================================================================\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: PROGRESSIVE SPATIAL RANDOMIZATION
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print(" EXPERIMENT 1: Progressive Spatial Randomization (0% -> 100%)")
    print("-------------------------------------------------------------------------")
    
    scramble_pcts = [0, 10, 25, 50, 75, 100]
    rand_data = []

    for pct in scramble_pcts:
        f = base.copy()
        if pct > 0:
            flat = f.ravel()
            num_swap = int(len(flat) * (pct / 100.0))
            idx = np.random.choice(len(flat), num_swap, replace=False)
            vals = flat[idx].copy()
            np.random.shuffle(vals)
            flat[idx] = vals
            f = flat.reshape(base.shape)

        rand_data.append({
            "Scramble_%": pct,
            "TB-v1": compute_tracebind_v1(f),
            "TB-v2": compute_tracebind_v2(f),
            "Moran I": compute_morans_i(f),
            "Geary C": compute_gearys_c(f)
        })

    df_rand = pd.DataFrame(rand_data)
    print(df_rand.to_string(index=False))
    
    csv_path1 = os.path.join(CSV_DIR, "randomization_response.csv")
    fig_path1 = os.path.join(FIG_DIR, "randomization_response.png")
    df_rand.to_csv(csv_path1, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scramble_pcts, df_rand["TB-v1"], marker="o", lw=2, label="TRACEBIND v1")
    ax.plot(scramble_pcts, df_rand["TB-v2"], marker="s", lw=2, label="TRACEBIND v2")
    ax.plot(scramble_pcts, df_rand["Moran I"], marker="^", lw=2, label="Moran's I")
    ax.plot(scramble_pcts, df_rand["Geary C"], marker="d", lw=2, label="Geary's C")
    ax.set_xlabel("Spatial Randomization (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Figure 1: Progressive Spatial Randomization Response", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path1, dpi=300)
    plt.close()

    v1_init, v1_final = df_rand.iloc[0]["TB-v1"], df_rand.iloc[-1]["TB-v1"]
    v2_init, v2_final = df_rand.iloc[0]["TB-v2"], df_rand.iloc[-1]["TB-v2"]
    
    print("\nSUMMARY OBSERVATIONS:")
    print(f"  * TB-v1 changed from {v1_init:.4f} to {v1_final:.4f} ({((v1_init - v1_final)/v1_init)*100:.1f}% decrease)")
    print(f"  * TB-v2 changed from {v2_init:.4f} to {v2_final:.4f} ({((v2_init - v2_final)/v2_init)*100:.1f}% decrease)")
    print(f"  Saved CSV : {csv_path1}")
    print(f"  Saved PNG : {fig_path1}\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: SIGNAL-TO-NOISE DEGRADATION
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print(" EXPERIMENT 2: Signal-to-Noise Degradation (Additive Gaussian Noise)")
    print("-------------------------------------------------------------------------")
    
    noise_sigmas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    noise_data = []

    for sigma in noise_sigmas:
        f_noisy = base + np.random.normal(0, sigma, base.shape)
        noise_data.append({
            "Sigma": sigma,
            "TB-v1": compute_tracebind_v1(f_noisy),
            "TB-v2": compute_tracebind_v2(f_noisy),
            "Moran I": compute_morans_i(f_noisy),
            "Geary C": compute_gearys_c(f_noisy)
        })

    df_noise = pd.DataFrame(noise_data)
    print(df_noise.to_string(index=False))

    csv_path2 = os.path.join(CSV_DIR, "noise_response.csv")
    fig_path2 = os.path.join(FIG_DIR, "noise_response.png")
    df_noise.to_csv(csv_path2, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_sigmas, df_noise["TB-v1"], marker="o", lw=2, label="TRACEBIND v1")
    ax.plot(noise_sigmas, df_noise["TB-v2"], marker="s", lw=2, label="TRACEBIND v2")
    ax.plot(noise_sigmas, df_noise["Moran I"], marker="^", lw=2, label="Moran's I")
    ax.plot(noise_sigmas, df_noise["Geary C"], marker="d", lw=2, label="Geary's C")
    ax.set_xlabel("Additive Noise Std (σ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Signal-to-Noise Degradation Curves", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=300)
    plt.close()

    print(f"\nSaved CSV : {csv_path2}")
    print(f"Saved PNG : {fig_path2}\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: RESOLUTION CONVERGENCE
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print(" EXPERIMENT 3: Grid Resolution Scaling Convergence")
    print("-------------------------------------------------------------------------")
    
    resolutions = [32, 64, 128, 256, 512]
    res_data = []

    for r in resolutions:
        f_res = generate_base_cyclone(shape=(r, r))
        res_data.append({
            "Resolution": f"{r}x{r}",
            "TB-v1": compute_tracebind_v1(f_res),
            "TB-v2": compute_tracebind_v2(f_res),
            "Moran I": compute_morans_i(f_res),
            "Geary C": compute_gearys_c(f_res)
        })

    df_res = pd.DataFrame(res_data)
    print(df_res[["Resolution", "TB-v1", "TB-v2", "Moran I", "Geary C"]].to_string(index=False))

    csv_path3 = os.path.join(CSV_DIR, "resolution_convergence.csv")
    fig_path3 = os.path.join(FIG_DIR, "resolution_convergence.png")
    df_res.to_csv(csv_path3, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(resolutions, df_res["TB-v1"], marker="o", lw=2, label="TRACEBIND v1")
    ax.plot(resolutions, df_res["TB-v2"], marker="s", lw=2, label="TRACEBIND v2")
    ax.plot(resolutions, df_res["Moran I"], marker="^", lw=2, label="Moran's I")
    ax.set_xscale("log", base=2)
    ax.set_xticks(resolutions)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Grid Resolution (pixels)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Figure 3: Grid Scale Numerical Convergence", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path3, dpi=300)
    plt.close()

    v2_std = np.std(df_res["TB-v2"])
    res_status = "PASS" if v2_std <= 0.050 else "FAIL"
    print("\nVALIDATION METRICS:")
    print(f"  * Resolution Standard Deviation (TB-v2): {v2_std:.4f}")
    print(f"  * Convergence Threshold: <= 0.0500 | Result: {res_status}")
    print(f"  Saved CSV : {csv_path3}")
    print(f"  Saved PNG : {fig_path3}\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: ROTATIONAL & TRANSLATIONAL INVARIANCE
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print(" EXPERIMENT 4: Rotational & Translational Physical Invariance")
    print("-------------------------------------------------------------------------")
    
    angles = [0, 45, 90, 135, 180]
    inv_data = []

    for a in angles:
        f_rot = rotate(base, a, reshape=False, mode='reflect')
        inv_data.append({
            "Transformation": f"Rotation {a}°",
            "TB-v1": compute_tracebind_v1(f_rot),
            "TB-v2": compute_tracebind_v2(f_rot),
            "Moran I": compute_morans_i(f_rot)
        })

    shifts = [(0, 0), (5, 5), (10, -10), (-15, 5)]
    for s_y, s_x in shifts:
        f_shift = shift(base, shift=(s_y, s_x), mode='reflect')
        inv_data.append({
            "Transformation": f"Shift ({s_y}, {s_x})",
            "TB-v1": compute_tracebind_v1(f_shift),
            "TB-v2": compute_tracebind_v2(f_shift),
            "Moran I": compute_morans_i(f_shift)
        })

    df_inv = pd.DataFrame(inv_data)
    print(df_inv.to_string(index=False))

    csv_path4 = os.path.join(CSV_DIR, "invariance.csv")
    df_inv.to_csv(csv_path4, index=False)

    rot_max_dev = float(np.max(np.abs(df_inv.iloc[:5]["TB-v2"] - df_inv.iloc[0]["TB-v2"])))
    rot_status = "PASS" if rot_max_dev <= 0.010 else "FAIL"
    print("\nVALIDATION METRICS:")
    print(f"  * Maximum Rotation Deviation (TB-v2): {rot_max_dev:.4f}")
    print(f"  * Invariance Threshold: <= 0.0100 | Result: {rot_status}")
    print(f"  Saved CSV : {csv_path4}\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 5: GAUSSIAN BLUR
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print(" EXPERIMENT 5: Smoothness vs. Organization (Gaussian Blur Response)")
    print("-------------------------------------------------------------------------")
    
    blur_sigmas = [0.0, 1.0, 2.0, 4.0, 8.0]
    blur_data = []

    for s in blur_sigmas:
        f_blur = gaussian_filter(base, sigma=s) if s > 0 else base.copy()
        blur_data.append({
            "Sigma": s,
            "TB-v1": compute_tracebind_v1(f_blur),
            "TB-v2": compute_tracebind_v2(f_blur),
            "Moran I": compute_morans_i(f_blur),
            "Grad Energy": compute_gradient_energy(f_blur)
        })

    df_blur = pd.DataFrame(blur_data)
    print(df_blur.to_string(index=False))

    csv_path5 = os.path.join(CSV_DIR, "gaussian_blur_response.csv")
    fig_path5 = os.path.join(FIG_DIR, "blur_response.png")
    df_blur.to_csv(csv_path5, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(blur_sigmas, df_blur["TB-v1"], marker="o", lw=2, label="TRACEBIND v1")
    ax.plot(blur_sigmas, df_blur["TB-v2"], marker="s", lw=2, label="TRACEBIND v2")
    ax.plot(blur_sigmas, df_blur["Moran I"], marker="^", lw=2, label="Moran's I")
    ax.set_xlabel("Gaussian Blur Kernel (σ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Smoothness vs Organization Response (Gaussian Blur)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path5, dpi=300)
    plt.close()

    print(f"\nSaved CSV : {csv_path5}")
    print(f"Saved PNG : {fig_path5}\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 6: CONTINUOUS LINEAR MIXTURE CONTINUUM
    # -------------------------------------------------------------------------
    print("-------------------------------------------------------------------------")
    print(" EXPERIMENT 6: Cyclone-to-Noise Continuous Linear Mixture Continuum")
    print("-------------------------------------------------------------------------")
    
    alphas = np.linspace(0.0, 1.0, 11)
    noise_field = np.random.normal(0, np.std(base), base.shape)
    mix_data = []

    for alpha in alphas:
        f_mix = (1.0 - alpha) * base + alpha * noise_field
        mix_data.append({
            "Alpha": round(alpha, 2),
            "TB-v1": compute_tracebind_v1(f_mix),
            "TB-v2": compute_tracebind_v2(f_mix),
            "Moran I": compute_morans_i(f_mix),
            "Geary C": compute_gearys_c(f_mix)
        })

    df_mix = pd.DataFrame(mix_data)
    print(df_mix.to_string(index=False))

    csv_path6 = os.path.join(CSV_DIR, "continuous_mixture_response.csv")
    fig_path6 = os.path.join(FIG_DIR, "continuous_mixture_response.png")
    df_mix.to_csv(csv_path6, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, df_mix["TB-v1"], marker="o", lw=2, label="TRACEBIND v1")
    ax.plot(alphas, df_mix["TB-v2"], marker="s", lw=2, label="TRACEBIND v2")
    ax.plot(alphas, df_mix["Moran I"], marker="^", lw=2, label="Moran's I")
    ax.plot(alphas, df_mix["Geary C"], marker="d", lw=2, label="Geary's C")
    ax.set_xlabel("Noise Fraction α (0.0 = Cyclone, 1.0 = White Noise)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5: Cyclone-to-Noise Linear Mixture Continuum", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_path6, dpi=300)
    plt.close()

    print(f"\nSaved CSV : {csv_path6}")
    print(f"Saved PNG : {fig_path6}\n")

    # Visual Summary
    print("=========================================================================")
    print(" FIGURES GENERATED:")
    print("  [✓] randomization_response.png")
    print("  [✓] noise_response.png")
    print("  [✓] resolution_convergence.png")
    print("  [✓] blur_response.png")
    print("  [✓] continuous_mixture_response.png")
    print("=========================================================================")

if __name__ == "__main__":
    run_all_property_tests()