"""
TRACEBIND Phase 8 - Phase 7 Metric Operator Characterization & Sensitivity Analysis
====================================================================================
Performs a dual-category evaluation of the frozen compute_phase_coherence() operator:

Category A: Mathematical Characterization (Static Baseline + Kinetic Descriptors)
Category B: Sensitivity Analysis (Parameter Sweeps, Multi-Type Perturbation Matrix)

Deliverables generated in REPO_ROOT / "phase8" / "operator_characterization":
- response_library_v1.0.json (Includes Operator SHA-256 Hash)
- perturbation_matrix.csv
- sensitivity_curves/*.png
"""

import sys
import json
import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Resilient Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent

# Walk up directory tree until REPO_ROOT containing 'phase7' is found
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT.parent != REPO_ROOT:
    if (REPO_ROOT / "phase7").exists():
        break
    REPO_ROOT = REPO_ROOT.parent

COHERENCE_MODULE_PATH = REPO_ROOT / "phase7" / "sandbox" / "metrics" / "coherence.py"

if not COHERENCE_MODULE_PATH.exists():
    raise FileNotFoundError(f"Target operator file not found: {COHERENCE_MODULE_PATH}")

# Direct dynamic load of coherence module (bypasses __init__.py package requirements)
spec = importlib.util.spec_from_file_location("coherence_module", COHERENCE_MODULE_PATH)
coherence_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coherence_module)

compute_phase_coherence = coherence_module.compute_phase_coherence


def get_operator_sha256(filepath):
    """Computes SHA-256 hash of the frozen phase7 operator source file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# --- Grid Setup ---
def build_grid(nx=121, ny=121, dx=2500.0):
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dx
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2) + 1e-5
    theta = np.arctan2(yy, xx)
    return xx, yy, r, theta, dx


# --- Kinematic Flow Descriptors ---
def compute_flow_descriptors(u, v, dx):
    spd = np.sqrt(u**2 + v**2)
    mean_speed = float(np.mean(spd))
    rms_speed = float(np.sqrt(np.mean(spd**2)))

    dudx, dudy = np.gradient(u, dx, axis=1), np.gradient(u, dx, axis=0)
    dvdx, dvdy = np.gradient(v, dx, axis=1), np.gradient(v, dx, axis=0)

    vorticity = dvdx - dudy
    divergence = dudx + dvdy

    return {
        "mean_speed": round(mean_speed, 4),
        "rms_speed": round(rms_speed, 4),
        "mean_vorticity": round(float(np.mean(vorticity)), 8),
        "mean_divergence": round(float(np.mean(divergence)), 8),
    }


# --- Canonical Cases Generator ---
def generate_canonical_cases(nx=121, ny=121):
    xx, yy, r, theta, dx = build_grid(nx, ny)
    cases = {}

    cases["Zero Wind"] = (np.zeros_like(xx), np.zeros_like(yy))
    cases["Uniform Flow"] = (np.full_like(xx, 10.0), np.zeros_like(yy))

    v_theta_3 = np.full_like(r, 20.0)
    cases["Pure Rotation"] = (-v_theta_3 * np.sin(theta), v_theta_3 * np.cos(theta))

    omega = 1e-4
    v_theta_4 = omega * r
    cases["Solid Body Rotation"] = (-v_theta_4 * np.sin(theta), v_theta_4 * np.cos(theta))

    C = 1e6
    v_theta_5 = C / r
    cases["Potential Vortex"] = (-v_theta_5 * np.sin(theta), v_theta_5 * np.cos(theta))

    r0, gamma = 50000.0, 1e5
    v_theta_6 = (gamma / (2 * np.pi * r)) * (1.0 - np.exp(-((r / r0) ** 2)))
    cases["Lamb-Oseen Vortex"] = (-v_theta_6 * np.sin(theta), v_theta_6 * np.cos(theta))

    v_rad_8, v_tan_8 = -8.0, 4.0
    u_8 = v_rad_8 * np.cos(theta) - v_tan_8 * np.sin(theta)
    v_8 = v_rad_8 * np.sin(theta) + v_tan_8 * np.cos(theta)
    cases["Monsoon Convergent Flow"] = (u_8, v_8)

    alpha = np.radians(15.0)
    v_rad_9 = -v_theta_6 * np.tan(alpha)
    u_9 = v_rad_9 * np.cos(theta) - v_theta_6 * np.sin(theta)
    v_9 = v_rad_9 * np.sin(theta) + v_theta_6 * np.cos(theta)
    cases["Cyclonic Spiral"] = (u_9, v_9)

    u_10 = (-v_theta_6 * np.sin(theta)) + 15.0
    v_10 = v_theta_6 * np.cos(theta)
    cases["Steering Flow"] = (u_10, v_10)

    return cases, dx


# --- Main Runner ---
def run_characterization():
    output_dir = REPO_ROOT / "phase8" / "operator_characterization"
    curves_dir = output_dir / "sensitivity_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    sha256_hash = get_operator_sha256(COHERENCE_MODULE_PATH)
    xx, yy, r, theta, dx = build_grid()

    # --- Category A: Static Mathematical Characterization ---
    cases, _ = generate_canonical_cases()
    static_results = {}

    print("==========================================================================")
    print(f" CATEGORY A: MATHEMATICAL CHARACTERIZATION [SHA256: {sha256_hash[:10]}...]")
    print("==========================================================================")
    print(f"{'Case':<25} | {'Mean Spd':<9} | {'RMS Spd':<9} | {'Mean Vort':<10} | {'Mean Div':<10} | {'C_phi':<8}")
    print("-" * 82)

    for name, (u, v) in cases.items():
        c_phi = float(compute_phase_coherence(u, v))
        desc = compute_flow_descriptors(u, v, dx)
        desc["c_phi"] = round(c_phi, 6)
        static_results[name] = desc
        print(f"{name:<25} | {desc['mean_speed']:<9.2f} | {desc['rms_speed']:<9.2f} | {desc['mean_vorticity']:<10.2e} | {desc['mean_divergence']:<10.2e} | {c_phi:.6f}")

    # --- Category B: Sensitivity Analysis (Sweeps) ---
    sensitivity_results = {}

    # 1. Random Noise Baseline (100 Realizations)
    np.random.seed(42)
    noise_c_phis = [
        float(compute_phase_coherence(np.random.uniform(-10, 10, xx.shape), np.random.uniform(-10, 10, yy.shape)))
        for _ in range(100)
    ]
    sensitivity_results["Random_Noise_100_Runs"] = {
        "mean_c_phi": round(float(np.mean(noise_c_phis)), 6),
        "std_c_phi": round(float(np.std(noise_c_phis)), 6),
        "min_c_phi": round(float(np.min(noise_c_phis)), 6),
        "max_c_phi": round(float(np.max(noise_c_phis)), 6),
    }

    plt.figure(figsize=(6, 4))
    plt.hist(noise_c_phis, bins=15, edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(noise_c_phis), color="red", linestyle="--", label=f"Mean: {np.mean(noise_c_phis):.4f}")
    plt.title("Random Noise C_phi Distribution (100 Runs)")
    plt.xlabel("C_phi")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / "noise_statistics.png", dpi=200)
    plt.close()

    # 2. Steering Flow Sweep
    r0_base, gamma_base = 50000.0, 1e5
    vt_base = (gamma_base / (2 * np.pi * r)) * (1.0 - np.exp(-((r / r0_base) ** 2)))
    u_vortex, v_vortex = -vt_base * np.sin(theta), vt_base * np.cos(theta)

    u_steer_speeds = [0, 5, 10, 15, 20, 30, 40]
    c_phi_steer = [float(compute_phase_coherence(u_vortex + u_s, v_vortex)) for u_s in u_steer_speeds]
    sensitivity_results["Steering_Flow_Sweep"] = dict(zip(u_steer_speeds, [round(c, 6) for c in c_phi_steer]))

    plt.figure(figsize=(6, 4))
    plt.plot(u_steer_speeds, c_phi_steer, "o-")
    plt.title("C_phi vs. Steering Flow Speed")
    plt.xlabel("Steering Speed U_steer (m/s)")
    plt.ylabel("C_phi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curves_dir / "steering.png", dpi=200)
    plt.close()

    # 3. Spiral Inflow Sweep
    angles_deg = [0, 5, 10, 15, 20, 30, 45]
    c_phi_spiral = []
    for alpha_deg in angles_deg:
        alpha = np.radians(alpha_deg)
        v_rad = -vt_base * np.tan(alpha)
        u_sp = v_rad * np.cos(theta) - vt_base * np.sin(theta)
        v_sp = v_rad * np.sin(theta) + vt_base * np.cos(theta)
        c_phi_spiral.append(float(compute_phase_coherence(u_sp, v_sp)))
    sensitivity_results["Spiral_Inflow_Sweep"] = dict(zip(angles_deg, [round(c, 6) for c in c_phi_spiral]))

    plt.figure(figsize=(6, 4))
    plt.plot(angles_deg, c_phi_spiral, "s-", color="orange")
    plt.title("C_phi vs. Spiral Inflow Angle")
    plt.xlabel("Inflow Angle alpha (degrees)")
    plt.ylabel("C_phi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curves_dir / "spiral_angle.png", dpi=200)
    plt.close()

    # 4. Lamb-Oseen Core Radius Sweep
    r0_list = [20000, 35000, 50000, 75000, 100000]
    c_phi_lo_r0 = []
    for r0_val in r0_list:
        vt = (gamma_base / (2 * np.pi * r)) * (1.0 - np.exp(-((r / r0_val) ** 2)))
        c_phi_lo_r0.append(float(compute_phase_coherence(-vt * np.sin(theta), vt * np.cos(theta))))
    sensitivity_results["Lamb_Oseen_Radius_Sweep"] = dict(zip(r0_list, [round(c, 6) for c in c_phi_lo_r0]))

    plt.figure(figsize=(6, 4))
    plt.plot([r_val / 1000 for r_val in r0_list], c_phi_lo_r0, "d-", color="green")
    plt.title("C_phi vs. Lamb-Oseen Core Radius")
    plt.xlabel("Core Radius r0 (km)")
    plt.ylabel("C_phi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curves_dir / "lamb_oseen_radius.png", dpi=200)
    plt.close()

    # 5. Lamb-Oseen Circulation Gamma Sweep
    gamma_values = [2e4, 5e4, 1e5, 2e5, 5e5]
    c_phi_gamma = []
    for g_val in gamma_values:
        vt = (g_val / (2 * np.pi * r)) * (1.0 - np.exp(-((r / r0_base) ** 2)))
        c_phi_gamma.append(float(compute_phase_coherence(-vt * np.sin(theta), vt * np.cos(theta))))
    sensitivity_results["Lamb_Oseen_Circulation_Sweep"] = dict(zip(gamma_values, [round(c, 6) for c in c_phi_gamma]))

    plt.figure(figsize=(6, 4))
    plt.plot(gamma_values, c_phi_gamma, "^-", color="purple")
    plt.title("C_phi vs. Vortex Circulation Gamma")
    plt.xlabel("Circulation Gamma (m^2/s)")
    plt.ylabel("C_phi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curves_dir / "lamb_oseen_circulation.png", dpi=200)
    plt.close()

    # --- Multi-Type Perturbation Matrix ---
    levels = [0.05, 0.10, 0.20, 0.30]
    matrix_rows = []

    for case_name, (u_orig, v_orig) in cases.items():
        base_c_phi = float(compute_phase_coherence(u_orig, v_orig))
        std_u = np.std(u_orig) if np.std(u_orig) > 1e-5 else 1.0
        std_v = np.std(v_orig) if np.std(v_orig) > 1e-5 else 1.0

        for p in levels:
            # Type 1: Gaussian Noise
            np.random.seed(42)
            u_p1 = u_orig + np.random.normal(0, p * std_u, u_orig.shape)
            v_p1 = v_orig + np.random.normal(0, p * std_v, v_orig.shape)
            c1 = float(compute_phase_coherence(u_p1, v_p1))
            matrix_rows.append({"Case": case_name, "PerturbationType": "Gaussian_Noise", "Level": p, "Base_C_phi": round(base_c_phi, 6), "Perturbed_C_phi": round(c1, 6), "Delta_C_phi": round(c1 - base_c_phi, 6)})

            # Type 2: Random Missing Cells
            np.random.seed(42)
            mask_cells = np.random.rand(*u_orig.shape) < p
            u_p2, v_p2 = u_orig.copy(), v_orig.copy()
            u_p2[mask_cells], v_p2[mask_cells] = 0.0, 0.0
            c2 = float(compute_phase_coherence(u_p2, v_p2))
            matrix_rows.append({"Case": case_name, "PerturbationType": "Missing_Cells", "Level": p, "Base_C_phi": round(base_c_phi, 6), "Perturbed_C_phi": round(c2, 6), "Delta_C_phi": round(c2 - base_c_phi, 6)})

            # Type 3: Missing Wedge
            u_p3, v_p3 = u_orig.copy(), v_orig.copy()
            wedge_angle = p * np.pi
            mask_wedge = np.abs(theta) < (wedge_angle / 2.0)
            u_p3[mask_wedge], v_p3[mask_wedge] = 0.0, 0.0
            c3 = float(compute_phase_coherence(u_p3, v_p3))
            matrix_rows.append({"Case": case_name, "PerturbationType": "Missing_Wedge", "Level": p, "Base_C_phi": round(base_c_phi, 6), "Perturbed_C_phi": round(c3, 6), "Delta_C_phi": round(c3 - base_c_phi, 6)})

            # Type 4: Uniform Speed Bias
            u_p4 = u_orig + (p * std_u)
            v_p4 = v_orig + (p * std_v)
            c4 = float(compute_phase_coherence(u_p4, v_p4))
            matrix_rows.append({"Case": case_name, "PerturbationType": "Uniform_Bias", "Level": p, "Base_C_phi": round(base_c_phi, 6), "Perturbed_C_phi": round(c4, 6), "Delta_C_phi": round(c4 - base_c_phi, 6)})

    csv_path = output_dir / "perturbation_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Case", "PerturbationType", "Level", "Base_C_phi", "Perturbed_C_phi", "Delta_C_phi"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    # --- Save Output JSON Library v1.0 ---
    library_data = {
        "version": "1.0",
        "phase7_operator_sha256": sha256_hash,
        "static_characterization": static_results,
        "sensitivity_analysis": sensitivity_results,
    }

    json_path = output_dir / "response_library_v1.0.json"
    with open(json_path, "w") as f:
        json.dump(library_data, f, indent=2)

    print("-" * 82)
    print(f"[✓] Validation Package Generated Successfully in {output_dir}:")
    print(f"    - JSON Library: {json_path.name}")
    print(f"    - Perturbation Matrix: {csv_path.name}")
    print(f"    - Sensitivity Curves: {curves_dir.name}/ (steering, spiral_angle, lamb_oseen_radius, lamb_oseen_circulation, noise_statistics)")


if __name__ == "__main__":
    run_characterization()