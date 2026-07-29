import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def execute_inference():
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / "unblinded_master_dataset.csv"
    output_report = script_dir / "inference_report.json"

    df = pd.read_csv(dataset_path)

    # Ensure c_phi column is present
    cphi_col = "c_phi" if "c_phi" in df.columns else "C_phi"
    
    cyclone = df[df["condition"] == "cyclone"][cphi_col].values
    control = df[df["condition"] == "control"][cphi_col].values

    if len(cyclone) == 0 or len(control) == 0:
        raise ValueError("Could not split data into 'cyclone' and 'control' groups. Check condition column.")

    # Welch's t-test
    t_stat, p_welch = stats.ttest_ind(cyclone, control, equal_var=False)

    # Mann-Whitney U Test (One-sided: cyclone > control as pre-registered)
    u_stat, p_mw = stats.mannwhitneyu(cyclone, control, alternative="greater")

    # Effect Size: Hedges' g
    n1, n2 = len(cyclone), len(control)
    s1, s2 = np.std(cyclone, ddof=1), np.std(control, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (np.mean(cyclone) - np.mean(control)) / s_pooled if s_pooled != 0 else 0.0
    hedges_g = d * (1 - (3 / (4 * (n1 + n2) - 9)))

    # Cliff's Delta (Non-parametric effect size)
    d_matrix = np.subtract.outer(cyclone, control)
    cliffs_delta = (np.sum(d_matrix > 0) - np.sum(d_matrix < 0)) / (n1 * n2)

    results = {
        "cohort_summary": {
            "n_cyclone": n1,
            "n_control": n2,
            "total_sample": n1 + n2
        },
        "group_statistics": {
            "cyclone_mean": float(np.mean(cyclone)),
            "cyclone_std": float(s1),
            "cyclone_median": float(np.median(cyclone)),
            "control_mean": float(np.mean(control)),
            "control_std": float(s2),
            "control_median": float(np.median(control))
        },
        "pre_registered_hypothesis_tests": {
            "directional_hypothesis": "cyclone_C_phi > control_C_phi",
            "mann_whitney_u": float(u_stat),
            "p_val_mann_whitney_one_sided": float(p_mw),
            "welch_t_stat": float(t_stat),
            "p_val_welch_two_sided": float(p_welch),
            "hedges_g": float(hedges_g),
            "cliffs_delta": float(cliffs_delta)
        }
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ STATISTICAL INFERENCE COMPLETE ================")
    print(f"Cyclone Mean C_phi : {np.mean(cyclone):.6f} (± {s1:.6f})")
    print(f"Control Mean C_phi : {np.mean(control):.6f} (± {s2:.6f})")
    print(f"Hedges' g          : {hedges_g:.4f}")
    print(f"Mann-Whitney U p-val: {p_mw:.6e}")
    print(f"Report written to  : {output_report.resolve()}\n")

if __name__ == "__main__":
    execute_inference()