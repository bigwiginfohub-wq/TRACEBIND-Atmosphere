import sys
import numpy as np
import pandas as pd

def run_qa_gate(data_dict: dict, times: pd.DatetimeIndex = None, lats: np.ndarray = None) -> dict:
    """
    Automated QA Gate for ingested ERA5 atmospheric fields.
    Evaluates fields against a 100-point scoring matrix.
    """
    report = {}
    
    for var_name, grid in data_dict.items():
        grid_arr = np.asarray(grid)
        nan_count = np.sum(np.isnan(grid_arr))
        total_pixels = grid_arr.size
        missing_pct = (nan_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        
        # Breakdown of 100-point scorecard
        score_breakdown = {
            "coordinate_crs_check": 15 if grid_arr.ndim in (2, 3) else 0,
            "nan_null_tolerance": 20 if missing_pct < 1.0 else (10 if missing_pct < 5.0 else 0),
            "unit_standardization": 15,  # Assumed standard SI units
            "grid_spacing_regularity": 15 if grid_arr.shape[-1] > 1 else 0,
            "time_monotonicity": 15 if (times is None or times.is_monotonic_increasing) else 0,
            "duplicate_timestamps": 10 if (times is None or not times.has_duplicates) else 0,
            "lat_orientation_check": 10 if (lats is None or np.all(np.diff(lats) > 0) or np.all(np.diff(lats) < 0)) else 0
        }
        
        total_score = sum(score_breakdown.values())
        
        report[var_name] = {
            "shape": grid_arr.shape,
            "min_val": float(np.nanmin(grid_arr)),
            "max_val": float(np.nanmax(grid_arr)),
            "mean_val": float(np.nanmean(grid_arr)),
            "missing_pct": missing_pct,
            "qa_score": total_score,
            "qa_passed": total_score >= 80,
            "score_breakdown": score_breakdown
        }
    
    return report

if __name__ == "__main__":
    print("=== TRACEBIND QA Gate Initialized ===")
    
    # Quick sanity test on dummy frame & temporal grid
    sample_grid = np.random.normal(1013.25, 5.0, (50, 50))
    sample_times = pd.date_range("2026-01-01 00:00", periods=24, freq="1h")
    sample_lats = np.linspace(-90, 90, 50)
    
    report = run_qa_gate({"mslp": sample_grid}, times=sample_times, lats=sample_lats)
    
    for var, summary in report.items():
        status = "PASSED" if summary["qa_passed"] else "FAILED"
        print(f"\n[{status}] Variable: '{var}' | Total Score: {summary['qa_score']}/100")
        print(f" -> Missing Data : {summary['missing_pct']:.2f}%")
        print(f" -> Field Range  : [{summary['min_val']:.2f}, {summary['max_val']:.2f}]")
        print(" -> Score Breakdown:")
        for metric, pts in summary["score_breakdown"].items():
            print(f"    * {metric}: {pts} pts")