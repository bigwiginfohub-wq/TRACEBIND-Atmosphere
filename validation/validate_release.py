#!/usr/bin/env python3
"""
TRACEBIND Release & Characterization Validation Pipeline (Strict Mode)
Runs regression, boundary, noise, and characterization suites, aggregates results,
and outputs a deterministic markdown report with system metrics.
"""

import sys
import os
import json
import subprocess
import platform
import datetime
import importlib
from pathlib import Path
import time

def get_git_info() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return branch, commit
    except Exception:
        return "UNKNOWN", "UNKNOWN"

def get_deps() -> dict[str, str]:
    deps = ["numpy", "scipy", "libpysal", "esda", "matplotlib"]
    res = {}
    for dep in deps:
        try:
            mod = importlib.import_module(dep)
            res[dep] = getattr(mod, '__version__', 'Installed')
        except ImportError:
            res[dep] = "NOT INSTALLED"
    return res

def run_suite(suite_id: str, relative_file_path: str) -> dict:
    start_time = time.time()
    
    script_dir = Path(__file__).resolve().parent
    candidate_1 = script_dir / relative_file_path
    candidate_2 = script_dir.parent / relative_file_path
    
    if candidate_1.exists():
        file_path = candidate_1
    elif candidate_2.exists():
        file_path = candidate_2
    else:
        return {
            "suite": suite_id,
            "status": "ERROR",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "runtime_sec": 0.0,
            "details": f"CRITICAL: Test file not found."
        }

    try:
        cmd = [sys.executable, "-m", "pytest", str(file_path), "--json-report", "--json-report-file=tmp_report.json", "-q"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = round(time.time() - start_time, 2)
        
        json_path = Path("tmp_report.json")
        if json_path.exists():
            data = json.loads(json_path.read_text())
            json_path.unlink()
            summary = data.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            total = summary.get("total", passed + failed)
            status = "PASS" if res.returncode == 0 and failed == 0 else "FAIL"
            return {
                "suite": suite_id,
                "status": status,
                "tests_run": total,
                "passed": passed,
                "failed": failed,
                "runtime_sec": elapsed,
                "details": f"{passed}/{total} passed."
            }
        else:
            # Fallback when pytest-json-report isn't installed
            status = "PASS" if res.returncode == 0 else "FAIL"
            return {
                "suite": suite_id,
                "status": status,
                "tests_run": -1,
                "passed": -1,
                "failed": -1,
                "runtime_sec": elapsed,
                "details": "Pytest passed (install pytest-json-report for exact test counts)" if status == "PASS" else f"Pytest failed with exit code {res.returncode}"
            }
    except Exception as e:
        return {
            "suite": suite_id,
            "status": "ERROR",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "runtime_sec": round(time.time() - start_time, 2),
            "details": f"Execution exception: {str(e)}"
        }

    try:
        # Run pytest directly on the resolved path
        cmd = [sys.executable, "-m", "pytest", str(file_path), "--json-report", "--json-report-file=tmp_report.json", "-q"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = round(time.time() - start_time, 2)
        
        json_path = Path("tmp_report.json")
        if json_path.exists():
            data = json.loads(json_path.read_text())
            json_path.unlink()
            summary = data.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            total = summary.get("total", passed + failed)
            status = "PASS" if res.returncode == 0 and failed == 0 else "FAIL"
            return {
                "suite": suite_id,
                "status": status,
                "tests_run": total,
                "passed": passed,
                "failed": failed,
                "runtime_sec": elapsed,
                "details": f"{passed}/{total} passed."
            }
        else:
            status = "PASS" if res.returncode == 0 else "FAIL"
            return {
                "suite": suite_id,
                "status": status,
                "tests_run": -1,
                "passed": -1,
                "failed": -1,
                "runtime_sec": elapsed,
                "details": "Pytest executed without pytest-json-report extension."
            }
    except Exception as e:
        return {
            "suite": suite_id,
            "status": "ERROR",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "runtime_sec": round(time.time() - start_time, 2),
            "details": f"Execution exception: {str(e)}"
        }

def run_pipeline():
    print("=" * 60)
    print("      TRACEBIND VALIDATION & RELEASE PIPELINE")
    print("=" * 60)
    
    branch, commit = get_git_info()
    deps = get_deps()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Verify tests/ directory exists or fall back to local test mapping
    suite_targets = {
        "Regression & Determinism": "tests/test_domain_validation.py",
        "Synthetic Autocorrelation": "tests/test_synthetic_autocorrelation.py",
        "Cross-Metric Benchmark": "tests/characterization/test_cross_metric_benchmark.py"
    }
    
    suite_results = []
    overall_pass = True
    
    for name, rel_path in suite_targets.items():
        print(f"Executing: {name} ({rel_path})...")
        res = run_suite(name, rel_path)
        suite_results.append(res)
        if res["status"] in ["FAIL", "ERROR"]:
            overall_pass = False
        print(f" -> Status: {res['status']} ({res['runtime_sec']}s) - {res['details']}")

    # Markdown Report Generation
    report_lines = [
        "# TRACEBIND Automated Validation Report",
        f"**Execution Timestamp:** `{timestamp}`  ",
        f"**Pipeline Result:** `{'PASS' if overall_pass else 'FAIL'}`",
        "",
        "## 1. System & Environment",
        f"- **Git Branch:** `{branch}`",
        f"- **Git Commit:** `{commit}`",
        f"- **Python Version:** `{sys.version.split()[0]}`",
        f"- **Platform:** `{platform.platform()}`",
        "",
        "### Key Dependencies",
    ]
    for dep, ver in deps.items():
        report_lines.append(f"- **{dep}:** `{ver}`")
        
    report_lines.extend([
        "",
        "## 2. Machine-Aggregated Verification Suites",
        "| Suite Name | Status | Tests Run | Runtime (s) | Summary |",
        "| :--- | :---: | :---: | :---: | :--- |"
    ])
    
    for r in suite_results:
        icon = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL/ERROR"
        tests_str = str(r["tests_run"]) if r["tests_run"] >= 0 else "N/A"
        report_lines.append(f"| **{r['suite']}** | {icon} | {tests_str} | {r['runtime_sec']}s | {r['details']} |")
        
    report_lines.append("\n---\n*Generated automatically by `validate_release.py`*")
    
    report_file = Path("VALIDATION_REPORT.md")
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[✓] Validation artifact compiled -> {report_file.resolve()}")
    
    sys.exit(0 if overall_pass else 1)

if __name__ == "__main__":
    run_pipeline()